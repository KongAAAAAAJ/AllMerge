"""V2.2.1 regression tests for one shared Loss switch across A0/A1/A2.

This test is deliberately self-contained apart from stable fixtures already
provided by the V2.1 focused test module. It does not assume private helpers
that are absent from that module.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
import yaml

# Reuse only helpers that are actually exported by the V2.1 test module.
# Importing it also installs the optional gymnasium registration shim.
from tests.test_trace_anchor_consistent_loss import (
    CTRL,
    SMALL,
    _features,
    _loss_config,
)
from highway_env.planner.diffusion.config.experiment import load_mode_yaml
from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner
from highway_env.vehicle.koopman.models.conditional_model import ConditionalKoopman

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = (
    ("configs/train/ablation/diffusion_a0_fixed.yaml", "fixed"),
    ("configs/train/diffusion.yaml", "a1"),
    ("configs/train/ablation/diffusion_a2_adaptive.yaml", "adaptive"),
)


def _k2_checkpoint(path: Path) -> None:
    """Create a tiny deterministic-shape K2 checkpoint for CPU regression tests."""
    norm = dict(
        state_mean=[0.0] * 5,
        state_std=[1.0] * 5,
        control_mean=[0.0] * 2,
        control_std=[1.0] * 2,
    )
    cond = dict(
        condition_mean=[3500.0, 0.6],
        condition_std=[1000.0, 0.1],
    )
    kwargs = dict(history_steps=5, latent_dim=8, hidden_dim=16)
    model = ConditionalKoopman(
        "K2", condition_hidden=12, **norm, **cond, **kwargs
    )
    torch.save(
        {
            "model_type": "K2",
            "model_kwargs": kwargs,
            "normalization": norm,
            "condition_normalization": cond,
            "condition_hidden": 12,
            "model_state": model.state_dict(),
        },
        path,
    )


def _context() -> dict:
    """Return the physical-history fields consumed by VehicleExecution/K2."""
    state = torch.zeros((1, 5, 5), dtype=torch.float32)
    state[:, :, 0] = 20.0
    return {
        "koopman_state_hist_si": state,
        "koopman_control_hist_si": torch.zeros((1, 4, 2), dtype=torch.float32),
        "koopman_pose0_si": torch.zeros((1, 3), dtype=torch.float32),
        "koopman_payload_kg": torch.tensor([3500.0], dtype=torch.float32),
        "koopman_road_mu": torch.tensor([0.6], dtype=torch.float32),
    }


@pytest.mark.parametrize(("config_path", "expected_strategy"), STRATEGIES)
def test_shared_yaml_switch_inherits_into_a0_a1_a2(
    tmp_path, config_path, expected_strategy
):
    baseline = load_mode_yaml(PROJECT_ROOT / config_path)
    assert baseline["anchors"]["strategy"] == expected_strategy
    assert baseline["losses"]["training_scheme"] == "goal_endpoint"
    assert baseline["losses"]["anchor_consistency"]["weight"] > 0
    assert "selected_mode_scale" not in baseline["losses"]["anchor_consistency"]
    assert baseline["losses"]["fuel"]["relative_weight"] > 0

    # Copy the config tree, change exactly one shared switch, and verify that
    # all three strategy configs inherit it through the existing extends tree.
    config_root = tmp_path / "configs"
    shutil.copytree(PROJECT_ROOT / "configs", config_root)
    shared = config_root / "train" / "diffusion.yaml"
    shared_data = yaml.safe_load(shared.read_text(encoding="utf-8"))
    shared_data["control"]["losses"]["training_scheme"] = "anchor_consistent"
    shared.write_text(
        yaml.safe_dump(shared_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    for relative, strategy in STRATEGIES:
        resolved = load_mode_yaml(config_root.parent / relative)
        assert resolved["anchors"]["strategy"] == strategy
        assert resolved["losses"]["training_scheme"] == "anchor_consistent"
        assert resolved["losses"]["anchor_consistency"]["weight"] > 0
        assert "selected_mode_scale" not in resolved["losses"]["anchor_consistency"]
        assert resolved["losses"]["fuel"]["relative_weight"] > 0


@pytest.mark.parametrize(
    ("strategy", "runtime_kind"),
    (("fixed", "a0"), ("a1", "a1"), ("adaptive", "a2")),
)
def test_anchor_consistent_forward_accepts_all_anchor_manager_contracts(
    tmp_path, monkeypatch, strategy, runtime_kind
):
    """All manager strategies expose the same [B,M,40,2] control contract."""
    import highway_env.planner.fixed_anchor_manager as fixed_module
    import highway_env.planner.a1_anchor_manager as a1_module
    import highway_env.planner.anchor_scoring.manager as a2_module

    class FakeAnchorManager:
        def __init__(self, config, **kwargs):
            self.config = dict(config)
            self.runtime_kind = runtime_kind
            self.bank_sha256 = "test-fixed-anchor-bank-sha256"

        @torch.no_grad()
        def select(self, features):
            batch = features["ego_state"].shape[0]
            modes = 10
            device = features["ego_state"].device
            dtype = features["ego_state"].dtype
            controls = torch.zeros((batch, modes, 40, 2), device=device, dtype=dtype)
            accelerations = torch.linspace(-0.5, 0.5, modes, device=device, dtype=dtype)
            controls[:, :, :, 0] = accelerations.view(1, modes, 1)
            coarse = torch.zeros((batch, modes, 8, 2), device=device, dtype=dtype)
            valid = torch.ones((batch, modes), device=device, dtype=torch.bool)
            selected = dict(features)
            selected["coarse_controls"] = controls
            selected["coarse_trajectories"] = coarse
            selected["mode_valid_mask"] = valid
            return selected

    monkeypatch.setattr(fixed_module, "FixedAnchorManager", FakeAnchorManager)
    monkeypatch.setattr(a1_module, "A1AnchorManager", FakeAnchorManager)
    monkeypatch.setattr(a2_module, "AdaptiveAnchorManager", FakeAnchorManager)

    checkpoint = tmp_path / "k2.pt"
    _k2_checkpoint(checkpoint)
    model = ControlDiffusionPlanner(
        SMALL,
        CTRL,
        execution_config={"model": "koopman_k2", "checkpoint": str(checkpoint)},
        loss_config=_loss_config(),
        anchors_config={"strategy": strategy},
    )
    model._validate_objectives()
    assert model.anchor_manager.runtime_kind == runtime_kind
    assert all(
        not parameter.requires_grad
        for parameter in model.vehicle_model.koopman.parameters()
    )

    features = _features()
    expert = torch.zeros((1, 40, 2), dtype=torch.float32)
    expert[0, :, 0] = torch.arange(1, 41, dtype=torch.float32) * 2.0
    platoon = torch.zeros((1, 2, 41, 3), dtype=torch.float32)
    platoon[:, :, :, 0] = 500.0
    result = model.forward_train(
        features,
        expert,
        {"platoon_reference_poses": platoon},
        generator=torch.Generator(device="cpu").manual_seed(123),
        context=_context(),
    )

    for key in (
        "loss", "trajectory_loss", "anchor_consistency_loss",
        "anchor_execution_gap_m", "fuel_relative_loss",
        "fuel_relative_ratio", "fuel_improvement_pct",
    ):
        assert torch.isfinite(result[key]), f"{strategy}: {key}"
    assert result["anchor_consistency_loss"].item() >= 0.0
    assert result["fuel_relative_ratio"].item() > 0.0
    result["loss"].backward()
    grads = [
        parameter.grad
        for parameter in model.denoiser.parameters()
        if parameter.grad is not None
    ]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert all(
        parameter.grad is None
        for parameter in model.vehicle_model.koopman.parameters()
    )
