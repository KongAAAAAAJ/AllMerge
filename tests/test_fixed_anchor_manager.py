"""A0 direct control-space K-means tests."""
from __future__ import annotations

import json
import sys
import types

# The focused planner tests need only gymnasium's registration symbol. Keep
# this local stub so the test module remains runnable in lightweight CI images.
try:
    import gymnasium  # noqa: F401
except ImportError:
    gymnasium = types.ModuleType("gymnasium")
    envs = types.ModuleType("gymnasium.envs")
    registration = types.ModuleType("gymnasium.envs.registration")
    registration.register = lambda *args, **kwargs: None
    envs.registration = registration
    gymnasium.envs = envs
    sys.modules["gymnasium"] = gymnasium
    sys.modules["gymnasium.envs"] = envs
    sys.modules["gymnasium.envs.registration"] = registration

import numpy as np
import pytest
import torch

from scripts.data.build_a0_fixed_anchors import build_fixed_anchor_bank
from highway_env.planner.fixed_anchor_manager import FixedAnchorManager
from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner
from highway_env.planner.diffusion.training.control import load_control_checkpoint


def _metadata():
    return {
        "implementation": "A0_FIXED_CONTROL_KMEANS_EXEMPLARS_V4",
        "num_anchors": 10,
        "source_split": "train",
        "cluster_space": "flattened_40_step_physical_control_sequence",
        "prototype_type": "nearest_expert_to_cluster_centroid",
        "medoid_selection_used": False,
        "trajectory_to_control_matching_used": False,
        "mode_valid_mask_used": False,
        "safety_scoring_used": False,
        "adaptive_selection_used": False,
    }


def _write_bank(path):
    rng = np.random.default_rng(10)
    anchors = rng.normal(0.0, 0.1, (10, 40, 2)).astype(np.float32)
    anchors[..., 0] *= 0.8
    anchors[..., 1] *= 0.04
    scale = np.asarray([2.5, 0.08], dtype=np.float32)
    metadata = _metadata()
    metadata_text = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    np.savez_compressed(
        path,
        anchors_controls=anchors,
        cluster_centers_controls=anchors,
        control_scale=scale,
        metadata_json=np.asarray(metadata_text),
    )
    return anchors


def _scene(batch=2):
    maps = torch.zeros((batch, 8, 32, 10), dtype=torch.float32)
    mask = torch.zeros((batch, 8), dtype=torch.bool)
    for lane in range(3):
        maps[:, lane, :, 0] = torch.linspace(0.0, 120.0, 32)
        maps[:, lane, :, 1] = lane * 3.7
        maps[:, lane, :, 4] = 3.7
        maps[:, lane, :, 5] = 33.0
        maps[:, lane, :, 6 + lane] = 1.0
        mask[:, lane] = True
    ego = torch.zeros((batch, 14), dtype=torch.float32)
    ego[:, 0] = torch.tensor([20.0, 15.0][:batch])
    features = {
        "ego_state": ego,
        "agent_states": torch.zeros((batch, 16, 11), dtype=torch.float32),
        "agent_valid_mask": torch.zeros((batch, 16), dtype=torch.bool),
        "map_polylines": maps,
        "map_valid_mask": mask,
        "target_point": torch.tensor([[80.0, 0.0]]).expand(batch, -1).clone(),
        "target_lane_polyline": maps[:, 0].clone(),
        "coarse_controls": torch.zeros((batch, 10, 40, 2), dtype=torch.float32),
        "coarse_trajectories": torch.zeros((batch, 10, 8, 2), dtype=torch.float32),
        "mode_valid_mask": torch.zeros((batch, 10), dtype=torch.bool),
    }
    return features


def test_fixed_anchor_manager_broadcasts_controls_and_rolls_out_per_scene(tmp_path):
    bank_path = tmp_path / "a0.npz"
    anchors = _write_bank(bank_path)
    manager = FixedAnchorManager(
        {"strategy": "fixed", "num_selected": 10, "bank_path": str(bank_path)},
        num_modes=10, dt=0.1, wheelbase_m=5.5,
    )
    features = _scene(batch=2)
    selected = manager.select(features)
    assert selected["coarse_controls"].shape == (2, 10, 40, 2)
    assert selected["coarse_trajectories"].shape == (2, 10, 8, 2)
    assert selected["mode_valid_mask"].shape == (2, 10)
    assert selected["mode_valid_mask"].all()
    torch.testing.assert_close(selected["coarse_controls"][0], torch.from_numpy(anchors))
    torch.testing.assert_close(selected["coarse_controls"][1], selected["coarse_controls"][0])
    assert not torch.allclose(selected["coarse_trajectories"][0], selected["coarse_trajectories"][1])
    assert not features["mode_valid_mask"].any()  # source features are not mutated


def test_fixed_control_strategy_training_inference_and_checkpoint_reload(tmp_path):
    bank_path = tmp_path / "a0.npz"
    _write_bank(bank_path)
    anchors_cfg = {"strategy": "fixed", "num_selected": 10, "bank_path": str(bank_path)}
    model_config = {
        "num_modes": 10, "max_agents": 16, "max_map_polylines": 8,
        "map_points": 32, "d_model": 32, "d_ffn": 64,
        "num_heads": 4, "num_scene_layers": 0,
        "num_denoiser_layers": 1, "dropout": 0.0,
    }
    loss_config = {
        "mode_selection": "goal_endpoint",
        "trajectory": {"weight": 1.0},
        "background_collision": {"weight": 1.0},
        "intertruck_collision": {"weight": 1.0},
        "ttc": {"weight": 1.0},
        "road_boundary": {"weight": 1.0},
        "safety": {
            "vehicle_length_m": 8.0, "vehicle_width_m": 2.6,
            "collision_margin_m": 0.2, "ttc_threshold_s": 1.0,
            "softplus_temperature_m": 0.2, "collision_distance_scale_m": 1.0,
            "ttc_distance_scale_m": 5.0, "road_distance_scale_m": 1.0,
        },
        "ltr": {"weight": 0.0, "limit": 0.6},
        "fuel": {"weight": 0.0},
        "progress": {"weight": 0.0, "minimum_progress_m": 60.0},
    }
    model = ControlDiffusionPlanner(
        model_config=model_config,
        control_config={"inference_steps": 2, "inference_start_timestep": 8},
        execution_config={"model": "kinematic"},
        loss_config=loss_config,
        anchors_config=anchors_cfg,
    )
    features = _scene(batch=2)
    from tests.test_a1_online_runtime import _live_geometry
    features.update({key: torch.as_tensor(value) for key, value in _live_geometry(batch=2).items()})
    target = torch.zeros((2, 40, 2), dtype=torch.float32)
    target[:, :, 0] = torch.arange(1, 41, dtype=torch.float32)[None, :] * 2.0
    platoon_reference = torch.zeros((2, 2, 41, 3), dtype=torch.float32)
    platoon_reference[:, :, :, 0] = 500.0
    train_out = model.forward_train(
        features, target, {"platoon_reference_poses": platoon_reference}
    )
    assert torch.isfinite(train_out["loss"])
    assert train_out["controls"].shape == (2, 40, 2)
    expected_goal_ade = torch.linalg.vector_norm(
        train_out["poses"][:, 1:, :2] - target, dim=-1
    ).mean()
    torch.testing.assert_close(train_out["goal_selected_ade"], expected_goal_ade)
    infer_out = model.infer_multimodal(features)
    assert infer_out["control_candidates"].shape == (2, 10, 40, 2)
    assert infer_out["trajectory_candidates"].shape == (2, 10, 8, 2)

    checkpoint = tmp_path / "a0_control.pt"
    torch.save({
        "model_config": model_config,
        "control_config": {"inference_steps": 2, "inference_start_timestep": 8},
        "execution_config": {"model": "kinematic"},
        "loss_config": loss_config,
        "koopman_metadata": None,
        "anchors_config": model.anchors_config,
        "model_state": model.state_dict(),
    }, checkpoint)
    loaded, payload = load_control_checkpoint(checkpoint, device="cpu")
    assert payload["anchors_config"]["strategy"] == "fixed"
    assert loaded.anchors_config["bank_sha256"] == model.anchors_config["bank_sha256"]
    torch.testing.assert_close(
        loaded._prepare_anchors(features)["coarse_controls"],
        model._prepare_anchors(features)["coarse_controls"],
    )


def test_fixed_manager_rejects_hash_mismatch_and_non_centroid_bank(tmp_path):
    bank_path = tmp_path / "a0.npz"
    _write_bank(bank_path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        FixedAnchorManager(
            {"bank_path": str(bank_path), "num_selected": 10, "bank_sha256": "0" * 64}
        )
    with np.load(bank_path, allow_pickle=False) as data:
        anchors = data["anchors_controls"].copy()
        scale = data["control_scale"].copy()
    metadata = _metadata()
    metadata["prototype_type"] = "medoid"
    metadata_text = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    np.savez_compressed(bank_path, anchors_controls=anchors,
        cluster_centers_controls=anchors, control_scale=scale,
        metadata_json=np.asarray(metadata_text))
    with pytest.raises(ValueError, match="observed expert cluster representatives"):
        FixedAnchorManager({"bank_path": str(bank_path), "num_selected": 10})
