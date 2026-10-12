"""AR V1 smoke: run from repo root: python test_ar_v1.py.
Requires normal AllMerge project dependencies; no expert dataset/GPU needed.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from highway_env.planner.diffusion.autoregressive import StructuredAutoregressivePlanner
from highway_env.planner.diffusion.config import build_structured_diffusion_config
from highway_env.planner.diffusion.runtime import DiffusionPlannerRuntime
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter
from pretraining.checkpoint_io import export_runtime_checkpoint


def synthetic_features(config, batch=2):
    device = torch.device("cpu")
    torch.manual_seed(1337)
    t = torch.arange(1, config.horizon_steps + 1).float() * config.trajectory_dt
    anchors = torch.zeros(batch, config.num_modes, config.horizon_steps, 2)
    anchors[..., 0] = t * 15.0
    anchors[..., 1] = torch.arange(config.num_modes).float()[None, :, None] * 0.2
    f = {
        "ego_state": torch.zeros(batch, config.ego_dim),
        "agent_states": torch.zeros(batch, config.max_agents, config.agent_dim),
        "agent_valid_mask": torch.ones(batch, config.max_agents, dtype=torch.bool),
        "map_polylines": torch.zeros(batch, config.max_map_polylines, config.map_points, config.map_dim),
        "map_valid_mask": torch.ones(batch, config.max_map_polylines, dtype=torch.bool),
        "target_point": torch.zeros(batch, 2),
        "target_lane_polyline": torch.zeros(batch, config.map_points, config.map_dim),
        "coarse_trajectories": anchors,
        "mode_valid_mask": torch.ones(batch, config.num_modes, dtype=torch.bool),
    }
    return f


def main():
    torch.set_num_threads(1)
    config = build_structured_diffusion_config(
        d_model=32, d_ffn=64, num_denoiser_layers=1,
        num_scene_layers=0, dropout=0.0, reg_head_type="gru",
    )
    adapter = PlannerTensorAdapter(config, torch.device("cpu"))
    torch.manual_seed(4)
    baseline = StructuredDiffusionPlanner(config, adapter)
    torch.manual_seed(4)
    ar = StructuredAutoregressivePlanner(config, adapter)
    assert list(baseline.state_dict()) == list(ar.state_dict())
    assert all(torch.equal(a, b) for a, b in
               zip(baseline.state_dict().values(), ar.state_dict().values()))
    assert sum(p.numel() for p in baseline.parameters()) == sum(p.numel() for p in ar.parameters())

    f = synthetic_features(config)
    target = f["coarse_trajectories"][:, 0, :].clone()
    sem = torch.zeros(2, dtype=torch.long)
    out = ar.forward_train(f, target, sem)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert ar.denoiser.reg_head[1].weight.grad is not None
    assert ar.denoiser.reg_head[1].weight.grad.abs().sum() > 0
    ar.zero_grad(set_to_none=True)

    # Dense V1 is still the original runtime-consistent correction objective.
    dense = torch.zeros(2, 40, 2)
    dense_out = ar.forward_train(
        f, target, sem, target_trajectory_dense=dense,
        dense_loss_enabled=True, dense_loss_lambda_p=0.1,
    )
    dense_out["loss"].backward()
    assert ar.dense_residual_head.net[-1].weight.grad is not None
    assert ar.dense_residual_head.net[-1].weight.grad.abs().sum() > 0

    ar.eval()
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    group1 = ar.sample_group(f, group_size=3, generator=g1)
    group2 = ar.sample_group(f, group_size=3, generator=g2)
    assert group1["candidates"].shape == (2, 3, config.num_modes, 8, 2)
    assert group1["step_log_prob"].shape == (2, 3, config.num_modes, 8)
    assert torch.equal(group1["candidates"], group2["candidates"])
    lp = ar.joint_trajectory_log_prob(f, group1["candidates"][:, 0])
    expected_lp = group1["step_log_prob"][:, 0].sum(dim=(-1, -2))
    assert lp.shape == (2,) and torch.isfinite(lp).all()
    assert torch.allclose(lp, expected_lp, rtol=1e-4, atol=1e-3), (lp, expected_lp)

    with tempfile.TemporaryDirectory() as tmp:
        training_path = Path(tmp) / "training.pt"
        runtime_path = Path(tmp) / "runtime.pt"
        torch.save({
            "planner_state_dict": ar.state_dict(),
            "allmerge_model_config": {
                "d_model": 32, "d_ffn": 64, "num_denoiser_layers": 1,
                "num_scene_layers": 0, "dropout": 0.0, "reg_head_type": "gru",
            },
            "generator_type": "ar_v1", "ar_sigma_xy_m": [2.0, 0.5],
        }, training_path)
        export_runtime_checkpoint(training_path, runtime_path)
        runtime = DiffusionPlannerRuntime({
            "checkpoint": str(runtime_path), "device": "cpu",
            "allow_random_weights": False, "strict_checkpoint": True,
        })
        assert runtime.generator_type == "ar_v1"
        pred = runtime.model.infer_multimodal(f, generator=torch.Generator().manual_seed(4))
        assert pred["trajectory_execution_candidates"].shape == (2, 10, 8, 2)
    print("[PASS] AR V1: identical network, NLL/GRU gradients, dense gradients, G3 reproducibility, joint log-prob, runtime checkpoint")


if __name__ == "__main__":
    main()
