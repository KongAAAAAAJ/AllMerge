from __future__ import annotations

import torch

from highway_env.planner.diffusion.config import build_structured_diffusion_config
from highway_env.planner.diffusion.diffusion_schedule import TruncatedDDIMSchedule
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter


def _features(cfg):
    batch = 2
    coarse = torch.randn(batch, cfg.num_modes, cfg.horizon_steps, 2)
    map_polylines = torch.zeros(
        batch,
        cfg.max_map_polylines,
        cfg.map_points,
        cfg.map_dim,
    )
    # Mark one geometrically existing current lane for semantic KEEP assignment.
    map_polylines[:, 0, :, 6] = 1.0
    return {
        "ego_state": torch.zeros(batch, cfg.ego_dim),
        "agent_states": torch.zeros(batch, cfg.max_agents, cfg.agent_dim),
        "agent_valid_mask": torch.ones(batch, cfg.max_agents, dtype=torch.bool),
        "map_polylines": map_polylines,
        "map_valid_mask": torch.ones(batch, cfg.max_map_polylines, dtype=torch.bool),
        "target_point": torch.zeros(batch, 2),
        "target_lane_polyline": map_polylines[:, 0].clone(),
        "coarse_trajectories": coarse,
        "mode_valid_mask": torch.ones(batch, cfg.num_modes, dtype=torch.bool),
    }


def test_schedule_epsilon_x0_roundtrip():
    torch.manual_seed(7)
    schedule = TruncatedDDIMSchedule(num_train_timesteps=1000)
    x0 = torch.randn(3, 10, 8, 2)
    epsilon = torch.randn_like(x0)
    timesteps = torch.tensor([0, 8, 49], dtype=torch.long)
    xt = schedule.add_noise(x0, epsilon, timesteps)

    x0_hat = schedule.predict_x0_from_epsilon(xt, epsilon, timesteps)
    epsilon_hat = schedule.predict_epsilon_from_x0(xt, x0, timesteps)

    torch.testing.assert_close(x0_hat, x0, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(epsilon_hat, epsilon, atol=2e-5, rtol=1e-5)


def test_epsilon_training_and_runtime_smoke():
    torch.manual_seed(11)
    cfg = build_structured_diffusion_config(
        prediction_type="epsilon",
        reg_head_type="gru",
        d_model=32,
        d_ffn=64,
        num_heads=4,
        num_scene_layers=1,
        num_denoiser_layers=1,
        dropout=0.0,
    )
    adapter = PlannerTensorAdapter(cfg, torch.device("cpu"))
    model = StructuredDiffusionPlanner(cfg, adapter)
    features = _features(cfg)
    target = features["coarse_trajectories"][:, 0].clone()
    semantic = torch.zeros(target.shape[0], dtype=torch.long)

    out = model.forward_train(features, target, target_semantic=semantic)
    assert torch.isfinite(out["loss"])
    assert float(out["noise_prediction_loss"].detach()) > 0.0
    torch.testing.assert_close(out["prediction_loss"], out["noise_prediction_loss"])

    with torch.no_grad():
        pred = model.infer_multimodal(features)
    assert pred["trajectory_candidates"].shape == (
        target.shape[0],
        cfg.num_modes,
        cfg.horizon_steps,
        2,
    )
    assert torch.isfinite(pred["trajectory_candidates"]).all()


def test_sample_path_keeps_original_objective():
    torch.manual_seed(13)
    cfg = build_structured_diffusion_config(
        prediction_type="sample",
        reg_head_type="gru",
        d_model=32,
        d_ffn=64,
        num_heads=4,
        num_scene_layers=1,
        num_denoiser_layers=1,
        dropout=0.0,
    )
    adapter = PlannerTensorAdapter(cfg, torch.device("cpu"))
    model = StructuredDiffusionPlanner(cfg, adapter)
    features = _features(cfg)
    target = features["coarse_trajectories"][:, 0].clone()
    semantic = torch.zeros(target.shape[0], dtype=torch.long)

    out = model.forward_train(features, target, target_semantic=semantic)
    assert float(out["noise_prediction_loss"].detach()) == 0.0
    torch.testing.assert_close(out["prediction_loss"], out["x0_reconstruction_loss"])
