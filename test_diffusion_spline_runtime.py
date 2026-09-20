# STAGE_A_SPLINE_RUNTIME_V1
from __future__ import annotations

import numpy as np
import torch

from highway_env.planner.diffusion.runtime import DiffusionPlannerRuntime


def _features(batch=2, modes=10, horizon=8):
    coarse = np.zeros((batch, modes, horizon, 2), dtype=np.float32)
    t = np.arange(1, horizon + 1, dtype=np.float32) * 0.5
    for b in range(batch):
        for m in range(modes):
            coarse[b, m, :, 0] = (10.0 + b) * t
            coarse[b, m, :, 1] = (m - modes // 2) * 0.15 * t

    ego_state = np.zeros((batch, 14), dtype=np.float32)
    ego_state[:, 0] = 10.0
    agent_mask = np.zeros((batch, 16), dtype=bool)
    agent_mask[:, 0] = True
    map_mask = np.zeros((batch, 8), dtype=bool)
    map_mask[:, 0] = True

    return {
        "ego_state": ego_state,
        "agent_states": np.zeros((batch, 16, 11), dtype=np.float32),
        "agent_valid_mask": agent_mask,
        "map_polylines": np.zeros((batch, 8, 32, 10), dtype=np.float32),
        "map_valid_mask": map_mask,
        "target_point": np.zeros((batch, 2), dtype=np.float32),
        "target_lane_polyline": np.zeros((batch, 32, 10), dtype=np.float32),
        "coarse_trajectories": coarse,
        "mode_valid_mask": np.ones((batch, modes), dtype=bool),
    }


def test_runtime_selected_trajectory_spline_postprocess_only():
    torch.manual_seed(123)
    runtime = DiffusionPlannerRuntime({
        "device": "cpu",
        "allow_random_weights": True,
        "deterministic_seed": 9,
        "diffusion_spline_enabled": True,
        "diffusion_execution_enabled": False,
        "spline_dense_dt": 0.1,
        "model": {
            "d_model": 32,
            "d_ffn": 64,
            "num_heads": 4,
            "num_scene_layers": 1,
            "num_denoiser_layers": 1,
        },
    })
    features = _features()
    enabled = runtime.infer(features)

    assert enabled["trajectory"].shape == (2, 8, 2)
    assert enabled["trajectory_dense"].shape == (2, 41, 2)
    assert enabled["trajectory_dense_velocity"].shape == (2, 41, 2)
    assert enabled["trajectory_dense_acceleration"].shape == (2, 41, 2)
    assert enabled["trajectory_dense_time_s"].shape == (41,)
    assert float(enabled["spline_max_abs_waypoint_interpolation_error"]) < 1e-5

    runtime.diffusion_spline_enabled = False
    disabled = runtime.infer(features)
    assert "trajectory_dense" not in disabled
    np.testing.assert_allclose(
        enabled["trajectory"],
        disabled["trajectory"],
        atol=1e-6,
        rtol=0.0,
    )
