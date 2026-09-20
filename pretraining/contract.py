"""Contract checks between the completed W1 dataset and frozen planner."""
from __future__ import annotations

from typing import Mapping

import torch

from expert_dataset import BOOL_FEATURE_KEYS, FEATURE_KEYS
from .dataset_adapter import unpack_w1_batch


def validate_model_schema(model_config) -> None:
    if int(model_config.horizon_steps) != 8:
        raise RuntimeError(f"Expected 8-step horizon, got {model_config.horizon_steps}")
    if int(model_config.num_modes) != 10:
        raise RuntimeError(f"Expected 10 modes, got {model_config.num_modes}")
    if abs(float(model_config.trajectory_dt) - 0.5) > 1e-9:
        raise RuntimeError(f"Expected trajectory_dt=0.5, got {model_config.trajectory_dt}")


def validate_training_batch(batch, model_config) -> int:
    normalized = unpack_w1_batch(batch)
    features: Mapping = normalized["features"]
    missing = [key for key in FEATURE_KEYS if key not in features]
    if missing:
        raise KeyError(f"W1 features missing keys: {missing}")

    batch_size = int(features["ego_state"].shape[0])
    expected_shapes = {
        "ego_state": (batch_size, int(model_config.ego_dim)),
        "agent_states": (
            batch_size,
            int(model_config.max_agents),
            int(model_config.agent_dim),
        ),
        "agent_valid_mask": (batch_size, int(model_config.max_agents)),
        "map_polylines": (
            batch_size,
            int(model_config.max_map_polylines),
            int(model_config.map_points),
            int(model_config.map_dim),
        ),
        "map_valid_mask": (batch_size, int(model_config.max_map_polylines)),
        "target_point": (batch_size, 2),
        "target_lane_polyline": (
            batch_size,
            int(model_config.map_points),
            int(model_config.map_dim),
        ),
        "coarse_trajectories": (
            batch_size,
            int(model_config.num_modes),
            int(model_config.horizon_steps),
            2,
        ),
        "mode_valid_mask": (batch_size, int(model_config.num_modes)),
    }
    for key, expected in expected_shapes.items():
        value = features[key]
        if tuple(value.shape) != expected:
            raise ValueError(f"features.{key}: shape={tuple(value.shape)}, expected={expected}")
        expected_dtype = torch.bool if key in BOOL_FEATURE_KEYS else torch.float32
        if value.dtype != expected_dtype:
            raise TypeError(f"features.{key}: dtype={value.dtype}, expected={expected_dtype}")

    target = normalized["expert_trajectory"]
    expected_target = (batch_size, int(model_config.horizon_steps), 2)
    if tuple(target.shape) != expected_target:
        raise ValueError(f"trajectory: shape={tuple(target.shape)}, expected={expected_target}")
    if target.dtype != torch.float32 or not torch.isfinite(target).all():
        raise ValueError("trajectory must be finite torch.float32")

    for key in ("expert_mode", "expert_semantic"):
        value = normalized[key]
        if tuple(value.shape) != (batch_size,) or value.dtype != torch.long:
            raise TypeError(
                f"{key}: shape={tuple(value.shape)} dtype={value.dtype}; "
                f"expected {(batch_size,)} torch.long"
            )
    return batch_size


# STAGED_DENSE_SUPERVISION_V1
def validate_dense_supervision_batch(
    batch,
    model_config,
    *,
    dense_dt: float = 0.1,
) -> int:
    normalized = unpack_w1_batch(batch)
    dense = normalized.get("expert_trajectory_dense")
    if dense is None:
        raise ValueError(
            "Dense loss is enabled, but this dataset does not contain the real "
            "Stage-3 10 Hz trajectory_dense target"
        )
    batch_size = int(normalized["features"]["ego_state"].shape[0])
    horizon_s = float(model_config.horizon_steps) * float(model_config.trajectory_dt)
    dense_steps = int(round(horizon_s / float(dense_dt)))
    expected = (batch_size, dense_steps, 2)
    if tuple(dense.shape) != expected:
        raise ValueError(
            f"trajectory_dense: shape={tuple(dense.shape)}, expected={expected}"
        )
    if dense.dtype != torch.float32 or not torch.isfinite(dense).all():
        raise ValueError("trajectory_dense must be finite torch.float32")

    dense_dt_tensor = normalized.get("dense_dt")
    horizon_tensor = normalized.get("trajectory_horizon_s")
    if dense_dt_tensor is None or horizon_tensor is None:
        raise ValueError(
            "Dense loss requires dense_dt and trajectory_horizon_s from Stage 3"
        )
    expected_dt = torch.full_like(dense_dt_tensor.float(), float(dense_dt))
    expected_horizon = torch.full_like(horizon_tensor.float(), horizon_s)
    if not torch.allclose(dense_dt_tensor.float(), expected_dt, atol=1e-6, rtol=0.0):
        raise ValueError(
            f"dense_dt must be {float(dense_dt):.6g}s for Stage D"
        )
    if not torch.allclose(horizon_tensor.float(), expected_horizon, atol=1e-6, rtol=0.0):
        raise ValueError(
            f"trajectory_horizon_s must be {horizon_s:.6g}s for Stage D"
        )
    return batch_size
