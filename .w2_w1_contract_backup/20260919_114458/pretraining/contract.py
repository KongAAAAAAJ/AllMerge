"""Cross-window contract checks for W1 dataset and the frozen AllMerge planner."""
from __future__ import annotations

from typing import Mapping

import torch

from data_pipeline.feature_schema import FEATURE_KEYS, MASK_KEYS, SCHEMA


def validate_model_schema(model_config) -> None:
    pairs = {
        "ego_dim": SCHEMA.d_ego,
        "agent_dim": SCHEMA.d_agent,
        "map_dim": SCHEMA.d_map,
        "max_agents": SCHEMA.n_agent,
        "max_map_polylines": SCHEMA.n_map,
        "map_points": SCHEMA.map_points,
        "num_modes": SCHEMA.n_modes,
        "horizon_steps": SCHEMA.horizon_steps,
    }
    mismatches = []
    for field, expected in pairs.items():
        actual = int(getattr(model_config, field))
        if actual != int(expected):
            mismatches.append(f"{field}={actual}, dataset={expected}")
    if abs(float(model_config.trajectory_dt) - float(SCHEMA.trajectory_dt)) > 1e-9:
        mismatches.append(
            f"trajectory_dt={model_config.trajectory_dt}, dataset={SCHEMA.trajectory_dt}"
        )
    if mismatches:
        raise RuntimeError("Planner/W1 dataset contract mismatch: " + "; ".join(mismatches))


def validate_training_batch(batch: Mapping) -> int:
    required = {"features", "expert_trajectory", "expert_mode", "expert_semantic", "metadata"}
    missing = sorted(required.difference(batch.keys()))
    if missing:
        raise KeyError(f"W1 batch missing keys: {missing}")

    features = batch["features"]
    missing_features = [key for key in FEATURE_KEYS if key not in features]
    if missing_features:
        raise KeyError(f"W1 features missing keys: {missing_features}")

    batch_size = int(features["ego_state"].shape[0])
    expected_shapes = {
        key: (batch_size, *SCHEMA.feature_shapes[key])
        for key in FEATURE_KEYS
    }
    for key, expected in expected_shapes.items():
        value = features[key]
        if tuple(value.shape) != expected:
            raise ValueError(f"features.{key}: shape={tuple(value.shape)}, expected={expected}")
        expected_dtype = torch.bool if key in MASK_KEYS else torch.float32
        if value.dtype != expected_dtype:
            raise TypeError(f"features.{key}: dtype={value.dtype}, expected={expected_dtype}")

    target = batch["expert_trajectory"]
    if tuple(target.shape) != (batch_size, SCHEMA.horizon_steps, 2):
        raise ValueError(
            f"expert_trajectory: shape={tuple(target.shape)}, expected={(batch_size, SCHEMA.horizon_steps, 2)}"
        )
    if target.dtype != torch.float32:
        raise TypeError(f"expert_trajectory dtype={target.dtype}, expected=torch.float32")
    if not torch.isfinite(target).all():
        raise ValueError("expert_trajectory contains NaN/Inf")

    for key in ("expert_mode", "expert_semantic"):
        value = batch[key]
        if tuple(value.shape) != (batch_size,):
            raise ValueError(f"{key}: shape={tuple(value.shape)}, expected={(batch_size,)}")
        if value.dtype != torch.long:
            raise TypeError(f"{key}: dtype={value.dtype}, expected=torch.long")

    return batch_size
