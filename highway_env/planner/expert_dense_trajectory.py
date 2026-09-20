from __future__ import annotations

from typing import Dict

import numpy as np

from highway_env.planner.geometry import world_to_ego_point


DENSE_DT_S = 0.1
DENSE_HORIZON_S = 4.0
SPARSE_DT_S = 0.5
SPARSE_STEPS = 8
DENSE_STEPS = 40


def _exact_time_indices(
    source_time_s: np.ndarray,
    target_time_s: np.ndarray,
    *,
    tolerance_s: float = 2e-4,
) -> np.ndarray:
    """Locate target timestamps in a native source grid without interpolation."""
    source_time_s = np.asarray(source_time_s, dtype=np.float64).reshape(-1)
    target_time_s = np.asarray(target_time_s, dtype=np.float64).reshape(-1)

    if source_time_s.ndim != 1 or source_time_s.size < 2:
        raise ValueError("source_time_s must be a 1-D grid with at least 2 points")
    if not np.isfinite(source_time_s).all():
        raise ValueError("source_time_s contains NaN/Inf")
    if np.any(np.diff(source_time_s) <= 0.0):
        raise ValueError("source_time_s must be strictly increasing")

    indices = []
    for target in target_time_s:
        matches = np.flatnonzero(
            np.isclose(source_time_s, target, atol=tolerance_s, rtol=0.0)
        )
        if matches.size != 1:
            raise ValueError(
                "Native 10 Hz expert path does not contain required timestamp "
                f"t={float(target):.3f}s (matches={matches.tolist()}). "
                "Stage 3 forbids interpolating sparse expert points to fake a dense target."
            )
        indices.append(int(matches[0]))

    return np.asarray(indices, dtype=np.int64)


def build_native_dense_expert_targets(
    *,
    source_time_s,
    world_xy,
    ego_position_world,
    ego_heading_world: float,
    horizon_steps: int = SPARSE_STEPS,
    sparse_dt: float = SPARSE_DT_S,
    dense_dt: float = DENSE_DT_S,
) -> Dict[str, np.ndarray | float]:
    """
    Build dense/sparse expert targets from the Polynomial planner's native 10 Hz path.

    Important: dense points are selected at exact native timestamps. There is no
    interpolation from the 8 sparse planner targets.
    """
    source_time_s = np.asarray(source_time_s, dtype=np.float64).reshape(-1)
    world_xy = np.asarray(world_xy, dtype=np.float32)
    ego_position_world = np.asarray(ego_position_world, dtype=np.float32).reshape(2)

    if world_xy.ndim != 2 or world_xy.shape[1] != 2:
        raise ValueError(f"world_xy must be [N,2], got {world_xy.shape}")
    if world_xy.shape[0] != source_time_s.shape[0]:
        raise ValueError("source_time_s/world_xy length mismatch")
    if not np.isfinite(world_xy).all():
        raise ValueError("world_xy contains NaN/Inf")

    horizon_steps = int(horizon_steps)
    sparse_dt = float(sparse_dt)
    dense_dt = float(dense_dt)
    horizon_s = horizon_steps * sparse_dt

    if horizon_steps <= 0 or sparse_dt <= 0.0 or dense_dt <= 0.0:
        raise ValueError("horizon_steps/sparse_dt/dense_dt must be positive")

    dense_steps_float = horizon_s / dense_dt
    dense_steps = int(round(dense_steps_float))
    if not np.isclose(dense_steps, dense_steps_float, atol=1e-7, rtol=0.0):
        raise ValueError("horizon must be an integer multiple of dense_dt")

    sparse_stride_float = sparse_dt / dense_dt
    sparse_stride = int(round(sparse_stride_float))
    if not np.isclose(sparse_stride, sparse_stride_float, atol=1e-7, rtol=0.0):
        raise ValueError("sparse_dt must be an integer multiple of dense_dt")

    dense_time_s = (
        np.arange(1, dense_steps + 1, dtype=np.float32) * np.float32(dense_dt)
    )
    native_indices = _exact_time_indices(source_time_s, dense_time_s)
    dense_world_xy = world_xy[native_indices].astype(np.float32, copy=True)
    dense_local_xy = world_to_ego_point(
        dense_world_xy,
        ego_position_world,
        float(ego_heading_world),
    ).astype(np.float32, copy=False)

    sparse_indices = (
        np.arange(1, horizon_steps + 1, dtype=np.int64) * sparse_stride - 1
    )
    sparse_time_s = (
        np.arange(1, horizon_steps + 1, dtype=np.float32) * np.float32(sparse_dt)
    )
    sparse_local_xy = dense_local_xy[sparse_indices].astype(np.float32, copy=True)

    if dense_local_xy.shape != (dense_steps, 2):
        raise RuntimeError(f"dense expert shape mismatch: {dense_local_xy.shape}")
    if sparse_local_xy.shape != (horizon_steps, 2):
        raise RuntimeError(f"sparse expert shape mismatch: {sparse_local_xy.shape}")

    return {
        "future_trajectory_dense": dense_local_xy,
        "dense_time_s": dense_time_s,
        "dense_dt": float(dense_dt),
        "trajectory_horizon_s": float(horizon_s),
        "sparse_xy": sparse_local_xy,
        "sparse_time_s": sparse_time_s,
        "dense_sparse_indices": sparse_indices,
    }
