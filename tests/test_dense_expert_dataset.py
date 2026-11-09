from __future__ import annotations

import numpy as np
import pytest

from highway_env.planner.expert_dense_trajectory import (
    build_native_dense_expert_targets,
)


def test_native_dense_exact_grid_and_sparse_consistency():
    t = np.arange(0, 4.0 + 1e-6, 0.1, dtype=np.float32)
    world = np.stack([10.0 + 8.0 * t, 3.0 + 0.5 * t], axis=-1).astype(np.float32)

    out = build_native_dense_expert_targets(
        source_time_s=t,
        world_xy=world,
        ego_position_world=np.array([10.0, 3.0], dtype=np.float32),
        ego_heading_world=0.0,
    )

    dense = out["future_trajectory_dense"]
    sparse = out["sparse_xy"]
    assert dense.shape == (40, 2)
    assert sparse.shape == (8, 2)
    assert np.isclose(out["dense_dt"], 0.1)
    assert np.isclose(out["trajectory_horizon_s"], 4.0)
    assert np.allclose(sparse, dense[[4, 9, 14, 19, 24, 29, 34, 39]], atol=1e-6)


def test_native_dense_rotated_ego_frame():
    t = np.arange(0, 4.0 + 1e-6, 0.1, dtype=np.float32)
    world = np.stack([5.0 + 0.0 * t, -2.0 + 2.0 * t], axis=-1).astype(np.float32)
    out = build_native_dense_expert_targets(
        source_time_s=t,
        world_xy=world,
        ego_position_world=np.array([5.0, -2.0], dtype=np.float32),
        ego_heading_world=np.pi / 2,
    )
    dense = out["future_trajectory_dense"]
    assert np.allclose(dense[:, 1], 0.0, atol=2e-5)
    assert np.all(dense[:, 0] > 0.0)


def test_missing_native_timestamp_fails_no_interpolation_fallback():
    t = np.arange(0, 4.0 + 1e-6, 0.1, dtype=np.float32)
    keep = ~np.isclose(t, 0.7, atol=1e-6)
    t = t[keep]
    world = np.stack([t, np.zeros_like(t)], axis=-1).astype(np.float32)

    with pytest.raises(ValueError, match="forbids interpolating"):
        build_native_dense_expert_targets(
            source_time_s=t,
            world_xy=world,
            ego_position_world=np.zeros(2, dtype=np.float32),
            ego_heading_world=0.0,
        )
