"""MATP's dense-curve adapter backed by the current AllMerge spline module.

This is the compatibility layer replacing the legacy
`trajectory_mode_reward.spline_dense` import from all_merge-guideDistill.
The interpolation equations and initial-velocity convention are retained.
"""
from __future__ import annotations

from functools import lru_cache

import torch

from highway_env.planner.diffusion.common.spline import ClampedCubicTrajectorySpline


@lru_cache(maxsize=12)
def _decoder(sparse_dt: float, dense_dt: float, device: str):
    return ClampedCubicTrajectorySpline(
        horizon_s=8 * sparse_dt,
        sparse_dt=sparse_dt,
        dense_dt=dense_dt,
    ).to(torch.device(device))


def evaluate_dense_spline(xy: torch.Tensor, config, *, initial_velocity_xy=None):
    """Decode ``[...,8,2]`` points to dense position/velocity/acceleration.

    The source worktree uses 8 points at 0.5 s and 41 points at 0.1 s.
    The same interface is retained while using the current `common.spline`
    implementation.  Production callers should pass the planning-time ego
    velocity explicitly.
    """
    if not torch.is_tensor(xy) or xy.ndim < 2 or tuple(xy.shape[-2:]) != (8, 2):
        raise ValueError(f"Expected tensor [...,8,2], received {getattr(xy, 'shape', None)}")
    if not xy.is_floating_point():
        raise TypeError("xy must use a floating dtype")
    if not torch.isfinite(xy).all():
        raise ValueError("xy contains NaN/Inf")

    dt = float(config.trajectory_dt_s)
    dense_dt = float(config.interpolation_dt_s)
    if dt <= 0.0 or dense_dt <= 0.0 or dense_dt > dt:
        raise ValueError("Invalid trajectory/interpolation time step")

    original_shape = xy.shape[:-2]
    flat = xy.reshape(-1, 8, 2)
    origin = torch.zeros_like(flat[:, 0])
    if initial_velocity_xy is None:
        # Retain the source branch's utility fallback.  Production use should
        # pass an explicit synchronized ego velocity.
        start_velocity = torch.stack(
            (flat[:, 0, 0].clamp_min(0.0) / dt,
             torch.zeros_like(flat[:, 0, 0])),
            dim=-1,
        )
    else:
        velocity = torch.as_tensor(
            initial_velocity_xy, dtype=xy.dtype, device=xy.device
        )
        try:
            start_velocity = torch.broadcast_to(
                velocity, (*original_shape, 2)
            ).reshape(-1, 2).detach()
        except RuntimeError as exc:
            raise ValueError(
                f"initial_velocity_xy shape {tuple(velocity.shape)} cannot broadcast "
                f"to {tuple((*original_shape, 2))}"
            ) from exc

    end_velocity = (flat[:, -1] - flat[:, -2]) / dt
    decoder = _decoder(dt, dense_dt, str(flat.device))
    dense, velocity, acceleration = decoder.evaluate(
        flat,
        start_xy=origin,
        start_velocity_xy=start_velocity,
        end_velocity_xy=end_velocity,
    )
    return tuple(
        value.reshape(*original_shape, value.shape[-2], 2)
        for value in (dense, velocity, acceleration)
    )
