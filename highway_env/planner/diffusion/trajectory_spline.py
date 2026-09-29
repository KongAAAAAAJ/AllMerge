# STAGE_A_SPLINE_RUNTIME_V1
"""Differentiable clamped cubic spline decoder for AllMerge trajectories.

The Diffusion planner output contract remains sparse ego-local waypoints:
    [..., 8, 2] at 0.5, 1.0, ..., 4.0 s.

This module prepends the current ego-local point at t=0 and evaluates one
clamped cubic spline through all 9 knots on a 10 Hz grid by default.
It is pure PyTorch so gradients can be reused by later dense supervision.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class ClampedCubicTrajectorySpline(nn.Module):
    """Batch-capable C2 clamped cubic interpolation in physical xy units."""

    def __init__(
        self,
        horizon_s: float = 4.0,
        sparse_dt: float = 0.5,
        dense_dt: float = 0.1,
    ) -> None:
        super().__init__()

        self.horizon_s = float(horizon_s)
        self.sparse_dt = float(sparse_dt)
        self.dense_dt = float(dense_dt)

        if self.horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")
        if self.sparse_dt <= 0.0 or self.dense_dt <= 0.0:
            raise ValueError("sparse_dt and dense_dt must be positive")

        sparse_steps_float = self.horizon_s / self.sparse_dt
        dense_steps_float = self.horizon_s / self.dense_dt
        self.sparse_steps = int(round(sparse_steps_float))
        self.dense_steps = int(round(dense_steps_float))

        if abs(sparse_steps_float - self.sparse_steps) > 1e-6:
            raise ValueError("horizon_s must be divisible by sparse_dt")
        if abs(dense_steps_float - self.dense_steps) > 1e-6:
            raise ValueError("horizon_s must be divisible by dense_dt")

        knot_time = torch.arange(
            self.sparse_steps + 1,
            dtype=torch.float32,
        ) * self.sparse_dt
        dense_time = torch.arange(
            self.dense_steps + 1,
            dtype=torch.float32,
        ) * self.dense_dt
        interval_h = knot_time[1:] - knot_time[:-1]

        system = torch.zeros(
            self.sparse_steps + 1,
            self.sparse_steps + 1,
            dtype=torch.float32,
        )

        # Clamped boundary row at t=0.
        system[0, 0] = 2.0 * interval_h[0]
        system[0, 1] = interval_h[0]

        # Interior C2 continuity rows.
        for i in range(1, self.sparse_steps):
            h_prev = interval_h[i - 1]
            h_next = interval_h[i]
            system[i, i - 1] = h_prev
            system[i, i] = 2.0 * (h_prev + h_next)
            system[i, i + 1] = h_next

        # Clamped boundary row at t=T.
        system[-1, -2] = interval_h[-1]
        system[-1, -1] = 2.0 * interval_h[-1]

        sparse_dense_index_float = (
            torch.arange(1, self.sparse_steps + 1, dtype=torch.float32)
            * self.sparse_dt
            / self.dense_dt
        )
        sparse_dense_index = torch.round(
            sparse_dense_index_float
        ).to(torch.long)

        if not torch.allclose(
            sparse_dense_index_float,
            sparse_dense_index.to(torch.float32),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(
                "dense_dt must place every sparse waypoint exactly on the dense grid"
            )

        self.register_buffer("knot_time_s", knot_time, persistent=False)
        self.register_buffer("dense_time_s", dense_time, persistent=False)
        self.register_buffer("interval_h_s", interval_h, persistent=False)
        self.register_buffer("system_matrix", system, persistent=False)
        self.register_buffer(
            "sparse_dense_indices",
            sparse_dense_index,
            persistent=False,
        )

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")

    @staticmethod
    def _broadcast_boundary(
        value: torch.Tensor,
        target_shape: torch.Size,
        *,
        name: str,
    ) -> torch.Tensor:
        try:
            return torch.broadcast_to(value, target_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} cannot broadcast to "
                f"{tuple(target_shape)}"
            ) from exc

    def _prepare_inputs(
        self,
        sparse_xy: torch.Tensor,
        start_xy: torch.Tensor,
        start_velocity_xy: torch.Tensor,
        end_velocity_xy: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(sparse_xy):
            raise TypeError("sparse_xy must be a torch.Tensor")
        if sparse_xy.ndim < 2:
            raise ValueError("sparse_xy must have shape [..., 8, 2]")
        if tuple(sparse_xy.shape[-2:]) != (self.sparse_steps, 2):
            raise ValueError(
                f"sparse_xy has shape {tuple(sparse_xy.shape)}, expected "
                f"[..., {self.sparse_steps}, 2]"
            )
        if not sparse_xy.is_floating_point():
            raise TypeError("sparse_xy must use a floating dtype")

        device = sparse_xy.device
        dtype = sparse_xy.dtype
        boundary_shape = sparse_xy.shape[:-2] + (2,)

        start_xy = torch.as_tensor(start_xy, device=device, dtype=dtype)
        start_velocity_xy = torch.as_tensor(
            start_velocity_xy,
            device=device,
            dtype=dtype,
        )
        start_xy = self._broadcast_boundary(
            start_xy,
            boundary_shape,
            name="start_xy",
        )
        start_velocity_xy = self._broadcast_boundary(
            start_velocity_xy,
            boundary_shape,
            name="start_velocity_xy",
        )

        knots = torch.cat(
            [start_xy.unsqueeze(-2), sparse_xy],
            dim=-2,
        )

        if end_velocity_xy is None:
            end_velocity_xy = (
                knots[..., -1, :] - knots[..., -2, :]
            ) / self.sparse_dt
        else:
            end_velocity_xy = torch.as_tensor(
                end_velocity_xy,
                device=device,
                dtype=dtype,
            )
            end_velocity_xy = self._broadcast_boundary(
                end_velocity_xy,
                boundary_shape,
                name="end_velocity_xy",
            )

        self._require_finite("sparse_xy", sparse_xy)
        self._require_finite("start_xy", start_xy)
        self._require_finite("start_velocity_xy", start_velocity_xy)
        self._require_finite("end_velocity_xy", end_velocity_xy)

        return knots, start_velocity_xy, end_velocity_xy

    def _second_derivatives(
        self,
        knots: torch.Tensor,
        start_velocity_xy: torch.Tensor,
        end_velocity_xy: torch.Tensor,
    ) -> torch.Tensor:
        h = self.interval_h_s.to(
            device=knots.device,
            dtype=knots.dtype,
        )
        slopes = (
            knots[..., 1:, :] - knots[..., :-1, :]
        ) / h.view(*([1] * (knots.ndim - 2)), -1, 1)

        left = 6.0 * (slopes[..., 0, :] - start_velocity_xy)
        middle = 6.0 * (slopes[..., 1:, :] - slopes[..., :-1, :])
        right = 6.0 * (end_velocity_xy - slopes[..., -1, :])
        rhs = torch.cat(
            [left.unsqueeze(-2), middle, right.unsqueeze(-2)],
            dim=-2,
        )

        system = self.system_matrix.to(
            device=knots.device,
            dtype=knots.dtype,
        )
        return torch.linalg.solve(system, rhs)

    def _evaluate_with_second_derivatives(
        self,
        knots: torch.Tensor,
        second: torch.Tensor,
        query_time_s: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = torch.as_tensor(
            query_time_s,
            device=knots.device,
            dtype=knots.dtype,
        )
        if query.ndim != 1:
            raise ValueError("query_time_s must be one-dimensional")
        self._require_finite("query_time_s", query)
        if (query < -1e-7).any() or (query > self.horizon_s + 1e-7).any():
            raise ValueError(
                f"query_time_s must stay inside [0, {self.horizon_s}]"
            )

        knot_time = self.knot_time_s.to(
            device=knots.device,
            dtype=knots.dtype,
        )
        interval_index = torch.bucketize(
            query,
            knot_time[1:-1],
            right=False,
        ).clamp(0, self.sparse_steps - 1)

        p0 = knots[..., interval_index, :]
        p1 = knots[..., interval_index + 1, :]
        m0 = second[..., interval_index, :]
        m1 = second[..., interval_index + 1, :]

        t0 = knot_time[interval_index]
        t1 = knot_time[interval_index + 1]
        h = t1 - t0
        a = t1 - query
        b = query - t0

        prefix_dims = knots.ndim - 2
        view_shape = (*([1] * prefix_dims), query.numel(), 1)
        h_v = h.view(view_shape)
        a_v = a.view(view_shape)
        b_v = b.view(view_shape)

        position = (
            m0 * a_v.pow(3) / (6.0 * h_v)
            + m1 * b_v.pow(3) / (6.0 * h_v)
            + (p0 - m0 * h_v.pow(2) / 6.0) * (a_v / h_v)
            + (p1 - m1 * h_v.pow(2) / 6.0) * (b_v / h_v)
        )

        velocity = (
            -m0 * a_v.pow(2) / (2.0 * h_v)
            + m1 * b_v.pow(2) / (2.0 * h_v)
            + (p1 - p0) / h_v
            - (m1 - m0) * h_v / 6.0
        )

        acceleration = m0 * (a_v / h_v) + m1 * (b_v / h_v)
        return position, velocity, acceleration

    def evaluate(
        self,
        sparse_xy: torch.Tensor,
        start_xy: torch.Tensor,
        start_velocity_xy: torch.Tensor,
        end_velocity_xy: Optional[torch.Tensor] = None,
        query_time_s: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return position, velocity and acceleration on the requested grid."""
        knots, start_velocity_xy, end_velocity_xy = self._prepare_inputs(
            sparse_xy,
            start_xy,
            start_velocity_xy,
            end_velocity_xy,
        )
        second = self._second_derivatives(
            knots,
            start_velocity_xy,
            end_velocity_xy,
        )
        if query_time_s is None:
            query_time_s = self.dense_time_s
        return self._evaluate_with_second_derivatives(
            knots,
            second,
            query_time_s,
        )

    def interpolate(
        self,
        sparse_xy: torch.Tensor,
        start_xy: torch.Tensor,
        start_velocity_xy: torch.Tensor,
        end_velocity_xy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        position, _, _ = self.evaluate(
            sparse_xy,
            start_xy,
            start_velocity_xy,
            end_velocity_xy,
        )
        return position

    def forward(
        self,
        sparse_xy: torch.Tensor,
        start_xy: torch.Tensor,
        start_velocity_xy: torch.Tensor,
        end_velocity_xy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.interpolate(
            sparse_xy,
            start_xy,
            start_velocity_xy,
            end_velocity_xy,
        )
