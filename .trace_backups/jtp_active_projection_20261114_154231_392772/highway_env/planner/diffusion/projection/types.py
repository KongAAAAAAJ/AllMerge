"""Typed geometric inputs and outputs for the independent JTP solver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RoadGeometry:
    """Drivable road corridors represented by one or more centerline polylines.

    Args:
        centerlines_xy: [P,2] or [L,P,2] world-frame road-corridor centerlines.
            Each polyline should represent a connected drivable carriageway region
            bounded by the outer road edges, rather than an individual lane centerline.
        left_width_m/right_width_m: scalar, [L], [P], or [L,P] half-widths.
            Widths are measured to the left/right of each oriented corridor centerline.
            For symmetric road corridors, use the same half-width on both sides.
    """

    centerlines_xy: torch.Tensor
    left_width_m: torch.Tensor | float
    right_width_m: torch.Tensor | float

    def tensors(self, *, device, dtype):
        center = torch.as_tensor(self.centerlines_xy, device=device, dtype=dtype)
        if center.ndim == 2:
            center = center.unsqueeze(0)
        if center.ndim != 3 or center.shape[-1] != 2 or center.shape[1] < 2:
            raise ValueError("Road centerlines must be [P>=2,2] or [L,P>=2,2]")
        lanes, points = center.shape[:2]

        def expand_width(value, name):
            width = torch.as_tensor(value, device=device, dtype=dtype)
            if width.ndim == 0:
                width = width.expand(lanes, points)
            elif width.ndim == 1:
                if width.numel() == lanes:
                    width = width[:, None].expand(lanes, points)
                elif width.numel() == points and lanes == 1:
                    width = width[None, :]
                else:
                    raise ValueError(f"{name} 1-D shape must be [L] or [P] for one lane")
            elif width.ndim == 2 and width.shape == (lanes, points):
                pass
            else:
                raise ValueError(f"{name} must be scalar, [L], [P] (single lane), or [L,P]")
            if not torch.isfinite(width).all() or (width <= 0).any():
                raise ValueError(f"{name} must contain finite positive widths")
            return width

        if not torch.isfinite(center).all():
            raise ValueError("Road centerlines contain NaN/Inf")
        segment_lengths = torch.linalg.vector_norm(center[:, 1:] - center[:, :-1], dim=-1)
        if (segment_lengths <= 1.0e-6).any():
            raise ValueError("Road centerlines must not contain duplicate consecutive points")
        left = expand_width(self.left_width_m, "left_width_m")
        right = expand_width(self.right_width_m, "right_width_m")
        return center, left, right


@dataclass
class ExternalTraffic:
    """Predicted surrounding-vehicle poses in world coordinates.

    `poses_world` is [A,H+1,3] with columns x, y, yaw. Optional circle offsets
    and radii can be shared ([C]/scalar) or actor-specific ([A,C]). If omitted,
    the configured passenger-vehicle circle cover is used.
    """

    poses_world: torch.Tensor
    circle_offsets_m: Optional[torch.Tensor] = None
    circle_radius_m: Optional[torch.Tensor | float] = None

    def tensors(self, *, device, dtype, expected_steps: int):
        poses = torch.as_tensor(self.poses_world, device=device, dtype=dtype)
        if poses.ndim != 3 or poses.shape[-1] != 3:
            raise ValueError("External poses must be [A,H+1,3]")
        if poses.shape[1] != expected_steps:
            raise ValueError(f"External poses need {expected_steps} time points")
        if not torch.isfinite(poses).all():
            raise ValueError("External poses contain NaN/Inf")
        offsets = None if self.circle_offsets_m is None else torch.as_tensor(
            self.circle_offsets_m, device=device, dtype=dtype)
        radius = None if self.circle_radius_m is None else torch.as_tensor(
            self.circle_radius_m, device=device, dtype=dtype)
        if offsets is not None:
            if offsets.ndim not in (1, 2) or offsets.shape[-1] == 0:
                raise ValueError("External circle offsets must be [C] or [A,C]")
            if offsets.ndim == 2 and offsets.shape[0] != poses.shape[0]:
                raise ValueError("Actor-specific circle offsets must have leading dimension A")
            if not torch.isfinite(offsets).all():
                raise ValueError("External footprint offsets contain NaN/Inf")
        if radius is not None:
            if radius.ndim not in (0, 1, 2):
                raise ValueError("External circle radii must be scalar, [C], or [A,C]")
            if radius.ndim == 2 and radius.shape[0] != poses.shape[0]:
                raise ValueError("Actor-specific circle radii must have leading dimension A")
            if not torch.isfinite(radius).all() or (radius <= 0).any():
                raise ValueError("External footprint radii must be finite and positive")
        if offsets is not None and radius is not None and radius.ndim > 0:
            if offsets.shape[-1] != radius.shape[-1]:
                raise ValueError("External circle offsets and radii must share C")
        return poses, offsets, radius


def constant_velocity_poses(initial_states, horizon_steps: int, dt_s: float, *, device=None, dtype=None):
    """Build [A,H+1,3] world poses from [x,y,vx,vy] or [x,y,vx,vy,yaw]."""
    state = torch.as_tensor(initial_states, device=device, dtype=dtype or torch.float32)
    if state.ndim != 2 or state.shape[1] not in (4, 5):
        raise ValueError("initial_states must be [A,4] or [A,5]: x,y,vx,vy[,yaw]")
    if horizon_steps < 1 or dt_s <= 0 or not torch.isfinite(state).all():
        raise ValueError("Invalid constant-velocity rollout inputs")
    times = torch.arange(horizon_steps + 1, device=state.device, dtype=state.dtype) * dt_s
    x = state[:, 0:1] + state[:, 2:3] * times[None]
    y = state[:, 1:2] + state[:, 3:4] * times[None]
    if state.shape[1] == 5:
        yaw = state[:, 4:5].expand(-1, horizon_steps + 1)
    else:
        yaw0 = torch.atan2(state[:, 3], state[:, 2])
        yaw = yaw0[:, None].expand(-1, horizon_steps + 1)
    return torch.stack((x, y, yaw), dim=-1)


@dataclass
class JTPResult:
    controls: torch.Tensor
    trajectories: torch.Tensor
    states: torch.Tensor
    ltr: torch.Tensor
    success: bool
    status: str
    iterations: int
    diagnostics: dict
