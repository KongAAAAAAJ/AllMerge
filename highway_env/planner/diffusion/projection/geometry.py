"""Differentiable road, curvature, and footprint geometry for JTP."""
from __future__ import annotations

import torch

from .types import RoadGeometry


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def footprint_circle_centers(poses: torch.Tensor, offsets_m: torch.Tensor) -> torch.Tensor:
    """Return [T,C,2] circle centers for one [T,3] pose sequence."""
    if poses.ndim != 2 or poses.shape[-1] != 3:
        raise ValueError("poses must be [T,3]")
    offsets = torch.as_tensor(offsets_m, device=poses.device, dtype=poses.dtype).reshape(-1)
    x = poses[:, None, 0] + offsets[None] * torch.cos(poses[:, None, 2])
    y = poses[:, None, 1] + offsets[None] * torch.sin(poses[:, None, 2])
    return torch.stack((x, y), dim=-1)


def nearest_road_association(points_xy: torch.Tensor, road: RoadGeometry) -> tuple[torch.Tensor, torch.Tensor]:
    """Freeze nearest corridor/segment IDs for one local projection update."""
    center, _, _ = road.tensors(device=points_xy.device, dtype=points_xy.dtype)
    starts = center[:, :-1]
    delta = center[:, 1:] - starts
    rel = points_xy[:, :, None, None, :] - starts[None, None]
    denom = delta.square().sum(-1).clamp_min(torch.finfo(points_xy.dtype).eps)
    fraction = (rel * delta[None, None]).sum(-1) / denom[None, None]
    fraction = fraction.clamp(0.0, 1.0)
    projection = starts[None, None] + fraction[..., None] * delta[None, None]
    distance2 = (points_xy[:, :, None, None, :] - projection).square().sum(-1)
    flat = distance2.reshape(points_xy.shape[0], points_xy.shape[1], -1).argmin(dim=-1)
    segments = center.shape[1] - 1
    lane_ids = torch.div(flat, segments, rounding_mode="floor")
    segment_ids = flat.remainder(segments)
    return lane_ids.detach(), segment_ids.detach()


def road_corridor_residual(
    poses: torch.Tensor,
    road: RoadGeometry,
    *,
    circle_offsets_m,
    circle_radius_m: float,
    clearance_m: float,
    association: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Two-sided road-boundary residuals for all footprint circles and t=1..H.

    Each residual must be <= 0. The closest corridor and polyline segment are
    selected from the current SCP iterate and held fixed during linearization.
    """
    centers = footprint_circle_centers(poses[1:], torch.as_tensor(
        circle_offsets_m, device=poses.device, dtype=poses.dtype))
    centerlines, left_width, right_width = road.tensors(device=poses.device, dtype=poses.dtype)
    lane_ids, segment_ids = association or nearest_road_association(centers.detach(), road)
    a = centerlines[lane_ids, segment_ids]
    b = centerlines[lane_ids, segment_ids + 1]
    delta = b - a
    denom = delta.square().sum(-1).clamp_min(torch.finfo(poses.dtype).eps)
    rel = centers - a
    fraction = ((rel * delta).sum(-1) / denom).clamp(0.0, 1.0)
    projection = a + fraction[..., None] * delta
    tangent = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-8)
    offset = centers - projection
    signed_d = tangent[..., 0] * offset[..., 1] - tangent[..., 1] * offset[..., 0]
    left = left_width[lane_ids, segment_ids] * (1.0 - fraction) + left_width[lane_ids, segment_ids + 1] * fraction
    right = right_width[lane_ids, segment_ids] * (1.0 - fraction) + right_width[lane_ids, segment_ids + 1] * fraction
    radius_margin = float(circle_radius_m) + float(clearance_m)
    # d_min + r + m <= d <= d_max - r - m
    left_residual = signed_d + radius_margin - left
    right_residual = -signed_d + radius_margin - right
    return torch.stack((left_residual, right_residual), dim=-1).reshape(-1)


def curvature_residual(poses: torch.Tensor, curvature_limit_per_m: float) -> torch.Tensor:
    """Discrete yaw-rate-over-distance curvature residual kappa^2-kappa_lim^2."""
    if poses.shape[0] < 3:
        return poses.new_empty((0,))
    delta_yaw = wrap_angle(poses[2:, 2] - poses[1:-1, 2])
    ds = torch.linalg.vector_norm(poses[2:, :2] - poses[1:-1, :2], dim=-1).clamp_min(1.0e-4)
    curvature = delta_yaw / ds
    return curvature.square() - float(curvature_limit_per_m) ** 2


def geometry_residual(
    poses: torch.Tensor,
    road: RoadGeometry,
    cfg: dict,
    *,
    association=None,
) -> tuple[torch.Tensor, dict]:
    geometry_cfg = cfg["geometry"]
    footprint = geometry_cfg["footprint"]
    road_res = road_corridor_residual(
        poses, road,
        circle_offsets_m=footprint["circle_offsets_m"],
        circle_radius_m=float(footprint["circle_radius_m"]),
        clearance_m=float(geometry_cfg["road_clearance_m"]),
        association=association,
    )
    curvature_res = curvature_residual(poses, float(geometry_cfg["curvature_limit_per_m"]))
    return torch.cat((road_res, curvature_res)), {
        "road_count": int(road_res.numel()),
        "curvature_count": int(curvature_res.numel()),
    }


