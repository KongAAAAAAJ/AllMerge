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
    selected from the current JTP projection iterate and held fixed during local linearization.
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




def _collision_actor_fields(actor):
    if isinstance(actor, dict):
        pose = actor.get("pose")
        length = actor.get("length_m", actor.get("length"))
        width = actor.get("width_m", actor.get("width"))
    else:
        pose = getattr(actor, "pose", None)
        length = getattr(actor, "length_m", getattr(actor, "length", None))
        width = getattr(actor, "width_m", getattr(actor, "width", None))
    if pose is None or length is None or width is None:
        raise ValueError("Each external collision actor requires pose, length_m and width_m")
    return pose, float(length), float(width)


def _actor_footprint_centers(pose: torch.Tensor, length_m: float, width_m: float):
    """Approximate a rectangular actor by overlapping circles along its length.

    This matches the independent JTP evaluator's circle-cover construction.
    """
    radius = max(float(width_m) / 2.0, 1.0e-3)
    start = -float(length_m) / 2.0 + radius
    stop = float(length_m) / 2.0 - radius
    if stop < start:
        offsets = pose.new_zeros((1,))
    else:
        import math
        count = max(2, int(math.ceil((stop - start) / (2.0 * radius))) + 1)
        offsets = torch.linspace(start, stop, count, device=pose.device, dtype=pose.dtype)
    return footprint_circle_centers(pose[1:], offsets), radius


def intertruck_collision_residual(trajectories: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Differentiable pairwise footprint-clearance residuals for controlled trucks.

    Input trajectories are world-frame [N,H+1,3]. All circle pairs between
    different trucks are checked at matching future time steps. Positive
    residuals indicate overlap of the configured safety envelopes.
    """
    if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
        raise ValueError("trajectories must be [N,H+1,3]")
    n = trajectories.shape[0]
    if n < 2:
        return trajectories.new_empty((0,))
    footprint = cfg["geometry"]["footprint"]
    offsets = torch.as_tensor(footprint["circle_offsets_m"], device=trajectories.device,
                              dtype=trajectories.dtype)
    radius = float(footprint["circle_radius_m"])
    clearance = float(cfg["collision"]["clearance_m"])
    scale = max(float(cfg["projection"]["residual_scales"]["collision_m"]), 1.0e-8)
    centers = [footprint_circle_centers(trajectories[i, 1:], offsets) for i in range(n)]
    residuals = []
    for i in range(n):
        for j in range(i + 1, n):
            delta = centers[i][:, :, None, :] - centers[j][:, None, :, :]
            distance = torch.sqrt(delta.square().sum(dim=-1).clamp_min(1.0e-12))
            residuals.append((2.0 * radius + clearance - distance).reshape(-1) / scale)
    return torch.cat(residuals) if residuals else trajectories.new_empty((0,))


def external_collision_residual(trajectories: torch.Tensor, actors, cfg: dict) -> torch.Tensor:
    """Differentiable controlled-truck/external-actor clearance residuals.

    ``actors`` is a sequence of objects or mappings exposing a world-frame
    ``pose [H+1,3]``, ``length_m`` and ``width_m``. Actor predictions are held
    fixed during a JTP solve (the evaluator/runtime currently supplies CV paths).
    """
    if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
        raise ValueError("trajectories must be [N,H+1,3]")
    actors = tuple(actors or ())
    if not actors:
        return trajectories.new_empty((0,))
    n, steps, _ = trajectories.shape
    footprint = cfg["geometry"]["footprint"]
    offsets = torch.as_tensor(footprint["circle_offsets_m"], device=trajectories.device,
                              dtype=trajectories.dtype)
    truck_radius = float(footprint["circle_radius_m"])
    clearance = float(cfg["collision"]["clearance_m"])
    scale = max(float(cfg["projection"]["residual_scales"]["collision_m"]), 1.0e-8)
    truck_centers = [footprint_circle_centers(trajectories[i, 1:], offsets) for i in range(n)]
    residuals = []
    for actor in actors:
        pose_raw, length, width = _collision_actor_fields(actor)
        pose = torch.as_tensor(pose_raw, device=trajectories.device, dtype=trajectories.dtype)
        if pose.shape != (steps, 3) or not torch.isfinite(pose).all():
            raise ValueError(f"External actor pose must be finite [{steps},3], got {tuple(pose.shape)}")
        actor_centers, actor_radius = _actor_footprint_centers(pose, length, width)
        for i in range(n):
            delta = truck_centers[i][:, :, None, :] - actor_centers[:, None, :, :]
            distance = torch.sqrt(delta.square().sum(dim=-1).clamp_min(1.0e-12))
            residuals.append((truck_radius + actor_radius + clearance - distance).reshape(-1) / scale)
    return torch.cat(residuals) if residuals else trajectories.new_empty((0,))
