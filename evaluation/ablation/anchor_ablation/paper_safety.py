from __future__ import annotations

"""Planning safety metrics on the selected, unprojected K2 trajectory.

The K2 execution model returns poses in the ego-at-planning-time local frame.
Background traffic is extrapolated with constant velocity, while synchronized
controlled trucks are compared using their own raw K2 predictions. The JTP
projector and its constraint evaluator are not used by this module.
"""

import math
from typing import Mapping, Sequence

import numpy as np


def _resample_polyline(points: np.ndarray, widths: np.ndarray, count: int = 64):
    points = np.asarray(points, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("Each valid drivable lane must contain at least two XY points")
    if widths.shape != (len(points),):
        raise ValueError("Lane widths must match the number of polyline points")
    finite = np.isfinite(points).all(axis=1) & np.isfinite(widths) & (widths > 0)
    points, widths = points[finite], widths[finite]
    if len(points) < 2:
        raise ValueError("Drivable lane has fewer than two finite points")
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5
    points, widths = points[keep], widths[keep]
    if len(points) < 2:
        raise ValueError("Drivable lane degenerates after duplicate-point removal")
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if arc[-1] <= 1e-5:
        raise ValueError("Drivable lane has zero arc length")
    target = np.linspace(0.0, arc[-1], int(count))
    xy = np.stack([np.interp(target, arc, points[:, d]) for d in range(2)], axis=-1)
    width = np.interp(target, arc, widths)
    if not np.isfinite(xy).all() or not np.isfinite(width).all() or np.any(width <= 0):
        raise ValueError("Invalid drivable-lane coordinates or widths")
    return xy, width


def road_geometry_from_features(features: Mapping, pose0: np.ndarray, *, point_count: int = 64) -> dict:
    """Build world-frame corridors from the map's recognized drivable lanes.

    Map features are expressed in the initial ego frame. The lane-class flags
    in channels 6:10 follow the A2 safety teacher's drivable-lane definition.
    """
    maps = np.asarray(features["map_polylines"], dtype=np.float64)
    valid = np.asarray(features["map_valid_mask"], dtype=bool)
    pose0 = np.asarray(pose0, dtype=np.float64).reshape(3)
    if maps.ndim != 3 or maps.shape[-1] < 10 or valid.shape != (maps.shape[0],):
        raise ValueError("Expected map_polylines [L,P,D>=10] and map_valid_mask [L]")
    drivable = (np.max(maps[..., 6:10], axis=(-2, -1)) > 0.5) & valid
    if not np.any(drivable):
        raise ValueError("DAC evaluation requires at least one recognized drivable lane")

    lanes, half_widths = [], []
    c, s = math.cos(float(pose0[2])), math.sin(float(pose0[2]))
    for lane_id in np.flatnonzero(drivable):
        local_xy = maps[lane_id, :, :2]
        lane_width = np.maximum(maps[lane_id, :, 4], 0.1)
        xy, width = _resample_polyline(local_xy, lane_width, point_count)
        world = np.empty_like(xy)
        world[:, 0] = pose0[0] + c * xy[:, 0] - s * xy[:, 1]
        world[:, 1] = pose0[1] + s * xy[:, 0] + c * xy[:, 1]
        lanes.append(world)
        half_widths.append(width * 0.5)
    widths = np.stack(half_widths)
    return {
        "centerlines_xy": np.stack(lanes),
        "left_width_m": widths,
        "right_width_m": widths,
    }


def local_to_world_poses(local_poses: np.ndarray, pose0: np.ndarray) -> np.ndarray:
    """Transform K2 ego-local [T,3] poses into the world frame."""
    local = np.asarray(local_poses, dtype=np.float64)
    origin = np.asarray(pose0, dtype=np.float64).reshape(3)
    if local.ndim != 2 or local.shape[1] != 3 or not np.isfinite(local).all():
        raise ValueError("local_poses must be a finite [T,3] K2 trajectory")
    c, s = math.cos(float(origin[2])), math.sin(float(origin[2]))
    world = np.empty_like(local)
    world[:, 0] = origin[0] + c * local[:, 0] - s * local[:, 1]
    world[:, 1] = origin[1] + s * local[:, 0] + c * local[:, 1]
    world[:, 2] = origin[2] + local[:, 2]
    return world


def _circle_offsets(length: float, width: float, count: int = 3):
    radius = max(float(width) * 0.5, 1e-3)
    start, end = -float(length) * 0.5 + radius, float(length) * 0.5 - radius
    if end <= start:
        return np.asarray([0.0]), radius
    required = int(math.ceil((end - start) / max(2.0 * radius, 1e-6))) + 1
    return np.linspace(start, end, max(int(count), required)), radius


def _circle_centers(poses: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1)
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    return poses[:, None, :2] + offsets[None, :, None] * np.stack((c, s), axis=-1)[:, None, :]


def _track_velocity(points: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference velocity for a [T,...,2] point track."""
    points = np.asarray(points, dtype=np.float64)
    velocity = np.zeros_like(points)
    if len(points) > 1:
        velocity[0] = (points[1] - points[0]) / dt
        velocity[-1] = (points[-1] - points[-2]) / dt
    if len(points) > 2:
        velocity[1:-1] = (points[2:] - points[:-2]) / (2.0 * dt)
    return velocity


def _minimum_ttc_between_circle_tracks(
    ego_centers: np.ndarray,
    actor_centers: np.ndarray,
    ego_velocity: np.ndarray,
    actor_velocity: np.ndarray,
    combined_radius_m: float,
    *,
    margin_m: float,
    closing_speed_epsilon_mps: float,
) -> float:
    """Minimum radial TTC over time and all footprint-circle pairs."""
    relative = actor_centers[:, None, :, :] - ego_centers[:, :, None, :]
    relative_velocity = actor_velocity[:, None, :, :] - ego_velocity[:, :, None, :]
    distance = np.linalg.norm(relative, axis=-1)
    gap = distance - float(combined_radius_m) - float(margin_m)
    closing_speed = -np.sum(relative * relative_velocity, axis=-1) / np.maximum(distance, 1e-8)
    ttc = np.full(gap.shape, np.inf, dtype=np.float64)
    ttc[gap <= 0.0] = 0.0
    approaching = (gap > 0.0) & (closing_speed > float(closing_speed_epsilon_mps))
    ttc[approaching] = gap[approaching] / closing_speed[approaching]
    return float(np.min(ttc)) if ttc.size else float("inf")


def _collision_parameters(vehicle_cfg: Mapping):
    footprint = vehicle_cfg["geometry"]["footprint"]
    offsets = np.asarray(footprint["circle_offsets_m"], dtype=np.float64)
    radius = float(footprint["circle_radius_m"])
    margin = float(vehicle_cfg.get("evaluation", {}).get("collision_margin_m", 0.25))
    dt = float(vehicle_cfg["execution"]["dt_s"])
    if offsets.ndim != 1 or offsets.size == 0 or radius <= 0 or dt <= 0 or margin < 0:
        raise ValueError("Invalid vehicle footprint or safety-evaluation parameters")
    if not np.isfinite(offsets).all() or not all(map(math.isfinite, (radius, margin, dt))):
        raise ValueError("Vehicle footprint and safety-evaluation parameters must be finite")
    return offsets, radius, margin, dt


def external_collision_free(local_poses: np.ndarray, features: Mapping, vehicle_cfg: Mapping) -> bool:
    """Check raw K2 ego poses against uncontrolled actors under CV prediction."""
    poses = np.asarray(local_poses, dtype=np.float64)
    ego = np.asarray(features["ego_state"], dtype=np.float64)
    agents = np.asarray(features["agent_states"], dtype=np.float64)
    valid = np.asarray(features["agent_valid_mask"], dtype=bool)
    ego_offsets, ego_radius, margin, dt = _collision_parameters(vehicle_cfg)
    times = np.arange(len(poses), dtype=np.float64) * dt
    for idx in np.flatnonzero(valid):
        actor = agents[idx]
        # Controlled trucks are checked from their synchronized raw K2 paths.
        if actor.shape[0] > 10 and actor[10] > 0.5:
            continue
        actor_velocity = actor[2:4] + ego[:2]
        actor_heading = math.atan2(float(actor[6]), float(actor[7]))
        actor_length = max(float(actor[8]), 0.5)
        actor_width = max(float(actor[9]), 0.3)
        actor_offsets, actor_radius = _circle_offsets(actor_length, actor_width, 3)
        direction = np.asarray([math.cos(actor_heading), math.sin(actor_heading)])
        for step, t in enumerate(times):
            ego_centers = _circle_centers(poses[step:step + 1], ego_offsets)[0]
            actor_centers = actor[:2][None] + actor_velocity[None] * t + actor_offsets[:, None] * direction[None]
            d2 = np.sum((ego_centers[:, None] - actor_centers[None]) ** 2, axis=-1)
            if np.any(d2 < (ego_radius + actor_radius + margin) ** 2):
                return False
    return True


def external_min_ttc_s(
    local_poses: np.ndarray,
    features: Mapping,
    vehicle_cfg: Mapping,
    *,
    closing_speed_epsilon_mps: float = 0.1,
) -> float:
    """Minimum predicted TTC to uncontrolled actors under constant-velocity motion."""
    poses = np.asarray(local_poses, dtype=np.float64)
    ego = np.asarray(features["ego_state"], dtype=np.float64)
    agents = np.asarray(features["agent_states"], dtype=np.float64)
    valid = np.asarray(features["agent_valid_mask"], dtype=bool)
    ego_offsets, ego_radius, margin, dt = _collision_parameters(vehicle_cfg)
    ego_centers = _circle_centers(poses, ego_offsets)
    ego_velocity = _track_velocity(ego_centers, dt)
    times = np.arange(len(poses), dtype=np.float64) * dt
    minimum = float("inf")
    for idx in np.flatnonzero(valid):
        actor = agents[idx]
        if actor.shape[0] > 10 and actor[10] > 0.5:
            continue
        actor_velocity = actor[2:4] + ego[:2]
        actor_heading = math.atan2(float(actor[6]), float(actor[7]))
        actor_length = max(float(actor[8]), 0.5)
        actor_width = max(float(actor[9]), 0.3)
        actor_offsets, actor_radius = _circle_offsets(actor_length, actor_width, 3)
        direction = np.asarray([math.cos(actor_heading), math.sin(actor_heading)])
        actor_centers = (
            actor[:2][None, None, :]
            + times[:, None, None] * actor_velocity[None, None, :]
            + actor_offsets[None, :, None] * direction[None, None, :]
        )
        actor_velocity_track = np.broadcast_to(actor_velocity[None, None, :], actor_centers.shape)
        value = _minimum_ttc_between_circle_tracks(
            ego_centers, actor_centers, ego_velocity, actor_velocity_track,
            ego_radius + actor_radius, margin_m=margin,
            closing_speed_epsilon_mps=closing_speed_epsilon_mps,
        )
        minimum = min(minimum, value)
    return minimum


def drivable_area_compliance(
    world_poses: np.ndarray,
    road: Mapping,
    vehicle_cfg: Mapping,
    *,
    tolerance_m: float = 1e-6,
) -> float:
    """Percentage of future steps with the full multi-circle vehicle footprint inside drivable area."""
    poses = np.asarray(world_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[-1] != 3 or poses.shape[0] < 2:
        raise ValueError("world_poses must be [H+1,3]")
    centerlines = np.asarray(road["centerlines_xy"], dtype=np.float64)
    left_width = np.asarray(road["left_width_m"], dtype=np.float64)
    right_width = np.asarray(road["right_width_m"], dtype=np.float64)
    if centerlines.ndim != 3 or centerlines.shape[-1] != 2:
        raise ValueError("road centerlines must be [L,P,2]")
    if left_width.shape != centerlines.shape[:2] or right_width.shape != centerlines.shape[:2]:
        raise ValueError("road corridor widths must be [L,P]")
    if not np.isfinite(centerlines).all() or not np.isfinite(left_width).all() or not np.isfinite(right_width).all():
        raise ValueError("Road geometry contains NaN/Inf")
    if tolerance_m < 0:
        raise ValueError("Drivable-area tolerance must be nonnegative")

    footprint = vehicle_cfg["geometry"]["footprint"]
    offsets = np.asarray(footprint["circle_offsets_m"], dtype=np.float64)
    radius = float(footprint["circle_radius_m"])
    centers = _circle_centers(poses[1:], offsets)  # Exclude the initial state: future horizon only.

    starts = centerlines[:, :-1]
    delta = centerlines[:, 1:] - starts
    denom = np.sum(delta * delta, axis=-1)
    if np.any(denom <= 1e-12):
        raise ValueError("Road centerlines contain duplicate consecutive points")
    rel = centers[:, :, None, None, :] - starts[None, None, :, :, :]
    fraction = np.sum(rel * delta[None, None, :, :, :], axis=-1) / denom[None, None, :, :]
    fraction = np.clip(fraction, 0.0, 1.0)
    projection = starts[None, None, :, :, :] + fraction[..., None] * delta[None, None, :, :, :]
    distance2 = np.sum((centers[:, :, None, None, :] - projection) ** 2, axis=-1)
    flat_index = np.argmin(distance2.reshape(*centers.shape[:2], -1), axis=-1)
    segment_count = centerlines.shape[1] - 1
    lane_ids = flat_index // segment_count
    segment_ids = flat_index % segment_count
    gather = (np.arange(centers.shape[0])[:, None], np.arange(centers.shape[1])[None, :])
    a = starts[lane_ids, segment_ids]
    d = delta[lane_ids, segment_ids]
    f = fraction[gather[0], gather[1], lane_ids, segment_ids]
    nearest = a + f[..., None] * d
    tangent = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-8)
    offset = centers - nearest
    signed = tangent[..., 0] * offset[..., 1] - tangent[..., 1] * offset[..., 0]
    left = left_width[lane_ids, segment_ids] * (1.0 - f) + left_width[lane_ids, segment_ids + 1] * f
    right = right_width[lane_ids, segment_ids] * (1.0 - f) + right_width[lane_ids, segment_ids + 1] * f
    left_residual = signed + radius - left
    right_residual = -signed + radius - right
    circle_inside = (left_residual <= tolerance_m) & (right_residual <= tolerance_m)
    step_inside = circle_inside.all(axis=1)
    return float(step_inside.mean() * 100.0)


def intertruck_group_collision_free(records: Sequence[dict], vehicle_cfg: Mapping) -> bool:
    """Check synchronized collisions between controlled trucks' raw K2 world trajectories."""
    if len(records) <= 1:
        return True
    offsets, radius, margin, _ = _collision_parameters(vehicle_cfg)
    for i in range(len(records)):
        pi = np.asarray(records[i]["raw_world_poses"], dtype=np.float64)
        for j in range(i + 1, len(records)):
            pj = np.asarray(records[j]["raw_world_poses"], dtype=np.float64)
            if pi.shape != pj.shape:
                raise ValueError("Synchronized platoon predictions must share the same time grid")
            ci = _circle_centers(pi, offsets)
            cj = _circle_centers(pj, offsets)
            d2 = np.sum((ci[:, :, None, :] - cj[:, None, :, :]) ** 2, axis=-1)
            if np.any(d2 < (2.0 * radius + margin) ** 2):
                return False
    return True


def intertruck_group_min_ttc_s(
    records: Sequence[dict],
    vehicle_cfg: Mapping,
    *,
    closing_speed_epsilon_mps: float = 0.1,
) -> list[float]:
    """Per-truck minimum TTC to other synchronized controlled trucks."""
    count = len(records)
    result = [float("inf")] * count
    if count <= 1:
        return result
    offsets, radius, margin, dt = _collision_parameters(vehicle_cfg)
    centers = [
        _circle_centers(np.asarray(item["raw_world_poses"], dtype=np.float64), offsets)
        for item in records
    ]
    velocities = [_track_velocity(value, dt) for value in centers]
    for i in range(count):
        for j in range(i + 1, count):
            if centers[i].shape != centers[j].shape:
                raise ValueError("Synchronized platoon predictions must share the same time grid")
            value = _minimum_ttc_between_circle_tracks(
                centers[i], centers[j], velocities[i], velocities[j],
                2.0 * radius, margin_m=margin,
                closing_speed_epsilon_mps=closing_speed_epsilon_mps,
            )
            result[i] = min(result[i], value)
            result[j] = min(result[j], value)
    return result
