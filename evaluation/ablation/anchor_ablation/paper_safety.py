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
from shapely.affinity import affine_transform

from highway_env.planner.road_area import road_area, road_clearance


def road_geometry_from_features(features: Mapping, pose0: np.ndarray) -> dict:
    """Build the world-frame road union from raw lane edges."""
    c, s = math.cos(float(pose0[2])), math.sin(float(pose0[2]))
    return {"area": affine_transform(
        road_area(features), [c, -s, s, c, float(pose0[0]), float(pose0[1])])}


def local_to_world_poses(local_poses: np.ndarray, pose0: np.ndarray) -> np.ndarray:
    """Transform ego-local [T,3] poses into the world frame."""
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


def expert_local_poses(future_xy: np.ndarray) -> np.ndarray:
    """Prepend the observed origin and estimate future yaw from path tangents.

    Raw expert labels contain XY rather than body yaw. The initial local yaw
    is zero; stationary future nodes retain the preceding tangent heading.
    """
    xy = np.asarray(future_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 1 or not np.isfinite(xy).all():
        raise ValueError('Expert future trajectory must be finite [H,2] XY points')
    poses = np.zeros((len(xy) + 1, 3), dtype=np.float64)
    poses[1:, :2] = xy
    tangent = np.gradient(poses[:, :2], axis=0)
    for index in range(1, len(poses)):
        if np.linalg.norm(tangent[index]) > 1e-8:
            poses[index, 2] = math.atan2(tangent[index, 1], tangent[index, 0])
        else:
            poses[index, 2] = poses[index - 1, 2]
    return poses


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
    """Check evaluated ego poses against uncontrolled actors under CV prediction."""
    poses = np.asarray(local_poses, dtype=np.float64)
    ego = np.asarray(features["ego_state"], dtype=np.float64)
    agents = np.asarray(features["agent_states"], dtype=np.float64)
    valid = np.asarray(features["agent_valid_mask"], dtype=bool)
    ego_offsets, ego_radius, margin, dt = _collision_parameters(vehicle_cfg)
    times = np.arange(len(poses), dtype=np.float64) * dt
    for idx in np.flatnonzero(valid):
        actor = agents[idx]
        # Controlled trucks are checked from their synchronized evaluated paths.
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
    footprint = vehicle_cfg["geometry"]["footprint"]
    offsets = np.asarray(footprint["circle_offsets_m"], dtype=np.float64)
    radius = float(footprint["circle_radius_m"])
    centers = _circle_centers(poses[1:], offsets)  # Exclude the initial state: future horizon only.

    circle_inside = road_clearance(road["area"], centers) + tolerance_m >= radius
    step_inside = circle_inside.all(axis=1)
    return float(step_inside.mean() * 100.0)


def intertruck_collision_free(records: Sequence[dict], vehicle_cfg: Mapping) -> list[bool]:
    """Per-vehicle flags: a colliding pair never marks an uninvolved truck."""
    result = [True] * len(records)
    if len(records) <= 1:
        return result
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
                result[i] = result[j] = False
    return result


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
