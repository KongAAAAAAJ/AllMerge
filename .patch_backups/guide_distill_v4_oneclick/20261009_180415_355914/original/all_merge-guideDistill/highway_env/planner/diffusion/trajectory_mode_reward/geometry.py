"""Trajectory interpolation, frame transforms, and AllMerge road geometry."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from highway_env.road.lane import CircularLane, SineLane, StraightLane

from .config import (
    TrajectoryModeRewardConfig,
    TrajectoryModeRewardError,
)
from .constants import (
    HORIZON_STEPS,
    NUM_VEHICLES,
)


def _heading_from_xy(xy: np.ndarray) -> np.ndarray:
    """Recover local heading from an XY trajectory tangent."""
    points = np.asarray(xy, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not np.isfinite(points).all()
    ):
        raise TrajectoryModeRewardError(
            "xy trajectory must be finite [N,2]"
        )

    origin = np.zeros((1, 2), dtype=np.float64)
    deltas = np.diff(
        np.concatenate((origin, points), axis=0),
        axis=0,
    )
    heading = np.zeros(len(points), dtype=np.float64)
    last = 0.0
    for index, delta in enumerate(deltas):
        if float(np.linalg.norm(delta)) > 1e-6:
            last = math.atan2(float(delta[1]), float(delta[0]))
        heading[index] = last
    return np.unwrap(heading)


def _dense_local_trajectories(
    trajectories: np.ndarray,
    config: TrajectoryModeRewardConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate [G,3,8,2] raw tau_d into dense [G,3,T,3] poses."""
    values = np.asarray(trajectories, dtype=np.float64)
    if (
        values.ndim != 4
        or values.shape[1] != NUM_VEHICLES
        or values.shape[2:] != (HORIZON_STEPS, 2)
        or not np.isfinite(values).all()
    ):
        raise TrajectoryModeRewardError(
            "trajectories must be finite [G,3,8,2]"
        )

    source_times = (
        np.arange(1, HORIZON_STEPS + 1, dtype=np.float64)
        * config.trajectory_dt_s
    )
    target_times = np.arange(
        0.0,
        source_times[-1] + 0.5 * config.interpolation_dt_s,
        config.interpolation_dt_s,
        dtype=np.float64,
    )
    source_with_origin = np.concatenate(([0.0], source_times))

    dense = np.empty(
        (values.shape[0], NUM_VEHICLES, len(target_times), 3),
        dtype=np.float64,
    )
    for group in range(values.shape[0]):
        for role in range(NUM_VEHICLES):
            trajectory = values[group, role]
            xy_with_origin = np.concatenate(
                (
                    np.zeros((1, 2), dtype=np.float64),
                    trajectory,
                ),
                axis=0,
            )
            dense_xy = np.column_stack(
                [
                    np.interp(
                        target_times,
                        source_with_origin,
                        xy_with_origin[:, axis],
                    )
                    for axis in range(2)
                ]
            )
            dense[group, role, :, :2] = dense_xy
            dense[group, role, :, 2] = _heading_from_xy(
                dense_xy
            )
    return dense, target_times


def local_to_world(
    local: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    values = np.asarray(local, dtype=np.float64)
    origin = np.asarray(pose, dtype=np.float64).reshape(-1)
    if values.shape[-1] != 3 or origin.shape != (3,):
        raise TrajectoryModeRewardError(
            "local poses must end in 3 and origin pose must be [3]"
        )
    cos_h = math.cos(float(origin[2]))
    sin_h = math.sin(float(origin[2]))
    world = np.empty_like(values, dtype=np.float64)
    world[..., 0] = (
        origin[0]
        + cos_h * values[..., 0]
        - sin_h * values[..., 1]
    )
    world[..., 1] = (
        origin[1]
        + sin_h * values[..., 0]
        + cos_h * values[..., 1]
    )
    world[..., 2] = np.arctan2(
        np.sin(values[..., 2] + origin[2]),
        np.cos(values[..., 2] + origin[2]),
    )
    return world


def tracking_aware_dimensions(
    config: TrajectoryModeRewardConfig,
) -> tuple[float, float]:
    heading = config.tracking_heading_margin_rad
    half_length = (
        0.5 * config.vehicle_length_m
        + config.tracking_longitudinal_margin_m
        + 0.5 * config.vehicle_width_m * math.sin(heading)
    )
    half_width = (
        0.5 * config.vehicle_width_m
        + config.tracking_lateral_margin_m
        + 0.5 * config.vehicle_length_m * math.sin(heading)
    )
    return 2.0 * half_length, 2.0 * half_width


def _footprint_corners(
    poses: np.ndarray,
    dimensions: tuple[float, float],
) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64)
    length, width = map(float, dimensions)
    local = np.asarray(
        [
            [0.5 * length, 0.5 * width],
            [0.5 * length, -0.5 * width],
            [-0.5 * length, 0.5 * width],
            [-0.5 * length, -0.5 * width],
        ],
        dtype=np.float64,
    )
    result = np.empty(
        (len(values), 4, 2),
        dtype=np.float64,
    )
    for index, pose in enumerate(values):
        cos_h = math.cos(float(pose[2]))
        sin_h = math.sin(float(pose[2]))
        rotation = np.asarray(
            [[cos_h, -sin_h], [sin_h, cos_h]],
            dtype=np.float64,
        )
        result[index] = (
            local @ rotation.T
            + pose[None, :2]
        )
    return result


def _iter_lanes(road: object) -> Iterable[object]:
    network = getattr(road, "network", None)
    graph = getattr(network, "graph", None)
    if not isinstance(graph, dict):
        raise TrajectoryModeRewardError(
            "AllMerge road is missing network.graph"
        )
    seen: set[int] = set()
    for outgoing in graph.values():
        if not isinstance(outgoing, dict):
            continue
        for lane_group in outgoing.values():
            if not isinstance(lane_group, (list, tuple)):
                continue
            for lane in lane_group:
                identifier = id(lane)
                if identifier not in seen:
                    seen.add(identifier)
                    yield lane


def _lane_width_at(lane: object, longitudinal: float) -> float:
    if hasattr(lane, "width_at"):
        return float(lane.width_at(float(longitudinal)))
    width = getattr(lane, "width", None)
    if width is None:
        width = getattr(lane, "DEFAULT_WIDTH", None)
    if width is None:
        raise TrajectoryModeRewardError(
            "lane does not provide width_at/width"
        )
    return float(width)


def _point_lane_margin(
    point: np.ndarray,
    lanes: list[object],
) -> float:
    best = -1.0e6
    for lane in lanes:
        try:
            longitudinal, lateral = lane.local_coordinates(point)
            longitudinal = float(longitudinal)
            lateral = float(lateral)
            length = float(lane.length)
            s_for_width = float(
                np.clip(longitudinal, 0.0, length)
            )
            lateral_margin = (
                0.5 * _lane_width_at(lane, s_for_width)
                - abs(lateral)
            )
            longitudinal_margin = min(
                longitudinal,
                length - longitudinal,
            )
            margin = min(
                lateral_margin,
                longitudinal_margin,
            )
            best = max(best, float(margin))
        except (AttributeError, TypeError, ValueError):
            continue
    return best


# GRPO_SPEED_V2B_BATCH_GEOMETRY_20261007

def _heading_from_xy_batch(xy: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of _heading_from_xy for [...,N,2] trajectories."""
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] != 2 or not np.isfinite(points).all():
        raise TrajectoryModeRewardError("xy trajectories must be finite [...,N,2]")
    origin = np.zeros((*points.shape[:-2], 1, 2), dtype=np.float64)
    deltas = np.diff(np.concatenate((origin, points), axis=-2), axis=-2)
    norms = np.linalg.norm(deltas, axis=-1)
    raw = np.arctan2(deltas[..., 1], deltas[..., 0])
    valid = norms > 1e-6
    time_idx = np.arange(points.shape[-2], dtype=np.int64)
    shape = (1,) * (valid.ndim - 1) + (points.shape[-2],)
    idx = np.where(valid, time_idx.reshape(shape), 0)
    idx = np.maximum.accumulate(idx, axis=-1)
    heading = np.take_along_axis(raw, idx, axis=-1)
    # Scalar implementation starts with last=0.0 before the first valid tangent.
    any_valid = np.maximum.accumulate(valid, axis=-1)
    heading = np.where(any_valid, heading, 0.0)
    return np.unwrap(heading, axis=-1)


def _dense_local_trajectories_batch(
    trajectories: np.ndarray,
    config: TrajectoryModeRewardConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch interpolate [G,3,8,2] into [G,3,T,3] without a G loop."""
    values = np.asarray(trajectories, dtype=np.float64)
    if (
        values.ndim != 4
        or values.shape[1] != NUM_VEHICLES
        or values.shape[2:] != (HORIZON_STEPS, 2)
        or not np.isfinite(values).all()
    ):
        raise TrajectoryModeRewardError("trajectories must be finite [G,3,8,2]")
    source_times = np.arange(1, HORIZON_STEPS + 1, dtype=np.float64) * config.trajectory_dt_s
    target_times = np.arange(
        0.0,
        source_times[-1] + 0.5 * config.interpolation_dt_s,
        config.interpolation_dt_s,
        dtype=np.float64,
    )
    source = np.concatenate(([0.0], source_times))
    values_with_origin = np.concatenate(
        (np.zeros((values.shape[0], NUM_VEHICLES, 1, 2), dtype=np.float64), values),
        axis=2,
    )
    left = np.searchsorted(source, target_times, side="right") - 1
    left = np.clip(left, 0, len(source) - 2)
    right = left + 1
    denom = source[right] - source[left]
    alpha = (target_times - source[left]) / denom
    xy0 = values_with_origin[:, :, left, :]
    xy1 = values_with_origin[:, :, right, :]
    dense_xy = xy0 + alpha[None, None, :, None] * (xy1 - xy0)
    dense = np.empty((*dense_xy.shape[:-1], 3), dtype=np.float64)
    dense[..., :2] = dense_xy
    dense[..., 2] = _heading_from_xy_batch(dense_xy)
    return dense, target_times


def _footprint_corners_batch(
    poses: np.ndarray,
    dimensions: tuple[float, float],
) -> np.ndarray:
    """Return [...,N,4,2] footprint corners for batched poses."""
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise TrajectoryModeRewardError("poses must be finite [...,N,3]")
    length, width = map(float, dimensions)
    local = np.asarray(
        [
            [0.5 * length, 0.5 * width],
            [0.5 * length, -0.5 * width],
            [-0.5 * length, 0.5 * width],
            [-0.5 * length, -0.5 * width],
        ],
        dtype=np.float64,
    )
    c = np.cos(values[..., 2])[..., None]
    s = np.sin(values[..., 2])[..., None]
    lx = local[:, 0]
    ly = local[:, 1]
    out = np.empty((*values.shape[:-1], 4, 2), dtype=np.float64)
    out[..., 0] = values[..., 0, None] + c * lx - s * ly
    out[..., 1] = values[..., 1, None] + s * lx + c * ly
    return out


def _lane_margin_points_batch(points: np.ndarray, lane: object) -> np.ndarray:
    """Vectorized point-to-lane signed margin for AllMerge scenario lane types."""
    pts = np.asarray(points, dtype=np.float64)
    flat = pts.reshape(-1, 2)
    if isinstance(lane, SineLane):
        delta = flat - np.asarray(lane.start, dtype=np.float64)
        longitudinal = delta @ np.asarray(lane.direction, dtype=np.float64)
        lateral = delta @ np.asarray(lane.direction_lateral, dtype=np.float64)
        lateral = lateral - float(lane.amplitude) * np.sin(
            float(lane.pulsation) * longitudinal + float(lane.phase)
        )
        width = np.full_like(longitudinal, float(lane.width), dtype=np.float64)
    elif isinstance(lane, StraightLane):
        delta = flat - np.asarray(lane.start, dtype=np.float64)
        longitudinal = delta @ np.asarray(lane.direction, dtype=np.float64)
        lateral = delta @ np.asarray(lane.direction_lateral, dtype=np.float64)
        width = np.full_like(longitudinal, float(lane.width), dtype=np.float64)
    elif isinstance(lane, CircularLane):
        delta = flat - np.asarray(lane.center, dtype=np.float64)
        phi = np.arctan2(delta[:, 1], delta[:, 0])
        start = float(lane.start_phase)
        phi = start + ((phi - start + np.pi) % (2.0 * np.pi) - np.pi)
        radius = float(lane.radius)
        direction = float(lane.direction)
        longitudinal = direction * (phi - start) * radius
        lateral = direction * (radius - np.linalg.norm(delta, axis=1))
        width = np.full_like(longitudinal, float(lane.width), dtype=np.float64)
    else:
        # Exact scalar fallback for lane classes outside the four AllMerge scenarios.
        out = np.empty(len(flat), dtype=np.float64)
        for i, point in enumerate(flat):
            try:
                longitudinal, lateral = lane.local_coordinates(point)
                longitudinal = float(longitudinal)
                lateral = float(lateral)
                length = float(lane.length)
                s_for_width = float(np.clip(longitudinal, 0.0, length))
                lateral_margin = 0.5 * _lane_width_at(lane, s_for_width) - abs(lateral)
                longitudinal_margin = min(longitudinal, length - longitudinal)
                out[i] = min(lateral_margin, longitudinal_margin)
            except (AttributeError, TypeError, ValueError):
                out[i] = -1.0e6
        return out.reshape(pts.shape[:-1])
    length = float(lane.length)
    lateral_margin = 0.5 * width - np.abs(lateral)
    longitudinal_margin = np.minimum(longitudinal, length - longitudinal)
    margin = np.minimum(lateral_margin, longitudinal_margin)
    return margin.reshape(pts.shape[:-1])


def road_margin_series_batch(
    world_poses: np.ndarray,
    road: object,
    config: TrajectoryModeRewardConfig,
    *,
    tracking_aware: bool,
) -> np.ndarray:
    """Batch equivalent of road_margin_series for [G,N,3] poses."""
    poses = np.asarray(world_poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-1] != 3 or not np.isfinite(poses).all():
        raise TrajectoryModeRewardError("world_poses must be finite [G,N,3]")
    dimensions = (
        tracking_aware_dimensions(config)
        if tracking_aware
        else (config.vehicle_length_m, config.vehicle_width_m)
    )
    lanes = list(_iter_lanes(road))
    if not lanes:
        raise TrajectoryModeRewardError("road network contains no lanes")
    corners = _footprint_corners_batch(poses, dimensions)
    best = np.full(corners.shape[:-1], -1.0e6, dtype=np.float64)
    for lane in lanes:
        best = np.maximum(best, _lane_margin_points_batch(corners, lane))
    return np.min(best, axis=-1)


def road_margin_series(
    world_poses: np.ndarray,
    road: object,
    config: TrajectoryModeRewardConfig,
    *,
    tracking_aware: bool,
) -> np.ndarray:
    """Approximate signed footprint margin to the union of AllMerge lanes."""
    poses = np.asarray(world_poses, dtype=np.float64)
    if (
        poses.ndim != 2
        or poses.shape[1] != 3
        or not np.isfinite(poses).all()
    ):
        raise TrajectoryModeRewardError(
            "world_poses must be finite [N,3]"
        )

    dimensions = (
        tracking_aware_dimensions(config)
        if tracking_aware
        else (
            config.vehicle_length_m,
            config.vehicle_width_m,
        )
    )
    lanes = list(_iter_lanes(road))
    if not lanes:
        raise TrajectoryModeRewardError(
            "road network contains no lanes"
        )

    corners = _footprint_corners(
        poses,
        dimensions,
    )
    margins = np.empty(len(poses), dtype=np.float64)
    for index, pose_corners in enumerate(corners):
        corner_margins = [
            _point_lane_margin(point, lanes)
            for point in pose_corners
        ]
        margins[index] = min(corner_margins)
    return margins
