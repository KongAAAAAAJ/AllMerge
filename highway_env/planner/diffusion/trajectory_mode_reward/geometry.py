"""Trajectory interpolation, frame transforms, and AllMerge road geometry."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

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
