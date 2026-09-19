"""Adapter from AllMerge scenario state to reward geometry context."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .config import (
    TrajectoryModeRewardConfig,
    TrajectoryModeRewardError,
)
from .constants import NUM_VEHICLES
from .results import RewardGeometryContext


def _vehicle_heading(vehicle: object) -> float:
    value = getattr(vehicle, "heading", None)
    if value is None:
        value = getattr(vehicle, "heading_theta", None)
    if value is None:
        raise TrajectoryModeRewardError(
            "vehicle is missing heading"
        )
    heading = float(value)
    if not math.isfinite(heading):
        raise TrajectoryModeRewardError(
            "vehicle heading must be finite"
        )
    return heading


def _vehicle_pose(vehicle: object) -> np.ndarray:
    position = np.asarray(
        getattr(vehicle, "position", ()),
        dtype=np.float64,
    ).reshape(-1)
    if position.size < 2 or not np.isfinite(position[:2]).all():
        raise TrajectoryModeRewardError(
            "vehicle position must contain finite x/y"
        )
    return np.asarray(
        [
            float(position[0]),
            float(position[1]),
            _vehicle_heading(vehicle),
        ],
        dtype=np.float64,
    )


def _vehicle_dimensions(
    vehicle: object,
    config: TrajectoryModeRewardConfig,
) -> tuple[float, float]:
    length = getattr(
        vehicle,
        "LENGTH",
        getattr(vehicle, "length", config.vehicle_length_m),
    )
    width = getattr(
        vehicle,
        "WIDTH",
        getattr(vehicle, "width", config.vehicle_width_m),
    )
    return float(length), float(width)


def _constant_lane_prediction(
    road: object,
    vehicle: object,
    times: np.ndarray,
) -> np.ndarray:
    """Predict a background actor on its current lane with constant speed."""
    pose = _vehicle_pose(vehicle)
    speed = float(getattr(vehicle, "speed", 0.0))
    lane_index = getattr(vehicle, "lane_index", None)

    if lane_index is not None:
        try:
            lane = road.network.get_lane(lane_index)
            s0, lateral0 = lane.local_coordinates(pose[:2])
            s0 = float(s0)
            lateral0 = float(lateral0)
            values = np.empty((len(times), 3), dtype=np.float64)
            all_inside = True
            for index, time_s in enumerate(times):
                longitudinal = s0 + speed * float(time_s)
                if longitudinal < 0.0 or longitudinal > float(lane.length):
                    all_inside = False
                    break
                values[index, :2] = lane.position(
                    longitudinal,
                    lateral0,
                )
                values[index, 2] = lane.heading_at(
                    longitudinal
                )
            if all_inside:
                return values
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            pass

    forward = np.asarray(
        [math.cos(pose[2]), math.sin(pose[2])],
        dtype=np.float64,
    )
    values = np.empty((len(times), 3), dtype=np.float64)
    values[:, :2] = (
        pose[None, :2]
        + times[:, None] * speed * forward[None, :]
    )
    values[:, 2] = pose[2]
    return values


class AllMergeRewardStateAdapter:
    """Build the process-local geometry cache for one unchanged live state."""

    def __init__(
        self,
        config: TrajectoryModeRewardConfig,
    ) -> None:
        self.config = config

    def build(
        self,
        env: object,
        times: np.ndarray,
    ) -> RewardGeometryContext:
        road = getattr(env, "road", None)
        if road is None:
            raise TrajectoryModeRewardError(
                "env.road is required for reward evaluation"
            )

        controlled = list(
            getattr(env, "controlled_vehicles", ()) or ()
        )
        if len(controlled) != NUM_VEHICLES:
            raise TrajectoryModeRewardError(
                f"expected {NUM_VEHICLES} controlled vehicles, "
                f"got {len(controlled)}"
            )

        poses = tuple(
            _vehicle_pose(vehicle)
            for vehicle in controlled
        )
        controlled_ids = {id(vehicle) for vehicle in controlled}

        background = list(
            getattr(env, "background_vehicles", ()) or ()
        )
        if not background:
            background = [
                vehicle
                for vehicle in list(
                    getattr(road, "vehicles", ()) or ()
                )
                if id(vehicle) not in controlled_ids
            ]

        predictions: dict[
            str,
            list[tuple[str, np.ndarray, tuple[float, float]]],
        ] = {}
        for index, vehicle in enumerate(background):
            actor = f"background_{index}"
            predicted = _constant_lane_prediction(
                road,
                vehicle,
                times,
            )
            predictions[actor] = [
                (
                    actor,
                    predicted,
                    _vehicle_dimensions(vehicle, self.config),
                )
            ]

        # The same scene background is visible to all three target roles.
        per_role: tuple[Mapping, ...] = tuple(
            predictions
            for _ in range(NUM_VEHICLES)
        )
        return RewardGeometryContext(
            poses=poses,
            backgrounds=per_role,
            road=road,
        )
