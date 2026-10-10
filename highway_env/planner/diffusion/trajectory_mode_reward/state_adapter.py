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


def _planning_state_snapshot(
    env: object,
) -> tuple[tuple[np.ndarray, ...], tuple[Mapping, ...]] | None:
    """Return the state frozen when latest_planner_features were built."""
    snapshot = getattr(
        env,
        "latest_planner_reward_state_snapshot",
        None,
    )
    if not isinstance(snapshot, Mapping):
        return None

    controlled = np.asarray(
        snapshot.get("controlled_poses"),
        dtype=np.float64,
    )
    if (
        controlled.shape != (NUM_VEHICLES, 3)
        or not np.isfinite(controlled).all()
    ):
        raise TrajectoryModeRewardError(
            "latest planner reward snapshot has invalid controlled_poses"
        )

    raw_background = snapshot.get("background_states", ())
    if raw_background is None:
        raw_background = ()
    background = tuple(raw_background)
    if not all(isinstance(item, Mapping) for item in background):
        raise TrajectoryModeRewardError(
            "latest planner reward snapshot has invalid background_states"
        )

    return (
        tuple(controlled[index].copy() for index in range(NUM_VEHICLES)),
        background,
    )


def _snapshot_background_prediction(
    road: object,
    state: Mapping,
    times: np.ndarray,
) -> np.ndarray:
    position = np.asarray(state.get("position"), dtype=np.float64).reshape(-1)
    heading = float(state.get("heading"))
    speed = float(state.get("speed", 0.0))
    if (
        position.size < 2
        or not np.isfinite(position[:2]).all()
        or not math.isfinite(heading)
        or not math.isfinite(speed)
    ):
        raise TrajectoryModeRewardError(
            "planning-time background snapshot contains invalid pose/speed"
        )

    lane_index = state.get("lane_index")
    if lane_index is not None:
        try:
            lane = road.network.get_lane(tuple(lane_index))
            s0, lateral0 = lane.local_coordinates(position[:2])
            s0 = float(s0)
            lateral0 = float(lateral0)
            values = np.empty((len(times), 3), dtype=np.float64)
            all_inside = True
            for index, time_s in enumerate(times):
                longitudinal = s0 + speed * float(time_s)
                if longitudinal < 0.0 or longitudinal > float(lane.length):
                    all_inside = False
                    break
                values[index, :2] = lane.position(longitudinal, lateral0)
                values[index, 2] = lane.heading_at(longitudinal)
            if all_inside:
                return values
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            pass

    forward = np.asarray(
        [math.cos(heading), math.sin(heading)],
        dtype=np.float64,
    )
    values = np.empty((len(times), 3), dtype=np.float64)
    values[:, :2] = (
        position[None, :2]
        + times[:, None] * speed * forward[None, :]
    )
    values[:, 2] = heading
    return values


def _snapshot_vehicle_dimensions(
    state: Mapping,
    config: TrajectoryModeRewardConfig,
) -> tuple[float, float]:
    length = float(state.get("length", config.vehicle_length_m))
    width = float(state.get("width", config.vehicle_width_m))
    if length <= 0.0:
        length = float(config.vehicle_length_m)
    if width <= 0.0:
        width = float(config.vehicle_width_m)
    return length, width


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

        # REWARD_FRAME_SNAPSHOT_V1
        # Candidates are expressed in the ego-local frame frozen when planner
        # features were built.  env.step() advances the simulator before GRPO
        # evaluates them, so using the live post-step pose would rotate/translate
        # the entire 4 s trajectory in world coordinates.  Prefer the exact
        # planning-time snapshot and keep the live-state path only as a legacy
        # fallback for callers that do not originate from planner features.
        planning_snapshot = _planning_state_snapshot(env)
        if planning_snapshot is not None:
            poses, background_snapshot = planning_snapshot
            background = None
        else:
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
            background_snapshot = ()

        predictions: dict[
            str,
            list[tuple[str, np.ndarray, tuple[float, float]]],
        ] = {}
        if planning_snapshot is not None:
            for index, actor_state in enumerate(background_snapshot):
                actor = f"background_{index}"
                predicted = _snapshot_background_prediction(
                    road,
                    actor_state,
                    times,
                )
                predictions[actor] = [
                    (
                        actor,
                        predicted,
                        _snapshot_vehicle_dimensions(
                            actor_state,
                            self.config,
                        ),
                    )
                ]
        else:
            for index, vehicle in enumerate(background or []):
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
        # Freeze the same role-specific target lane used to condition Diffusion.
        # These are already in the corresponding planning-time ego frame.
        target_lines = None
        frozen_features = getattr(env, 'latest_planner_features', None)
        if isinstance(frozen_features, Mapping) and 'target_lane_polyline' in frozen_features:
            poly = np.asarray(frozen_features['target_lane_polyline'], dtype=np.float64)
            if (poly.ndim != 3 or poly.shape[0] != NUM_VEHICLES or
                    poly.shape[1] < 2 or poly.shape[2] < 2 or not np.isfinite(poly).all()):
                raise TrajectoryModeRewardError('Invalid planning-time target_lane_polyline features')
            target_lines = tuple(poly[role, :, :2].copy() for role in range(NUM_VEHICLES))
        return RewardGeometryContext(
            poses=poses,
            backgrounds=per_role,
            road=road,
            target_centerlines=target_lines,
        )
