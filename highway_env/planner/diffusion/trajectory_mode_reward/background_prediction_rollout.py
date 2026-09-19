"""Simulator rollout helpers for background-prediction diagnostics."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from .collision_geometry import (
    obb_overlap_series,
    shared_corridor_gap_series,
)
from .config import TrajectoryModeRewardConfig
from .interaction_diagnostics_geometry import (
    dense_joint_world,
    tracking_dimensions,
)
from .risk import closing_ttc_from_gap_series


@dataclass(frozen=True)
class BackgroundRollout:
    poses: np.ndarray
    speeds_mps: np.ndarray
    times_s: np.ndarray


@dataclass(frozen=True)
class PairComparison:
    target_role: int
    actor_id: str

    predicted_collision: bool
    actual_collision: bool
    predicted_clearance: bool
    actual_clearance: bool

    predicted_first_collision_time_s: float | None
    actual_first_collision_time_s: float | None
    predicted_first_clearance_time_s: float | None
    actual_first_clearance_time_s: float | None

    predicted_min_gap_m: float
    actual_min_gap_m: float
    predicted_min_ttc_s: float
    actual_min_ttc_s: float

    endpoint_position_error_m: float
    max_position_error_m: float
    mean_position_error_m: float
    endpoint_longitudinal_error_m: float
    endpoint_lateral_error_m: float
    endpoint_speed_error_mps: float
    max_speed_error_mps: float


def _vehicle_pose(vehicle: object) -> np.ndarray:
    position = np.asarray(
        getattr(vehicle, "position", ()),
        dtype=np.float64,
    ).reshape(-1)
    heading = float(getattr(vehicle, "heading", np.nan))
    if (
        position.size < 2
        or not np.isfinite(position[:2]).all()
        or not math.isfinite(heading)
    ):
        raise AssertionError(
            "background vehicle must expose finite position/heading"
        )
    return np.asarray(
        [position[0], position[1], heading],
        dtype=np.float64,
    )


def rollout_backgrounds(
    env: object,
    *,
    horizon_s: float,
    dt_s: float,
) -> dict[str, BackgroundRollout]:
    """Deep-copy the live state and roll AllMerge vehicle dynamics forward.

    The current Polynomial controls already installed in controlled vehicles are
    held; no planner replan occurs. Background actors continue to execute their
    native ``road.act()`` behavior at every simulator step.
    """
    if horizon_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("horizon_s and dt_s must be positive")

    step_count = int(round(horizon_s / dt_s))
    if not math.isclose(
        step_count * dt_s,
        horizon_s,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "horizon_s must be an integer multiple of dt_s"
        )

    try:
        cloned = copy.deepcopy(env)
    except Exception as exc:
        raise RuntimeError(
            "copy.deepcopy(env) failed. Background predictor diagnostics "
            "requires an isolated simulator copy so the live validation "
            "trajectory is not mutated."
        ) from exc

    backgrounds = list(
        getattr(cloned, "background_vehicles", ()) or ()
    )
    if not backgrounds:
        raise AssertionError(
            "diagnostic rollout requires background_vehicles"
        )

    times = np.arange(
        step_count + 1,
        dtype=np.float64,
    ) * float(dt_s)

    poses = {
        f"background_{index}": np.empty(
            (step_count + 1, 3),
            dtype=np.float64,
        )
        for index in range(len(backgrounds))
    }
    speeds = {
        f"background_{index}": np.empty(
            step_count + 1,
            dtype=np.float64,
        )
        for index in range(len(backgrounds))
    }

    for index, vehicle in enumerate(backgrounds):
        actor = f"background_{index}"
        poses[actor][0] = _vehicle_pose(vehicle)
        speeds[actor][0] = float(
            getattr(vehicle, "speed", 0.0)
        )

    for step in range(1, step_count + 1):
        cloned.road.act()
        cloned.road.step(float(dt_s))
        if hasattr(cloned, "steps"):
            cloned.steps += 1
        if hasattr(cloned, "time"):
            cloned.time += float(dt_s)

        current = list(
            getattr(cloned, "background_vehicles", ()) or ()
        )
        if len(current) != len(backgrounds):
            raise AssertionError(
                "background_vehicles length changed during diagnostic rollout"
            )

        for index, vehicle in enumerate(current):
            actor = f"background_{index}"
            poses[actor][step] = _vehicle_pose(vehicle)
            speeds[actor][step] = float(
                getattr(vehicle, "speed", 0.0)
            )

    return {
        actor: BackgroundRollout(
            poses=np.ascontiguousarray(poses[actor]),
            speeds_mps=np.ascontiguousarray(speeds[actor]),
            times_s=np.ascontiguousarray(times),
        )
        for actor in poses
    }


def _first_true_time(
    values: np.ndarray,
    times: np.ndarray,
) -> float | None:
    indices = np.flatnonzero(
        np.asarray(values, dtype=np.bool_)
    )
    if not len(indices):
        return None
    return float(times[int(indices[0])])


def _collision_mask(
    target: np.ndarray,
    target_dimensions: tuple[float, float],
    other: np.ndarray,
    other_dimensions: tuple[float, float],
) -> np.ndarray:
    """Get first-collision timing while preserving reward collision semantics."""
    from .interaction_diagnostics_geometry import _obb_overlap_mask

    mask = np.zeros(len(target), dtype=np.bool_)
    mask[1:] = _obb_overlap_mask(
        target[1:],
        target_dimensions,
        other[1:],
        other_dimensions,
    )

    aggregate = obb_overlap_series(
        target[1:],
        target_dimensions,
        other[1:],
        other_dimensions,
        0.0,
    )
    if bool(np.any(mask)) != bool(aggregate):
        raise AssertionError(
            "per-step collision mask disagrees with reward geometry"
        )
    return mask


def _interaction_metrics(
    target_world: np.ndarray,
    background_world: np.ndarray,
    *,
    target_gap_dimensions: tuple[float, float],
    background_dimensions: tuple[float, float],
    times: np.ndarray,
    config: TrajectoryModeRewardConfig,
):
    gap = shared_corridor_gap_series(
        target_world,
        target_gap_dimensions,
        background_world,
        background_dimensions,
        no_risk_gap_m=config.no_risk_gap_m,
    )
    ttc = closing_ttc_from_gap_series(
        gap,
        dt_s=config.interpolation_dt_s,
        closing_speed_epsilon_mps=(
            config.closing_speed_epsilon_mps
        ),
        no_risk_gap_m=config.no_risk_gap_m,
        no_risk_ttc_s=config.no_risk_ttc_s,
    )
    collision_mask = _collision_mask(
        target_world,
        (
            config.vehicle_length_m,
            config.vehicle_width_m,
        ),
        background_world,
        background_dimensions,
    )
    clearance_mask = (
        gap < config.background_safe_gap_m
    )
    return {
        "collision": bool(np.any(collision_mask)),
        "clearance": bool(np.any(clearance_mask)),
        "first_collision_time_s": _first_true_time(
            collision_mask,
            times,
        ),
        "first_clearance_time_s": _first_true_time(
            clearance_mask,
            times,
        ),
        "min_gap_m": float(np.min(gap)),
        "min_ttc_s": float(np.min(ttc)),
    }


def compare_prediction_to_rollout(
    *,
    target_role: int,
    actor_id: str,
    target_world: np.ndarray,
    predicted_background: np.ndarray,
    actual_background: BackgroundRollout,
    background_dimensions: tuple[float, float],
    config: TrajectoryModeRewardConfig,
) -> PairComparison:
    predicted = np.asarray(
        predicted_background,
        dtype=np.float64,
    )
    actual = np.asarray(
        actual_background.poses,
        dtype=np.float64,
    )
    times = np.asarray(
        actual_background.times_s,
        dtype=np.float64,
    )

    if (
        predicted.shape != actual.shape
        or predicted.shape != target_world.shape
        or predicted.shape[1] != 3
    ):
        raise AssertionError(
            "target/predicted/actual trajectories must share [T,3]"
        )

    gap_dimensions = tracking_dimensions(config)
    predicted_metrics = _interaction_metrics(
        target_world,
        predicted,
        target_gap_dimensions=gap_dimensions,
        background_dimensions=background_dimensions,
        times=times,
        config=config,
    )
    actual_metrics = _interaction_metrics(
        target_world,
        actual,
        target_gap_dimensions=gap_dimensions,
        background_dimensions=background_dimensions,
        times=times,
        config=config,
    )

    position_delta = (
        predicted[:, :2] - actual[:, :2]
    )
    position_error = np.linalg.norm(
        position_delta,
        axis=1,
    )

    actual_forward = np.column_stack(
        [
            np.cos(actual[:, 2]),
            np.sin(actual[:, 2]),
        ]
    )
    actual_lateral = np.column_stack(
        [
            -actual_forward[:, 1],
            actual_forward[:, 0],
        ]
    )
    longitudinal_error = np.einsum(
        "ij,ij->i",
        position_delta,
        actual_forward,
    )
    lateral_error = np.einsum(
        "ij,ij->i",
        position_delta,
        actual_lateral,
    )

    predicted_speed = np.full(
        len(times),
        float(actual_background.speeds_mps[0]),
        dtype=np.float64,
    )
    speed_error = (
        predicted_speed
        - actual_background.speeds_mps
    )

    return PairComparison(
        target_role=int(target_role),
        actor_id=actor_id,
        predicted_collision=predicted_metrics["collision"],
        actual_collision=actual_metrics["collision"],
        predicted_clearance=predicted_metrics["clearance"],
        actual_clearance=actual_metrics["clearance"],
        predicted_first_collision_time_s=(
            predicted_metrics["first_collision_time_s"]
        ),
        actual_first_collision_time_s=(
            actual_metrics["first_collision_time_s"]
        ),
        predicted_first_clearance_time_s=(
            predicted_metrics["first_clearance_time_s"]
        ),
        actual_first_clearance_time_s=(
            actual_metrics["first_clearance_time_s"]
        ),
        predicted_min_gap_m=predicted_metrics["min_gap_m"],
        actual_min_gap_m=actual_metrics["min_gap_m"],
        predicted_min_ttc_s=predicted_metrics["min_ttc_s"],
        actual_min_ttc_s=actual_metrics["min_ttc_s"],
        endpoint_position_error_m=float(position_error[-1]),
        max_position_error_m=float(np.max(position_error)),
        mean_position_error_m=float(np.mean(position_error)),
        endpoint_longitudinal_error_m=float(
            longitudinal_error[-1]
        ),
        endpoint_lateral_error_m=float(
            lateral_error[-1]
        ),
        endpoint_speed_error_mps=float(speed_error[-1]),
        max_speed_error_mps=float(
            np.max(np.abs(speed_error))
        ),
    )
