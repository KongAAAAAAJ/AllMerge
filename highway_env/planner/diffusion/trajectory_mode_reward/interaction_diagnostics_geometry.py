"""Geometry-level source attribution for reward interaction diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .collision_geometry import (
    obb_overlap_series,
    shared_corridor_gap_series,
)
from .config import TrajectoryModeRewardConfig
from .geometry import (
    _dense_local_trajectories,
    local_to_world,
    tracking_aware_dimensions,
)
from .interaction_diagnostics_report import InteractionEvent
from .risk import closing_ttc_from_gap_series


def _obb_overlap_mask(
    first: np.ndarray,
    first_dimensions: tuple[float, float],
    second: np.ndarray,
    second_dimensions: tuple[float, float],
) -> np.ndarray:
    """Per-timestep SAT overlap mask matching obb_overlap_series(..., 0)."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (
        first.shape != second.shape
        or first.ndim != 2
        or first.shape[1] != 3
    ):
        raise ValueError("OBB trajectories must be matching [N,3]")

    a_size = np.asarray(first_dimensions, dtype=np.float64)
    b_size = np.asarray(second_dimensions, dtype=np.float64)
    if (
        a_size.shape != (2,)
        or b_size.shape != (2,)
        or np.any(a_size <= 0)
        or np.any(b_size <= 0)
    ):
        raise ValueError("OBB dimensions must be positive [2]")

    a_half = 0.5 * a_size[None, :]
    b_half = 0.5 * b_size[None, :]

    a_long = np.column_stack(
        [np.cos(first[:, 2]), np.sin(first[:, 2])]
    )
    a_lat = np.column_stack([-a_long[:, 1], a_long[:, 0]])
    b_long = np.column_stack(
        [np.cos(second[:, 2]), np.sin(second[:, 2])]
    )
    b_lat = np.column_stack([-b_long[:, 1], b_long[:, 0]])
    delta = second[:, :2] - first[:, :2]

    separated = np.zeros(len(first), dtype=np.bool_)
    for axis in (a_long, a_lat, b_long, b_lat):
        a_radius = (
            a_half[:, 0]
            * np.abs(np.einsum("ij,ij->i", a_long, axis))
            + a_half[:, 1]
            * np.abs(np.einsum("ij,ij->i", a_lat, axis))
        )
        b_radius = (
            b_half[:, 0]
            * np.abs(np.einsum("ij,ij->i", b_long, axis))
            + b_half[:, 1]
            * np.abs(np.einsum("ij,ij->i", b_lat, axis))
        )
        separated |= (
            np.abs(np.einsum("ij,ij->i", delta, axis))
            > a_radius + b_radius
        )
    return np.ascontiguousarray(~separated, dtype=np.bool_)


def _first_true(
    mask: np.ndarray,
    times: np.ndarray,
) -> tuple[int | None, float | None]:
    indices = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
    if not len(indices):
        return None, None
    index = int(indices[0])
    return index, float(times[index])


def _minimum_with_time(
    values: np.ndarray,
    times: np.ndarray,
) -> tuple[float, int, float]:
    values = np.asarray(values, dtype=np.float64)
    index = int(np.argmin(values))
    return float(values[index]), index, float(times[index])


def background_metadata(env: object) -> dict[str, dict[str, Any]]:
    metadata = {}
    for index, vehicle in enumerate(
        list(getattr(env, "background_vehicles", ()) or ())
    ):
        position = np.asarray(
            getattr(vehicle, "position", (np.nan, np.nan)),
            dtype=np.float64,
        ).reshape(-1)
        lane_index = getattr(vehicle, "lane_index", None)
        metadata[f"background_{index}"] = {
            "current_x_m": (
                float(position[0]) if position.size >= 1 else None
            ),
            "current_y_m": (
                float(position[1]) if position.size >= 2 else None
            ),
            "current_speed_mps": float(
                getattr(vehicle, "speed", np.nan)
            ),
            "lane_index": (
                repr(lane_index) if lane_index is not None else None
            ),
        }
    return metadata


def dense_joint_world(
    expert_xy: np.ndarray,
    poses: tuple[np.ndarray, ...],
    config: TrajectoryModeRewardConfig,
) -> tuple[np.ndarray, np.ndarray]:
    dense_local, times = _dense_local_trajectories(
        expert_xy[None],
        config,
    )
    dense_local = dense_local[0]
    dense_world = np.empty_like(dense_local, dtype=np.float64)
    for role in range(dense_local.shape[0]):
        dense_world[role] = local_to_world(
            dense_local[role],
            poses[role],
        )
    return dense_world, times


def build_interaction_event(
    *,
    scenario: str,
    seed: int,
    rollout_step: int,
    simulation_time_s: float,
    target_role: int,
    selected_mode_idx: int,
    source_type: str,
    source_id: str,
    source_role: int | None,
    source_metadata: dict[str, Any] | None,
    target_world: np.ndarray,
    other_world: np.ndarray,
    target_gap_dimensions: tuple[float, float],
    other_dimensions: tuple[float, float],
    safe_gap_m: float,
    times: np.ndarray,
    config: TrajectoryModeRewardConfig,
) -> InteractionEvent | None:
    gap = shared_corridor_gap_series(
        target_world,
        target_gap_dimensions,
        other_world,
        other_dimensions,
        no_risk_gap_m=config.no_risk_gap_m,
    )
    ttc = closing_ttc_from_gap_series(
        gap,
        dt_s=config.interpolation_dt_s,
        closing_speed_epsilon_mps=config.closing_speed_epsilon_mps,
        no_risk_gap_m=config.no_risk_gap_m,
        no_risk_ttc_s=config.no_risk_ttc_s,
    )
    clearance_mask = gap < float(safe_gap_m)

    physical_target = (
        config.vehicle_length_m,
        config.vehicle_width_m,
    )
    future_mask = _obb_overlap_mask(
        target_world[1:],
        physical_target,
        other_world[1:],
        other_dimensions,
    )
    expected_collision = obb_overlap_series(
        target_world[1:],
        physical_target,
        other_world[1:],
        other_dimensions,
        0.0,
    )
    if bool(np.any(future_mask)) != bool(expected_collision):
        raise AssertionError(
            "diagnostic overlap mask disagrees with reward overlap"
        )

    collision_mask = np.zeros(len(times), dtype=np.bool_)
    collision_mask[1:] = future_mask
    collision = bool(np.any(collision_mask))
    clearance = bool(np.any(clearance_mask))
    if not collision and not clearance:
        return None

    collision_index, collision_time = _first_true(
        collision_mask,
        times,
    )
    clearance_index, clearance_time = _first_true(
        clearance_mask,
        times,
    )
    min_gap, min_gap_index, min_gap_time = _minimum_with_time(
        gap,
        times,
    )
    min_ttc, min_ttc_index, min_ttc_time = _minimum_with_time(
        ttc,
        times,
    )
    meta = source_metadata or {}

    return InteractionEvent(
        scenario=scenario,
        seed=int(seed),
        rollout_step=int(rollout_step),
        simulation_time_s=float(simulation_time_s),
        target_role=int(target_role),
        selected_mode_idx=int(selected_mode_idx),
        source_type=source_type,
        source_id=source_id,
        source_role=source_role,
        collision=collision,
        clearance_violation=clearance,
        first_collision_dense_index=collision_index,
        first_collision_time_s=collision_time,
        first_clearance_dense_index=clearance_index,
        first_clearance_time_s=clearance_time,
        minimum_gap_m=min_gap,
        minimum_gap_dense_index=min_gap_index,
        minimum_gap_time_s=min_gap_time,
        minimum_ttc_s=min_ttc,
        minimum_ttc_dense_index=min_ttc_index,
        minimum_ttc_time_s=min_ttc_time,
        safe_gap_threshold_m=float(safe_gap_m),
        source_current_x_m=meta.get("current_x_m"),
        source_current_y_m=meta.get("current_y_m"),
        source_current_speed_mps=meta.get("current_speed_mps"),
        source_lane_index=meta.get("lane_index"),
    )


def tracking_dimensions(
    config: TrajectoryModeRewardConfig,
) -> tuple[float, float]:
    return tracking_aware_dimensions(config)
