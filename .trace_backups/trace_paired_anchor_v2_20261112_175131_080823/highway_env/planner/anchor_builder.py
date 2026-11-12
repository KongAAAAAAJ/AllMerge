"""AllMerge-native dynamic anchor generation.

This module mirrors the semantic structure used by the Diffusion-metadrive
BEV branch:

    KEEP_HIGH / KEEP_MEDIUM / KEEP_LOW
    LEFT_LC_HIGH / LEFT_LC_MEDIUM / LEFT_LC_LOW
    RIGHT_LC_HIGH / RIGHT_LC_MEDIUM / RIGHT_LC_LOW
    STOP

The implementation is adapted to AllMerge:
- input is the structured ground-truth feature dictionary;
- speed levels are centered dynamically around the current ego speed and the
  RuleMaker longitudinal acceleration command;
- lane-change anchors use the vectorized current/left/right lane polylines;
- every geometrically existing mode always keeps its coarse trajectory;
- mode_valid_mask expresses traffic/geometry selectability only;
- traffic validity uses a constant-velocity prediction plus longitudinal/
  lateral oriented-rectangle overlap checks;
- output remains NumPy and is independent of PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from highway_env.planner.mode_definitions import (
    MODE_SLOTS,
    NUM_MODE_SLOTS,
    BehaviorType,
    ModeSlot,
    SpeedProfile,
)


@dataclass
class AnchorOutput:
    coarse_trajectories: np.ndarray
    mode_valid_mask: np.ndarray
    diagnostics: Optional[Dict[int, Dict[str, Any]]] = None


def _polyline_arc_lengths(polyline: np.ndarray) -> np.ndarray:
    polyline = np.asarray(polyline, dtype=np.float32)
    if len(polyline) <= 1:
        return np.zeros((len(polyline),), dtype=np.float32)
    deltas = np.diff(polyline, axis=0)
    seg = np.linalg.norm(deltas, axis=1)
    return np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(seg, dtype=np.float32)]
    )


def _project_origin_to_polyline(polyline: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return closest projection of ego origin onto a polyline and segment index."""
    polyline = np.asarray(polyline, dtype=np.float32)

    if polyline.shape[0] == 1:
        return polyline[0].copy(), 0

    best_point = polyline[0].copy()
    best_segment = 0
    best_dist2 = float(np.dot(best_point, best_point))

    for i in range(polyline.shape[0] - 1):
        p0 = polyline[i]
        p1 = polyline[i + 1]
        d = p1 - p0
        denom = float(np.dot(d, d))
        if denom <= 1e-9:
            candidate = p0
        else:
            t = float(np.clip(-np.dot(p0, d) / denom, 0.0, 1.0))
            candidate = p0 + t * d

        dist2 = float(np.dot(candidate, candidate))
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_point = candidate.astype(np.float32)
            best_segment = i

    return best_point, best_segment


def _forward_polyline(polyline: np.ndarray) -> np.ndarray:
    """Trim map polyline so arc-length zero is the closest lane point to ego."""
    polyline = np.asarray(polyline, dtype=np.float32)

    if polyline.ndim != 2 or polyline.shape[1] != 2 or polyline.shape[0] == 0:
        raise ValueError(f"Invalid polyline shape: {polyline.shape}")

    if polyline.shape[0] == 1:
        return polyline.copy()

    projection, segment_idx = _project_origin_to_polyline(polyline)

    tail = polyline[segment_idx + 1 :]
    forward = np.concatenate([projection[None, :], tail], axis=0)

    # Remove consecutive duplicate points.
    if forward.shape[0] > 1:
        keep = [0]
        for i in range(1, forward.shape[0]):
            if np.linalg.norm(forward[i] - forward[keep[-1]]) > 1e-4:
                keep.append(i)
        forward = forward[keep]

    if forward.shape[0] == 1:
        forward = np.concatenate(
            [forward, forward + np.asarray([[1.0, 0.0]], dtype=np.float32)],
            axis=0,
        )

    return forward.astype(np.float32, copy=False)


def _sample_polyline_at_distances(
    polyline: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Arc-length sample with linear extrapolation beyond the last map point."""
    polyline = np.asarray(polyline, dtype=np.float32)
    distances = np.asarray(distances, dtype=np.float32)

    if polyline.shape[0] == 1:
        return np.repeat(polyline, len(distances), axis=0)

    arc = _polyline_arc_lengths(polyline)
    total = float(arc[-1])

    if total <= 1e-6:
        return np.repeat(polyline[:1], len(distances), axis=0)

    last_dir = polyline[-1] - polyline[-2]
    last_norm = float(np.linalg.norm(last_dir))
    last_unit = (
        last_dir / last_norm
        if last_norm > 1e-6
        else np.asarray([1.0, 0.0], dtype=np.float32)
    )

    output = []
    for raw_d in distances:
        d = max(float(raw_d), 0.0)

        if d >= total:
            output.append(polyline[-1] + (d - total) * last_unit)
            continue

        idx = int(np.searchsorted(arc, d, side="right") - 1)
        idx = max(0, min(idx, polyline.shape[0] - 2))

        denom = float(arc[idx + 1] - arc[idx])
        alpha = 0.0 if denom <= 1e-9 else (d - float(arc[idx])) / denom
        output.append(
            (1.0 - alpha) * polyline[idx] + alpha * polyline[idx + 1]
        )

    return np.asarray(output, dtype=np.float32)


def _quintic_blend(progress: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(progress, dtype=np.float32), 0.0, 1.0)
    return (10.0 * p**3 - 15.0 * p**4 + 6.0 * p**5).astype(np.float32)


class AllMergeAnchorBuilder:
    """Generate dynamic multi-modal anchors from AllMerge planner features."""

    DEFAULT_CONFIG = {
        "horizon_steps": 8,
        "trajectory_dt": 0.5,

        # Dynamic speed levels.
        # The center speed is shifted by:
        #   target_acceleration * acceleration_preview_s
        "speed_delta_mps": 4.0,
        "acceleration_preview_s": 1.0,
        "minimum_lane_change_speed_mps": 3.0,

        # Emergency stop.
        "emergency_decel_mps2": 4.5,

        # Collision-aware validity.
        #
        # Collision is checked in the instantaneous local frame of each
        # anchor point. Agent boxes are projected onto the anchor's
        # longitudinal/lateral axes using the relative heading.
        "collision_check_enabled": True,
        "ego_length_m": 5.0,
        "ego_width_m": 2.0,
        "collision_longitudinal_margin_m": 1.0,
        "collision_lateral_margin_m": 0.3,

        # KEEP_LOW and STOP remain available as fallback modes.
        "keep_low_always_valid": True,
        "stop_always_valid": True,

        # Collision diagnosis is debug-only and is disabled by default.
        # When enabled, the first concrete collision evidence for each
        # traffic-invalid mode is saved in self.last_diagnostics.
        "diagnostics_enabled": False,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)

        self.horizon_steps = int(self.config["horizon_steps"])
        self.dt = float(self.config["trajectory_dt"])
        self.speed_delta = float(self.config["speed_delta_mps"])
        self.acceleration_preview_s = float(
            self.config["acceleration_preview_s"]
        )
        self.min_lc_speed = float(
            self.config["minimum_lane_change_speed_mps"]
        )
        self.emergency_decel = float(
            self.config["emergency_decel_mps2"]
        )

        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if self.dt <= 0.0:
            raise ValueError("trajectory_dt must be positive")

        # List[ego_idx -> Dict[mode_idx -> diagnostic record]]
        # Updated after every build_batch(). Kept outside planner features
        # so model inputs remain clean.
        self.last_diagnostics = []

    @property
    def num_modes(self) -> int:
        return NUM_MODE_SLOTS

    def build_batch(
        self,
        features: Dict[str, np.ndarray],
        target_accelerations: Optional[Sequence[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Build [B, M, 8, 2] anchors and [B, M] validity mask."""
        batch_size = int(features["ego_state"].shape[0])

        if target_accelerations is None:
            target_accelerations = [0.0] * batch_size

        if len(target_accelerations) != batch_size:
            raise ValueError(
                f"target_accelerations has {len(target_accelerations)} items, "
                f"expected {batch_size}"
            )

        coarse = np.zeros(
            (
                batch_size,
                self.num_modes,
                self.horizon_steps,
                2,
            ),
            dtype=np.float32,
        )
        valid = np.zeros(
            (batch_size, self.num_modes),
            dtype=bool,
        )
        diagnostics_batch = []

        for b in range(batch_size):
            out = self.build_single(
                ego_state=features["ego_state"][b],
                agent_states=features["agent_states"][b],
                agent_valid_mask=features["agent_valid_mask"][b],
                map_polylines=features["map_polylines"][b],
                map_valid_mask=features["map_valid_mask"][b],
                target_acceleration=float(target_accelerations[b]),
            )
            coarse[b] = out.coarse_trajectories
            valid[b] = out.mode_valid_mask
            diagnostics_batch.append(
                out.diagnostics if out.diagnostics is not None else {}
            )

        self.last_diagnostics = diagnostics_batch

        self._validate_batch(coarse, valid, batch_size)

        return {
            "coarse_trajectories": coarse,
            "mode_valid_mask": valid,
        }

    def build_single(
        self,
        *,
        ego_state: np.ndarray,
        agent_states: np.ndarray,
        agent_valid_mask: np.ndarray,
        map_polylines: np.ndarray,
        map_valid_mask: np.ndarray,
        target_acceleration: float = 0.0,
    ) -> AnchorOutput:
        ego_state = np.asarray(ego_state, dtype=np.float32)
        agent_states = np.asarray(agent_states, dtype=np.float32)
        agent_valid_mask = np.asarray(agent_valid_mask, dtype=bool)
        map_polylines = np.asarray(map_polylines, dtype=np.float32)
        map_valid_mask = np.asarray(map_valid_mask, dtype=bool)

        current, left, right = self._extract_lane_polylines(
            map_polylines,
            map_valid_mask,
        )

        if current is None:
            # Safe fallback: straight line in ego x direction.
            x = np.linspace(
                0.0,
                120.0,
                64,
                dtype=np.float32,
            )
            current = np.stack(
                [x, np.zeros_like(x)],
                axis=1,
            )

        current = _forward_polyline(current)
        left = _forward_polyline(left) if left is not None else None
        right = _forward_polyline(right) if right is not None else None

        ego_speed = max(float(ego_state[0]), 0.0)
        speed_limit = self._resolve_speed_limit(
            map_polylines,
            map_valid_mask,
            fallback=max(ego_speed + self.speed_delta, 30.0),
        )
        speed_levels = self._dynamic_speed_levels(
            ego_speed=ego_speed,
            target_acceleration=target_acceleration,
            speed_limit=speed_limit,
        )

        coarse = np.zeros(
            (self.num_modes, self.horizon_steps, 2),
            dtype=np.float32,
        )
        valid = np.zeros((self.num_modes,), dtype=bool)
        diagnostics_enabled = bool(
            self.config.get("diagnostics_enabled", False)
        )
        mode_diagnostics: Dict[int, Dict[str, Any]] = {}

        for slot in MODE_SLOTS:
            if slot.semantic_group == "STOP":
                trajectory = self._generate_stop(
                    current,
                    ego_speed,
                )
                is_valid = True

            elif slot.semantic_group == "KEEP":
                target_speed = speed_levels[slot.speed_profile]
                trajectory = self._generate_lane_follow(
                    current,
                    ego_speed,
                    target_speed,
                )
                is_valid = True

            elif slot.semantic_group == "LEFT_LC":
                if left is None:
                    trajectory = None
                    is_valid = False
                else:
                    target_speed = speed_levels[slot.speed_profile]
                    trajectory = self._generate_lane_change(
                        current,
                        left,
                        ego_speed,
                        target_speed,
                    )
                    is_valid = True

            elif slot.semantic_group == "RIGHT_LC":
                if right is None:
                    trajectory = None
                    is_valid = False
                else:
                    target_speed = speed_levels[slot.speed_profile]
                    trajectory = self._generate_lane_change(
                        current,
                        right,
                        ego_speed,
                        target_speed,
                    )
                    is_valid = True

            else:
                trajectory = None
                is_valid = False

            if trajectory is None:
                if diagnostics_enabled:
                    mode_diagnostics[slot.index] = {
                        "mode_index": int(slot.index),
                        "mode_name": slot.name,
                        "semantic_group": slot.semantic_group,
                        "geometry_exists": False,
                        "valid": False,
                        "invalid_reason": "geometry_unavailable",
                        "collision": None,
                    }
                continue

            collision_record = None
            invalid_reason = None

            if is_valid and self._should_collision_filter(slot):
                collision_free, collision_record = self._is_collision_free(
                    trajectory,
                    ego_state,
                    agent_states,
                    agent_valid_mask,
                    collect_diagnostic=diagnostics_enabled,
                )
                if not collision_free:
                    is_valid = False
                    invalid_reason = "collision"

            # KEEP_LOW and STOP are retained as fallback modes.
            if (
                slot.name == "KEEP_LOW"
                and bool(self.config["keep_low_always_valid"])
            ):
                is_valid = True
                invalid_reason = None
                collision_record = None

            if (
                slot.semantic_group == "STOP"
                and bool(self.config["stop_always_valid"])
            ):
                is_valid = True
                invalid_reason = None
                collision_record = None

            # Keep geometry and selectability as two separate concepts.
            #
            # A geometrically existing mode always retains its trajectory,
            # even when traffic makes it invalid.
            coarse[slot.index] = trajectory
            valid[slot.index] = bool(is_valid)

            if diagnostics_enabled:
                if collision_record is not None:
                    collision_record["mode_index"] = int(slot.index)
                    collision_record["mode_name"] = slot.name
                    collision_record["semantic_group"] = slot.semantic_group

                mode_diagnostics[slot.index] = {
                    "mode_index": int(slot.index),
                    "mode_name": slot.name,
                    "semantic_group": slot.semantic_group,
                    "geometry_exists": True,
                    "valid": bool(is_valid),
                    "invalid_reason": invalid_reason,
                    "collision": collision_record,
                }

        # Guarantee at least one valid nonzero trajectory.
        if not np.any(valid):
            keep_low = next(
                slot for slot in MODE_SLOTS if slot.name == "KEEP_LOW"
            )
            trajectory = self._generate_lane_follow(
                current,
                ego_speed,
                max(ego_speed - self.speed_delta, 0.0),
            )
            coarse[keep_low.index] = trajectory
            valid[keep_low.index] = True

        return AnchorOutput(
            coarse_trajectories=coarse,
            mode_valid_mask=valid,
            diagnostics=mode_diagnostics if diagnostics_enabled else None,
        )

    # ------------------------------------------------------------------
    # Dynamic speed levels
    # ------------------------------------------------------------------

    def _dynamic_speed_levels(
        self,
        *,
        ego_speed: float,
        target_acceleration: float,
        speed_limit: float,
    ) -> Dict[SpeedProfile, float]:
        speed_limit = max(float(speed_limit), 1.0)

        # The RuleMaker acceleration only shifts the center of the anchor bank
        # over a short preview; it is not assumed to stay constant for all 4 s.
        center = (
            float(ego_speed)
            + float(target_acceleration) * self.acceleration_preview_s
        )
        center = float(np.clip(center, 0.0, speed_limit))

        high = float(np.clip(center + self.speed_delta, 0.0, speed_limit))
        medium = center
        low = float(np.clip(center - self.speed_delta, 0.0, speed_limit))

        return {
            SpeedProfile.HIGH: high,
            SpeedProfile.MEDIUM: medium,
            SpeedProfile.LOW: low,
        }

    def _distance_profile(
        self,
        start_speed: float,
        target_speed: float,
    ) -> np.ndarray:
        """Smoothly change v0 to target speed and integrate distance."""
        v0 = max(float(start_speed), 0.0)
        vt = max(float(target_speed), 0.0)

        # Speed at t=0 plus one value at each future step.
        velocities = np.linspace(
            v0,
            vt,
            self.horizon_steps + 1,
            dtype=np.float32,
        )

        step_distances = (
            0.5
            * (velocities[:-1] + velocities[1:])
            * self.dt
        )
        return np.cumsum(
            step_distances,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def _generate_lane_follow(
        self,
        lane_polyline: np.ndarray,
        start_speed: float,
        target_speed: float,
    ) -> np.ndarray:
        distances = self._distance_profile(
            start_speed,
            target_speed,
        )
        return _sample_polyline_at_distances(
            lane_polyline,
            distances,
        )

    def _generate_lane_change(
        self,
        current_polyline: np.ndarray,
        target_polyline: np.ndarray,
        start_speed: float,
        target_speed: float,
    ) -> np.ndarray:
        target_speed = max(
            float(target_speed),
            self.min_lc_speed,
        )

        distances = self._distance_profile(
            start_speed,
            target_speed,
        )

        base = _sample_polyline_at_distances(
            current_polyline,
            distances,
        )
        target = _sample_polyline_at_distances(
            target_polyline,
            distances,
        )

        progress = np.linspace(
            1.0 / self.horizon_steps,
            1.0,
            self.horizon_steps,
            dtype=np.float32,
        )
        blend = _quintic_blend(progress)[:, None]

        return (
            (1.0 - blend) * base
            + blend * target
        ).astype(np.float32)

    def _generate_stop(
        self,
        current_polyline: np.ndarray,
        start_speed: float,
    ) -> np.ndarray:
        speed = max(float(start_speed), 0.0)
        distances = []
        distance = 0.0

        for _ in range(self.horizon_steps):
            next_speed = max(
                speed - self.emergency_decel * self.dt,
                0.0,
            )
            distance += 0.5 * (speed + next_speed) * self.dt
            distances.append(distance)
            speed = next_speed

        return _sample_polyline_at_distances(
            current_polyline,
            np.asarray(distances, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Lane extraction from vectorized map
    # ------------------------------------------------------------------

    def _extract_lane_polylines(
        self,
        map_polylines: np.ndarray,
        map_valid_mask: np.ndarray,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        current = None
        left_candidates = []
        right_candidates = []

        for idx in np.where(map_valid_mask)[0]:
            poly = map_polylines[idx]
            xy = poly[:, :2]

            is_current = bool(np.max(poly[:, 6]) > 0.5)
            is_left = bool(np.max(poly[:, 7]) > 0.5)
            is_right = bool(np.max(poly[:, 8]) > 0.5)

            if is_current:
                current = xy.copy()
            elif is_left:
                left_candidates.append(xy.copy())
            elif is_right:
                right_candidates.append(xy.copy())

        left = self._nearest_lateral_lane(
            current,
            left_candidates,
        )
        right = self._nearest_lateral_lane(
            current,
            right_candidates,
        )

        return current, left, right

    @staticmethod
    def _nearest_lateral_lane(
        current: Optional[np.ndarray],
        candidates: Sequence[np.ndarray],
    ) -> Optional[np.ndarray]:
        if not candidates:
            return None

        if current is None:
            return candidates[0]

        current_proj, _ = _project_origin_to_polyline(current)

        scored = []
        for candidate in candidates:
            candidate_proj, _ = _project_origin_to_polyline(candidate)
            lateral_gap = abs(
                float(candidate_proj[1] - current_proj[1])
            )
            scored.append((lateral_gap, candidate))

        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    @staticmethod
    def _resolve_speed_limit(
        map_polylines: np.ndarray,
        map_valid_mask: np.ndarray,
        fallback: float,
    ) -> float:
        values = []

        for idx in np.where(map_valid_mask)[0]:
            poly = map_polylines[idx]
            # Prefer current and target lane speed limits.
            is_relevant = (
                np.max(poly[:, 6]) > 0.5
                or np.max(poly[:, 9]) > 0.5
            )
            if not is_relevant:
                continue

            speed = float(poly[0, 5])
            if speed > 0.0 and np.isfinite(speed):
                values.append(speed)

        return min(values) if values else float(fallback)

    # ------------------------------------------------------------------
    # Lightweight traffic validity
    # ------------------------------------------------------------------

    def _should_collision_filter(self, slot: ModeSlot) -> bool:
        if not bool(self.config["collision_check_enabled"]):
            return False

        # Match the spirit of the BEV generator:
        # fast/medium keep modes and lane-change modes are filtered,
        # while KEEP_LOW and STOP remain fallbacks.
        if slot.semantic_group == "STOP":
            return False
        if slot.name == "KEEP_LOW":
            return False
        return True

    @staticmethod
    def _trajectory_headings(
        trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate the tangent heading of every future anchor point.

        The returned heading is expressed in the initial ego frame.
        Central differences are used where possible so lane-change
        curvature is represented more smoothly than with a one-step
        forward difference.
        """
        trajectory = np.asarray(
            trajectory,
            dtype=np.float32,
        )

        count = trajectory.shape[0]
        headings = np.zeros(
            (count,),
            dtype=np.float32,
        )

        if count <= 1:
            return headings

        for i in range(count):
            if i == 0:
                direction = (
                    trajectory[1]
                    - trajectory[0]
                )
            elif i == count - 1:
                direction = (
                    trajectory[-1]
                    - trajectory[-2]
                )
            else:
                direction = (
                    trajectory[i + 1]
                    - trajectory[i - 1]
                )

            norm = float(
                np.linalg.norm(direction)
            )

            if norm <= 1e-6:
                headings[i] = (
                    headings[i - 1]
                    if i > 0
                    else 0.0
                )
            else:
                headings[i] = np.arctan2(
                    direction[1],
                    direction[0],
                )

        return headings

    def _is_collision_free(
        self,
        trajectory: np.ndarray,
        ego_state: np.ndarray,
        agent_states: np.ndarray,
        agent_valid_mask: np.ndarray,
        *,
        collect_diagnostic: bool = False,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Oriented longitudinal/lateral rectangle collision test.

        Returns:
            collision_free:
                True when no overlap is found.

            diagnostic:
                None in normal operation.

                When collect_diagnostic=True and an overlap is found,
                the first concrete collision evidence is returned. The
                evidence is sufficient to inspect why mode_valid_mask
                became False:

                    agent_idx
                    future step / time
                    signed longitudinal/lateral separation
                    longitudinal/lateral overlap thresholds
                    overlap penetration margins
                    anchor and predicted-agent points
                    anchor and agent headings
                    agent dimensions / controlled flag

        The test remains a lightweight predictor:
            agent motion = constant velocity in the initial ego frame.

        It is deliberately separate from the later Risk-PACT / execution
        surrogate safety model.
        """
        trajectory = np.asarray(
            trajectory,
            dtype=np.float32,
        )
        agent_states = np.asarray(
            agent_states,
            dtype=np.float32,
        )
        agent_valid_mask = np.asarray(
            agent_valid_mask,
            dtype=bool,
        )

        valid_agent_indices = np.flatnonzero(
            agent_valid_mask
        )

        if valid_agent_indices.size == 0:
            return True, None

        times = (
            np.arange(
                self.horizon_steps,
                dtype=np.float32,
            )
            + 1.0
        ) * self.dt

        # agent_states stores relative velocity:
        #     v_agent - v_ego
        #
        # Recover agent velocity represented in the initial ego frame.
        ego_velocity_local = np.asarray(
            [
                ego_state[0],
                ego_state[1],
            ],
            dtype=np.float32,
        )

        anchor_headings = self._trajectory_headings(
            trajectory
        )

        ego_length = float(
            self.config["ego_length_m"]
        )
        ego_width = float(
            self.config["ego_width_m"]
        )
        ego_half_length = 0.5 * ego_length
        ego_half_width = 0.5 * ego_width

        longitudinal_margin = float(
            self.config[
                "collision_longitudinal_margin_m"
            ]
        )
        lateral_margin = float(
            self.config[
                "collision_lateral_margin_m"
            ]
        )

        for agent_idx in valid_agent_indices:
            agent = agent_states[int(agent_idx)]

            initial_position = (
                agent[0:2].astype(
                    np.float32,
                    copy=False,
                )
            )

            agent_relative_velocity = (
                agent[2:4].astype(
                    np.float32,
                    copy=False,
                )
            )

            agent_velocity_local = (
                agent_relative_velocity
                + ego_velocity_local
            )

            predicted_positions = (
                initial_position[None, :]
                + times[:, None]
                * agent_velocity_local[
                    None, :
                ]
            )

            agent_length = max(
                float(agent[8]),
                0.1,
            )
            agent_width = max(
                float(agent[9]),
                0.1,
            )

            agent_half_length = (
                0.5 * agent_length
            )
            agent_half_width = (
                0.5 * agent_width
            )

            # Agent heading relative to the initial ego frame.
            agent_heading = float(
                np.arctan2(
                    agent[6],
                    agent[7],
                )
            )

            for step_idx in range(
                self.horizon_steps
            ):
                anchor_heading = float(
                    anchor_headings[
                        step_idx
                    ]
                )

                anchor_point = trajectory[
                    step_idx
                ]
                agent_point = predicted_positions[
                    step_idx
                ]

                delta = (
                    agent_point
                    - anchor_point
                )

                c = float(
                    np.cos(anchor_heading)
                )
                s = float(
                    np.sin(anchor_heading)
                )

                longitudinal = (
                    c * float(delta[0])
                    + s * float(delta[1])
                )
                lateral = (
                    -s * float(delta[0])
                    + c * float(delta[1])
                )

                relative_heading = (
                    agent_heading
                    - anchor_heading
                )

                cr = abs(
                    float(
                        np.cos(
                            relative_heading
                        )
                    )
                )
                sr = abs(
                    float(
                        np.sin(
                            relative_heading
                        )
                    )
                )

                # Project the agent oriented box onto the instantaneous
                # anchor longitudinal/lateral axes.
                agent_longitudinal_extent = (
                    cr * agent_half_length
                    + sr * agent_half_width
                )
                agent_lateral_extent = (
                    sr * agent_half_length
                    + cr * agent_half_width
                )

                longitudinal_threshold = (
                    ego_half_length
                    + agent_longitudinal_extent
                    + longitudinal_margin
                )
                lateral_threshold = (
                    ego_half_width
                    + agent_lateral_extent
                    + lateral_margin
                )

                abs_longitudinal = abs(
                    longitudinal
                )
                abs_lateral = abs(
                    lateral
                )

                longitudinal_overlap = (
                    abs_longitudinal
                    < longitudinal_threshold
                )
                lateral_overlap = (
                    abs_lateral
                    < lateral_threshold
                )

                if (
                    longitudinal_overlap
                    and lateral_overlap
                ):
                    if not collect_diagnostic:
                        return False, None

                    diagnostic = {
                        "agent_idx": int(agent_idx),
                        "agent_is_controlled": bool(
                            agent[10] > 0.5
                        ),
                        "future_step": int(
                            step_idx + 1
                        ),
                        "step_index": int(
                            step_idx
                        ),
                        "time_s": float(
                            times[step_idx]
                        ),

                        "longitudinal_m": float(
                            longitudinal
                        ),
                        "lateral_m": float(
                            lateral
                        ),
                        "abs_longitudinal_m": float(
                            abs_longitudinal
                        ),
                        "abs_lateral_m": float(
                            abs_lateral
                        ),

                        "longitudinal_threshold_m": float(
                            longitudinal_threshold
                        ),
                        "lateral_threshold_m": float(
                            lateral_threshold
                        ),

                        # Positive values mean penetration into the
                        # forbidden rectangle-overlap region.
                        "longitudinal_penetration_m": float(
                            longitudinal_threshold
                            - abs_longitudinal
                        ),
                        "lateral_penetration_m": float(
                            lateral_threshold
                            - abs_lateral
                        ),

                        "anchor_point": [
                            float(anchor_point[0]),
                            float(anchor_point[1]),
                        ],
                        "agent_predicted_point": [
                            float(agent_point[0]),
                            float(agent_point[1]),
                        ],
                        "agent_initial_point": [
                            float(initial_position[0]),
                            float(initial_position[1]),
                        ],

                        "anchor_heading_rad": float(
                            anchor_heading
                        ),
                        "agent_heading_rad": float(
                            agent_heading
                        ),
                        "relative_heading_rad": float(
                            relative_heading
                        ),

                        "ego_length_m": float(
                            ego_length
                        ),
                        "ego_width_m": float(
                            ego_width
                        ),
                        "agent_length_m": float(
                            agent_length
                        ),
                        "agent_width_m": float(
                            agent_width
                        ),

                        "agent_velocity_local_mps": [
                            float(
                                agent_velocity_local[0]
                            ),
                            float(
                                agent_velocity_local[1]
                            ),
                        ],
                    }

                    return False, diagnostic

        return True, None

    def set_diagnostics_enabled(
        self,
        enabled: bool = True,
    ) -> None:
        """Enable/disable collision evidence collection at runtime."""
        self.config["diagnostics_enabled"] = bool(
            enabled
        )

    def get_last_diagnostics(self):
        """
        Return diagnostics from the most recent build_batch().

        Structure:
            List[
                ego_idx -> {
                    mode_idx -> {
                        mode_name,
                        geometry_exists,
                        valid,
                        invalid_reason,
                        collision
                    }
                }
            ]
        """
        return self.last_diagnostics

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_batch(
        self,
        coarse: np.ndarray,
        valid: np.ndarray,
        batch_size: int,
    ) -> None:
        expected_coarse = (
            batch_size,
            self.num_modes,
            self.horizon_steps,
            2,
        )
        expected_valid = (
            batch_size,
            self.num_modes,
        )

        if coarse.shape != expected_coarse:
            raise RuntimeError(
                f"coarse_trajectories shape={coarse.shape}, "
                f"expected={expected_coarse}"
            )

        if valid.shape != expected_valid:
            raise RuntimeError(
                f"mode_valid_mask shape={valid.shape}, "
                f"expected={expected_valid}"
            )

        if not np.all(np.isfinite(coarse)):
            raise RuntimeError(
                "coarse_trajectories contains NaN/Inf"
            )
