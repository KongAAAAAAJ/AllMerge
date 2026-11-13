"""Live, synchronized K2 and geometric context for online JTP inference.

The K2 predictor was trained on 10-Hz SI histories:
  state = [vx, vy, body yaw rate, roll, Euler roll rate]
  control = [longitudinal acceleration reference, front-wheel angle reference]

At planner time t, the vehicle's current ``action`` is the command applied
between the previous observation and the current one. It is therefore paired
with the newly observed state when constructing the four transition controls
for the five-state history.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import yaml

from highway_env.planner.diffusion.projection.types import (
    ExternalTraffic,
    RoadGeometry,
    constant_velocity_poses,
)


def load_jtp_runtime_config(path) -> dict:
    """Load and validate online context settings, separate from JTP constraints."""
    from pathlib import Path

    filename = Path(path).expanduser()
    if not filename.is_file():
        raise FileNotFoundError(f"JTP runtime context configuration not found: {filename}")
    config = yaml.safe_load(filename.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("execution"), dict):
        raise ValueError("JTP runtime YAML must define an execution mapping")
    execution = config["execution"]
    if float(execution.get("dt_s", 0.0)) <= 0 or int(execution.get("horizon_steps", 0)) < 1:
        raise ValueError("Invalid execution dt_s/horizon_steps in JTP runtime YAML")
    if int(execution.get("control_steps", 0)) < 1:
        raise ValueError("Invalid execution.control_steps in JTP runtime YAML")
    if int(execution["control_steps"]) != int(execution["horizon_steps"]):
        raise ValueError("Current TRACE JTP runtime requires identity control interpolation (K=H)")
    if not isinstance(config.get("runtime"), dict):
        raise ValueError("JTP runtime YAML must define a runtime mapping")
    return config


@dataclass
class _VehicleHistory:
    states: deque = field(default_factory=lambda: deque(maxlen=5))
    controls: deque = field(default_factory=lambda: deque(maxlen=4))
    last_time_s: float | None = None
    last_pose: np.ndarray | None = None


class LiveK2ContextBuilder:
    """Build synchronized online K2 context and optional JTP scene geometry.

    ``payload_kg`` and ``road_mu`` are never guessed. They must be supplied in
    the JTP YAML runtime context, on the vehicle, or in the environment's
    TruckSim configuration.
    """

    def __init__(self, config: dict):
        self.config = dict(config)
        runtime = self.config.get("runtime", {}) or {}
        self.context_cfg = dict(runtime.get("context", {}) or {})
        self.road_cfg = dict(runtime.get("road", {}) or {})
        self.external_cfg = dict(runtime.get("external_traffic", {}) or {})
        self._histories: dict[int, _VehicleHistory] = {}

        self.sample_dt_s = float(self.context_cfg.get("sample_dt_s", 0.1))
        self.max_gap_tolerance_s = float(self.context_cfg.get("max_gap_tolerance_s", 0.02))
        self.required_states = int(self.context_cfg.get("history_state_steps", 5))
        execution = self.config.get("execution", {}) or {}
        self.execution_dt_s = float(execution.get("dt_s", self.sample_dt_s))
        self.execution_horizon_steps = int(execution.get("horizon_steps", 40))
        self.control_steps = int(execution.get("control_steps", self.execution_horizon_steps))
        if self.sample_dt_s <= 0 or self.max_gap_tolerance_s < 0:
            raise ValueError("Invalid JTP live-context sampling configuration")
        if self.required_states != 5:
            raise ValueError("The current K2 checkpoint requires exactly five observed states")
        if not np.isclose(self.execution_dt_s, self.sample_dt_s, atol=1.0e-8):
            raise ValueError("JTP runtime sample_dt_s must match execution.dt_s")
        if self.control_steps != self.execution_horizon_steps:
            raise ValueError("Current JTP runtime requires identity control interpolation (K=H)")

    def reset(self) -> None:
        self._histories.clear()

    @staticmethod
    def _read_signal(vehicle, name: str, fallback_index: int | None = None):
        export = np.asarray(getattr(vehicle, "export_array", []), dtype=np.float64).reshape(-1)
        signal_map = getattr(vehicle, "signal_map", None)
        if isinstance(signal_map, dict):
            index = signal_map.get("exports", {}).get(name)
            if index is not None and 0 <= int(index) < export.size:
                return float(export[int(index)])
            # A custom export order must provide an explicit signal map; do
            # not silently interpret it using the legacy fixed indices.
            return None
        if fallback_index is not None and fallback_index < export.size:
            return float(export[fallback_index])
        return None

    @classmethod
    def _state_from_vehicle(cls, vehicle) -> np.ndarray:
        """Extract the exact five SI channels used by Koopman data processing."""
        explicit = getattr(vehicle, "koopman_state_si", None)
        if explicit is not None:
            state = np.asarray(explicit, dtype=np.float64).reshape(-1)
            if state.shape == (5,) and np.isfinite(state).all():
                return state.astype(np.float32)

        # Canonical legacy TruckSim export order from data/processing.py:
        # vx, vy, Roll, AVx, Pitch, AVy, AVz are at indices 2,3,6,7,8,9,11.
        vx_kmh = cls._read_signal(vehicle, "vx_kmh", 2)
        vy_kmh = cls._read_signal(vehicle, "vy_kmh", 3)
        roll_deg = cls._read_signal(vehicle, "roll_deg", 6)
        p_deg_s = cls._read_signal(vehicle, "body_p_deg_s", 7)
        pitch_deg = cls._read_signal(vehicle, "pitch_deg", 8)
        q_deg_s = cls._read_signal(vehicle, "body_q_deg_s", 9)
        r_deg_s = cls._read_signal(vehicle, "yaw_rate_deg_s", 11)

        required = (vx_kmh, vy_kmh, roll_deg, pitch_deg, r_deg_s)
        if any(value is None for value in required):
            raise RuntimeError(
                "JTP K2 context requires TruckSim SI state channels. Expected the "
                "canonical 15/19-signal export or koopman_state_si [5]."
            )

        vx = float(vx_kmh) / 3.6
        vy = float(vy_kmh) / 3.6
        roll = float(np.deg2rad(roll_deg))
        pitch = float(np.deg2rad(pitch_deg))
        r = float(np.deg2rad(r_deg_s))
        if p_deg_s is not None and q_deg_s is not None:
            p = float(np.deg2rad(p_deg_s))
            q = float(np.deg2rad(q_deg_s))
            cos_pitch = float(np.cos(pitch))
            if abs(cos_pitch) < 1.0e-3:
                raise RuntimeError("K2 state extraction is near the Euler pitch singularity")
            # Match processing.to_si(): Euler roll derivative, not raw body AVx.
            roll_rate = p + (q * np.sin(roll) + r * np.cos(roll)) * np.tan(pitch)
        else:
            direct = getattr(vehicle, "roll_rate_euler_rad_s", None)
            if direct is None:
                raise RuntimeError(
                    "K2 state extraction needs body_p/body_q channels to reproduce "
                    "the training Euler roll-rate definition, or roll_rate_euler_rad_s."
                )
            roll_rate = float(direct)
        state = np.asarray([vx, vy, r, roll, roll_rate], dtype=np.float32)
        if not np.isfinite(state).all():
            raise RuntimeError("Non-finite TruckSim state while building K2 history")
        return state

    @staticmethod
    def _previous_control(vehicle) -> np.ndarray:
        action = getattr(vehicle, "action", None)
        if not isinstance(action, dict) or "acceleration" not in action or "steering" not in action:
            raise RuntimeError(
                "JTP K2 history requires the previously applied vehicle.action "
                "with acceleration and front-wheel steering references."
            )
        control = np.asarray([action["acceleration"], action["steering"]], dtype=np.float32)
        if not np.isfinite(control).all():
            raise RuntimeError("Non-finite previous control in JTP K2 history")
        return control

    @staticmethod
    def _pose_from_vehicle(vehicle) -> np.ndarray:
        pose = np.asarray([vehicle.position[0], vehicle.position[1], vehicle.heading], dtype=np.float32)
        if not np.isfinite(pose).all():
            raise RuntimeError("Non-finite vehicle pose while building JTP context")
        return pose

    def _resolve_condition(self, name: str, vehicle, env, index: int) -> float:
        aliases = {
            "payload_kg": ("koopman_payload_kg", "payload_kg"),
            "road_mu": ("koopman_road_mu", "road_mu", "friction_coefficient", "mu"),
        }
        value = self.context_cfg.get(name)
        if isinstance(value, (list, tuple)):
            if index >= len(value):
                raise ValueError(f"runtime.context.{name} has no entry for controlled vehicle {index}")
            value = value[index]
        elif isinstance(value, dict):
            value = value.get(str(index), value.get("default"))

        if value is None:
            for attr in aliases[name]:
                candidate = getattr(vehicle, attr, None)
                if candidate is not None:
                    value = candidate
                    break
        if value is None:
            truck_cfg = (getattr(env, "config", {}) or {}).get("TruckSim", {}) or {}
            for attr in (name, *aliases[name]):
                if attr in truck_cfg:
                    value = truck_cfg[attr]
                    break
            if isinstance(value, (list, tuple)):
                if index >= len(value):
                    raise ValueError(f"TruckSim.{name} has no entry for controlled vehicle {index}")
                value = value[index]
        if value is None:
            raise RuntimeError(
                f"JTP requires a known {name}. Set runtime.context.{name} in "
                "configs/projection/jtp.yaml or expose it on each controlled vehicle."
            )
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"Non-finite JTP runtime condition {name}")
        if name == "payload_kg" and result < 0:
            raise ValueError("JTP payload_kg must be nonnegative")
        if name == "road_mu" and not 0 < result <= 1:
            raise ValueError("JTP road_mu must be in (0,1]")
        return result

    def _update_vehicle_history(self, vehicle, timestamp: float) -> tuple[_VehicleHistory, bool]:
        key = id(vehicle)
        state = self._state_from_vehicle(vehicle)
        pose = self._pose_from_vehicle(vehicle)
        history = self._histories.get(key)
        if history is None:
            history = _VehicleHistory()
            history.states.append(state)
            history.last_time_s = timestamp
            history.last_pose = pose
            self._histories[key] = history
            return history, False

        gap = timestamp - float(history.last_time_s)
        if gap <= 0.0 or abs(gap - self.sample_dt_s) > self.max_gap_tolerance_s:
            # A reset, duplicate policy call, or non-10-Hz sample invalidates
            # the old temporal window. Start a fresh warm-up rather than
            # silently feeding misaligned K2 history.
            history = _VehicleHistory()
            history.states.append(state)
            history.last_time_s = timestamp
            history.last_pose = pose
            self._histories[key] = history
            return history, False

        # At observation t_k, vehicle.action is u_{k-1}, the command that
        # acted over [t_{k-1}, t_k). This matches the K2 training alignment.
        history.controls.append(self._previous_control(vehicle))
        history.states.append(state)
        history.last_time_s = timestamp
        history.last_pose = pose
        return history, len(history.states) == 5 and len(history.controls) == 4

    def _build_road_geometry(self, env, vehicle, target_lane_index) -> RoadGeometry:
        network = env.road.network
        lane_dict = network.lanes_dict()
        current_index = tuple(vehicle.lane_index)
        target_index = tuple(target_lane_index) if target_lane_index is not None else current_index
        groups = []
        for index in (current_index, target_index):
            group = tuple(index[:2])
            if group not in groups:
                groups.append(group)

        sample_count = int(self.road_cfg.get("sample_points", 64))
        backward = float(self.road_cfg.get("backward_range_m", 20.0))
        forward = float(self.road_cfg.get("forward_range_m", 180.0))
        if sample_count < 3 or backward < 0 or forward <= 0:
            raise ValueError("Invalid JTP road sampling configuration")

        # Include nearby successor/merge road segments so that the corridor
        # covers the complete four-second execution horizon, including cases
        # where the current lane segment ends before the prediction horizon.
        if bool(self.road_cfg.get("include_connected_road_segments", True)):
            ego_heading = float(vehicle.heading)
            ego_tangent = np.asarray([np.cos(ego_heading), np.sin(ego_heading)])
            ego_position = np.asarray(vehicle.position, dtype=np.float64)
            lateral_range = float(self.road_cfg.get("search_lateral_range_m", 40.0))
            for lane_index, lane in lane_dict.items():
                group = tuple(lane_index[:2])
                if group in groups or bool(getattr(lane, "forbidden", False)):
                    continue
                start_point = np.asarray(lane.position(0.0, 0.0), dtype=np.float64)
                delta = start_point - ego_position
                forward_distance = float(np.dot(delta, ego_tangent))
                lateral_distance = float(ego_tangent[0] * delta[1] - ego_tangent[1] * delta[0])
                heading_error = float(np.arctan2(
                    np.sin(float(lane.heading_at(0.0)) - ego_heading),
                    np.cos(float(lane.heading_at(0.0)) - ego_heading)))
                if (-backward <= forward_distance <= forward + 30.0
                        and abs(lateral_distance) <= lateral_range
                        and abs(heading_error) <= 0.5 * np.pi):
                    groups.append(group)

        centerlines, left_widths, right_widths = [], [], []
        for group in groups:
            lanes = [(tuple(index), lane) for index, lane in lane_dict.items()
                     if tuple(index[:2]) == group and not bool(getattr(lane, "forbidden", False))]
            if not lanes:
                continue

            # Determine physical left/right using the local road tangent,
            # rather than assuming that lane IDs encode a world-y ordering.
            reference_lane = network.get_lane(current_index) if tuple(current_index[:2]) == group else lanes[0][1]
            s_ref = float(np.clip(reference_lane.local_coordinates(vehicle.position)[0],
                                  0.0, float(reference_lane.length)))
            reference_position = np.asarray(reference_lane.position(s_ref, 0.0), dtype=np.float64)
            reference_heading = float(reference_lane.heading_at(s_ref))
            tangent = np.asarray([np.cos(reference_heading), np.sin(reference_heading)])

            lateral_positions = []
            for lane_index, lane in lanes:
                s0 = float(np.clip(lane.local_coordinates(vehicle.position)[0],
                                   0.0, float(lane.length)))
                center = np.asarray(lane.position(s0, 0.0), dtype=np.float64)
                delta = center - reference_position
                lateral = float(tangent[0] * delta[1] - tangent[1] * delta[0])
                lateral_positions.append((lateral, lane_index, lane, s0))
            lateral_positions.sort(key=lambda item: item[0])
            right_lane, left_lane = lateral_positions[0], lateral_positions[-1]
            right_obj, right_s0 = right_lane[2], right_lane[3]
            left_obj, left_s0 = left_lane[2], left_lane[3]

            left_start, left_end = max(0.0, left_s0 - backward), min(float(left_obj.length), left_s0 + forward)
            right_start, right_end = max(0.0, right_s0 - backward), min(float(right_obj.length), right_s0 + forward)
            if min(left_end - left_start, right_end - right_start) < 1.0:
                continue
            left_s = np.linspace(left_start, left_end, sample_count)
            right_s = np.linspace(right_start, right_end, sample_count)
            left_boundary, right_boundary = [], []
            for s_value in left_s:
                width = float(left_obj.width_at(float(s_value)))
                left_boundary.append(left_obj.position(float(s_value), 0.5 * width))
            for s_value in right_s:
                width = float(right_obj.width_at(float(s_value)))
                right_boundary.append(right_obj.position(float(s_value), -0.5 * width))
            left_boundary = np.asarray(left_boundary, dtype=np.float32)
            right_boundary = np.asarray(right_boundary, dtype=np.float32)
            centerline = 0.5 * (left_boundary + right_boundary)
            half_width = 0.5 * np.linalg.norm(left_boundary - right_boundary, axis=-1)
            if (np.linalg.norm(np.diff(centerline, axis=0), axis=-1) <= 1.0e-5).any():
                raise RuntimeError(f"Degenerate JTP road corridor for road group {group}")
            if not np.isfinite(centerline).all() or not np.isfinite(half_width).all() or (half_width <= 0).any():
                raise RuntimeError(f"Invalid JTP road corridor for road group {group}")
            centerlines.append(centerline)
            left_widths.append(half_width)
            right_widths.append(half_width)

        if not centerlines:
            raise RuntimeError(
                f"Cannot construct a JTP drivable corridor for current/target lanes "
                f"{current_index} / {target_index}"
            )
        return RoadGeometry(
            centerlines_xy=torch.as_tensor(np.stack(centerlines), dtype=torch.float32),
            left_width_m=torch.as_tensor(np.stack(left_widths), dtype=torch.float32),
            right_width_m=torch.as_tensor(np.stack(right_widths), dtype=torch.float32),
        )

    def _build_external_traffic(self, env, feature_builder, controlled_vehicles) -> ExternalTraffic | None:
        if not bool(self.external_cfg.get("enabled", True)):
            return None
        max_distance = float(self.external_cfg.get("max_distance_m", 120.0))
        max_actors = int(self.external_cfg.get("max_actors", 16))
        if max_distance <= 0 or max_actors < 0:
            raise ValueError("Invalid JTP external-traffic configuration")
        controlled_ids = {id(vehicle) for vehicle in controlled_vehicles}
        controlled_positions = [np.asarray(vehicle.position, dtype=np.float64) for vehicle in controlled_vehicles]
        actors = []
        for actor in getattr(env.road, "vehicles", []):
            if id(actor) in controlled_ids:
                continue
            position = np.asarray(getattr(actor, "position", [np.nan, np.nan]), dtype=np.float64)
            if position.shape != (2,) or not np.isfinite(position).all():
                continue
            distance = min(float(np.linalg.norm(position - p)) for p in controlled_positions)
            if distance > max_distance:
                continue
            if feature_builder is None or not hasattr(feature_builder, "_vehicle_world_motion"):
                raise RuntimeError(
                    "JTP external-traffic constraints require the shared AllMergeFeatureBuilder"
                )
            try:
                world_velocity, _ = feature_builder._vehicle_world_motion(actor)
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot construct constant-velocity prediction for nearby actor "
                    f"{type(actor).__name__}: {exc}"
                ) from exc
            velocity = np.asarray(world_velocity, dtype=np.float64).reshape(2)
            yaw = float(getattr(actor, "heading", 0.0))
            length = float(getattr(actor, "LENGTH", 4.5))
            width = float(getattr(actor, "WIDTH", 1.8))
            if not np.isfinite([*velocity, yaw, length, width]).all() or length <= 0 or width <= 0:
                continue
            actors.append((distance, np.asarray([position[0], position[1], velocity[0], velocity[1], yaw]),
                           length, width))
        actors.sort(key=lambda item: item[0])
        actors = actors[:max_actors]
        if not actors:
            return None

        initial = torch.as_tensor(np.stack([item[1] for item in actors]), dtype=torch.float32)
        horizon = int(self.config["execution"]["horizon_steps"])
        dt = float(self.config["execution"]["dt_s"])
        poses = constant_velocity_poses(initial, horizon, dt)
        offsets, radii = [], []
        for _, _, length, width in actors:
            # Three overlapping circles exactly cover the oriented rectangle
            # when each radius spans half the inter-circle gap and half width.
            offset = np.asarray([-length / 3.0, 0.0, length / 3.0], dtype=np.float32)
            radius = float(np.sqrt((length / 6.0) ** 2 + (width / 2.0) ** 2))
            offsets.append(offset)
            radii.append(np.full(3, radius, dtype=np.float32))
        return ExternalTraffic(
            poses_world=poses,
            circle_offsets_m=torch.as_tensor(np.stack(offsets), dtype=torch.float32),
            circle_radius_m=torch.as_tensor(np.stack(radii), dtype=torch.float32),
        )

    def build(self, env, feature_builder, target_lane_indices, *, include_projection_inputs: bool) -> dict[str, Any]:
        vehicles = list(env.controlled_vehicles)
        if not vehicles:
            raise RuntimeError("JTP live context requires at least one controlled vehicle")
        if target_lane_indices is None:
            target_lane_indices = [vehicle.lane_index for vehicle in vehicles]
        if len(target_lane_indices) != len(vehicles):
            raise ValueError("JTP target lane count must match controlled vehicles")

        env_config = getattr(env, "config", {}) or {}
        policy_hz = float(env_config.get("policy_frequency", 1.0 / self.sample_dt_s))
        simulation_hz = float(env_config.get("simulation_frequency", policy_hz))
        if (not np.isclose(1.0 / policy_hz, self.sample_dt_s, atol=1.0e-6)
                or not np.isclose(simulation_hz, policy_hz, atol=1.0e-6)):
            raise RuntimeError(
                "Online K2/JTP currently requires synchronized 10-Hz policy and simulation "
                f"steps (expected dt={self.sample_dt_s:g}s, got policy={policy_hz:g}Hz, "
                f"simulation={simulation_hz:g}Hz)."
            )
        timestamp = float(getattr(env, "time", np.nan))
        if not np.isfinite(timestamp):
            raise RuntimeError("JTP live K2 context requires env.time")
        histories, ready_flags = [], []
        for vehicle in vehicles:
            history, ready = self._update_vehicle_history(vehicle, timestamp)
            histories.append(history)
            ready_flags.append(ready)

        # Remove stale vehicle objects after reset/reconfiguration.
        active_ids = {id(vehicle) for vehicle in vehicles}
        self._histories = {key: value for key, value in self._histories.items() if key in active_ids}
        if not all(ready_flags):
            return {
                "ready": False,
                "status": "history_warmup",
                "history_state_counts": [len(item.states) for item in histories],
                "history_control_counts": [len(item.controls) for item in histories],
                "required_state_count": 5,
                "required_control_count": 4,
            }

        state_hist = np.stack([np.stack(list(item.states), axis=0) for item in histories]).astype(np.float32)
        control_hist = np.stack([np.stack(list(item.controls), axis=0) for item in histories]).astype(np.float32)
        pose0 = np.stack([self._pose_from_vehicle(vehicle) for vehicle in vehicles]).astype(np.float32)
        payload = np.asarray([self._resolve_condition("payload_kg", vehicle, env, i)
                              for i, vehicle in enumerate(vehicles)], dtype=np.float32)
        road_mu = np.asarray([self._resolve_condition("road_mu", vehicle, env, i)
                              for i, vehicle in enumerate(vehicles)], dtype=np.float32)
        context = {
            "koopman_state_hist_si": state_hist,
            "koopman_control_hist_si": control_hist,
            "koopman_pose0_si": pose0,
            "koopman_payload_kg": payload,
            "koopman_road_mu": road_mu,
        }

        result = {"ready": True, "status": "ready", "context": context}
        if include_projection_inputs:
            result["road_geometry"] = [
                self._build_road_geometry(env, vehicle, lane_index)
                for vehicle, lane_index in zip(vehicles, target_lane_indices)
            ]
            result["external_traffic"] = self._build_external_traffic(
                env, feature_builder, vehicles)
        return result
