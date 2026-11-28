"""Live, synchronized K2 context for online JTP inference.

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

    def build(self, env) -> dict[str, Any]:
        vehicles = list(env.controlled_vehicles)
        if not vehicles:
            raise RuntimeError("JTP live context requires at least one controlled vehicle")
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
        return result
