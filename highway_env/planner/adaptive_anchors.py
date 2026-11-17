"""Shared, identity-free control-space candidate generation for A1 and A2.

The configured candidate pool is generated once by this module. A1 selects
anchors by deterministic joint diversity only; A2 applies a learned safety
ranking before the same diversity selector. The diffusion planner always
receives the fixed number K_a of selected anchors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from highway_env.planner.anchor_builder import AllMergeAnchorBuilder, _forward_polyline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CANDIDATE_CONFIG = {
    # pool_scale is the only knob needed to increase/decrease the shared pool.
    # Each scale unit contributes 36 candidates.
    "pool_scale": 1,
    "base_current_lane_speed_samples": 5,
    "base_adjacent_lane_speed_samples": 3,
    "base_brake_candidate_count": 3,
    "brake_fraction_range": [0.15, 0.85],
    "steering_residuals_rad": [-0.006, 0.0, 0.006],
    "include_emergency_stop": True,
    "base": {
        "horizon_steps": 8,
        "trajectory_dt": 0.5,
        "control_steps": 40,
        "control_dt": 0.1,
        "wheelbase_m": 5.5,
        "max_accel_mps2": 2.5,
        "max_decel_mps2": 4.0,
        "max_steer_rad": 0.08,
        "speed_delta_mps": 6.0,
        "acceleration_preview_s": 1.0,
        "minimum_lane_change_speed_mps": 3.0,
        "lookahead_s": 0.65,
        "min_lookahead_m": 7.0,
        "ego_length_m": 8.0,
        "ego_width_m": 2.6,
        "collision_check_enabled": False,
    },
}


def load_candidate_generation_config(config=None) -> dict:
    """Resolve a shared YAML config path plus optional inline overrides.

    ``pool_scale`` is the single candidate-count control. Scale=1 reproduces
    the original 36-slot pool; scale=3 produces 108 slots. A1 and A2 consume
    this same resolved dictionary, and cache fingerprints include it.
    """
    raw = dict(config or {}) if isinstance(config, Mapping) else {}
    config_path = raw.pop("config_path", None)
    resolved = dict(DEFAULT_CANDIDATE_CONFIG)
    resolved["base"] = dict(DEFAULT_CANDIDATE_CONFIG["base"])
    if config_path:
        path = Path(config_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Shared candidate-pool config not found: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Candidate-pool YAML must be a mapping: {path}")
        if "candidate_generation" in loaded:
            loaded = loaded["candidate_generation"]
        if not isinstance(loaded, dict):
            raise ValueError("candidate_generation must be a mapping")
        resolved.update({k: v for k, v in loaded.items() if k != "base"})
        resolved["base"].update(loaded.get("base", {}) or {})
    resolved.update({k: v for k, v in raw.items() if k != "base"})
    resolved["base"].update(raw.get("base", {}) or {})

    scale = int(resolved["pool_scale"])
    base_current = int(resolved["base_current_lane_speed_samples"])
    base_adjacent = int(resolved["base_adjacent_lane_speed_samples"])
    base_brakes = int(resolved["base_brake_candidate_count"])
    if scale <= 0 or min(base_current, base_adjacent, base_brakes) <= 0:
        raise ValueError("pool_scale and base candidate counts must be positive")

    resolved["pool_scale"] = scale
    resolved["current_lane_speed_samples"] = base_current * scale
    resolved["adjacent_lane_speed_samples"] = base_adjacent * scale
    resolved["steering_residuals_rad"] = [
        float(x) for x in resolved["steering_residuals_rad"]
    ]
    if not resolved["steering_residuals_rad"] or 0.0 not in resolved["steering_residuals_rad"]:
        raise ValueError("steering_residuals_rad must include 0.0")
    if any(abs(x) > float(resolved["base"]["max_steer_rad"])
           for x in resolved["steering_residuals_rad"]):
        raise ValueError("Steering residual exceeds the configured steering limit")

    resolved["include_emergency_stop"] = bool(resolved["include_emergency_stop"])
    brake_total = base_brakes * scale
    fraction_count = brake_total - int(resolved["include_emergency_stop"])
    if fraction_count < 0:
        raise ValueError("Invalid brake candidate count")
    low, high = [float(x) for x in resolved["brake_fraction_range"]]
    if not 0.0 < low <= high <= 1.0:
        raise ValueError("brake_fraction_range must satisfy 0 < low <= high <= 1")
    resolved["brake_fractions"] = (
        np.linspace(low, high, fraction_count, dtype=np.float64).tolist()
        if fraction_count else []
    )

    derived = (
        (resolved["current_lane_speed_samples"]
         + 2 * resolved["adjacent_lane_speed_samples"])
        * len(resolved["steering_residuals_rad"])
        + len(resolved["brake_fractions"])
        + int(resolved["include_emergency_stop"])
    )
    resolved["candidate_pool_size"] = int(derived)
    return resolved


class AdaptiveCandidateBuilder:
    """Generate a configurable pool of paired control/trajectory candidates.

    Default 108-candidate layout:
      current lane: 15 speeds x 3 steering residuals = 45
      left lane:     9 speeds x 3 steering residuals = 27
      right lane:    9 speeds x 3 steering residuals = 27
      braking:       8 deceleration levels + 1 emergency stop = 9

    Missing adjacent lanes retain zero-filled slots with valid=False. Thus the
    tensor size stays fixed at 108 while geometric availability depends on road
    topology. Candidate positions have no semantic embedding or learned ID.
    """

    def __init__(self, config=None):
        self.resolved_config = load_candidate_generation_config(config)
        self.config = self.resolved_config
        self.base = AllMergeAnchorBuilder(self.config["base"])
        self.control_steps = self.base.control_steps
        self.horizon_steps = self.base.horizon_steps
        self.steer_residuals = tuple(self.config["steering_residuals_rad"])
        self.brake_fractions = tuple(self.config["brake_fractions"])
        self.current_speed_samples = self.config["current_lane_speed_samples"]
        self.adjacent_speed_samples = self.config["adjacent_lane_speed_samples"]
        self.include_emergency_stop = self.config["include_emergency_stop"]
        self.num_candidates = self.config["candidate_pool_size"]

    def _rollout_many(self, controls, initial_speed, *, return_dense=False):
        controls = np.asarray(controls, dtype=np.float32)
        if controls.ndim != 3 or controls.shape[1:] != (self.control_steps, 2):
            raise ValueError(
                f"controls must be [M,{self.control_steps},2], got {controls.shape}"
            )
        count = controls.shape[0]
        dt, wb = self.base.control_dt, self.base.wheelbase
        x = np.zeros(count, dtype=np.float64)
        y = np.zeros(count, dtype=np.float64)
        psi = np.zeros(count, dtype=np.float64)
        velocity = np.full(count, max(float(initial_speed), 0.0), dtype=np.float64)
        dense = np.zeros((count, self.control_steps, 2), dtype=np.float32)
        for step in range(self.control_steps):
            acceleration = controls[:, step, 0].astype(np.float64, copy=False)
            steering = controls[:, step, 1].astype(np.float64, copy=False)
            next_v = np.maximum(0.0, velocity + acceleration * dt)
            vmid = 0.5 * (velocity + next_v)
            yaw = vmid * np.tan(steering) * dt / wb
            heading_mid = psi + 0.5 * yaw
            x += vmid * np.cos(heading_mid) * dt
            y += vmid * np.sin(heading_mid) * dt
            psi += yaw
            velocity = next_v
            dense[:, step, 0] = x
            dense[:, step, 1] = y
        stride = self.control_steps // self.horizon_steps
        coarse = dense[:, stride - 1::stride].copy()
        return (coarse, dense.copy()) if return_dense else coarse

    def build_single(self, *, ego_state, map_polylines, map_valid_mask,
                     target_acceleration=0.0):
        base = self.base
        ego_state = np.asarray(ego_state, dtype=np.float32)
        polylines = np.asarray(map_polylines, dtype=np.float32)
        mask = np.asarray(map_valid_mask, dtype=bool)
        current, left, right = base._extract_lane_polylines(polylines, mask)
        if current is None:
            x = np.linspace(0.0, 120.0, 64, dtype=np.float32)
            current = np.stack((x, np.zeros_like(x)), axis=1)
        current = _forward_polyline(current)
        left = _forward_polyline(left) if left is not None else None
        right = _forward_polyline(right) if right is not None else None

        v0 = max(0.0, float(ego_state[0]))
        limit = base._resolve_speed_limit(polylines, mask, max(v0 + base.speed_delta, 30.0))
        levels = base._dynamic_speed_levels(
            ego_speed=v0, target_acceleration=target_acceleration, speed_limit=limit
        )
        low, high = min(levels.values()), max(levels.values())

        residuals = self.steer_residuals
        residual_phase = np.sin(
            np.pi * (np.arange(self.control_steps, dtype=np.float32) + 1.0)
            / self.control_steps
        ).astype(np.float32)
        controls = np.zeros((self.num_candidates, self.control_steps, 2), dtype=np.float32)
        valid = np.zeros(self.num_candidates, dtype=bool)
        index = 0

        lane_specs = (
            (current, self.current_speed_samples),
            (left, self.adjacent_speed_samples),
            (right, self.adjacent_speed_samples),
        )
        for lane, speed_count in lane_specs:
            speeds = np.linspace(low, high, speed_count, dtype=np.float32)
            if lane is None:
                index += speed_count * len(residuals)
                continue
            for speed in speeds:
                base_u, _ = base._sample_and_rollout(
                    current, lane, v0, float(speed),
                    lane_change=(lane is not current), emergency_stop=False,
                )
                for residual in residuals:
                    u = base_u.copy()
                    if residual:
                        u[:, 1] = np.clip(
                            u[:, 1] + residual * residual_phase,
                            -base.max_steer, base.max_steer,
                        )
                    controls[index] = u
                    valid[index] = True
                    index += 1

        base_brake_u, _ = base._sample_and_rollout(
            current, current, v0, 0.0,
            lane_change=False, emergency_stop=False,
        )
        for fraction in self.brake_fractions:
            u = base_brake_u.copy()
            u[:, 0] = np.maximum(u[:, 0], -base.max_decel * fraction)
            controls[index] = u
            valid[index] = True
            index += 1

        if self.include_emergency_stop:
            stop_u, _ = base._sample_and_rollout(
                current, current, v0, 0.0,
                lane_change=False, emergency_stop=True,
            )
            controls[index] = stop_u
            valid[index] = True
            index += 1

        if index != self.num_candidates:
            raise RuntimeError(
                f"Candidate generator filled {index} slots, expected {self.num_candidates}"
            )
        tracks = self._rollout_many(controls, v0)
        tracks[~valid] = 0.0
        return {
            "coarse_controls": controls,
            "coarse_trajectories": tracks,
            "mode_valid_mask": valid,
        }

    def build_batch(self, features, target_accelerations=None):
        def array(name):
            value = features[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value)
        ego = array("ego_state")
        maps = array("map_polylines")
        mask = array("map_valid_mask")
        n = len(ego)
        if target_accelerations is None:
            accelerations = np.asarray(ego[:, 2], dtype=np.float32)
        else:
            accelerations = np.asarray(target_accelerations, dtype=np.float32)
        if accelerations.shape != (n,):
            raise ValueError(f"target_accelerations must be [{n}]")
        rows = [
            self.build_single(
                ego_state=ego[i], map_polylines=maps[i],
                map_valid_mask=mask[i], target_acceleration=float(accelerations[i]),
            )
            for i in range(n)
        ]
        return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def _joint_pairwise_distance(controls, trajectories, *, trajectory_weight=1.0,
                             control_weight=1.0):
    trajectory = trajectories / trajectories.new_tensor([120.0, 24.0])
    control = controls / controls.new_tensor([2.5, 0.08])
    d_traj = (trajectory[:, None] - trajectory[None, :]).norm(dim=-1).mean(-1)
    d_ctrl = (control[:, None] - control[None, :]).norm(dim=-1).mean(-1)
    return trajectory_weight * d_traj + control_weight * d_ctrl


def select_diverse_anchors(controls, trajectories, valid,
                           *, num_selected=10, trajectory_weight=1.0,
                           control_weight=1.0):
    """Deterministic A1 diversity-only selection over the complete candidate pool.

    The first anchor is the medoid of the joint control/trajectory distance
    matrix. Remaining anchors maximize their minimum distance to the selected
    set. No safety score or candidate semantic identity is used.
    """
    import torch
    if controls.ndim != 4 or controls.shape[-2:] != (40, 2):
        raise ValueError("controls must be [B,M,40,2]")
    if trajectories.shape != (controls.shape[0], controls.shape[1], 8, 2):
        raise ValueError("trajectories must be [B,M,8,2]")
    if valid.shape != controls.shape[:2]:
        raise ValueError("valid must be [B,M]")
    if int(num_selected) <= 0 or (valid.sum(1) < int(num_selected)).any():
        raise ValueError("Insufficient valid candidates for requested A1 anchor count")
    output = torch.empty(
        (controls.shape[0], int(num_selected)), dtype=torch.long, device=controls.device
    )
    for b in range(controls.shape[0]):
        ids = torch.nonzero(valid[b].bool(), as_tuple=False).flatten()
        c = controls[b, ids]
        a = trajectories[b, ids]
        dist = _joint_pairwise_distance(
            c, a, trajectory_weight=trajectory_weight, control_weight=control_weight
        )
        # Deterministic medoid seed; torch.argmin resolves ties by first index.
        selected_local = [int(dist.mean(dim=1).argmin().item())]
        min_dist = dist[selected_local[0]].clone()
        for _ in range(1, int(num_selected)):
            available = torch.ones(len(ids), dtype=torch.bool, device=ids.device)
            available[selected_local] = False
            next_local = int(min_dist.masked_fill(~available, -torch.inf).argmax().item())
            selected_local.append(next_local)
            min_dist = torch.minimum(min_dist, dist[next_local])
        output[b] = ids[torch.as_tensor(selected_local, device=ids.device)]
    return output
