"""Control-space candidate bank with no semantic-mode identities.

The existing A1 road-conditioned control sampler is reused for lateral road
references and speed targets. Every candidate is a physical control sequence
paired with its own forward kinematic rollout. Candidate array positions carry
no learned identity and can be reordered freely.
"""
from __future__ import annotations

import numpy as np

from highway_env.planner.anchor_builder import (
    AllMergeAnchorBuilder, _forward_polyline,
)


class AdaptiveCandidateBuilder:
    """Deterministic 36-candidate pool: 15 current, 9 per neighbor, 3 brakes.

    Missing adjacent lanes are represented by masked zero-valued candidates;
    at least 18 geometrically defined candidates remain on a single-lane road.
    The candidate count is independent of the trained planner's final K_a.
    """

    def __init__(self, config=None):
        self.base = AllMergeAnchorBuilder(config)
        self.control_steps = self.base.control_steps
        self.horizon_steps = self.base.horizon_steps
        self.num_candidates = 36
        self.steer_residuals = (-0.006, 0.0, 0.006)

    def _rollout(self, controls, initial_speed):
        """Use the same midpoint bicycle integration as the A1 sampler."""
        dt, wb = self.base.control_dt, self.base.wheelbase
        x = y = psi = 0.0
        v = max(float(initial_speed), 0.0)
        dense = np.zeros((self.control_steps, 2), dtype=np.float32)
        for j, (acceleration, steering) in enumerate(controls):
            next_v = max(0.0, v + float(acceleration) * dt)
            vmid = 0.5 * (v + next_v)
            yaw = vmid * np.tan(float(steering)) * dt / wb
            x += vmid * np.cos(psi + 0.5 * yaw) * dt
            y += vmid * np.sin(psi + 0.5 * yaw) * dt
            psi += yaw
            v = next_v
            dense[j] = (x, y)
        stride = self.control_steps // self.horizon_steps
        return dense[stride - 1 :: stride].copy()

    def build_single(self, *, ego_state, map_polylines, map_valid_mask,
                     target_acceleration=0.0):
        base = self.base
        ego_state = np.asarray(ego_state, dtype=np.float32)
        polylines = np.asarray(map_polylines, dtype=np.float32)
        mask = np.asarray(map_valid_mask, dtype=bool)
        current, left, right = base._extract_lane_polylines(polylines, mask)
        if current is None:
            # Same no-map fallback as A1.
            x = np.linspace(0.0, 120.0, 64, dtype=np.float32)
            current = np.stack((x, np.zeros_like(x)), axis=1)
        current = _forward_polyline(current)
        left = _forward_polyline(left) if left is not None else None
        right = _forward_polyline(right) if right is not None else None
        v0 = max(0.0, float(ego_state[0]))
        limit = base._resolve_speed_limit(polylines, mask, max(v0 + base.speed_delta, 30.0))
        levels = base._dynamic_speed_levels(ego_speed=v0,
                    target_acceleration=target_acceleration, speed_limit=limit)
        low, high = min(levels.values()), max(levels.values())
        # Road-conditioned longitudinal target acceleration and continuous
        # steering residuals. No semantic mode slot, index, or embedding.
        lanes = ((current, np.linspace(low, high, 5)),
                 (left, np.linspace(low, high, 3)),
                 (right, np.linspace(low, high, 3)))
        controls = np.zeros((36, self.control_steps, 2), dtype=np.float32)
        tracks = np.zeros((36, self.horizon_steps, 2), dtype=np.float32)
        valid = np.zeros(36, dtype=bool)
        index = 0
        for lane, speeds in lanes:
            for speed in speeds:
                for residual in self.steer_residuals:
                    if lane is not None:
                        u, _ = base._sample_and_rollout(
                            current, lane, v0, float(speed),
                            lane_change=(lane is not current), emergency_stop=False,
                        )
                        if residual:
                            # Steering residual vanishes at t=0 and t=T,
                            # preventing a discontinuous initial steering jump.
                            phase = np.sin(np.pi * (np.arange(self.control_steps) + 1)
                                           / self.control_steps).astype(np.float32)
                            u[:, 1] = np.clip(u[:, 1] + residual * phase,
                                               -base.max_steer, base.max_steer)
                        controls[index] = u
                        tracks[index] = self._rollout(u, v0)
                        valid[index] = True
                    index += 1
        for brake in (0.35, 0.65, 1.0):
            u, _ = base._sample_and_rollout(
                current, current, v0, 0.0,
                lane_change=False, emergency_stop=(brake == 1.0),
            )
            if brake != 1.0:
                u[:, 0] = np.maximum(u[:, 0], -base.max_decel * brake)
            controls[index] = u
            tracks[index] = self._rollout(u, v0)
            valid[index] = True
            index += 1
        assert index == self.num_candidates
        return {'coarse_controls': controls,
                'coarse_trajectories': tracks,
                'mode_valid_mask': valid}

    def build_batch(self, features, target_accelerations=None):
        def array(name):
            value = features[name]
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            return np.asarray(value)
        ego, maps, mask = array('ego_state'), array('map_polylines'), array('map_valid_mask')
        n = len(ego)
        if target_accelerations is None:
            # Observed ego longitudinal acceleration; no expert-future leakage.
            accelerations = np.asarray(ego[:, 2], dtype=np.float32)
        else:
            accelerations = np.asarray(target_accelerations, dtype=np.float32)
        if accelerations.shape != (n,):
            raise ValueError(f'target_accelerations must be [{n}]')
        rows = [self.build_single(ego_state=ego[i], map_polylines=maps[i],
                                 map_valid_mask=mask[i],
                                 target_acceleration=float(accelerations[i])) for i in range(n)]
        return {key: np.stack([row[key] for row in rows]) for key in rows[0]}
