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

    def _rollout_many(self, controls, initial_speed):
        """Vectorized midpoint bicycle rollout over all candidate anchors."""
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
        return dense[:, stride - 1 :: stride].copy()

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
        valid = np.zeros(36, dtype=bool)
        index = 0
        # The steering residual is applied *after* the road-conditioned
        # base control has been sampled.  Therefore all three residual variants
        # share exactly the same base control sequence.  Sample it once and
        # branch by copy instead of repeating the 40-step pure-pursuit rollout
        # three times.
        phase = np.sin(
            np.pi * (np.arange(self.control_steps, dtype=np.float32) + 1.0)
            / self.control_steps
        ).astype(np.float32)
        for lane, speeds in lanes:
            for speed in speeds:
                if lane is not None:
                    base_u, _ = base._sample_and_rollout(
                        current, lane, v0, float(speed),
                        lane_change=(lane is not current), emergency_stop=False,
                    )
                    for residual in self.steer_residuals:
                        u = base_u.copy()
                        if residual:
                            # Steering residual vanishes at t=0 and t=T,
                            # preventing a discontinuous initial steering jump.
                            u[:, 1] = np.clip(
                                u[:, 1] + residual * phase,
                                -base.max_steer, base.max_steer,
                            )
                        controls[index] = u
                        valid[index] = True
                        index += 1
                else:
                    # Preserve the three masked candidate slots for a missing
                    # adjacent lane.
                    index += len(self.steer_residuals)

        # The 0.35/0.65 braking candidates share the same non-emergency base
        # control.  Only the longitudinal clipping differs.  The emergency
        # stop remains a separate sample because its acceleration law differs.
        base_brake_u, _ = base._sample_and_rollout(
            current, current, v0, 0.0,
            lane_change=False, emergency_stop=False,
        )
        for brake in (0.35, 0.65):
            u = base_brake_u.copy()
            u[:, 0] = np.maximum(u[:, 0], -base.max_decel * brake)
            controls[index] = u
            valid[index] = True
            index += 1

        stop_u, _ = base._sample_and_rollout(
            current, current, v0, 0.0,
            lane_change=False, emergency_stop=True,
        )
        controls[index] = stop_u
        valid[index] = True
        index += 1
        assert index == self.num_candidates
        tracks = self._rollout_many(controls, v0)
        tracks[~valid] = 0.0
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
