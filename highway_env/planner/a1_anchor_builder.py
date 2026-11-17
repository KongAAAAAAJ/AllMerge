"""Offline A1 lattice anchors from raw expert samples and exact lane geometry."""
from __future__ import annotations

import numpy as np

MODE_NAMES = ("KEEP_HIGH", "KEEP_MEDIUM", "KEEP_LOW",
              "LEFT_LC_HIGH", "LEFT_LC_MEDIUM", "LEFT_LC_LOW",
              "RIGHT_LC_HIGH", "RIGHT_LC_MEDIUM", "RIGHT_LC_LOW", "STOP")


def _resample_lane(geometry: np.ndarray, mask: np.ndarray, column: int = 8):
    lane = np.asarray(geometry[mask], np.float64)
    order = np.argsort(lane[:, column])
    lane = lane[order]
    _, unique = np.unique(lane[:, column], return_index=True)
    return lane[np.sort(unique)]


def _interp(lane: np.ndarray, s: np.ndarray, columns=(0, 1)) -> np.ndarray:
    ss = lane[:, 8]
    return np.stack([np.interp(s, ss, lane[:, c]) for c in columns], axis=-1)


def _quintic(x):
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return 10*x**3 - 15*x**4 + 6*x**5


class A1AnchorBuilder:
    """Build 10 road-bounded trajectory/control anchors per raw sample."""

    def __init__(self, config: dict):
        self.cfg = config
        self.steps = int(config.get("control_steps", 40))
        self.dt = float(config.get("dt", 0.1))
        self.wheelbase = float(config.get("wheelbase_m", 5.5))
        self.length = float(config.get("ego_length_m", 8.0))
        self.width = float(config.get("ego_width_m", 2.6))
        self.max_accel = float(config.get("max_accel_mps2", 2.5))
        self.max_decel = float(config.get("max_decel_mps2", 4.0))
        self.max_steer = float(config.get("max_steer_rad", 0.08))
        self.lane_change_s = float(config.get("lane_change_distance_m", 65.0))
        self.edge_margin = float(config.get("road_edge_margin_m", 0.15))
        self.speed_delta = float(config.get("speed_delta_mps", 3.0))
        self.min_speed = float(config.get("minimum_lane_change_speed_mps", 3.0))

    def build(self, sample: dict) -> dict:
        geom = sample["road_lane_geometry"]
        point_mask = sample["road_lane_point_mask"]
        lane_mask = sample["road_lane_mask"].astype(bool)
        current_i = int(sample["road_current_lane_slot"])
        target_i = int(sample["road_target_lane_slot"])
        lanes = {i: _resample_lane(geom[i], point_mask[i]) for i in np.flatnonzero(lane_mask)}
        current = lanes[current_i]
        origin = np.zeros(2, np.float64)
        current_s = float(current[np.argmin(np.sum(current[:, :2]**2, axis=1)), 8])
        current_center = _interp(current, np.asarray([current_s]))[0]
        speed = max(float(sample["ego_state"][0]), 0.0)
        speed_limit = float(sample["road_lane_meta"][current_i, 0])
        if speed_limit <= 0:
            speed_limit = speed + self.speed_delta
        speeds = np.clip([speed + self.speed_delta, speed, max(speed-self.speed_delta, 0)], 0, speed_limit)

        # Lateral semantics come from signed displacement in the ego frame.
        # Positive local y is left; lane-index numbering is never interpreted as direction.
        adjacent = {"left": None, "right": None}
        connected = sample["road_lane_connections"].astype(bool)[current_i]
        for i in lanes:
            if i == current_i or not connected[i]:
                continue
            lane = lanes[i]
            j = int(np.argmin(np.sum((lane[:, :2]-current_center)**2, axis=1)))
            lateral = float(lane[j, 1]-current_center[1])
            side = "left" if lateral > 0.5 else "right" if lateral < -0.5 else None
            if side and (adjacent[side] is None or abs(lateral) < adjacent[side][0]):
                adjacent[side] = (abs(lateral), i)

        # Explicit target lane wins when it is an adjacent lane on that side.
        if target_i != current_i and target_i in lanes:
            target_lane = lanes[target_i]
            j = int(np.argmin(np.sum((target_lane[:, :2]-current_center)**2, axis=1)))
            dy = float(target_lane[j, 1]-current_center[1])
            side = "left" if dy > 0.5 else "right" if dy < -0.5 else None
            if side:
                adjacent[side] = (abs(dy), target_i)

        controls = np.zeros((10, self.steps, 2), np.float32)
        trajectories = np.zeros((10, 8, 2), np.float32)
        valid = np.zeros(10, bool)
        dense = np.zeros((10, self.steps+1, 2), np.float32)
        reasons = ["unavailable_lane"] * 10
        for mode in range(10):
            if mode == 9:
                target_lane_i, speed_target, lc, stop = current_i, 0.0, False, True
            elif mode < 3:
                target_lane_i, speed_target, lc, stop = current_i, float(speeds[mode]), False, False
            else:
                side = "left" if mode < 6 else "right"
                entry = adjacent[side]
                if entry is None:
                    continue
                target_lane_i = entry[1]
                speed_target = max(float(speeds[(mode-3) % 3]), self.min_speed)
                lc, stop = True, False
            path, ok = self._trajectory(current, lanes[target_lane_i], current_s, speed,
                                        speed_target, lane_change=lc, stop=stop)
            if not ok:
                reasons[mode] = "road_boundary_or_control_limit"
                continue
            u = self._inverse_kinematics(path, speed, stop)
            if not np.isfinite(u).all() or np.max(np.abs(u[:, 1])) > self.max_steer + 1e-5:
                reasons[mode] = "control_limit"
                continue
            controls[mode] = u
            dense[mode] = path.astype(np.float32)
            trajectories[mode] = path[5::5, :2].astype(np.float32)
            valid[mode] = True
            reasons[mode] = "valid"

        return {"a1_coarse_controls": controls,
                "a1_coarse_trajectories": trajectories,
                "a1_dense_trajectories": dense.astype(np.float32),
                "a1_valid_mask": valid,
                "a1_invalid_reasons": np.asarray(reasons, dtype="U40")}

    def _trajectory(self, current, target, s0, v0, v_target, *, lane_change, stop):
        t = np.arange(self.steps+1, dtype=np.float64)*self.dt
        if stop:
            accel = -min(self.max_decel, max(v0, 0.0)/max(t[-1], self.dt))
        else:
            accel = np.clip((v_target-v0)/max(t[-1], self.dt), -self.max_decel, self.max_accel)
        velocity = np.maximum(v0 + accel*t, 0.0)
        if stop and velocity[-1] > 0:
            velocity = np.maximum(v0 - self.max_decel*t, 0.0)
        distance = np.concatenate([[0.0], np.cumsum(0.5*(velocity[1:]+velocity[:-1])*self.dt)])
        s = np.clip(s0 + distance, current[0, 8], current[-1, 8])
        base = _interp(current, s)
        # Start at the ego origin while smoothly joining the current lane centerline.
        join = _quintic(distance / 12.0)
        base = base - (1.0-join[:, None])*base[0]
        if lane_change:
            target_s0 = float(target[np.argmin(np.sum(target[:, :2]**2, axis=1)), 8])
            ts = np.clip(target_s0 + distance, target[0, 8], target[-1, 8])
            target_xy = _interp(target, ts)
            target_xy = target_xy - (1.0-join[:, None])*target_xy[0]
            blend = _quintic(distance / self.lane_change_s)[:, None]
            path = base*(1.0-blend) + target_xy*blend
        else:
            path = base
        path[0] = (0.0, 0.0)
        # Blend the lane-edge corridor with the same progress as the centerline.
        left = _interp(current, s, (2, 3)) - (1.0-join[:, None])*base[0]
        right = _interp(current, s, (4, 5)) - (1.0-join[:, None])*base[0]
        if lane_change:
            target_origin = _interp(target, np.asarray([target_s0]))[0]
            tl = _interp(target, ts, (2, 3)) - (1.0-join[:, None])*target_origin
            tr = _interp(target, ts, (4, 5)) - (1.0-join[:, None])*target_origin
            b = _quintic(distance/self.lane_change_s)[:, None]
            left, right = left*(1-b)+tl*b, right*(1-b)+tr*b
        center = 0.5*(left+right)
        tangent = np.gradient(center, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        lateral = np.sum((path-center)*normal, axis=1)
        half_width = 0.5*self.width + self.edge_margin
        corridor = 0.5*np.linalg.norm(left-right, axis=1)
        longitudinal_margin = 0.5*self.length
        longitudinal_ok = (s0-current[0, 8] >= longitudinal_margin and
                           current[-1, 8]-s[-1] >= longitudinal_margin)
        if lane_change:
            longitudinal_ok = longitudinal_ok and (target_s0-target[0, 8] >= longitudinal_margin and
                                                     target[-1, 8]-ts[-1] >= longitudinal_margin)
        return path, bool(longitudinal_ok and np.all(np.abs(lateral)+half_width <= corridor+1e-4))

    def _inverse_kinematics(self, path, initial_speed, stop):
        d = np.diff(path, axis=0)
        segment = np.linalg.norm(d, axis=1)
        heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        # Smooth sub-meter map discretization noise before differentiating curvature.
        padded = np.pad(heading, (2, 2), mode="edge")
        heading = np.convolve(padded, np.ones(5, dtype=np.float64)/5.0, mode="valid")
        heading = np.concatenate([heading[:1], heading])
        ds = np.maximum(segment, 1e-5)
        dpsi = np.diff(heading)
        curvature = np.concatenate([dpsi/ds, (dpsi/ds)[-1:]])
        steering = np.arctan(self.wheelbase*curvature)
        velocity = segment/self.dt
        velocity = np.concatenate([[max(initial_speed, 0.0)], velocity])
        acceleration = np.diff(velocity)/self.dt
        if stop:
            acceleration = np.minimum(acceleration, 0.0)
        acceleration = np.clip(acceleration, -self.max_decel, self.max_accel)
        return np.stack([acceleration, steering[:self.steps]], axis=-1).astype(np.float32)
