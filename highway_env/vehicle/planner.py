import math
import numpy as np
import copy

from highway_env.vehicle.dynamics_EightDoF_Model import dynamic_check
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


class PolyPlanner:
    """
    Polynomial path planning with selectable longitudinal speed prediction
    """
    """换道时间阈值"""
    T_LC_MIN = 1
    T_LC_MAX = 4

    """间隔时间"""
    DT = 0.1

    """第二段轨迹规划时间"""
    T_NEXT = 1

    """单次规划轨迹数"""
    NUM_PATHS = 5

    def __init__(
            self,
            controlled_vehicle,
            vehicle_type,
            road,
            target_lane_index,
            lead_vehicle = None,
            follow_index = None,
            group = None,
            planner_mode = "legacy",
            aligned_horizon_s = 4.0,
    ):
        self.controlled_vehicle = controlled_vehicle
        self.vehicle_type = vehicle_type
        self.lead_vehicle = lead_vehicle
        self.follow_index = follow_index
        self.group = group
        self.road = road
        self.target_lane_index = target_lane_index

        if planner_mode not in {"legacy", "aligned"}:
            raise ValueError(
                "PolyPlanner planner_mode must be 'legacy' or 'aligned'"
            )

        self.planner_mode = planner_mode
        self.aligned_horizon_s = float(aligned_horizon_s)

        if self.aligned_horizon_s <= 0.0:
            raise ValueError("aligned_horizon_s must be positive")

        self.last_time_s = None
    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _quintic_coefficients(p0, v0, a0, pf, vf, af, tf):
        tf = max(float(tf), 1e-3)
        matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [1.0, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                [0.0, 1.0, 2.0 * tf, 3.0 * tf ** 2,
                 4.0 * tf ** 3, 5.0 * tf ** 4],
                [0.0, 0.0, 2.0, 6.0 * tf,
                 12.0 * tf ** 2, 20.0 * tf ** 3],
            ],
            dtype=float,
        )
        boundary = np.array([p0, v0, a0, pf, vf, af], dtype=float)
        return np.linalg.solve(matrix, boundary)

    @staticmethod
    def _eval_quintic(coefficients, t):
        c = coefficients
        t = float(t)
        position = (
            c[0] + c[1] * t + c[2] * t ** 2
            + c[3] * t ** 3 + c[4] * t ** 4 + c[5] * t ** 5
        )
        velocity = (
            c[1] + 2.0 * c[2] * t + 3.0 * c[3] * t ** 2
            + 4.0 * c[4] * t ** 3 + 5.0 * c[5] * t ** 4
        )
        return float(position), float(velocity)

    def _frenet_time_grid(self, count):
        return np.arange(int(count), dtype=float) * float(self.DT)

    def _target_lane_frenet_state(self):
        target_lane = self.road.network.get_lane(self.target_lane_index)
        s0, d0 = target_lane.local_coordinates(
            self.controlled_vehicle.position
        )
        s0 = float(s0)
        d0 = float(d0)

        lane_heading = float(
            target_lane.heading_at(
                np.clip(s0, 0.0, target_lane.length)
            )
        )
        heading_error = self._wrap_angle(
            float(self.controlled_vehicle.heading) - lane_heading
        )
        speed = float(self.controlled_vehicle.speed)
        s_dot0 = speed * math.cos(heading_error)
        d_dot0 = speed * math.sin(heading_error)
        return target_lane, s0, d0, s_dot0, d_dot0

    def _frenet_lateral_profile(self, times, d0, d_dot0, t_lc):
        times = np.asarray(times, dtype=float)
        if abs(float(d0)) < 1e-6 and abs(float(d_dot0)) < 1e-6:
            return np.zeros_like(times), np.zeros_like(times)

        coefficients = self._quintic_coefficients(
            p0=float(d0),
            v0=float(d_dot0),
            a0=0.0,
            pf=0.0,
            vf=0.0,
            af=0.0,
            tf=float(t_lc),
        )

        d_values = np.zeros_like(times)
        d_speeds = np.zeros_like(times)
        for i, t in enumerate(times):
            if t <= t_lc:
                d_values[i], d_speeds[i] = self._eval_quintic(
                    coefficients, t
                )
            else:
                d_values[i] = 0.0
                d_speeds[i] = 0.0
        return d_values, d_speeds

    def _integrate_frenet_longitudinal(self, s0, s_speeds):
        s_speeds = np.asarray(s_speeds, dtype=float).reshape(-1)
        count = len(s_speeds)
        s_positions = np.zeros(count, dtype=float)
        if count == 0:
            return s_positions

        s_positions[0] = float(s0)
        for i in range(1, count):
            s_positions[i] = (
                s_positions[i - 1]
                + 0.5 * (s_speeds[i - 1] + s_speeds[i]) * float(self.DT)
            )
        return s_positions

    def _frenet_to_world(self, lane, s_positions, d_positions):
        """Follow connected lanes; extend a terminal tangent instead of clipping.

        The finite extent of a scenario lane is not a zero-speed target. A
        forecast beyond a terminal segment remains moving along its tangent;
        this does not add a road to the simulator or guarantee road compliance.
        """
        points = []
        for s_value, d_value in zip(s_positions, d_positions):
            current, index = lane, self.target_lane_index
            distance = float(s_value)
            while distance > float(current.length):
                successor = self._next_prediction_lane(current, index)
                if successor is None:
                    break
                if current.length <= 0:
                    raise ValueError('Expert planning requires positive lane lengths')
                distance -= float(current.length)
                index, current = successor
            boundary = float(np.clip(distance, 0.0, current.length))
            point = np.asarray(current.position(boundary, float(d_value)), dtype=float)
            if distance != boundary:
                heading = float(current.heading_at(boundary))
                point = point + (distance - boundary) * np.array([math.cos(heading), math.sin(heading)])
            points.append(point)
        return np.asarray(points, dtype=float)

    def _next_prediction_lane(self, lane, index):
        """Choose a deterministic geometric continuation, respecting the route."""
        network = self.road.network
        edges = network.graph.get(index[1], {})
        route = getattr(self.controlled_vehicle, 'route', None) or []
        planned = next((entry for entry in route if entry[0] == index[1]), None)
        if planned is not None:
            edges = {planned[1]: edges[planned[1]]} if planned[1] in edges else {}
        endpoint = np.asarray(lane.position(lane.length, 0.0))
        heading = float(lane.heading_at(lane.length))
        candidates = []
        for target, lanes in edges.items():
            for lane_id, successor in enumerate(lanes):
                if planned is not None and planned[2] is not None and lane_id != planned[2]:
                    continue
                gap = float(np.linalg.norm(np.asarray(successor.position(0.0, 0.0)) - endpoint))
                # Do not jump across an unmodelled gap or onto a parallel road.
                if gap > 1e-3:
                    continue
                turn = abs(self._wrap_angle(float(successor.heading_at(0.0)) - heading))
                candidates.append((turn, str(target), lane_id, target, successor))
        if not candidates:
            return None
        _, _, lane_id, target, successor = min(candidates, key=lambda value: value[:3])
        return (index[1], target, lane_id), successor

    def _world_path_kinematics(
            self,
            world_xy,
            fallback_lane,
            s_positions,
            s_speeds,
            d_speeds,
    ):
        world_xy = np.asarray(world_xy, dtype=float)
        count = len(world_xy)
        if count == 0:
            return [], [], []

        if count == 1:
            s0 = float(np.clip(s_positions[0], 0.0, fallback_lane.length))
            heading = float(fallback_lane.heading_at(s0))
            speed = float(math.hypot(s_speeds[0], d_speeds[0]))
            return [heading], [speed], []

        tangent = np.zeros_like(world_xy)
        tangent[0] = world_xy[1] - world_xy[0]
        tangent[-1] = world_xy[-1] - world_xy[-2]
        if count > 2:
            tangent[1:-1] = world_xy[2:] - world_xy[:-2]

        headings = []
        for i, direction in enumerate(tangent):
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-9:
                s_value = float(
                    np.clip(s_positions[i], 0.0, fallback_lane.length)
                )
                heading = float(fallback_lane.heading_at(s_value))
            else:
                heading = math.atan2(
                    float(direction[1]),
                    float(direction[0]),
                )
            headings.append(heading)

        headings_unwrapped = np.unwrap(np.asarray(headings, dtype=float))
        speeds = np.hypot(
            np.asarray(s_speeds, dtype=float),
            np.asarray(d_speeds, dtype=float),
        )

        steering_angles = []
        for i in range(1, count):
            yaw_rate = (
                headings_unwrapped[i] - headings_unwrapped[i - 1]
            ) / float(self.DT)
            speed = max(float(speeds[i - 1]), 1e-3)
            steering = math.atan(
                float(self.controlled_vehicle.LENGTH) * yaw_rate / speed
            )
            steering_angles.append(steering)

        return (
            headings_unwrapped.tolist(),
            speeds.tolist(),
            steering_angles,
        )

    def _build_frenet_candidate(self, t_lc, total_horizon):
        target_lane, s0, d0, _, d_dot0 = self._target_lane_frenet_state()

        accelerations, s_speeds, _legacy_x_positions = (
            self.speed_planning(float(total_horizon))
        )
        accelerations = np.asarray(accelerations, dtype=float).reshape(-1)
        s_speeds = np.asarray(s_speeds, dtype=float).reshape(-1)

        if len(s_speeds) == 0:
            raise RuntimeError(
                "PolyPlanner.speed_planning returned an empty speed profile"
            )

        times = self._frenet_time_grid(len(s_speeds))
        d_positions, d_speeds = self._frenet_lateral_profile(
            times=times,
            d0=d0,
            d_dot0=d_dot0,
            t_lc=float(t_lc),
        )
        s_positions = self._integrate_frenet_longitudinal(
            s0=s0,
            s_speeds=s_speeds,
        )
        world_xy = self._frenet_to_world(
            lane=target_lane,
            s_positions=s_positions,
            d_positions=d_positions,
        )
        headings, speeds, steering_angles = self._world_path_kinematics(
            world_xy=world_xy,
            fallback_lane=target_lane,
            s_positions=s_positions,
            s_speeds=s_speeds,
            d_speeds=d_speeds,
        )

        steering_count = len(steering_angles)
        vx = s_speeds[:steering_count]
        ax = accelerations[:steering_count]
        if len(ax) < steering_count:
            if len(ax) == 0:
                ax = np.zeros(steering_count, dtype=float)
            else:
                ax = np.pad(
                    ax,
                    (0, steering_count - len(ax)),
                    mode="edge",
                )

        check_time = np.arange(steering_count, dtype=float) * float(self.DT)
        safe, ltr_values = dynamic_check(
            t=check_time,
            delta_f=np.asarray(steering_angles, dtype=float),
            vx=np.asarray(vx, dtype=float),
            ax=np.asarray(ax, dtype=float),
        )

        return {
            "safe": bool(safe),
            "ltr_values": ltr_values,
            "times": times,
            "headings": headings,
            "speeds": speeds,
            "steering_angles": steering_angles,
            "world_x": world_xy[:, 0].tolist(),
            "world_y": world_xy[:, 1].tolist(),
            "d0": float(d0),
            "lane_width": float(
                target_lane.width_at(
                    np.clip(s0, 0.0, target_lane.length)
                )
            ),
        }

    def planning(self):
        target_lane, s0, d0, _, _ = self._target_lane_frenet_state()

        lane_width = max(
            float(
                target_lane.width_at(
                    np.clip(s0, 0.0, target_lane.length)
                )
            ),
            1e-3,
        )
        lateral_ratio = abs(float(d0)) / lane_width

        if lateral_ratio < 1e-3:
            candidate_times = [max(float(self.DT), 1e-3)]
        else:
            t_min = max(
                float(self.DT),
                lateral_ratio * float(self.T_LC_MIN),
            )
            t_max = max(
                t_min,
                lateral_ratio * float(self.T_LC_MAX),
            )
            candidate_times = np.linspace(
                t_min,
                t_max,
                int(self.NUM_PATHS),
            )

        last_candidate = None

        for t_lc in candidate_times:
            legacy_total_horizon = float(t_lc) + float(self.T_NEXT)
            if self.planner_mode == "aligned":
                total_horizon = max(
                    float(self.aligned_horizon_s),
                    legacy_total_horizon,
                )
            else:
                total_horizon = legacy_total_horizon

            candidate = self._build_frenet_candidate(
                t_lc=float(t_lc),
                total_horizon=total_horizon,
            )
            last_candidate = candidate

            if not candidate["safe"]:
                continue

            self.last_time_s = np.asarray(
                candidate["times"],
                dtype=np.float32,
            )

            return (
                candidate["headings"],
                candidate["speeds"],
                candidate["steering_angles"],
                candidate["world_x"],
                candidate["world_y"],
                candidate["ltr_values"],
            )

        if last_candidate is None:
            self.last_time_s = None

        return None, None, None, None, None, None
    def lc_planning(self, target_y):
        lc_coefficient = abs(
            self.controlled_vehicle.position[1] - target_y
        ) / 4

        if lc_coefficient == 0:
            lc_coefficient = 1

        t_min = lc_coefficient * self.T_LC_MIN
        t_max = lc_coefficient * self.T_LC_MAX
        t_min = self.DT if t_min < self.DT else t_min
        t_max = self.DT if t_max < self.DT else t_max

        # Keep legacy candidate selection unchanged.
        for t_lc in np.linspace(t_min, t_max, self.NUM_PATHS):
            legacy_total_horizon = t_lc + self.T_NEXT

            accelerations, x_speeds, x_positions = self.speed_planning(
                legacy_total_horizon
            )

            headings, y_speeds, y_positions, speeds = self.path_planning(
                t_lc,
                target_y,
                x_positions,
                x_speeds,
                post_horizon_s=self.T_NEXT,
            )

            steering_angles = []

            for i in range(1, len(headings)):
                steering = math.atan(
                    self.controlled_vehicle.LENGTH
                    * (headings[i] - headings[i - 1])
                    / speeds[i - 1]
                    / self.DT
                )
                steering_angles.append(steering)

            times = np.linspace(0, t_lc, len(steering_angles))

            isRollover, LTR_values = dynamic_check(
                t=times,
                delta_f=np.array(steering_angles),
                vx=np.array(x_speeds[:-1]),
                ax=np.array(accelerations[:-1]),
            )

            if isRollover:
                if self.planner_mode == "legacy":
                    returned_count = max(len(x_positions) - 1, 0)

                    self.last_time_s = (
                        np.arange(
                            returned_count,
                            dtype=np.float32,
                        )
                        * self.DT
                    )

                    return (
                        headings[:-1],
                        speeds[:-1],
                        steering_angles,
                        x_positions[:-1],
                        y_positions[:-1],
                        LTR_values,
                    )

                # The same candidate is already selected. Only extend
                # the returned benchmark path to the common 4 s horizon.
                aligned_total_horizon = max(
                    float(self.aligned_horizon_s),
                    float(legacy_total_horizon),
                )

                aligned_post_horizon = max(
                    float(self.T_NEXT),
                    aligned_total_horizon - float(t_lc),
                )

                (
                    aligned_accelerations,
                    aligned_x_speeds,
                    aligned_x_positions,
                ) = self.speed_planning(aligned_total_horizon)

                (
                    aligned_headings,
                    aligned_y_speeds,
                    aligned_y_positions,
                    aligned_speeds,
                ) = self.path_planning(
                    t_lc,
                    target_y,
                    aligned_x_positions,
                    aligned_x_speeds,
                    post_horizon_s=aligned_post_horizon,
                )

                aligned_steering_angles = []

                for i in range(1, len(aligned_headings)):
                    steering = math.atan(
                        self.controlled_vehicle.LENGTH
                        * (
                            aligned_headings[i]
                            - aligned_headings[i - 1]
                        )
                        / aligned_speeds[i - 1]
                        / self.DT
                    )
                    aligned_steering_angles.append(steering)

                self.last_time_s = (
                    np.arange(
                        len(aligned_x_positions),
                        dtype=np.float32,
                    )
                    * self.DT
                )

                return (
                    aligned_headings,
                    aligned_speeds,
                    aligned_steering_angles,
                    aligned_x_positions,
                    aligned_y_positions,
                    LTR_values,
                )

        return None, None, None, None, None, None
    def speed_planning(self, t_lim) -> tuple:
        """
        Speed planning with the selected longitudinal controller
        """
        accelerations, x_speeds, x_positions = [], [], []
        copy_controlled_vehicle = copy.deepcopy(self.controlled_vehicle)
        context = getattr(copy_controlled_vehicle.road, "longitudinal_control", None)
        use_lmpc = context is not None and context.kind == "lmpc"
        if use_lmpc:
            copy_controlled_vehicle._lon_dt = self.DT
            ego_id = getattr(copy_controlled_vehicle, "_lon_id", None)
            if ego_id is not None:
                for index, vehicle in enumerate(copy_controlled_vehicle.road.vehicles):
                    if getattr(vehicle, "_lon_id", None) == ego_id:
                        copy_controlled_vehicle.road.vehicles[index] = copy_controlled_vehicle
        lead_copy = self.lead_vehicle
        if use_lmpc:
            # Use references from the same copied road and advance that road's
            # states. A follower's neighbour lookup must see moving references.
            def planning_reference(original):
                if original is None:
                    return None
                for index, vehicle in enumerate(self.road.vehicles):
                    if vehicle is original:
                        return copy_controlled_vehicle.road.vehicles[index]
                return copy.deepcopy(original)
            lead_copy = planning_reference(self.lead_vehicle)

        front_v, _ = self.road.neighbour_vehicles(
            vehicle=self.controlled_vehicle, lane_index=self.controlled_vehicle.lane_index
        )
        front_v_copy = planning_reference(front_v) if use_lmpc else copy.deepcopy(front_v)

        for _ in np.linspace(0, t_lim, int(t_lim / self.DT) + 1):
            # Is it a leader vehicle?
            if self.vehicle_type == 'Leader':
                acc = copy_controlled_vehicle.integrated_longitudinal_control(
                    ego_vehicle=copy_controlled_vehicle if use_lmpc else self.controlled_vehicle, front_vehicle=front_v_copy
                )
            else:
                acc = copy_controlled_vehicle.integrated_longitudinal_control(
                    lead_vehicle=lead_copy, follow_index=self.follow_index, group=self.group
                )
            if use_lmpc:
                copy_controlled_vehicle.action["acceleration"] = acc
                # Planning uses a kinematic snapshot, not stale TruckSim feedback.
                if hasattr(copy_controlled_vehicle, "measured_acceleration"):
                    copy_controlled_vehicle.measured_acceleration[0] = acc
            accelerations.append(acc)
            x_speeds.append(copy_controlled_vehicle.speed)
            x_positions.append(copy_controlled_vehicle.position[0])

            # Update front vehicle, if it exits
            if use_lmpc:
                references = list(copy_controlled_vehicle.road.vehicles)
                for reference in (lead_copy, front_v_copy):
                    if reference is not None and all(reference is not other for other in references):
                        references.append(reference)
                for reference in references:
                    if reference is copy_controlled_vehicle:
                        continue
                    reference_a = getattr(reference, "action", {}).get("acceleration", 0)
                    reference.position[0] += reference.speed * self.DT + 0.5 * reference_a * self.DT ** 2
                    reference.speed += reference_a * self.DT
            elif front_v_copy is not None:
                front_a = front_v_copy.action['acceleration']
                front_v_copy.position[0] = front_v_copy.position[0] + front_v_copy.speed * self.DT + 0.5 * front_a * self.DT ** 2
                front_v_copy.speed = front_v_copy.speed + front_a * self.DT

            # Update ego vehicle
            copy_controlled_vehicle.position[0] = copy_controlled_vehicle.position[0] + copy_controlled_vehicle.speed * self.DT + 0.5 * acc * self.DT ** 2
            copy_controlled_vehicle.speed = copy_controlled_vehicle.speed + acc * self.DT

        return accelerations, x_speeds, x_positions

    def path_planning(
            self,
            t_lc,
            target_y,
            x_positions,
            x_speeds,
            post_horizon_s=None,
    ):
        if post_horizon_s is None:
            post_horizon_s = self.T_NEXT

        y_positions = []
        y_speeds = []
        headings = []
        speeds = []

        if self.controlled_vehicle.position[1] != target_y:
            y0 = self.controlled_vehicle.position[1]
            vy0 = 0
            ay0 = self.controlled_vehicle.action["acceleration"]
            t0 = 0
            yf = target_y
            vyf = 0
            ayf = 0
            tf = t_lc

            T = np.array([
                [1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
                [0, 1, 2 * t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
                [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
                [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                [0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3],
            ])

            Y = np.array([
                [y0],
                [vy0],
                [ay0],
                [yf],
                [vyf],
                [ayf],
            ])

            b = np.linalg.solve(T, Y)

            lane_change_count = int(t_lc / self.DT) + 1

            for t, vx in zip(
                np.linspace(0, t_lc, lane_change_count),
                x_speeds[:lane_change_count],
            ):
                vy = (
                    b[1]
                    + 2 * b[2] * t
                    + 3 * b[3] * t ** 2
                    + 4 * b[4] * t ** 3
                    + 5 * b[5] * t ** 4
                )

                y = (
                    b[0]
                    + b[1] * t
                    + b[2] * t ** 2
                    + b[3] * t ** 3
                    + b[4] * t ** 4
                    + b[5] * t ** 5
                )

                heading = math.atan(vy / vx)
                speed = math.sqrt(vx ** 2 + vy ** 2)

                headings.append(heading)
                y_speeds.append(vy)
                speeds.append(speed)
                y_positions.append(y)

            for vx in x_speeds[lane_change_count:]:
                headings.append(0)
                y_speeds.append(0)
                speeds.append(vx)
                y_positions.append(yf)

        else:
            headings = [0] * len(x_positions)
            y_speeds = [0] * len(x_positions)
            speeds = x_speeds
            y_positions = [
                self.controlled_vehicle.position[1]
            ] * len(x_positions)

        return headings, y_speeds, y_positions, speeds
