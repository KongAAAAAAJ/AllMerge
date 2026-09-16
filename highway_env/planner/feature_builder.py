"""Build Diffusion-planner inputs directly from AllMerge ground truth.

This module deliberately returns NumPy arrays only. Torch conversion,
normalization and model execution belong to the planner runtime, not the
environment feature builder.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from highway_env.planner.geometry import (
    ego_to_world_vector,
    world_to_ego_point,
    world_to_ego_vector,
    wrap_angle,
)


EGO_FEATURE_NAMES = (
    "vx_body",
    "vy_body",
    "ax_body",
    "ay_body",
    "yaw_rate",
    "steering_command",
    "steering_feedback",
    "lane_lateral_offset",
    "sin_heading_error",
    "cos_heading_error",
    "roll",
    "roll_rate",
    "pitch",
    "pitch_rate",
)

AGENT_FEATURE_NAMES = (
    "x_rel",
    "y_rel",
    "vx_rel",
    "vy_rel",
    "ax_rel",
    "ay_rel",
    "sin_heading_rel",
    "cos_heading_rel",
    "length",
    "width",
    "is_controlled",
)

MAP_FEATURE_NAMES = (
    "x_rel",
    "y_rel",
    "sin_heading_rel",
    "cos_heading_rel",
    "lane_width",
    "speed_limit",
    "is_current_lane",
    "is_left_lane",
    "is_right_lane",
    "is_target_lane",
)


class AllMergeFeatureBuilder:
    """Construct ego-centric planner features for every controlled vehicle."""

    DEFAULT_CONFIG = {
        "max_agents": 16,
        "agent_radius": 120.0,
        "max_map_polylines": 8,
        "map_points": 32,
        "map_backward_range": 20.0,
        "map_forward_range": 120.0,
        "map_lateral_range": 24.0,
        "target_point_horizon": 4.0,
        "target_point_min_lookahead": 20.0,
        "target_point_max_lookahead": 120.0,
        # Current TruckSim projects usually export Roll/Pitch/AV* in degrees.
        # Keep configurable because the .sim export unit can be changed.
        "trucksim_angular_unit": "deg",
    }

    def __init__(self, env, config: Optional[dict] = None) -> None:
        self.env = env
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)

        self.max_agents = int(self.config["max_agents"])
        self.agent_radius = float(self.config["agent_radius"])
        self.max_map_polylines = int(self.config["max_map_polylines"])
        self.map_points = int(self.config["map_points"])
        self.map_backward_range = float(self.config["map_backward_range"])
        self.map_forward_range = float(self.config["map_forward_range"])
        self.map_lateral_range = float(self.config["map_lateral_range"])

        if self.max_agents <= 0:
            raise ValueError("Planner.features.max_agents must be positive")
        if self.max_map_polylines <= 0:
            raise ValueError("Planner.features.max_map_polylines must be positive")
        if self.map_points < 2:
            raise ValueError("Planner.features.map_points must be >= 2")

    @property
    def ego_dim(self) -> int:
        return len(EGO_FEATURE_NAMES)

    @property
    def agent_dim(self) -> int:
        return len(AGENT_FEATURE_NAMES)

    @property
    def map_dim(self) -> int:
        return len(MAP_FEATURE_NAMES)

    def build_batch(
        self,
        target_lane_indices: Optional[Sequence] = None,
    ) -> Dict[str, np.ndarray]:
        """Build one ego-centric sample per controlled vehicle and stack B."""
        vehicles = list(self.env.controlled_vehicles)
        batch_size = len(vehicles)
        if batch_size == 0:
            raise RuntimeError("Cannot build planner features without controlled vehicles")

        if target_lane_indices is None:
            target_lane_indices = [None] * batch_size
        if len(target_lane_indices) != batch_size:
            raise ValueError(
                f"target_lane_indices has {len(target_lane_indices)} items, "
                f"expected {batch_size}"
            )

        samples = [
            self.build_single(ego=ego, target_lane_index=target_lane_index)
            for ego, target_lane_index in zip(vehicles, target_lane_indices)
        ]

        batch = {
            key: np.stack([sample[key] for sample in samples], axis=0)
            for key in samples[0]
        }
        self._validate_batch(batch)
        return batch

    def build_single(self, ego, target_lane_index=None) -> Dict[str, np.ndarray]:
        """Build planner features from one controlled vehicle's viewpoint."""
        target_lane_index = self._resolve_target_lane_index(ego, target_lane_index)
        current_lane_index = tuple(ego.lane_index)

        ego_state = self._build_ego_state(ego)
        agent_states, agent_valid_mask = self._build_agent_states(ego)
        map_polylines, map_valid_mask = self._build_map_polylines(
            ego, target_lane_index
        )
        target_lane_polyline = self._sample_lane_polyline(
            ego=ego,
            lane_index=target_lane_index,
            current_lane_index=current_lane_index,
            target_lane_index=target_lane_index,
        )
        target_point = self._build_target_point(ego, target_lane_index)

        return {
            "ego_state": ego_state,
            "agent_states": agent_states,
            "agent_valid_mask": agent_valid_mask,
            "map_polylines": map_polylines,
            "map_valid_mask": map_valid_mask,
            "target_point": target_point,
            "target_lane_polyline": target_lane_polyline,
        }

    # ------------------------------------------------------------------
    # Ego state
    # ------------------------------------------------------------------
    def _build_ego_state(self, ego) -> np.ndarray:
        body = self._vehicle_body_state(ego)
        lane_offset = np.asarray(ego.lane_offset, dtype=np.float32)
        heading_error = float(lane_offset[2])

        state = np.asarray(
            [
                body["vx"],
                body["vy"],
                body["ax"],
                body["ay"],
                body["yaw_rate"],
                body["steering_command"],
                body["steering_feedback"],
                float(lane_offset[1]),
                np.sin(heading_error),
                np.cos(heading_error),
                body["roll"],
                body["roll_rate"],
                body["pitch"],
                body["pitch_rate"],
            ],
            dtype=np.float32,
        )
        self._assert_finite("ego_state", state)
        return state

    def _vehicle_body_state(self, vehicle) -> Dict[str, float]:
        """Return SI body-frame state for both kinematics and TruckSim vehicles."""
        action = getattr(vehicle, "action", {})
        if not isinstance(action, dict):
            action = {}
        steering_command = float(action.get("steering", 0.0))

        export = getattr(vehicle, "export_array", None)
        if export is not None:
            export = np.asarray(export, dtype=float)
        is_trucksim = export is not None and export.size >= 15

        if is_trucksim:
            # AllMerge TruckSim export order:
            # Ax, Ay, Vx, Vy, Xo, Yo, Roll, AVx, Pitch, AVy,
            # Yaw, AVz, AV_Eng, steer_l1, steer_r1
            measured_acceleration = np.asarray(
                getattr(vehicle, "measured_acceleration", export[:2] * 9.8),
                dtype=float,
            )
            steering_feedback_deg = float(
                getattr(
                    vehicle,
                    "front_wheel_angle_deg",
                    (export[13] + export[14]) / 2.0,
                )
            )
            angular_unit = str(
                self.config.get("trucksim_angular_unit", "deg")
            ).lower()
            if angular_unit == "deg":
                angular = np.deg2rad
            elif angular_unit == "rad":
                angular = float
            else:
                raise ValueError(
                    "Planner.features.trucksim_angular_unit must be 'deg' or 'rad'"
                )

            return {
                "vx": float(export[2] / 3.6),
                "vy": float(export[3] / 3.6),
                "ax": float(measured_acceleration[0]),
                "ay": float(measured_acceleration[1]),
                "yaw_rate": float(angular(export[11])),
                "steering_command": steering_command,
                "steering_feedback": float(np.deg2rad(steering_feedback_deg)),
                "roll": float(angular(export[6])),
                "roll_rate": float(angular(export[7])),
                "pitch": float(angular(export[8])),
                "pitch_rate": float(angular(export[9])),
            }

        speed = float(getattr(vehicle, "speed", 0.0))
        length = float(getattr(vehicle, "LENGTH", 5.0))
        acceleration = float(action.get("acceleration", 0.0))

        beta = float(np.arctan(0.5 * np.tan(steering_command)))
        vx = speed * np.cos(beta)
        vy = speed * np.sin(beta)
        yaw_rate = 0.0
        if abs(length) > 1e-6:
            yaw_rate = speed * np.sin(beta) / (length / 2.0)

        # Kinematic bicycle model: commanded longitudinal acceleration and
        # instantaneous centripetal acceleration are the available ground truth.
        ay = speed * yaw_rate

        return {
            "vx": float(vx),
            "vy": float(vy),
            "ax": acceleration,
            "ay": float(ay),
            "yaw_rate": float(yaw_rate),
            "steering_command": steering_command,
            "steering_feedback": steering_command,
            "roll": 0.0,
            "roll_rate": 0.0,
            "pitch": 0.0,
            "pitch_rate": 0.0,
        }

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------
    def _build_agent_states(self, ego) -> Tuple[np.ndarray, np.ndarray]:
        states = np.zeros((self.max_agents, self.agent_dim), dtype=np.float32)
        valid_mask = np.zeros((self.max_agents,), dtype=bool)

        nearby = self.env.road.close_vehicles_to(
            ego,
            distance=self.agent_radius,
            count=None,
            see_behind=True,
            sort=False,
        )
        nearby = sorted(
            nearby,
            key=lambda vehicle: float(
                np.linalg.norm(np.asarray(vehicle.position) - np.asarray(ego.position))
            ),
        )[: self.max_agents]

        ego_world_velocity, ego_world_acceleration = self._vehicle_world_motion(ego)

        for index, agent in enumerate(nearby):
            rel_position = world_to_ego_point(
                agent.position, ego.position, ego.heading
            )
            agent_world_velocity, agent_world_acceleration = self._vehicle_world_motion(
                agent
            )
            rel_velocity = world_to_ego_vector(
                agent_world_velocity - ego_world_velocity, ego.heading
            )
            rel_acceleration = world_to_ego_vector(
                agent_world_acceleration - ego_world_acceleration, ego.heading
            )
            heading_rel = float(wrap_angle(agent.heading - ego.heading))

            is_controlled = float(
                any(agent is controlled for controlled in self.env.controlled_vehicles)
            )

            states[index] = np.asarray(
                [
                    rel_position[0],
                    rel_position[1],
                    rel_velocity[0],
                    rel_velocity[1],
                    rel_acceleration[0],
                    rel_acceleration[1],
                    np.sin(heading_rel),
                    np.cos(heading_rel),
                    float(getattr(agent, "LENGTH", 0.0)),
                    float(getattr(agent, "WIDTH", 0.0)),
                    is_controlled,
                ],
                dtype=np.float32,
            )
            valid_mask[index] = True

        self._assert_finite("agent_states", states)
        return states, valid_mask

    def _vehicle_world_motion(self, vehicle) -> Tuple[np.ndarray, np.ndarray]:
        body = self._vehicle_body_state(vehicle)
        world_velocity = ego_to_world_vector(
            np.asarray([body["vx"], body["vy"]], dtype=np.float32),
            vehicle.heading,
        )
        world_acceleration = ego_to_world_vector(
            np.asarray([body["ax"], body["ay"]], dtype=np.float32),
            vehicle.heading,
        )
        return world_velocity, world_acceleration

    # ------------------------------------------------------------------
    # Vectorized map
    # ------------------------------------------------------------------
    def _build_map_polylines(
        self, ego, target_lane_index
    ) -> Tuple[np.ndarray, np.ndarray]:
        polylines = np.zeros(
            (self.max_map_polylines, self.map_points, self.map_dim),
            dtype=np.float32,
        )
        valid_mask = np.zeros((self.max_map_polylines,), dtype=bool)

        current_lane_index = tuple(ego.lane_index)
        candidates = self._select_map_lane_indices(
            ego=ego,
            current_lane_index=current_lane_index,
            target_lane_index=target_lane_index,
        )

        for index, lane_index in enumerate(candidates[: self.max_map_polylines]):
            polylines[index] = self._sample_lane_polyline(
                ego=ego,
                lane_index=lane_index,
                current_lane_index=current_lane_index,
                target_lane_index=target_lane_index,
            )
            valid_mask[index] = True

        self._assert_finite("map_polylines", polylines)
        return polylines, valid_mask

    def _select_map_lane_indices(
        self,
        ego,
        current_lane_index,
        target_lane_index,
    ):
        network = self.env.road.network
        lane_dict = network.lanes_dict()
        scores = {}

        for raw_lane_index, lane in lane_dict.items():
            lane_index = tuple(raw_lane_index)
            score, local_closest = self._lane_distance_score(ego, lane)
            in_roi = (
                -self.map_backward_range
                <= local_closest[0]
                <= self.map_forward_range
                and abs(local_closest[1]) <= self.map_lateral_range
            )
            if in_roi or lane_index in {current_lane_index, target_lane_index}:
                scores[lane_index] = score

        # Always retain current and target lane when present in the network.
        for lane_index in (current_lane_index, target_lane_index):
            if lane_index in lane_dict and lane_index not in scores:
                scores[lane_index] = 0.0

        def priority(lane_index):
            if lane_index == current_lane_index:
                special = 0
            elif lane_index == target_lane_index:
                special = 1
            else:
                special = 2
            return special, scores[lane_index]

        return sorted(scores, key=priority)

    def _lane_distance_score(self, ego, lane) -> Tuple[float, np.ndarray]:
        longitudinal, _ = lane.local_coordinates(ego.position)
        longitudinal = float(np.clip(longitudinal, 0.0, lane.length))
        closest_world = lane.position(longitudinal, 0.0)
        closest_local = world_to_ego_point(
            closest_world, ego.position, ego.heading
        )
        return float(np.linalg.norm(closest_local)), closest_local

    def _sample_lane_polyline(
        self,
        ego,
        lane_index,
        current_lane_index,
        target_lane_index,
    ) -> np.ndarray:
        lane_index = tuple(lane_index)
        lane = self.env.road.network.get_lane(lane_index)
        longitudinal, _ = lane.local_coordinates(ego.position)
        center_s = float(np.clip(longitudinal, 0.0, lane.length))

        start_s = max(0.0, center_s - self.map_backward_range)
        end_s = min(float(lane.length), center_s + self.map_forward_range)
        if end_s <= start_s:
            end_s = min(float(lane.length), start_s + 1e-3)

        s_values = np.linspace(start_s, end_s, self.map_points, dtype=np.float32)
        polyline = np.zeros((self.map_points, self.map_dim), dtype=np.float32)

        same_road = lane_index[:2] == current_lane_index[:2]
        is_current = float(lane_index == current_lane_index)
        is_target = float(lane_index == target_lane_index)
        is_left = float(same_road and lane_index[2] < current_lane_index[2])
        is_right = float(same_road and lane_index[2] > current_lane_index[2])
        speed_limit = (
            float(lane.speed_limit) if lane.speed_limit is not None else 0.0
        )

        for point_index, s in enumerate(s_values):
            world_point = lane.position(float(s), 0.0)
            local_point = world_to_ego_point(
                world_point, ego.position, ego.heading
            )
            heading_rel = float(
                wrap_angle(lane.heading_at(float(s)) - ego.heading)
            )
            polyline[point_index] = np.asarray(
                [
                    local_point[0],
                    local_point[1],
                    np.sin(heading_rel),
                    np.cos(heading_rel),
                    float(lane.width_at(float(s))),
                    speed_limit,
                    is_current,
                    is_left,
                    is_right,
                    is_target,
                ],
                dtype=np.float32,
            )

        return polyline

    # ------------------------------------------------------------------
    # Planning condition
    # ------------------------------------------------------------------
    def _build_target_point(self, ego, target_lane_index) -> np.ndarray:
        lane = self.env.road.network.get_lane(target_lane_index)
        longitudinal, _ = lane.local_coordinates(ego.position)
        longitudinal = float(np.clip(longitudinal, 0.0, lane.length))

        lookahead = float(ego.speed) * float(self.config["target_point_horizon"])
        lookahead = float(
            np.clip(
                lookahead,
                float(self.config["target_point_min_lookahead"]),
                float(self.config["target_point_max_lookahead"]),
            )
        )
        target_s = float(np.clip(longitudinal + lookahead, 0.0, lane.length))
        target_world = lane.position(target_s, 0.0)
        target_point = world_to_ego_point(
            target_world, ego.position, ego.heading
        ).astype(np.float32)
        self._assert_finite("target_point", target_point)
        return target_point

    def _resolve_target_lane_index(self, ego, target_lane_index):
        if target_lane_index is None:
            target_lane_index = getattr(ego, "target_lane_index", None)
        if target_lane_index is None:
            target_lane_index = ego.lane_index
        target_lane_index = tuple(target_lane_index)

        try:
            self.env.road.network.get_lane(target_lane_index)
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(
                f"Invalid planner target lane {target_lane_index!r} "
                f"for ego lane {ego.lane_index!r}"
            ) from error
        return target_lane_index

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _assert_finite(name: str, value: np.ndarray) -> None:
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"Planner feature {name} contains NaN/Inf")

    def _validate_batch(self, batch: Dict[str, np.ndarray]) -> None:
        batch_size = len(self.env.controlled_vehicles)
        expected_shapes = {
            "ego_state": (batch_size, self.ego_dim),
            "agent_states": (batch_size, self.max_agents, self.agent_dim),
            "agent_valid_mask": (batch_size, self.max_agents),
            "map_polylines": (
                batch_size,
                self.max_map_polylines,
                self.map_points,
                self.map_dim,
            ),
            "map_valid_mask": (batch_size, self.max_map_polylines),
            "target_point": (batch_size, 2),
            "target_lane_polyline": (
                batch_size,
                self.map_points,
                self.map_dim,
            ),
        }
        for name, shape in expected_shapes.items():
            if batch[name].shape != shape:
                raise RuntimeError(
                    f"Planner feature {name} shape={batch[name].shape}, expected={shape}"
                )
            if not name.endswith("mask"):
                self._assert_finite(name, batch[name])
