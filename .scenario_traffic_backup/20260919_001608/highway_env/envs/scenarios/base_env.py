from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Text, Tuple

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import LaneIndex, Road

Observation = np.ndarray

class ScenarioRoad(Road):
    """Road variant with geometry-based neighbour lookup.

    The current project-level ``Road.neighbour_vehicles`` contains straight-
    road assumptions based on global x/y coordinates. Scenario environments
    need lane-local Frenet coordinates so the same API also works on curved
    lanes and auxiliary merge/exit lanes. This override is local to the new
    scenarios package and does not change the legacy test environments.
    """

    def neighbour_vehicles(
        self, vehicle, lane_index: Optional[LaneIndex] = None, group: list = None
    ):
        lane_index = lane_index or vehicle.lane_index
        if lane_index is None:
            return None, None

        lane = self.network.get_lane(lane_index)
        ego_s, _ = lane.local_coordinates(vehicle.position)
        front_s = rear_s = None
        front_vehicle = rear_vehicle = None
        route = getattr(vehicle, "route", None)

        for candidate in self.vehicles + self.objects:
            if candidate is vehicle or (group is not None and candidate in group):
                continue

            candidate_lane_index = getattr(candidate, "lane_index", None)
            if candidate_lane_index is not None and not self.network.is_connected_road(
                candidate_lane_index,
                lane_index,
                route=route,
                same_lane=True,
                depth=1,
            ):
                continue

            candidate_s, candidate_lat = lane.local_coordinates(candidate.position)
            if not lane.on_lane(
                candidate.position, candidate_s, candidate_lat, margin=1.0
            ):
                continue

            if ego_s < candidate_s and (front_s is None or candidate_s <= front_s):
                front_s = candidate_s
                front_vehicle = candidate
            if candidate_s < ego_s and (rear_s is None or candidate_s > rear_s):
                rear_s = candidate_s
                rear_vehicle = candidate

        return front_vehicle, rear_vehicle


class BaseScenarioEnv(AbstractEnv, ABC):
    """Common base class for AllMerge data-collection/training scenarios.

    Design goals
    ------------
    1. Keep scenario environments independent from ``MyHighwayEnv`` and
       ``MyMergeEnv``. Those two environments remain system-rollout/test envs.
    2. Use one common scene API for expert collection, Diffusion inference,
       GRPO rollout and evaluation.
    3. Always create the controlled platoon before background traffic.
    4. Keep phase-1 scene initialization deterministic. Constrained scenario
       randomization is intentionally added in the next step.
    """

    SCENARIO_NAME = "base"
    MANEUVER = "undefined"

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()

        # Keep compatibility with the current Rule decision maker.
        sum_split_weights = float(np.sum(config["w_game split weights"]))
        sum_merge_weights = float(np.sum(config["w_game merge weights"]))

        config.update(
            {
                # Unified 10 Hz interface. AbstractEnv currently defaults to
                # simulation_frequency=1 and policy_frequency=10, which is not
                # appropriate for these scenario rollouts.
                "simulation_frequency": 10,
                "policy_frequency": 10,
                "duration": 5.0,
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "Kinematics"},
                },
                "action_num": {
                    "type": "MultiAgentAction",
                    "action_config": {"type": "DiscreteAndContinuousMetaAction"},
                },
                # Expert default. Later Diffusion/GRPO entrypoints should
                # override Planner without changing the scenario definition.
                "Planner": {
                    "state": True,
                    "type": "Polynomial",
                    "Polynomial": {
                        "mode": "aligned",
                        "aligned_horizon_s": 4.0,
                    },
                    "ExpertAlignment": {
                        "enabled": True,
                        "require_aligned_polynomial": True,
                    },
                    "features": {
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
                        "trucksim_angular_unit": "deg",
                        "anchors": {
                            "horizon_steps": 8,
                            "trajectory_dt": 0.5,
                            "speed_delta_mps": 4.0,
                            "acceleration_preview_s": 1.0,
                            "minimum_lane_change_speed_mps": 3.0,
                            "emergency_decel_mps2": 4.5,
                            "collision_check_enabled": True,
                            "ego_length_m": 5.0,
                            "ego_width_m": 2.0,
                            "collision_longitudinal_margin_m": 1.0,
                            "collision_lateral_margin_m": 0.3,
                            "keep_low_always_valid": True,
                            "stop_always_valid": True,
                            "diagnostics_enabled": False,
                        },
                    },
                },
                "Decision maker": "Rule",
                "Controller": {
                    "type": "lqr",
                    "vehicle_model": "kinematics",
                },
                "controlled_vehicles": 3,
                "initial_controlled_vehicle_speed": [25.0, 25.0, 25.0],
                "initial_controlled_vehicle_acc": [0.0, 0.0, 0.0],
                "offroad_terminal": True,
                "Split weights": [
                    x / sum_split_weights for x in config["w_game split weights"]
                ],
                "Merge weights": [
                    x / sum_merge_weights for x in config["w_game merge weights"]
                ],
                "screen_width": 1200,
                "screen_height": 260,
                "centering_position": [0.45, 0.5],
                "scaling": 4.5,
                "show_trajectories": False,
                "show_future_trajectories": True,
                # Phase-1 deterministic platoon. Randomized state ranges are
                # deliberately deferred to the next implementation step.
                "platoon_longitudinal": [150.0, 135.0, 120.0],
                # === SIMPLE RANDOM TRAFFIC V1 START ===
                "traffic_randomization": {
                    "enabled": True,

                    # Each concrete scenario overrides this interval.
                    "leader_spawn_s_range": [120.0, 160.0],

                    # Random platoon initial speed.
                    "leader_speed_range": [22.0, 26.0],

                    # Followers are placed behind the sampled leader.
                    "platoon_spacing": 15.0,

                    # Fixed number of background vehicles.
                    "background_vehicle_count": 8,

                    # Background longitudinal position relative to leader_s.
                    "background_s_offset_range": [-60.0, 100.0],

                    # Background initial speed relative to leader speed.
                    "background_speed_delta_range": [-4.0, 4.0],

                    # Minimal spawn constraints.
                    "background_min_speed": 15.0,
                    "min_spawn_gap": 15.0,
                    "road_edge_margin": 5.0,
                    "max_spawn_attempts": 200,

                    # V1 randomizes initial traffic only.
                    "randomize_background_behavior": False,
                },
                # === SIMPLE RANDOM TRAFFIC V1 END ===
                "scenario": {
                    "name": cls.SCENARIO_NAME,
                    "maneuver": cls.MANEUVER,
                    "target_lane_id": None,
                },
            }
        )
        return config

    # ------------------------------------------------------------------
    # Scene lifecycle
    # ------------------------------------------------------------------
    def _reset(self) -> None:
        self._create_road()
        # Order is intentional and is part of the scenarios API contract.
        self._create_controlled_platoon()
        self._create_background_traffic()
        self._after_scene_created()

    @abstractmethod
    def _create_road(self) -> None:
        """Build road geometry and assign ``self.road``."""

    @abstractmethod
    def _initial_lane_index(self) -> LaneIndex:
        """Lane where the controlled platoon is initialized."""

    def _route_destination(self) -> Optional[str]:
        """Optional road-network destination for route-aware scenarios."""
        return None

    def _create_controlled_platoon(self) -> None:
        lane_index = self._initial_lane_index()
        lane = self.road.network.get_lane(lane_index)

        num_vehicles = int(self.config["controlled_vehicles"])
        traffic_cfg = self.config.get("traffic_randomization", {})

        if traffic_cfg.get("enabled", False):
            leader_s_min, leader_s_max = traffic_cfg["leader_spawn_s_range"]
            leader_v_min, leader_v_max = traffic_cfg["leader_speed_range"]
            spacing = float(traffic_cfg["platoon_spacing"])

            leader_s = float(
                self.np_random.uniform(
                    float(leader_s_min),
                    float(leader_s_max),
                )
            )
            leader_speed = float(
                self.np_random.uniform(
                    float(leader_v_min),
                    float(leader_v_max),
                )
            )

            longitudinal = [
                leader_s - vehicle_index * spacing
                for vehicle_index in range(num_vehicles)
            ]

            # Start the platoon in a stable state.
            speeds = [leader_speed] * num_vehicles

        else:
            longitudinal = list(self.config["platoon_longitudinal"])
            speeds = list(self.config["initial_controlled_vehicle_speed"])

            if len(longitudinal) != num_vehicles:
                raise ValueError(
                    "platoon_longitudinal length must equal controlled_vehicles: "
                    f"{len(longitudinal)} != {num_vehicles}"
                )

            if len(speeds) != num_vehicles:
                raise ValueError(
                    "initial_controlled_vehicle_speed length must equal "
                    "controlled_vehicles: "
                    f"{len(speeds)} != {num_vehicles}"
                )

        # Fail early if the sampled platoon would leave the initial lane.
        for s in longitudinal:
            if s < 0.0 or s > float(lane.length):
                raise ValueError(
                    f"controlled platoon spawn s={s:.3f} is outside "
                    f"initial lane length={float(lane.length):.3f}"
                )

        # Stored for local background-traffic generation.
        self._platoon_leader_s = float(longitudinal[0])
        self._platoon_leader_speed = float(speeds[0])

        self.controlled_vehicles = []
        destination = self._route_destination()

        for index in range(num_vehicles):
            s = float(longitudinal[index])
            vehicle = self.action_type.vehicle_class(
                self.road,
                lane.position(s, 0.0),
                lane.heading_at(s),
                float(speeds[index]),
            )
            if destination is not None and hasattr(vehicle, "plan_route_to"):
                vehicle.plan_route_to(destination)
            self._configure_controlled_vehicle(vehicle, index)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

    def _configure_controlled_vehicle(self, vehicle, index: int) -> None:
        """Subclass hook for per-role controlled-vehicle initialization."""
        return None

    def _create_background_traffic(self) -> None:
        """Spawn a fixed number of simple random background vehicles.

        Background vehicles are sampled around the current platoon leader:
            s_bg = s_leader + Uniform(offset_min, offset_max)
            v_bg = v_leader + Uniform(delta_v_min, delta_v_max)

        A lane is sampled uniformly from all side lanes of the initial road
        segment. Only same-lane longitudinal spawn distance is checked.
        """
        traffic_cfg = self.config.get("traffic_randomization", {})
        self.background_vehicles = []

        if not traffic_cfg.get("enabled", False):
            return None

        vehicle_count = int(traffic_cfg["background_vehicle_count"])
        if vehicle_count <= 0:
            return None

        lane_indices = list(
            self.road.network.all_side_lanes(
                self._initial_lane_index()
            )
        )
        if not lane_indices:
            lane_indices = [self._initial_lane_index()]

        offset_min, offset_max = traffic_cfg[
            "background_s_offset_range"
        ]
        speed_delta_min, speed_delta_max = traffic_cfg[
            "background_speed_delta_range"
        ]

        min_speed = float(traffic_cfg["background_min_speed"])
        min_gap = float(traffic_cfg["min_spawn_gap"])
        edge_margin = float(traffic_cfg["road_edge_margin"])
        max_attempts = int(traffic_cfg["max_spawn_attempts"])
        randomize_behavior = bool(
            traffic_cfg.get(
                "randomize_background_behavior",
                False,
            )
        )

        leader_s = float(self._platoon_leader_s)
        leader_speed = float(self._platoon_leader_speed)

        for background_index in range(vehicle_count):
            spawned = False

            for _ in range(max_attempts):
                lane_index = lane_indices[
                    int(self.np_random.integers(0, len(lane_indices)))
                ]
                lane = self.road.network.get_lane(lane_index)

                longitudinal = (
                    leader_s
                    + float(
                        self.np_random.uniform(
                            float(offset_min),
                            float(offset_max),
                        )
                    )
                )

                if longitudinal < edge_margin:
                    continue

                if longitudinal > float(lane.length) - edge_margin:
                    continue

                if not self._background_spawn_position_is_free(
                    lane_index=lane_index,
                    longitudinal=longitudinal,
                    min_gap=min_gap,
                ):
                    continue

                speed = (
                    leader_speed
                    + float(
                        self.np_random.uniform(
                            float(speed_delta_min),
                            float(speed_delta_max),
                        )
                    )
                )

                lane_speed_limit = getattr(
                    lane,
                    "speed_limit",
                    None,
                )
                if lane_speed_limit is not None:
                    speed = min(
                        speed,
                        float(lane_speed_limit),
                    )

                speed = max(
                    speed,
                    min_speed,
                )

                vehicle = self._spawn_background_vehicle(
                    lane_index=lane_index,
                    longitudinal=longitudinal,
                    speed=float(speed),
                    target_speed=float(speed),
                    randomize_behavior=randomize_behavior,
                )

                self.background_vehicles.append(vehicle)
                spawned = True
                break

            if not spawned:
                raise RuntimeError(
                    "Failed to spawn the requested fixed number of "
                    "background vehicles. "
                    f"placed={background_index}, requested={vehicle_count}. "
                    "Increase background_s_offset_range or "
                    "max_spawn_attempts, or reduce min_spawn_gap."
                )

        return None

    def _background_spawn_position_is_free(
        self,
        lane_index,
        longitudinal: float,
        min_gap: float,
    ) -> bool:
        """Check only same-lane longitudinal spawn separation."""
        lane = self.road.network.get_lane(lane_index)

        for vehicle in self.road.vehicles:
            existing_lane_index = getattr(
                vehicle,
                "lane_index",
                None,
            )

            if existing_lane_index is None:
                continue

            if tuple(existing_lane_index) != tuple(lane_index):
                continue

            existing_s, _ = lane.local_coordinates(
                vehicle.position
            )

            if (
                abs(float(existing_s) - float(longitudinal))
                < float(min_gap)
            ):
                return False

        return True

    def _after_scene_created(self) -> None:
        """Optional subclass hook executed after all vehicles are created."""
        return None

    # ------------------------------------------------------------------
    # Reusable vehicle helper for phase 2
    # ------------------------------------------------------------------
    def _spawn_background_vehicle(
        self,
        lane_index: LaneIndex,
        longitudinal: float,
        speed: float,
        *,
        target_speed: Optional[float] = None,
        randomize_behavior: bool = False,
    ):
        """Create a background vehicle from lane-local coordinates.

        This helper is already geometry-agnostic and will be used by the
        constrained random traffic sampler in phase 2.
        """
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        lane = self.road.network.get_lane(lane_index)
        s = float(longitudinal)
        vehicle = vehicle_type(
            self.road,
            lane.position(s, 0.0),
            lane.heading_at(s),
            float(speed),
        )
        if target_speed is not None and hasattr(vehicle, "target_speed"):
            vehicle.target_speed = float(target_speed)
        if randomize_behavior and hasattr(vehicle, "randomize_behavior"):
            vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    # ------------------------------------------------------------------
    # Episode semantics
    # ------------------------------------------------------------------
    def _task_success(self) -> bool:
        """Scenario-specific completion signal; subclasses may override."""
        return False

    def _is_terminated(self) -> bool:
        if any(vehicle.crashed for vehicle in self.controlled_vehicles):
            return True
        if self.config.get("offroad_terminal", False) and any(
            not vehicle.on_road for vehicle in self.controlled_vehicles
        ):
            return True
        return self._task_success()

    def _is_truncated(self) -> bool:
        return self.time >= float(self.config["duration"])

    # Data-collection scenarios do not train against the legacy environment
    # reward. Diffusion/GRPO reward is added in their own training pipeline.
    def _reward(self, action: Action) -> float:
        return 0.0

    def _rewards(self, action: Action) -> Dict[Text, float]:
        if not self.controlled_vehicles:
            return {
                "collision": 0.0,
                "on_road": 1.0,
                "task_success": 0.0,
            }
        return {
            "collision": float(any(v.crashed for v in self.controlled_vehicles)),
            "on_road": float(all(v.on_road for v in self.controlled_vehicles)),
            "task_success": float(self._task_success()),
        }

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        info = super()._info(obs, action)
        info.update(
            {
                "scenario_name": self.SCENARIO_NAME,
                "maneuver": self.MANEUVER,
                "target_lane_id": self.config.get("scenario", {}).get(
                    "target_lane_id"
                ),
                "task_success": self._task_success(),
                "vehicle_speed": [float(v.speed) for v in self.controlled_vehicles],
                "vehicle_position": [
                    [float(v.position[0]), float(v.position[1])]
                    for v in self.controlled_vehicles
                ],
                "vehicle_lane_index": [
                    tuple(v.lane_index) for v in self.controlled_vehicles
                ],
            }
        )
        return info

    @staticmethod
    def _make_road(network, np_random, config: dict) -> Road:
        return ScenarioRoad(
            network=network,
            np_random=np_random,
            record_history=config["show_trajectories"],
            show_future_trajectory=config["show_future_trajectories"],
        )
