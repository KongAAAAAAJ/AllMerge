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

        longitudinal = list(self.config["platoon_longitudinal"])
        speeds = list(self.config["initial_controlled_vehicle_speed"])
        num_vehicles = int(self.config["controlled_vehicles"])

        if len(longitudinal) != num_vehicles:
            raise ValueError(
                "platoon_longitudinal length must equal controlled_vehicles: "
                f"{len(longitudinal)} != {num_vehicles}"
            )
        if len(speeds) != num_vehicles:
            raise ValueError(
                "initial_controlled_vehicle_speed length must equal controlled_vehicles: "
                f"{len(speeds)} != {num_vehicles}"
            )

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
        """Phase-1 placeholder.

        Background traffic is intentionally empty here. The next step will
        implement constrained traffic-condition sampling after the platoon has
        been created.
        """
        return None

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
