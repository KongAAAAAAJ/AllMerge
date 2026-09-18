from __future__ import annotations

from highway_env.envs.scenarios.base_env import BaseScenarioEnv
from highway_env.road.road import LaneIndex, RoadNetwork


class StraightLaneChangeEnv(BaseScenarioEnv):
    """Three-lane straight-road lane-change scenario skeleton."""

    SCENARIO_NAME = "straight_lane_change"
    MANEUVER = "lane_change_left"

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "lanes_count": 3,
                "road_length": 700.0,
                "speed_limit": 33.0,
                "initial_lane_id": 1,
                "platoon_longitudinal": [170.0, 155.0, 140.0],
            }
        )
        config["scenario"] = {
            "name": cls.SCENARIO_NAME,
            "maneuver": cls.MANEUVER,
            "road_geometry": "straight",
            "initial_lane_id": 1,
            "target_lane_id": 0,
        }
        return config

    def _create_road(self) -> None:
        network = RoadNetwork.straight_road_network(
            lanes=int(self.config["lanes_count"]),
            start=0.0,
            length=float(self.config["road_length"]),
            speed_limit=float(self.config["speed_limit"]),
            nodes_str=("s", "e"),
        )
        self.road = self._make_road(network, self.np_random, self.config)

    def _initial_lane_index(self) -> LaneIndex:
        return "s", "e", int(self.config["initial_lane_id"])

    def _task_success(self) -> bool:
        target_lane_id = int(self.config["scenario"]["target_lane_id"])
        return bool(self.controlled_vehicles) and all(
            vehicle.lane_index[2] == target_lane_id
            for vehicle in self.controlled_vehicles
        )
