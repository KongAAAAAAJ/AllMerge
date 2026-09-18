from __future__ import annotations

import numpy as np

from highway_env.envs.scenarios.base_env import BaseScenarioEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import LaneIndex, RoadNetwork


class MergeOutEnv(BaseScenarioEnv):
    """Mainline platoon moving right into an auxiliary exit lane."""

    SCENARIO_NAME = "merge_out"
    MANEUVER = "merge_right"

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "speed_limit": 30.0,
                "initial_lane_id": 2,
                "platoon_longitudinal": [125.0, 110.0, 95.0],
                "duration": 5.0,
            }
        )
        # LOCAL RANDOM TRAFFIC V2: fixed leader spawn region.
        config["traffic_randomization"]["leader_spawn_s_range"] = [110.0, 140.0]
        config["scenario"] = {
            "name": cls.SCENARIO_NAME,
            "maneuver": cls.MANEUVER,
            "road_geometry": "off_ramp_exit",
            "initial_lane_id": 2,
            "target_lane_id": 3,
        }
        return config

    def _create_road(self) -> None:
        network = RoadNetwork()
        c = LineType.CONTINUOUS_LINE
        s = LineType.STRIPED
        n = LineType.NONE
        lane_width = 4.0
        speed_limit = float(self.config["speed_limit"])

        exit_start = 0.0
        exit_split = 320.0

        # Mainline plus an auxiliary exit lane on the shared pre-exit segment.
        for lane_id in range(4):
            y = lane_id * lane_width
            network.add_lane(
                "b",
                "c",
                StraightLane(
                    [exit_start, y],
                    [exit_split, y],
                    line_types=[c if lane_id == 0 else n,
                                c if lane_id == 3 else s],
                    speed_limit=(25.0 if lane_id == 3 else speed_limit),
                ),
            )

        # Three-lane mainline continues straight.
        for lane_id in range(3):
            y = lane_id * lane_width
            network.add_lane(
                "c",
                "d",
                StraightLane(
                    [exit_split, y],
                    [700.0, y],
                    line_types=[c if lane_id == 0 else n,
                                c if lane_id == 2 else s],
                    speed_limit=speed_limit,
                ),
            )

        # Exit branch bends away from the highway from lane 3.
        exit_length = 300.0
        network.add_lane(
            "c",
            "e",
            SineLane(
                [exit_split, 3 * lane_width],
                [exit_split + exit_length, 3 * lane_width + 24.0],
                amplitude=2.0,
                pulsation=2.0 * np.pi / (2.0 * exit_length),
                phase=-np.pi / 2.0,
                line_types=[s, c],
                speed_limit=25.0,
            ),
        )

        self.road = self._make_road(network, self.np_random, self.config)

    def _initial_lane_index(self) -> LaneIndex:
        return "b", "c", int(self.config["initial_lane_id"])

    def _route_destination(self):
        return "e"

    def _task_success(self) -> bool:
        target_lane_id = int(self.config["scenario"]["target_lane_id"])
        if not self.controlled_vehicles:
            return False
        return all(
            (
                vehicle.lane_index[:2] == ("b", "c")
                and vehicle.lane_index[2] == target_lane_id
            )
            or vehicle.lane_index[:2] == ("c", "e")
            for vehicle in self.controlled_vehicles
        )
