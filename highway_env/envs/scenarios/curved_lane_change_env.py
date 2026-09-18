from __future__ import annotations

import numpy as np

from highway_env.envs.scenarios.base_env import BaseScenarioEnv
from highway_env.road.lane import CircularLane, LineType
from highway_env.road.road import LaneIndex, RoadNetwork


class CurvedLaneChangeEnv(BaseScenarioEnv):
    """Constant-curvature three-lane lane-change scenario skeleton.

    The road geometry is intentionally isolated here. Background traffic and
    difficulty randomization are added in phase 2.
    """

    SCENARIO_NAME = "curved_lane_change"
    MANEUVER = "lane_change_left"

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "lanes_count": 3,
                "curve_radius": 260.0,
                "curve_angle": 1.45,
                "speed_limit": 30.0,
                "initial_lane_id": 1,
                "platoon_longitudinal": [105.0, 90.0, 75.0],
            }
        )
        config["scenario"] = {
            "name": cls.SCENARIO_NAME,
            "maneuver": cls.MANEUVER,
            "road_geometry": "constant_curvature",
            "initial_lane_id": 1,
            "target_lane_id": 0,
        }
        return config

    def _create_road(self) -> None:
        network = RoadNetwork()
        lane_width = 4.0
        base_radius = float(self.config["curve_radius"])
        start_phase = -np.pi / 2
        end_phase = start_phase + float(self.config["curve_angle"])
        center = [0.0, base_radius]
        lane_count = int(self.config["lanes_count"])

        for lane_id in range(lane_count):
            radius = base_radius + lane_id * lane_width
            left_line = (
                LineType.CONTINUOUS_LINE
                if lane_id == 0
                else LineType.STRIPED
            )
            right_line = (
                LineType.CONTINUOUS_LINE
                if lane_id == lane_count - 1
                else LineType.NONE
            )
            network.add_lane(
                "c0",
                "c1",
                CircularLane(
                    center=center,
                    radius=radius,
                    start_phase=start_phase,
                    end_phase=end_phase,
                    clockwise=True,
                    width=lane_width,
                    line_types=[left_line, right_line],
                    speed_limit=float(self.config["speed_limit"]),
                ),
            )

        self.road = self._make_road(network, self.np_random, self.config)

    def _initial_lane_index(self) -> LaneIndex:
        return "c0", "c1", int(self.config["initial_lane_id"])

    def _task_success(self) -> bool:
        target_lane_id = int(self.config["scenario"]["target_lane_id"])
        return bool(self.controlled_vehicles) and all(
            vehicle.lane_index[2] == target_lane_id
            for vehicle in self.controlled_vehicles
        )
