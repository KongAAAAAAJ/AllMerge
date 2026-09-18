from __future__ import annotations

import numpy as np

from highway_env.envs.scenarios.base_env import BaseScenarioEnv
from highway_env.road.lane import CircularLane, LineType
from highway_env.road.road import LaneIndex, RoadNetwork


class CurvedLaneChangeEnv(BaseScenarioEnv):
    """Constant-curvature three-lane lane-change scenario."""

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

        # SIMPLE RANDOM TRAFFIC V1: fixed leader spawn region.
        config["traffic_randomization"]["leader_spawn_s_range"] = [95.0, 125.0]
        config["scenario"] = {
            "name": cls.SCENARIO_NAME,
            "maneuver": cls.MANEUVER,
            "road_geometry": "constant_curvature",
            "initial_lane_id": 1,
            "target_lane_id": 0,
        }
        return config

    def _create_road(self) -> None:
        """Create a three-lane constant-curvature road.

        Geometry
        --------
        With the default configuration:

            lane 0 center radius = 260 m
            lane 1 center radius = 264 m
            lane 2 center radius = 268 m
            lane width           =   4 m

        Since this CircularLane uses clockwise=True, positive Frenet lateral
        offset points toward the circle center, i.e. toward a smaller radius.

        Therefore the physical road boundaries are:

            R = 258 m : inner continuous boundary
            R = 262 m : lane 0 / lane 1 striped boundary
            R = 266 m : lane 1 / lane 2 striped boundary
            R = 270 m : outer continuous boundary

        `line_types[0]` is rendered on the negative-lateral side and
        `line_types[1]` on the positive-lateral side. For clockwise=True,
        this means:

            line_types[0] -> larger radius
            line_types[1] -> smaller radius

        The line assignment below draws every shared boundary only once and
        keeps the rendered lane centers aligned with the CircularLane geometry.
        """
        network = RoadNetwork()

        lane_width = 4.0
        base_radius = float(self.config["curve_radius"])
        lane_count = int(self.config["lanes_count"])

        start_phase = -np.pi / 2
        end_phase = start_phase + float(self.config["curve_angle"])
        center = [0.0, base_radius]

        for lane_id in range(lane_count):
            radius = base_radius + lane_id * lane_width

            # For clockwise CircularLane:
            #   index 0 -> negative lateral -> larger radius
            #   index 1 -> positive lateral -> smaller radius
            #
            # Desired boundaries for 3 lanes:
            #   lane 0: outer/shared R=262 striped, inner R=258 continuous
            #   lane 1: outer/shared R=266 striped, inner R=262 already drawn
            #   lane 2: outer R=270 continuous, inner R=266 already drawn
            if lane_count == 1:
                line_types = [
                    LineType.CONTINUOUS_LINE,
                    LineType.CONTINUOUS_LINE,
                ]
            elif lane_id == 0:
                line_types = [
                    LineType.STRIPED,
                    LineType.CONTINUOUS_LINE,
                ]
            elif lane_id == lane_count - 1:
                line_types = [
                    LineType.CONTINUOUS_LINE,
                    LineType.NONE,
                ]
            else:
                line_types = [
                    LineType.STRIPED,
                    LineType.NONE,
                ]

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
                    line_types=line_types,
                    speed_limit=float(self.config["speed_limit"]),
                ),
            )

        self.road = self._make_road(
            network,
            self.np_random,
            self.config,
        )

    def _initial_lane_index(self) -> LaneIndex:
        return (
            "c0",
            "c1",
            int(self.config["initial_lane_id"]),
        )

    def _task_success(self) -> bool:
        target_lane_id = int(
            self.config["scenario"]["target_lane_id"]
        )

        return bool(self.controlled_vehicles) and all(
            vehicle.lane_index[2] == target_lane_id
            for vehicle in self.controlled_vehicles
        )
