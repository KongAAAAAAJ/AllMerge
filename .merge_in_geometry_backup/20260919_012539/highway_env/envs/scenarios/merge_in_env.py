from __future__ import annotations

from highway_env.envs.scenarios.base_env import BaseScenarioEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import LaneIndex, RoadNetwork


class MergeInEnv(BaseScenarioEnv):
    """On-ramp platoon merging into the rightmost mainline lane.

    The controlled platoon starts on a dedicated auxiliary lane (lane 3) on
    the common merge segment and targets mainline lane 2.
    """

    SCENARIO_NAME = "merge_in"
    MANEUVER = "merge_left"

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "speed_limit": 30.0,
                "initial_lane_id": 3,
                "platoon_longitudinal": [125.0, 110.0, 95.0],
                "duration": 5.0,
            }
        )
        # LOCAL RANDOM TRAFFIC V2: fixed leader spawn region.
        config["traffic_randomization"]["leader_spawn_s_range"] = [110.0, 140.0]
        config["scenario"] = {
            "name": cls.SCENARIO_NAME,
            "maneuver": cls.MANEUVER,
            "road_geometry": "on_ramp_merge",
            "initial_lane_id": 3,
            "target_lane_id": 2,
        }
        return config

    def _create_road(self) -> None:
        network = RoadNetwork()
        c = LineType.CONTINUOUS_LINE
        s = LineType.STRIPED
        n = LineType.NONE
        lane_width = 4.0
        speed_limit = float(self.config["speed_limit"])

        # Common merge segment. Mainline lanes are 0..2; lane 3 is the
        # auxiliary merge lane. Keeping them on the same graph edge makes the
        # lane-change topology explicit for the planner.
        merge_start = 0.0
        merge_end = 320.0
        for lane_id in range(3):
            y = lane_id * lane_width
            network.add_lane(
                "b",
                "c",
                StraightLane(
                    [merge_start, y],
                    [merge_end, y],
                    line_types=[c if lane_id == 0 else n,
                                c if lane_id == 2 else s],
                    speed_limit=speed_limit,
                ),
            )

        # Auxiliary lane converges smoothly toward the mainline. It is still
        # represented as lane id 3 over the same b->c edge.
        network.add_lane(
            "b",
            "c",
            SineLane(
                [merge_start, 3 * lane_width],
                [merge_end, 3 * lane_width],
                amplitude=2.0,
                pulsation=2.0 * 3.141592653589793 / (2.0 * merge_end),
                phase=3.141592653589793 / 2.0,
                line_types=[s, c],
                speed_limit=25.0,
            ),
        )

        # Mainline continuation after the merge. The auxiliary lane disappears.
        for lane_id in range(3):
            y = lane_id * lane_width
            network.add_lane(
                "c",
                "d",
                StraightLane(
                    [merge_end, y],
                    [700.0, y],
                    line_types=[c if lane_id == 0 else n,
                                c if lane_id == 2 else s],
                    speed_limit=speed_limit,
                ),
            )

        self.road = self._make_road(network, self.np_random, self.config)

    def _initial_lane_index(self) -> LaneIndex:
        return "b", "c", int(self.config["initial_lane_id"])

    def _route_destination(self):
        return "d"

    def _task_success(self) -> bool:
        target_lane_id = int(self.config["scenario"]["target_lane_id"])
        if not self.controlled_vehicles:
            return False
        return all(
            (
                vehicle.lane_index[:2] == ("b", "c")
                and vehicle.lane_index[2] == target_lane_id
            )
            or vehicle.lane_index[:2] == ("c", "d")
            for vehicle in self.controlled_vehicles
        )
