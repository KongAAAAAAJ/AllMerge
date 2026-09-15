from typing import Dict, Text

import numpy as np
from matplotlib.rcsetup import validate_whiskers
from networkx.algorithms.bipartite import color

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.graphics import VehicleGraphics

from itertools import groupby

Observation = np.ndarray


class TestEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {
                        "type": "Kinematics",
                    }
                },
                "action_num": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteAndContinuousMetaAction",
                    }
                },

                "forward_speed_reward": 50,  # The vehicle was punished when speed < 18
                "on_road_reward": 0.05,
                "distance_reward": -300,
                "same_lane_reward": 50,  # The reward received when vehicles in the same lane
                'far_reward': 400,
                # "equal_speed_reward": 0,
                "collision_reward": -800,  # The reward received when colliding with a vehicle.
                "action_infeasible_reward": 0,  # The reward received when got infeasible action.
                "merge_reward": 300,  # The vehicle was punished when merge slowly
                # "action_reward": 0,  # The reward received when select not-split action.
                "high_speed_reward": 200,  # 50 The reward received when driving at full speed, linearly mapped to zero
                # for lower speeds according to config["reward_speed_range"].
                # "same_action_reward": 0,  # The vehicle is punished when always select same action.

                "reward_speed_range": [19, 26],
                "lanes_count": 3,
                "vehicles_count": 30,
                "controlled_vehicles": 3,
                "initial_lane_id": 1,
                "duration": 60,  # [s] The time for each training episode
                "ego_spacing": 0.3,
                "vehicles_density": 3,
                "initial_controlled_vehicle_speed": [25, 25, 25],
                "initial_controlled_vehicle_acc": [0, 0, 0],
                "lane_change_reward": 0,  # The reward received at each lane change action_num.
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    # TODO: Training Scene 1 (generate fixed position background vehicles)
    # def _create_vehicles(self) -> None:
    #     """Create some new random vehicles of a given type, and add them on the road."""
    #     other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     other_per_controlled = near_split(
    #         self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
    #     )
    #
    #     # TODO：Scene 1
    #     # Scene 1: add obstacle vehicle
    #     obstacle_vehicle_position = [210, 4]
    #     obstacle_lane_id = 1
    #     obstacle_vehicle = other_vehicles_type.create_settled(
    #         road=self.road,
    #         speed=self.np_random.uniform(low=14, high=16),
    #         lane_id=obstacle_lane_id,
    #         spacing=self.config["ego_spacing"],
    #         lead_x=obstacle_vehicle_position[0] + 10,
    #     )
    #     obstacle_vehicle.target_speed = 16
    #     obstacle_vehicle.randomize_behavior()
    #     self.road.vehicles.append(obstacle_vehicle)
    #
    #     # Scene 1：add jam vehicles
    #     jam_vehicles = [
    #         [155.17794036, 4],
    #         [100, 4],
    #         [85, 4],
    #         [65, 4],
    #         [40, 4],
    #         [189.17794036, 8],
    #         [176.17794036, 8],
    #         [165, 8],
    #         [149, 8],
    #         [130, 8],
    #         [118, 8],
    #         [109, 8],
    #         [93, 8],
    #         [85, 8],
    #         [62, 8],
    #         [39, 8],
    #         [182.17794036, 0],
    #         [195, 0],
    #         [160, 0],
    #         [140, 0],
    #         [115, 0],
    #         [95, 0],
    #         [60, 0],
    #         [40, 0],
    #         [214.25969631, 8],  # ————————————————————————
    #         [226.66740035, 0],
    #         [239.85809492, 8],  # ————————————————————————
    #         # [250.18592516, 4],
    #         [261.04299318, 8],
    #         [272.88207444, 0],
    #         # [284.71364398, 4],  # ————————————————————————
    #         # [296.16237801, 4],
    #         [308.41429593, 8],
    #         # [318.74016859, 4],
    #         [330.87969265, 8],
    #         [342.84076556, 0],
    #         [355.6141274, 4],
    #         [368.19753676, 4],
    #         [379.96224606, 8],
    #         [390.57787876, 4],
    #         [402.45314909, 8],  # ————————————————————————
    #         [413.42083242, 8],
    #         [424.30668435, 0],
    #         [436.92094203, 4],
    #         [448.51674235, 4],  # ——————————————————————————
    #         [461.78977624, 0],
    #         [473.68293203, 0],
    #         [486.36064451, 8],
    #         [497.28796333, 0],  # ——————————————————————————
    #         [509.65172481, 4],
    #         [521.19009605, 4],  # ——————————————————————————
    #         [533.50159833, 8],
    #         [544.63208692, 0],
    #         [556.00686753, 4]
    #     ]
    #     for jam_position in jam_vehicles:
    #         jam_lane_id = int(jam_position[1]/4)
    #         jam_speed = self.np_random.uniform(low=22, high=25)
    #         jam_vehicle = other_vehicles_type.create_settled(
    #             road=self.road,
    #             speed=jam_speed,
    #             lane_id=jam_lane_id,
    #             spacing=self.config["ego_spacing"],
    #             lead_x=jam_position[0] + 10,
    #         )
    #         jam_vehicle.target_speed = 27
    #         jam_vehicle.randomize_behavior()
    #         self.road.vehicles.append(jam_vehicle)
    #
    #     self.controlled_vehicles = []
    #     idx = 0
    #     for others in other_per_controlled:
    #         # generate leader settled
    #         if idx == 0:
    #             vehicle = Vehicle.create_settled(
    #                 self.road,
    #                 speed=self.config["initial_controlled_vehicle_speed"][idx],
    #                 lane_id=self.config["initial_lane_id"],
    #                 spacing=self.config["ego_spacing"],
    #                 lead_x=204,
    #             )
    #         else:
    #             # generate follower according to leader
    #             vehicle = Vehicle.create_settled(
    #                 road=self.road,
    #                 speed=self.config["initial_controlled_vehicle_speed"][idx],
    #                 lane_id=self.controlled_vehicles[-1].lane_index[2],
    #                 spacing=self.config["ego_spacing"],
    #                 lead_x=self.controlled_vehicles[-1].position[0],
    #             )
    #         vehicle = self.action_type.vehicle_class(
    #             self.road, vehicle.position, vehicle.heading, vehicle.speed
    #         )
    #         self.controlled_vehicles.append(vehicle)
    #         self.road.vehicles.append(vehicle)
    #         idx += 1

    # TODO: Training Scene 2 (generate random position background vehicles)
    # def _create_vehicles(self) -> None:
    #     """Create some new random vehicles of a given type, and add them on the road."""
    #     other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     other_per_controlled = near_split(
    #         self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
    #     )
    #
    #     self.controlled_vehicles = []
    #     idx = 0
    #     for others in other_per_controlled:
    #         # generate leader settled
    #         if idx == 0:
    #             vehicle = Vehicle.create_settled(
    #                 self.road,
    #                 speed=self.config["initial_controlled_vehicle_speed"][idx],
    #                 lane_id=self.config["initial_lane_id"],
    #                 spacing=self.config["ego_spacing"],
    #                 lead_x=204,
    #             )
    #         else:
    #             # generate follower according to leader
    #             vehicle = Vehicle.create_settled(
    #                 road=self.road,
    #                 speed=self.config["initial_controlled_vehicle_speed"][idx],
    #                 lane_id=self.controlled_vehicles[-1].lane_index[2],
    #                 spacing=self.config["ego_spacing"],
    #                 lead_x=self.controlled_vehicles[-1].position[0],
    #             )
    #         vehicle = self.action_type.vehicle_class(
    #             self.road, vehicle.position, vehicle.heading, vehicle.speed
    #         )
    #         self.controlled_vehicles.append(vehicle)
    #         self.road.vehicles.append(vehicle)
    #         idx += 1
    #
    #     # TODO: Scene 2
    #     # Scene 2: Add obstacle vehicle
    #     obs_lane_id = 1
    #     obstacle_vehicle_position = [210, 4]
    #     obs_vehicle = other_vehicles_type.create_settled(
    #         road=self.road,
    #         speed=self.np_random.uniform(low=14, high=16),
    #         lane_id=obs_lane_id,
    #         spacing=self.config["ego_spacing"],
    #         lead_x=obstacle_vehicle_position[0] + 10,
    #     )
    #     obs_vehicle.target_speed = 16
    #     obs_vehicle.randomize_behavior()
    #     self.road.vehicles.append(obs_vehicle)
    #
    #     # Scene 2: Add front vehicles
    #     for _ in range(self.config["vehicles_count"]):
    #         vehicle = other_vehicles_type.create_settled(
    #             road=self.road,
    #             speed=self.np_random.uniform(low=22, high=25),
    #             spacing=1 / self.config["vehicles_density"],
    #         )
    #         vehicle.target_speed = 27
    #         vehicle.randomize_behavior()
    #         self.road.vehicles.append(vehicle)
    #
    #     # Scene 2: Add rear vehicles
    #     rear_vehicle_count = 5
    #     for lane_id in [0, 1, 2]:
    #         if lane_id == 1:
    #             x_position = utils.generate_random_numbers(
    #                 20, 150, rear_vehicle_count, 30, 5
    #             )
    #         else:
    #             x_position = utils.generate_random_numbers(
    #                 50, 190, rear_vehicle_count, 30, 5
    #             )
    #         for x in x_position:
    #             vehicle = other_vehicles_type.create_settled(
    #                 road=self.road,
    #                 speed=self.np_random.uniform(low=22, high=25),
    #                 lane_id=lane_id,
    #                 lead_x=x + 10,
    #             )
    #             vehicle.target_speed = 27
    #             vehicle.randomize_behavior()
    #             self.road.vehicles.append(vehicle)

    # TODO: Test Scene 1 (generate random position background vehicles and several low-speed vehicles)
    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        idx = 0
        for others in other_per_controlled:
            # generate leader settled
            if idx == 0:
                vehicle = Vehicle.create_settled(
                    self.road,
                    speed=self.config["initial_controlled_vehicle_speed"][idx],
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"],
                    lead_x=204,
                )
            else:
                # generate follower according to leader
                vehicle = Vehicle.create_settled(
                    road=self.road,
                    speed=self.config["initial_controlled_vehicle_speed"][idx],
                    lane_id=self.controlled_vehicles[-1].lane_index[2],
                    spacing=self.config["ego_spacing"],
                    lead_x=self.controlled_vehicles[-1].position[0],
                )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)
            idx += 1

        # TODO: Scene 2
        # Scene 2: Add obstacle vehicle
        obs_lane_id = 1
        obstacle_vehicle_position = [210, 4]
        obs_vehicle = other_vehicles_type.create_settled(
            road=self.road,
            speed=self.np_random.uniform(low=14, high=16),
            lane_id=obs_lane_id,
            spacing=self.config["ego_spacing"],
            lead_x=obstacle_vehicle_position[0] + 10,
        )
        obs_vehicle.target_speed = 16
        obs_vehicle.randomize_behavior()
        self.road.vehicles.append(obs_vehicle)

        # Scene 2: Add front vehicles
        for _ in range(self.config["vehicles_count"]):
            vehicle = other_vehicles_type.create_settled(
                road=self.road,
                speed=self.np_random.uniform(low=22, high=25),
                spacing=1 / self.config["vehicles_density"],
            )
            vehicle.target_speed = 27
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Scene 2: Add rear vehicles
        rear_vehicle_count = 5
        for lane_id in [0, 1, 2]:
            if lane_id == 1:
                x_position = utils.generate_random_numbers(
                    20, 150, rear_vehicle_count, 30, 5
                )
            else:
                x_position = utils.generate_random_numbers(
                    50, 190, rear_vehicle_count, 30, 5
                )
            for x in x_position:
                vehicle = other_vehicles_type.create_settled(
                    road=self.road,
                    speed=self.np_random.uniform(low=22, high=25),
                    lane_id=lane_id,
                    lead_x=x + 10,
                )
                vehicle.target_speed = 27
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

        # 每个车道随机挑一辆车，作为obstacle car:
        v_0 = None
        v_2 = None
        for vehicle in self.road.vehicles:
            if v_0 and v_2:
                break
            if vehicle in self.controlled_vehicles:
                continue
            if v_0 is None and vehicle.lane_index[2] == 0:
                v_0 = vehicle
                vehicle.target_speed = 16
                vehicle.speed = self.np_random.uniform(low=20, high=22)
            if v_2 is None and vehicle.lane_index[2] == 2:
                v_2 = vehicle
                vehicle.target_speed = 16
                vehicle.speed = self.np_random.uniform(low=20, high=22)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] +
                                 self.config["on_road_reward"] +
                                 self.config["same_lane_reward"] +
                                 self.config["distance_reward"] +
                                 self.config["far_reward"] +
                                 # self.config["equal_speed_reward"] +
                                 self.config["forward_speed_reward"] +
                                 self.config["action_infeasible_reward"] +
                                 self.config["merge_reward"]
                                 # self.config["action_reward"]
                                 # self.config["same_action_reward"]
                                 ],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    # Kong add
    def _rewards(self, action: Action) -> Dict[Text, float]:
        forward_speed = []
        high_speed_reward = []
        collision_reward = []
        on_road_reward = []
        lane_index = []
        speeds = []
        acceleration = []
        x_position = []
        action_feasible_reward = []
        merge_waite_time = []
        group_action_list = []

        for vehicle in self.controlled_vehicles:
            # neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
            # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
            forward_speed.append(vehicle.speed * np.cos(vehicle.heading))
            scaled_speed = utils.lmap(np.mean(forward_speed), self.config["reward_speed_range"], [0, 1])
            high_speed_reward.append(np.clip(scaled_speed, 0, 1))
            on_road_reward.append(float(vehicle.on_road))
            collision_reward.append(float(vehicle.crashed))
            lane_index.append(vehicle.lane_index[2])
            speeds.append(vehicle.speed)
            acceleration.append(vehicle.action["acceleration"])
            x_position.append(vehicle.position[0])
            action_feasible_reward.append(float(vehicle.action_infeasible))
            merge_waite_time.append(vehicle.waite_timer)
            group_action_list.append(vehicle.action_list)

        return {
            "collision_reward": max(collision_reward),
            "high_speed_reward": float(sum(high_speed_reward) / len(high_speed_reward)),
            "on_road_reward": min(on_road_reward),
            "distance_reward": float(self._distance_reward(x_position, lane_index)),
            "same_lane_reward": self._samelane_reward(lane_index, forward_speed),
            # "equal_speed_reward": float(1 / (np.var(speeds) + 1)),
            "forward_speed_reward": self._forward_speed_reward(forward_speed),
            "far_reward": self._far(),
            "action_infeasible_reward": max(action_feasible_reward),
            "merge_reward": self._merge_reward(group_action_list),
            # "action_reward": self._action_reward(action),
            # "same_action_reward": self._same_action_reward(group_action_list)
        }

    def _same_action_reward(self, group_action_list) -> float:
        reward = 0
        for g_list in group_action_list:
            if g_list and sum(utils.diff(g_list)) == 0:
                reward -= 1/len(self.controlled_vehicles)
        return reward

    def _action_reward(self, action) -> float:
        last_index = self.action_type.index_before_merge
        last_action = utils.find_keys_by_value(self.action_type.GROUPS, last_index) if last_index is not None else action
        # Is it a split action?
        if self.action_type.SPLIT_A_MATRIX[last_action, action] == 1:
            reward = 0.04
        # Is it a merge action?
        elif self.action_type.MERGE_A_MATRIX[last_action, action] == 1:
            reward = 0.1
        # Is it a keep action?
        elif self.action_type.KEEP_A_MATRIX[last_action, action] == 1:
            reward = 0
        else:
            reward = -1
        return reward

    def _merge_reward(self, group_action_list) -> float:
        split_action = [0, 1, 2]
        max_num = 0
        for i in split_action:
            if i in group_action_list[0]:
                max_num += self.max_consecutive_count(group_action_list[0], i)
        if 5 < max_num / 15 <= 10:
            return -(max_num/15 - 5) / 5 * 1
        if max_num / 15 > 10:
            return -1
        return 0

    def max_consecutive_count(self, lst, element):
        groups = [sum(1 for _ in group) for key, group in groupby(lst) if key == element]

        return max(groups, default=0)

    def _far(self) -> float:
        far=[]
        for i in range(len(self.controlled_vehicles)):
            far.append(self.controlled_vehicles[i].position[0])
        min_far = min(far)
        if min_far < 480:
            return (min_far / 480) * 0.4
        else:
            return 0.6 + ((min_far-480) / 200) * 0.4

    def _forward_speed_reward(self, forward_speed) -> float:
        reward = 0
        min_speed = 16
        for speed in forward_speed:
            if min_speed < speed < 18:
                reward -= 1/(len(forward_speed)+speed-min_speed)
            elif speed <= min_speed:
                reward -= 1/len(forward_speed)
        return reward

    def _samelane_reward(self, lane_index, forward_speed) -> float:
        num_lane_index = len(set(lane_index))
        if num_lane_index > 1:
            reward = 0
        else:
            if min(forward_speed) <= 16:
                reward = 0.1
            elif min(forward_speed) <= 25:
                reward = 0.1 + (min(forward_speed)-16) / (25-16) * 0.9
            else:
                reward = 1
        return reward

    def _distance_reward(self, x_position, lane_index) -> float:
        sorted_zip = sorted(zip(x_position, lane_index))
        x_position, lane_index = zip(*sorted_zip)
        diff_x = utils.diff(x_position)
        if abs(max(diff_x)) > 30:
            reward = (max(diff_x) - 30) / (60-30) * 1
        elif abs(min(diff_x)) > 60:
            reward = 1.0
        else:
            reward = 0
        return reward

    def _is_terminated(self) -> bool:
        """The episode is over if one of controlled vehicles crashed or out of road."""
        for v in self.controlled_vehicles:
            if v.crashed or self.config["offroad_terminal"] and not v.on_road:
                return True
        return False
        # return (
        #     self.vehicle.crashed
        #     or self.config["offroad_terminal"]
        #     and not self.vehicle.on_road
        # )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class TestEnvFast(TestEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
