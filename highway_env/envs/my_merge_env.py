from pickle import FROZENSET
from typing import Dict, Text, Optional

import numpy as np
from itertools import groupby
from highway_env.envs.common.action import Action
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Obstacle
import random
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import RISK_FIELD
Observation = np.ndarray

class MyMergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        sum_split_weights = np.sum(config["w_game split weights"])
        sum_merge_weights = np.sum(config["w_game merge weights"])

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

                "Planner": {
                    "state": True,
                    "type": "Polynomial",
                },

                "Controller": {
                    "type": "lqr", # lqr / lmpc (longitudinal only)
                    "vehicle_model": "kinematics",  # kinematics / 8-DoF / trucksim
                },

                "Decision maker": "Rule",  # Game or Rule

                "forward_speed_reward": -10,  # 100 The vehicle was punished when speed < 19
                "on_road_reward": 0.05,  # 1
                "long_distance_reward": -30,  # 300
                "short_distance_reward": 20,  # 100
                "same_lane_reward": 15,  # 150 The reward received when vehicles in the same lane
                'far_reward': 40,  # 400
                "collision_reward": -80,  # 800 The reward received when colliding with a vehicle.
                # "action_infeasible_reward": 0,  # The reward received when got infeasible action.
                "split_reward": -90,  # -900 The vehicle was punished when splitting for long time
                "merge_reward": -3,  # -30 The vehicle was punished when merging slowly
                "high_speed_reward": 30,  # 300 The reward received when driving at full speed, linearly mapped to zero
                "safe_reward": -0,  #  -80 Risk value
                "action_reward": 10,  # 100
                "game_decision_reward": 0,  # 0 reward of game process

                "Split weights": config["w_game split weights"],
                "Merge weights": config["w_game merge weights"],

                "reward_speed_range": [18, 33],
                "lanes_count": 3,
                "vehicles_count": 10,  # 30
                "controlled_vehicles": 3,
                "initial_lane_id": 1,
                "duration": 10,  # [s] The time for each training episode
                "ego_spacing": 0.3,
                "max_vehicles_density": 0.5,
                "min_vehicles_density": 1,
                "initial_controlled_vehicle_speed": [25, 25, 25],
                "initial_controlled_vehicle_acc": [0, 0, 0],
                "lane_change_reward": 0,  # The reward received at each lane change action_num.
                "normalize_reward": True,
                "offroad_terminal": False,

                "screen_width": 1200,  # [px]
                "screen_height": 200,  # [px]
                "centering_position": [0.5, 0.5],
                "show_trajectories": False,  # Show history
                "show_future_trajectories": True,
            }
        )
        return config

    def _is_terminated(self) -> bool:
        """The episode is over if one of controlled vehicles crashed or out of road."""
        for v in self.controlled_vehicles:
            if v.crashed or self.config["offroad_terminal"] and not v.on_road:
                return True
        return False

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action_num
        :return: info dict
        """
        speeds = []
        x_position = []
        y_position = []
        accelerations = []
        steering_angles = []
        follow_distances = []
        follow_ttcs = []

        for vehicle in self.controlled_vehicles:
            speeds.append(vehicle.speed)
            x_position.append(vehicle.position[0])
            y_position.append(vehicle.position[1])
            accelerations.append(vehicle.action["acceleration"])
            steering_angles.append(vehicle.action["steering"])

            front_v, _ = self.road.neighbour_vehicles(vehicle=vehicle, lane_index=vehicle.lane_index)
            if front_v is None:
                follow_distances.append(np.inf)
                follow_ttcs.append(np.inf)
            else:
                follow_distances.append(front_v.position[0] - vehicle.position[0])
                follow_ttcs.append(utils.ttc(front_vehicle=front_v, rear_vehicle=vehicle))
        info = {
            "speed": speeds,
            "x_position": x_position,
            "y_position": y_position,
            "acceleration": accelerations,
            "steering_angle": steering_angles,
            "crashed": self.vehicle.crashed,
            "follow_distance": follow_distances,
            "follow_ttc": follow_ttcs,
        }
        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        context = getattr(self.road, "longitudinal_control", None)
        if context is not None:
            info["longitudinal_control"] = context.diagnostics()
        return info

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 1000]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH, 2 * StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, s], [n, c]]
        line_type_merge = [[c, s], [n, s], [n, c]]
        for i in range(3):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i], speed_limit=33),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                    speed_limit=33
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i], speed_limit=33
                ),
            )

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4 + 4], [ends[0], 6.5 + 4 + 4 + 4], line_types=[c, c], forbidden=True, speed_limit=20
        )
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
            speed_limit=20
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
            speed_limit=25
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
            show_future_trajectory=self.config["show_future_trajectories"]
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        """Ego vehicles"""
        lead_x_position = [204, 189, 174]
        self.controlled_vehicles = []
        for idx in range(self.config["controlled_vehicles"]):
            vehicle = Vehicle.create_settled(
                self.road,
                speed=self.config["initial_controlled_vehicle_speed"][idx],
                lane_from="a",
                lane_to="b",
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
                lead_x=lead_x_position[idx],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        """Cut-in vehicle"""
        cut_in_vehicle_type = utils.class_from_path("highway_env.vehicle.behavior.SpecialControlledVehicle")
        obs_lane_id = 2
        obstacle_vehicle_position = [188, 4]
        obs_vehicle = cut_in_vehicle_type.create_settled(
            road=self.road,
            speed=self.np_random.uniform(low=26, high=28),
            lane_from="a",
            lane_to="b",
            lane_id=obs_lane_id,
            spacing=1 / self.config["max_vehicles_density"],
            lead_x=obstacle_vehicle_position[0] + 10,
        )
        obs_vehicle.task = {
            "trigger time": 0.5,
            "type": "left lane change",
        }
        obs_vehicle.env = self
        obs_vehicle.target_speed = self.np_random.uniform(low=23, high=25)
        obs_vehicle.set_lane_index = ["a", "b", obs_lane_id]
        obs_vehicle.randomize_behavior()
        self.road.vehicles.append(obs_vehicle)

        """Front background vehicles in lane 0 & 1"""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            vehicle = other_vehicles_type.create_random(
                road=self.road,
                speed=self.np_random.uniform(low=22, high=25),
                lane_from="a",
                lane_to="b",
                lane_id=random.randint(0, 1),
                shortest_spacing=1 / self.config["max_vehicles_density"],
                longest_spacing=1 / self.config["min_vehicles_density"],
                controlled_vehicles=self.controlled_vehicles,
            )
            if vehicle.lane_index[2] == 0:
                vehicle.target_speed = self.np_random.uniform(low=28, high=30)
            else:
                vehicle.target_speed = self.np_random.uniform(low=24, high=26)
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        """Front background vehicles in lane 2"""
        vehicle_count = 4
        x_position = utils.generate_random_numbers(
            obs_vehicle.position[0] + 20, obs_vehicle.position[0] + 100, vehicle_count, 15, 5
        )
        for x_p in x_position:
            vehicle = other_vehicles_type.create_settled(
                road=self.road,
                speed=self.np_random.uniform(low=18, high=22),
                lane_from="a",
                lane_to="b",
                lane_id=2,
                lead_x=x_p + 10,
            )
            vehicle.target_speed = self.np_random.uniform(low=28, high=30)
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        """Rear background vehicles"""
        rear_vehicle_count = 6
        for lane_id in [0, 1, 2]:
            if lane_id == 0:
                x_position = utils.generate_random_numbers(
                    -30, 140, rear_vehicle_count, 30, 5
                )
            elif lane_id == 1:
                x_position = utils.generate_random_numbers(
                    -50, 150, rear_vehicle_count, 30, 5
                )
            else:
                x_position = utils.generate_random_numbers(
                    -80, 120, rear_vehicle_count, 30, 5
                )
            for x in x_position:
                vehicle = other_vehicles_type.create_settled(
                    road=self.road,
                    speed=self.np_random.uniform(low=22, high=25),
                    lane_from="a",
                    lane_to="b",
                    lane_id=lane_id,
                    lead_x=x + 10,
                )
                if lane_id == 0:
                    vehicle.target_speed = self.np_random.uniform(low=28, high=30)
                elif lane_id == 1:
                    vehicle.target_speed = self.np_random.uniform(low=24, high=26)
                else:
                    vehicle.target_speed = self.np_random.uniform(low=28, high=30)
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

        """Ramp vehicles"""
        merge_vehicle_type = utils.class_from_path("highway_env.vehicle.behavior.MergeIDMVehicle")
        ramp_vehicle_count = [3]  # Vehicles in straight lane, ramp lane and merging area
        lanes = [("k", "b", 0)]
        position_limits = [(20, 75)]
        for count, lane, p_lim in zip(ramp_vehicle_count, lanes, position_limits):
            x_positions = utils.generate_random_numbers(
                p_lim[0], p_lim[1], count, 30, 10
            )
            for x in x_positions:
                vehicle = merge_vehicle_type.create_settled(
                    road=self.road,
                    speed=self.np_random.uniform(low=18, high=22),
                    lane_from=lane[0],
                    lane_to=lane[1],
                    lane_id=lane[2],
                    lead_x=x + 10,
                )
                vehicle.target_speed = self.np_random.uniform(low=25, high=30)
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)


    def _reward(self, action: int) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward
            for name, reward in rewards.items()
        )
        reward = utils.lmap(
            reward,
            [
                self.config["forward_speed_reward"] +
                self.config["long_distance_reward"] +
                self.config["collision_reward"] +
                self.config["split_reward"] +
                self.config["merge_reward"] +
                self.config["safe_reward"],

                self.config["on_road_reward"] +
                self.config["short_distance_reward"] +
                self.config["same_lane_reward"] +
                self.config["far_reward"] +
                self.config["high_speed_reward"] +
                self.config["action_reward"] +
                self.config["game_decision_reward"],
            ],
            [0, 1],
        )
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: int) -> Dict[Text, float]:
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
        env_state_list = []

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
            env_state_list.append(vehicle.env_state_list)

        return {
            "collision_reward": max(collision_reward),
            "high_speed_reward": float(sum(high_speed_reward) / len(high_speed_reward)),
            "on_road_reward": min(on_road_reward),
            "long_distance_reward": float(self._distance_reward(x_position, lane_index)[0]),
            "short_distance_reward": float(self._distance_reward(x_position, lane_index)[1]),
            "same_lane_reward": self._samelane_reward(lane_index, forward_speed),
            # "equal_speed_reward": float(1 / (np.var(speeds) + 1)),
            "forward_speed_reward": self._forward_speed_reward(forward_speed),
            "far_reward": self._far(),
            "action_infeasible_reward": max(action_feasible_reward),
            "split_reward": self._split_reward(group_action_list),
            "merge_reward": self._merge_reward(env_state_list, group_action_list),
            "action_reward": self._action_reward(action),
            # "same_action_reward": self._same_action_reward(group_action_list),
            "safe_reward": self._safe_reward(),
            "game_decision_reward": self._game_decision_reward(),
        }

    def _game_decision_reward(self) -> float:
        best_cost = self.controlled_vehicles[0].best_cost
        if best_cost is not None:
            reward = best_cost / sum(self.config["w_game merge weights"])
        else:
            reward = 0
        return reward

    def _same_action_reward(self, group_action_list) -> float:
        reward = 0
        for g_list in group_action_list:
            if g_list and sum(utils.diff(g_list)) == 0:
                reward -= 1/len(self.controlled_vehicles)
        return reward

    def _action_reward(self, action) -> float:
        if action == 3:
            reward = 1
        else:
            reward = 0
        return reward

    def _split_reward(self, group_action_list) -> float:
        split_action = [0, 1, 2]
        array = np.array(group_action_list[0])
        if 3 in array:
            last_merge_index = np.where(array == 3)[0][-1]
            if last_merge_index < len(array) - 1:
                array = array[last_merge_index + 1:]
            else:
                array = []

        max_num = 0
        fre = self.config["simulation_frequency"]
        for i in split_action:
            if i in group_action_list[0]:
                max_num += self.max_consecutive_count(array, i)
        if max_num / fre <= 5:
            reward = (max_num / fre) / 5 * 0.2
        elif 5 < max_num / fre <= 10:
            reward = 0.2 + (max_num / fre - 5) / 5 * 0.6
        else:
            reward = 1
        return reward

    def _merge_reward(self, env_state_list, group_action_list) -> float:
        array = np.array(env_state_list[0]) - np.array(group_action_list[0])
        if len(array) > 0:
            last_zero_index = np.where(array == 0)[0][-1]
            count = self.max_consecutive_non_zero_count(array[last_zero_index+1:]) \
                if last_zero_index < len(array) - 1 else 0
        else:
            count = 0

        fre = self.config["simulation_frequency"]
        if count / fre <= 5:
            reward = 0
        elif 5 < count / fre <= 8:
            reward = (count / fre) / 5 * 0.2
        elif 8 < count/fre <= 15:
            reward = 0.2 + (count/fre - 5) / (10 - 5) * 0.6
        else:
            reward = 1
        return reward

    def max_consecutive_count(self, lst, element):
        groups = [sum(1 for _ in group) for key, group in groupby(lst) if key == element]

        return max(groups, default=0)

    def max_consecutive_non_zero_count(self, arr):
        max_count = 0
        current_count = 0

        for num in arr:
            if num != 0:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def _far(self) -> float:
        far=[]
        for i in range(len(self.controlled_vehicles)):
            far.append(self.controlled_vehicles[i].position[0])
        min_far = min(far)
        if min_far < 700:
            return (min_far / 700) * 0.1
        else:
            return 0.6 + ((min_far-700) / 200) * 0.4

    def _forward_speed_reward(self, forward_speed) -> float:
        min_speed = 16
        ave_speed = np.mean(forward_speed)
        if min_speed < ave_speed < 19:
            return -(ave_speed - min_speed) / (19-16) * 1
        elif ave_speed <= min_speed:
            return -1
        else:
            return 0

    def _samelane_reward(self, lane_index, forward_speed) -> float:
        num_lane_index = len(set(lane_index))
        if num_lane_index > 1:
            reward = 0
        else:
            if min(forward_speed) <= 16:
                reward = 0.1
            elif min(forward_speed) <= 22:
                reward = 0.1 + (min(forward_speed)-16) / (22-16) * 0.9
            else:
                reward = 1
        return reward

    def _distance_reward(self, x_position, lane_index) -> tuple:
        sorted_zip = sorted(zip(x_position, lane_index), reverse=True)
        x_position, lane_index = zip(*sorted_zip)
        diff_x = np.inf
        if len(np.unique(lane_index)) == 1:
            diff_x = max(utils.diff(x_position))
        elif len(np.unique(lane_index)) == 2:
            l = np.array(lane_index)
            l1 = np.where(l == np.unique(lane_index)[0])
            l2 = np.where(l == np.unique(lane_index)[1])
            if len(l1[0]) > 1:
                diff_x = abs(x_position[l1[0][0]] - x_position[l1[0][1]])
            if len(l2[0]) > 1:
                diff_x = abs(x_position[l2[0][0]] - x_position[l2[0][1]])

        # long distance punish
        if 13 <= diff_x <= 25:
            l_r = (diff_x - 10) / (25 - 10) * 0.3
        elif 25 < abs(diff_x) <= 50:
            l_r = 0.3 + (diff_x - 25) / (50 - 25) * 0.6
        elif abs(diff_x) > 50:
            l_r = 1
        else:
            l_r = 0

        # short_distance_reward
        if 10 < diff_x < 13:
            s_r = 0.8 - (diff_x - 10) / (13 - 10) * 0.8
        elif 7 < diff_x <= 10:
            s_r = 1 - (diff_x - 7) / (10 - 7) * 0.2
        elif diff_x <= 7:
            s_r = 0
        else:
            s_r = 0

        return l_r, s_r

    def _safe_reward(self):
        """
        Risk Value
        """
        risk_value = []
        rf = RISK_FIELD()
        for vehicle in self.controlled_vehicles:
            close_vehicles = self.road.close_objects_to(
                vehicle,
                self.PERCEPTION_DISTANCE,
                count=4,
                see_behind=False,
                sort=True,
                vehicles_only=False,
            )
            risk_value.append(
                rf.average_safety_risk(target_position=vehicle.position, risk_vehicles=close_vehicles)
            )
        reward = (max(risk_value)) / rf.max_risk_value
        return reward




