import numpy as np
from highway_env import utils
from highway_env.vehicle.controller import LEADVehicle, FOLLOWVehicle
from highway_env.vehicle.longitudinal import bind_longitudinal_source
import gurobipy as gp
from gurobipy import GRB, quicksum
import copy
import math


class COALITION_GAME_MAKER:
    """
    Decision made by coalition game
    """
    GROUPS = {
        0: [0, [1, 2]],
        1: [[0, 1], 2],
        2: [0, 1, 2],
        3: [[0, 1, 2]],
    }
    LANE_CENTER_Y = {
        0: 0,
        1: 4,
        2: 8,
        3: 12,
    }
    LANE_WIDTH = 4

    ACC_MAX = 6
    ACC_MIN = -6

    TTC_MIN = 0  # 0.2

    MIN_REWARD = -1e6

    # A_matrix = split_A_matrix + merge_A_matrix + keep_A_matrix
    A_MATRIX = np.ones(4)

    SPLIT_A_MATRIX = np.array([[0, 0, 1, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 0],
                               [1, 1, 1, 0]])

    MERGE_A_MATRIX = np.array([[0, 1, 0, 1],
                               [1, 0, 0, 1],
                               [1, 1, 0, 1],
                               [0, 0, 0, 0]])

    SPLIT_AND_MERGE_MATRIX = np.array([[0, 1, 0, 0],
                                       [1, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0]])

    KEEP_A_MATRIX = np.eye(4)

    def __init__(self, env):
        self.env = env
        self.ref_actions = None
        self.ref_lane_index = None
        self.action_list = None
        self.merge_state = "Done"

    def merge_state_update(self, merge_group, ideal_env_state, env_state: int = None) -> int:
        controlled_vehicles = self.env.controlled_vehicles
        x_positions = []
        y_positions = []
        lane_indexes = []
        all_merge = True

        # Do all vehicles successfully merge?
        for group in merge_group:
            if isinstance(group, list):
                x_positions = []
                lane_indexes = []
                for idx in group:
                    x_positions.append(controlled_vehicles[idx].position[0])
                    lane_indexes.append(controlled_vehicles[idx].lane_index)
            else:
                continue

            # If group's vehicles are in the same lane?
            if not all(x == lane_indexes[0] for x in lane_indexes):
                all_merge = False

            # If there is no other vehicles inside group?
            for other_v in self.env.road.vehicles:
                if other_v not in controlled_vehicles:
                    # If other_v position_x inside platoon and other_v lane_index equal to platoon
                    if utils.inside_list(other_v.position[0], x_positions, reverse=True) and other_v.lane_index == lane_indexes[0]:
                        all_merge = False

        if all_merge:
            # Are vehicles not close enough?
            distances = utils.diff(x_positions)
            ref_distance = 10
            if max(abs(np.array(distances) - ref_distance)) > 2:
                self.merge_state = "Space Tuning"
                return env_state
            # All is done!
            self.merge_state = "Done"
            return ideal_env_state

        # Do partial vehicles successfully merge?
        # Ex: ideal is [[0, 1, 2]], partial is [[0, 1], 2] or [0, [1, 2]]
        if env_state is not None:
            if len(merge_group) == 1:
                for idx in merge_group[0]:
                    x_positions.append(controlled_vehicles[idx].position[0])
                    y_positions.append(controlled_vehicles[idx].position[1])
                    lane_indexes.append(controlled_vehicles[idx].lane_index)
                sorted_x_positions, sorted_y_positions, sorted_lane_index, sorted_index = zip(
                    *sorted(zip(x_positions, y_positions, lane_indexes, list(range(len(x_positions)))),
                            key=lambda x: x[0], reverse=True)
                )
                # prob : [[0, 1], 2], [0, [1, 2]]
                for index_group, i in zip([[0, 1], [1, 2]], [0, 1]):
                    # same lane?
                    if (sorted_lane_index[index_group[0]] == sorted_lane_index[index_group[1]]
                            and abs(sorted_y_positions[index_group[0]] - sorted_y_positions[
                                index_group[1]] <= 0.5)):
                        for other_v in self.env.road.vehicles:
                            if other_v not in controlled_vehicles:
                                # If other_v position_x inside platoon and other_v lane_index equal to platoon
                                x_list = [sorted_x_positions[index_group[0]],
                                          sorted_x_positions[index_group[1]]]
                                if utils.inside_list(other_v.position[0], x_list,
                                                     reverse=True) and other_v.lane_index == \
                                        sorted_lane_index[index_group[0]]:
                                    continue
                                # close enough?
                                distance = sorted_x_positions[index_group[0]] - sorted_x_positions[
                                    index_group[1]]
                                ref_distance = 10
                                self.merge_state = "Still Merging"
                                if abs(distance - ref_distance) <= 2:
                                    return 1 if i == 0 else 0
                                else:
                                    return env_state

        self.merge_state = "Still Merging"
        return env_state

    def background_vehicles_predict(self, copy_all_vehicles, pre_time):
        delete_list = []
        predict_all_vehicles = copy.deepcopy(copy_all_vehicles)
        for vehicle, i in zip(predict_all_vehicles, range(len(predict_all_vehicles))):
            if type(vehicle).__name__ != "FOLLOWVehicle":
                vehicle.position[0] = (vehicle.position[0]
                                       + vehicle.speed * pre_time
                                       + 0.5 * vehicle.action["acceleration"] * pre_time ** 2)
            else:
                delete_list.append(i)
        del predict_all_vehicles[0: len(delete_list)]
        del copy_all_vehicles[0: len(delete_list)]
        return copy_all_vehicles, predict_all_vehicles

    def coalition_game_decision(self, index_group, weights, platoon_state, ideal_index_group: list = None):
        """
        Using coalition game to decide each car's lateral action.
        """
        ref_actions = [None] * len(self.env.controlled_vehicles)
        ref_lane_index = [None] * len(self.env.controlled_vehicles)

        # The prediction of each action, generate cost matrix
        basic_actions = ["left", "right", "keep"]

        if len(index_group) > 1:
            lead_vehicles = []
            lead_v_x = []
            lead_idx = []
            for index in index_group:
                if isinstance(index, list):
                    vehicle = self.env.controlled_vehicles[index[0]]
                    lead_idx.append(index[0])
                else:
                    vehicle = self.env.controlled_vehicles[index]
                    lead_idx.append(index)
                lead_vehicles.append(vehicle)
                lead_v_x.append(vehicle.position[0])

            sorted_v_x, sorted_lead_vehicles, sorted_lead_idx, sorted_index_group = zip(
                *sorted(list(zip(lead_v_x, lead_vehicles, lead_idx, index_group)), key=lambda x: x[0], reverse=True)
            )

            """Reward list"""
            action_list = self.actions_combination(len(sorted_index_group), basic_actions)
            reward_list, new_action_list = self.new_generate_rewards(action_list, sorted_index_group, weights)
            # reward_list, new_action_list, acc_reward_list, lat_reward_list, connect_reward_list, x_reward_list, y_reward_list = self.generate_rewards(action_list, sorted_index_group, lead_vehicles, sorted_lead_idx, weights)

            best_action, best_cost = self.game_optimization(reward_list, new_action_list, sorted_index_group)

            for index, action in zip(sorted_index_group, best_action):
                if action == "left":
                    _dir, lateral_action = -1, 0
                elif action == "right":
                    _dir, lateral_action = 1, 1
                else:
                    _dir, lateral_action = 0, 2
                if isinstance(index, list):
                    target_l_idx = tuple(list(self.env.controlled_vehicles[index[0]].lane_index)[:2] + [
                        (self.env.controlled_vehicles[index[0]].lane_index[2]) + _dir])
                    for i in index:
                        ref_actions[i] = lateral_action
                        ref_lane_index[i] = target_l_idx
                else:
                    target_l_idx = tuple(list(self.env.controlled_vehicles[index].lane_index)[:2] + [
                        (self.env.controlled_vehicles[index].lane_index[2]) + _dir])
                    ref_actions[index] = lateral_action
                    ref_lane_index[index] = target_l_idx

        else:
            # reward_list, new_action_list = self._rewards(basic_actions, index_group[0], weights)
            reward_list, new_action_list = self.new_generate_rewards(basic_actions, index_group, weights)
            best_action, best_cost = self.game_optimization(reward_list, new_action_list, index_group)

            if best_action == "left":
                _dir, lateral_action = -1, 0
            elif best_action == "right":
                _dir, lateral_action = 1, 1
            else:
                _dir, lateral_action = 0, 2
            target_l_idx = tuple(list(self.env.controlled_vehicles[index_group[0][0]].lane_index)[:2] + [
                    (self.env.controlled_vehicles[index_group[0][0]].lane_index[2]) + _dir])
            for i in index_group[0]:
                ref_actions[i] = lateral_action
                ref_lane_index[i] = target_l_idx

        # Update best_cost feature
        for v in self.env.controlled_vehicles:
            if platoon_state == "merge":
                v.best_cost = best_cost / sum(self.env.config["Merge weights"])
            else:
                v.best_cost = best_cost / sum(self.env.config["Split weights"])

        return ref_actions, ref_lane_index

    # def generate_rewards(self, all_action_list, sorted_index_group, lead_vehicles, sorted_lead_idx, weights):
    #     """
    #     Predict acceleration reward
    #     """
    #     pre_vehicles = copy.deepcopy(self.env.controlled_vehicles)
    #     pre_time = 0.1
    #     reward_list = []
    #     acc_reward_list = []
    #     lat_reward_list = []
    #     connect_reward_list = []
    #     x_reward_list = []
    #     y_reward_list = []
    #
    #     for actions, action_idx in zip(all_action_list, range(len(all_action_list))):
    #         acc_reward = 0
    #         lat_reward = 0
    #         pre_x = []
    #         pre_y = []
    #         infeasible = False
    #
    #         for action, lead_v, lead_idx, idx in zip(actions, lead_vehicles, sorted_lead_idx, sorted_index_group):
    #             """If action is infeasible, cut."""
    #             infeasible, target_lane_index = self.action_cut(lead_idx, action)
    #             if infeasible:
    #                 break
    #
    #             """acceleration"""
    #             acc_min = -20
    #             acc_max = 20
    #             new_v = LEADVehicle(
    #                 road=self.env.road,
    #                 position=lead_v.position,
    #                 speed=lead_v.speed,
    #                 target_speed=33,
    #                 timer=lead_v.timer,
    #                 target_lane_index=lead_v.target_lane_index,
    #             )
    #             front_v, rear_v = self.env.road.neighbour_vehicles(vehicle=lead_v, lane_index=target_lane_index)
    #             acc = new_v.lqr_control(ego_vehicle=lead_v, front_vehicle=front_v, is_limited=False)
    #             acc_reward += utils.normalization([acc_min, acc_max], acc)
    #
    #             """pre_x, pre_y"""
    #             if isinstance(idx, int):
    #                 pre_vehicles[idx].position[0] = pre_vehicles[idx].position[0] + pre_vehicles[idx].speed * pre_time + 0.5 * acc * pre_time ** 2
    #                 pre_vehicles[idx].position[1] = self.LANE_CENTER_Y[target_lane_index[2]]
    #                 pre_vehicles[idx].lane_ndex = target_lane_index
    #                 pre_x.append(pre_vehicles[idx].position[0])
    #                 pre_y.append(pre_vehicles[idx].position[1])
    #             else:
    #                 for i in idx:
    #                     pre_vehicles[i].position[0] = pre_vehicles[i].position[0] + pre_vehicles[
    #                         i].speed * pre_time + 0.5 * acc * pre_time ** 2
    #                     pre_vehicles[i].position[1] = self.LANE_CENTER_Y[target_lane_index[2]]
    #                     pre_vehicles[i].lane_index = target_lane_index
    #                     pre_x.append(pre_vehicles[i].position[0])
    #                     pre_y.append(pre_vehicles[i].position[1])
    #
    #             """lateral distance"""
    #             lat_reward += np.sign(lead_v.heading) * (self.LANE_CENTER_Y[target_lane_index[2]] - lead_v.position[1]) / self.LANE_WIDTH
    #
    #         if infeasible:
    #             all_action_list[action_idx] = None
    #             continue
    #
    #         """x/y distance error"""
    #         max_distance = 50
    #         max_y_err = 8
    #         x_error = sum(np.array(utils.diff(sorted(pre_x, reverse=True))) - 10) / (len(pre_x) - 1)
    #         x_reward = 1 - x_error / max_distance
    #         y_error = sum(utils.diff(sorted(pre_y, reverse=True))) / (len(pre_y) - 1)
    #         y_reward = 1 - y_error / max_y_err
    #
    #         """connection"""
    #         connect_reward = self.connect_nodes(pre_vehicles, self.env.controlled_vehicles)
    #
    #         reward = sum([w * r for w, r in zip(weights, [acc_reward, connect_reward, x_reward, y_reward, 0, lat_reward])])
    #         reward_list.append(reward)
    #         acc_reward_list.append(reward)
    #         lat_reward_list.append(lat_reward)
    #         x_reward_list.append(x_reward)
    #         y_reward_list.append(y_reward)
    #         connect_reward_list.append(connect_reward)
    #
    #     new_action_list = list(filter(lambda x: x is not None, all_action_list))
    #     return reward_list, new_action_list, acc_reward_list, lat_reward_list, connect_reward_list, x_reward_list, y_reward_list
    #
    # def _rewards(self, all_action_list, index_group, weights):
    #     """
    #     Predict acceleration reward
    #     """
    #     pre_vehicles = copy.deepcopy(self.env.controlled_vehicles)
    #     pre_time = 0.1
    #     reward_list = []
    #     acc_reward_list = []
    #     lat_reward_list = []
    #     connect_reward_list = []
    #     x_reward_list = []
    #     y_reward_list = []
    #
    #     for action, action_idx in zip(all_action_list, range(len(all_action_list))):
    #         acc_reward = 0
    #         lat_reward = 0
    #         pre_x = []
    #         pre_y = []
    #         infeasible = False
    #
    #         """If action is infeasible, cut."""
    #         infeasible, target_lane_index = self.action_cut(index_group[0], action)
    #         if infeasible:
    #             all_action_list[action_idx] = None
    #             continue
    #
    #         """acceleration"""
    #         acc_min = -20
    #         acc_max = 20
    #         lead_v = self.env.controlled_vehicles[index_group[0]]
    #         new_v = LEADVehicle(
    #             road=self.env.road,
    #             position=lead_v.position,
    #             speed=lead_v.speed,
    #             target_speed=33,
    #             timer=lead_v.timer,
    #             target_lane_index=lead_v.target_lane_index,
    #         )
    #         front_v, rear_v = self.env.road.neighbour_vehicles(vehicle=lead_v, lane_index=target_lane_index)
    #         acc = new_v.lqr_control(ego_vehicle=lead_v, front_vehicle=front_v, is_limited=False)
    #         acc_reward = utils.normalization([acc_min, acc_max], acc)
    #
    #         """pre_x, pre_y"""
    #         for i in index_group:
    #             pre_vehicles[i].position[0] = pre_vehicles[i].position[0] + pre_vehicles[
    #                 i].speed * pre_time + 0.5 * acc * pre_time ** 2
    #             pre_vehicles[i].position[1] = self.LANE_CENTER_Y[target_lane_index[2]]
    #             pre_vehicles[i].lane_index = target_lane_index
    #             pre_x.append(pre_vehicles[i].position[0])
    #             pre_y.append(pre_vehicles[i].position[1])
    #
    #         """lateral distance"""
    #         lat_reward = np.sign(lead_v.heading) * (
    #                     self.LANE_CENTER_Y[target_lane_index[2]] - lead_v.position[1]) / self.LANE_WIDTH
    #
    #         """x/y distance error"""
    #         max_distance = 50
    #         max_y_err = 8
    #         x_error = sum(np.array(utils.diff(sorted(pre_x, reverse=True))) - 10) / (len(pre_x) - 1)
    #         x_reward = 1 - x_error / max_distance
    #         y_error = sum(utils.diff(sorted(pre_y, reverse=True))) / (len(pre_y) - 1)
    #         y_reward = 1 - y_error / max_y_err
    #
    #         """connection"""
    #         connect_reward = self.connect_nodes(pre_vehicles, self.env.controlled_vehicles)
    #
    #         reward = sum([w * r for w, r in zip(weights, [acc_reward, connect_reward, x_reward, y_reward, 0, lat_reward])])
    #         reward_list.append(reward)
    #         acc_reward_list.append(reward)
    #         lat_reward_list.append(lat_reward)
    #         x_reward_list.append(x_reward)
    #         y_reward_list.append(y_reward)
    #         connect_reward_list.append(connect_reward)
    #
    #     new_action_list = list(filter(lambda x: x is not None, all_action_list))
    #     return reward_list, new_action_list

    def connect_nodes(self, pre_vehicles, vehicles):
        con_value = {
            "Connected": 1,
            "Weakly Connected": 0.6,
            "Very Weakly Connect": 0.1,
            "Disconnect": 0,
        }

        sorted_vehicles = sorted(vehicles, key=lambda x: x.position[0])
        sorted_pre_vehicles = sorted(pre_vehicles, key=lambda x: x.position[0])
        v1, v2, v3 = sorted_vehicles[0], sorted_vehicles[1], sorted_vehicles[2]
        obs12, id12, obs23, id23 = [], [], [], []
        pre_v1, pre_v2, pre_v3 = sorted_vehicles[0], sorted_vehicles[1], sorted_vehicles[2]
        pre_obs12, pre_id12, pre_obs23, pre_id23 = [], [], [], []

        for v in self.env.road.vehicles:
            if type(v).__name__ == "FOLLOWVehicle":
                continue

            if v1.position[0] <= v.position[0] <= v2.position[0] and (v.lane_index == v1.lane_index or v.lane_index == v2.lane_index):
                obs12.append(v)
                id12.append(v.lane_index[2])
            if v2.position[0] <= v.position[0] <= v3.position[0] and (v.lane_index == v2.lane_index or v.lane_index == v3.lane_index):
                obs12.append(v)
                id23.append(v.lane_index[2])

            if pre_v1.position[0] <= v.position[0] <= pre_v2.position[0] and (v.lane_index == pre_v1.lane_index or v.lane_index == pre_v2.lane_index):
                pre_obs12.append(v)
                pre_id12.append(v.lane_index[2])
            if pre_v2.position[0] <= v.position[0] <= pre_v3.position[0] and (v.lane_index == pre_v2.lane_index or v.lane_index == pre_v3.lane_index):
                pre_obs12.append(v)
                pre_id23.append(v.lane_index[2])

        connect_value = 0
        for obs, _id, v1, v2 in zip([obs12, obs23], [id12, id23], sorted_vehicles[:2], sorted_vehicles[1:]):
            if len(obs) == 0:
                connect_value += con_value["Connected"]
            elif len(set(_id)) == 1:
                connect_value += 1 - (1 - con_value["Weakly Connected"]) * len(_id) / 5
            elif v1.lane_index == v2.lane_index:
                connect_value += con_value["Disconnect"]
            else:
                connect_value += con_value["Weakly Connected"] - (con_value["Weakly Connected"] - con_value["Very Weakly Connect"]) * len(_id) / 5

        pre_connect_value = 0
        for obs, _id, pre_v1, pre_v2 in zip([pre_obs12, pre_obs23], [pre_id12, pre_id23], sorted_pre_vehicles[:2], sorted_pre_vehicles[1:]):
            if len(obs) == 0:
                pre_connect_value += con_value["Connected"]
            elif len(set(_id)) == 1:
                pre_connect_value += 1 - (1 - con_value["Weakly Connected"]) * len(_id) / 5
            elif pre_v1.lane_index == pre_v2.lane_index:
                pre_connect_value += con_value["Disconnect"]
            else:
                pre_connect_value += con_value["Weakly Connected"] - (con_value["Weakly Connected"] - con_value["Very Weakly Connect"]) * len(_id) / 5

        connect_reward = (pre_connect_value - connect_value) / 2

        return connect_reward

    def actions_combination(self, player_num, basic_actions):
        """
        Combine all possible actions of all cars
        """
        all_action_list = None
        car_num = len(self.env.controlled_vehicles)
        if player_num == 1:
            all_action_list = basic_actions
        elif player_num == 2:
            list1 = [basic_actions[0]] * car_num + [basic_actions[1]] * car_num + [basic_actions[2]] * car_num
            list2 = basic_actions * car_num
            all_action_list = [[a, b] for a, b in zip(list1, list2)]
        elif player_num == 3:
            list1 = [basic_actions[0]] * car_num ** 2 + [basic_actions[1]] * car_num ** 2 + [basic_actions[2]] * car_num ** 2
            list2 = [basic_actions[0]] * car_num + [basic_actions[1]] * car_num + [basic_actions[2]] * car_num
            list2 = list2 * car_num
            list3 = basic_actions * car_num ** 2
            all_action_list = [[a, b, c] for a, b, c in zip(list1, list2, list3)]

        return all_action_list

    def action_cut(self, index, action) -> tuple:
        if isinstance(index, int):
            vehicle = self.env.controlled_vehicles[index]
            end_vehicle = vehicle
        else:
            vehicle = self.env.controlled_vehicles[index[0]]
            end_vehicle = self.env.controlled_vehicles[index[-1]]
        if action == "left":
            dir = -1
        elif action == "right":
            dir = 1
        else:
            dir = 0
        target_lane_index = tuple(list(vehicle.lane_index[:2]) + [vehicle.lane_index[2] + dir])

        # Is target lane out of current road?
        if target_lane_index[2] < 0 or target_lane_index[2] >= len(
                self.env.road.network.graph[target_lane_index[0]][target_lane_index[1]]):
            return True, target_lane_index

        # Is target lane near ramp merge area?
        if target_lane_index[2] == 3 or (target_lane_index[2] == 2 and 150 <= vehicle.position[0] <= 310):
            return True, target_lane_index

        # Is the spacing and ttc unsafe to change?
        lc_t = 0.5
        if target_lane_index != vehicle.lane_index:
            front_v, rear_v = self.env.road.neighbour_vehicles(vehicle=vehicle, lane_index=target_lane_index)
            ttc_safe = 0.3
            if front_v is not None:
                # front safety spacing
                s0 = front_v.position[0] - vehicle.position[0]
                s1 = vehicle.speed * lc_t + 0.5 * vehicle.action["acceleration"] * lc_t ** 2
                s2 = front_v.speed * lc_t + 0.5 * front_v.action["acceleration"] * lc_t ** 2
                if s0 + s2 <= s1 + vehicle.LENGTH:
                    return True, target_lane_index

                # front safety ttc
                ttc_front = utils.ttc(front_v, vehicle)
                if front_v.position[0] - vehicle.position[0] <= 1.5 * vehicle.LENGTH or ttc_front <= ttc_safe:
                    return True, target_lane_index
            if rear_v is not None:
                if rear_v.position[0] < end_vehicle.position[0]:
                    # rear safety spacing
                    s0 = end_vehicle.position[0] - rear_v.position[0]
                    s1 = end_vehicle.speed * lc_t + 0.5 * end_vehicle.action["acceleration"] * lc_t ** 2
                    s2 = rear_v.speed * lc_t + 0.5 * rear_v.action["acceleration"] * lc_t ** 2
                    if s0 + s1 <= s2 + vehicle.LENGTH:
                        return True, target_lane_index

                    # rear spacing ttc
                    ttc_rear = utils.ttc(end_vehicle, rear_v)
                    if end_vehicle.position[0] - rear_v.position[0] <= 1.5 * end_vehicle.LENGTH or ttc_rear <= ttc_safe:
                        return True, target_lane_index
                else:
                    return True, target_lane_index


        return False, target_lane_index

    def group_action_to_vehicle_action(self, action) -> dict:
        """
        根据分组动作计算单车横向动作，通过mobile模型计算横向动作，通过lqr模型计算加速度
        param: action: 分组动作
        return: (target_lateral_action, target_acceleration): 单车动作列表
        """
        # TODO: Is current action split or merge?
        env_state = self.env.controlled_vehicles[0].env_state
        if self.SPLIT_A_MATRIX[env_state, action]:
            context = getattr(self.env.road, "longitudinal_control", None)
            if context is not None and context.kind == "lmpc":
                context.set_groups(self.GROUPS[action])
            higher_action = self.split(action)
            # update env_state
            for v in self.env.controlled_vehicles:
                v.env_state = action

        elif self.MERGE_A_MATRIX[env_state, action]:
            if self.SPLIT_AND_MERGE_MATRIX[env_state, action]:
                split_and_merge = True
            else:
                split_and_merge = False
            context = getattr(self.env.road, "longitudinal_control", None)
            if context is not None and context.kind == "lmpc":
                context.set_groups(self.GROUPS[action])
            higher_action = self.merge(env_state, action, split_and_merge)
            env_state = self.merge_state_update(merge_group=self.GROUPS[action], ideal_env_state=action, env_state=env_state)
            if self.merge_state == "Done":
                # update env_state
                for v in self.env.controlled_vehicles:
                    v.env_state = env_state
        else:
            context = getattr(self.env.road, "longitudinal_control", None)
            if context is not None and context.kind == "lmpc":
                context.set_groups(self.GROUPS[action])
            higher_action = self.keep(action)
            # update env_state
            for v in self.env.controlled_vehicles:
                v.env_state = action

        # update group_action
        for v in self.env.controlled_vehicles:
            v.group_action = action

        return higher_action

    def decision_execution(self, index_group, ref_lane_index, ref_actions):
        controlled_num = self.env.config["controlled_vehicles"]
        target_actions = [None] * controlled_num
        target_lane_index = [None] * controlled_num
        target_acceleration = [None] * controlled_num

        for v_list in index_group:
            # TODO: Leader acceleration
            if isinstance(v_list, int):
                front_id = v_list
                group = None
            else:
                front_id = v_list[0]
                group = []
                for i in v_list:
                    group.append(self.env.controlled_vehicles[i])
            first_v = self.env.controlled_vehicles[front_id]
            leader_v = LEADVehicle(
                road=self.env.road,
                position=first_v.position,
                speed=first_v.speed,
                target_speed=33,
                timer=first_v.timer,
                target_lane_index=first_v.target_lane_index,
            )
            bind_longitudinal_source(leader_v, first_v)

            # If the reference lateral action is safe to execute?
            # [lead_action, lead_target_lane_index] = leader_v.change_lane_policy(
            #     controlled_vehicle=first_v, group=group, target_lane_index=ref_lane_index[front_id]
            # )
            lead_action = ref_actions[front_id]
            lead_target_lane_index = ref_lane_index[front_id]
            first_v.timer = leader_v.timer

            # Update small team
            if isinstance(v_list, list):
                for j in v_list:
                    target_actions[j] = lead_action
                    target_lane_index[j] = lead_target_lane_index
            else:
                target_actions[v_list] = lead_action
                target_lane_index[v_list] = lead_target_lane_index
            target_actions[front_id] = lead_action
            target_lane_index[front_id] = lead_target_lane_index

            # LQR计算leader车纵向加速度：
            lead_acceleration = leader_v.integrated_longitudinal_control()
            first_v.action["acceleration"] = lead_acceleration  # update acceleration to let the followers know
            target_acceleration[front_id] = lead_acceleration

            # TODO: Follower acceleration
            if isinstance(v_list, list):
                follow_list = v_list[1:]
                follow_index = 1
                for follow_id in follow_list:
                    v = self.env.controlled_vehicles[follow_id]
                    follower_v = LEADVehicle(
                        road=self.env.road,
                        position=v.position,
                        speed=v.speed,
                        target_speed=33,
                        timer=v.timer,
                        target_lane_index=v.target_lane_index,
                    )
                    bind_longitudinal_source(follower_v, v)

                    # 计算follower车横向动作：
                    [follow_action, follow_target_lane_index] = follower_v.change_lane_policy(
                        controlled_vehicle = follower_v, group = group, target_lane_index = ref_lane_index[follow_id]
                    )
                    target_actions[follow_id] = follow_action
                    target_lane_index[follow_id] = follow_target_lane_index

                    # 计算follower车纵向加速度：
                    follow_acceleration = v.integrated_longitudinal_control(first_v, follow_index, group)
                    v.action["acceleration"] = follow_acceleration
                    target_acceleration[follow_id] = follow_acceleration
                    follow_index += 1
        return target_actions, target_lane_index, target_acceleration

    def merge_decision_execution(
            self, index_group, ref_lane_index, ref_actions, target_actions, target_lane_index, target_acceleration
    ):

        if isinstance(index_group, int):
            vehicle = self.env.controlled_vehicles[index_group]
            group = None
            v = LEADVehicle(
                road=self.env.road,
                position=vehicle.position,
                speed=vehicle.speed,
                target_speed=33,
                timer=vehicle.timer,
                target_lane_index=vehicle.target_lane_index,
            )
            bind_longitudinal_source(v, vehicle)
            # 横向
            # [front_leader_action, front_leader_target_lane_index] = v.change_lane_policy(
            #     controlled_vehicle=vehicle, group=group, target_lane_index=ref_lane_index[index_group]
            # )
            front_leader_action = ref_actions[index_group]
            front_leader_target_lane_index = ref_lane_index[index_group]

            vehicle.timer = v.timer
            target_actions[index_group] = front_leader_action
            target_lane_index[index_group] = front_leader_target_lane_index

            front_leader_acceleration = v.integrated_longitudinal_control(
                controlled_vehicles=self.env.controlled_vehicles
            )
            target_acceleration[index_group] = front_leader_acceleration
            return target_actions, target_lane_index, target_acceleration

        merge_num = len(index_group)
        # TODO: Front leader planning
        group = []
        if isinstance(index_group[0], list):
            lead_idx = index_group[0][0]
            for i in index_group[0]:
                group.append(self.env.controlled_vehicles[i])
        else:
            lead_idx = index_group[0]
            group = None
        front_leader = self.env.controlled_vehicles[lead_idx]

        front_leader_v = LEADVehicle(
            road=self.env.road,
            position=front_leader.position,
            speed=front_leader.speed,
            target_speed=33,
            timer=front_leader.timer,
            target_lane_index=front_leader.target_lane_index,
        )
        bind_longitudinal_source(front_leader_v, front_leader)

        # TODO: 判断合并状态
        preceding_vs = []
        rear_leaders = []
        for v_list in index_group[1:]:
            if isinstance(v_list, list):
                rear_leader = self.env.controlled_vehicles[v_list[0]]
            else:
                rear_leader = self.env.controlled_vehicles[v_list]
            rear_leaders.append(rear_leader)
            preceding_v, _ = self.env.road.neighbour_vehicles(rear_leader, front_leader.lane_index)
            preceding_vs.append(preceding_v)

        front_speed_control_type = "no control"
        # If: front_follower在rear_leader_1之前n辆车或rear_leader_1在rear_leader_2之前n辆车,或间距 > 30m，front_leader减速
        if merge_num == 3:
            if (preceding_vs[0] is not None and front_leader.position[0] > preceding_vs[0].position[0]
                    or
                    front_leader.position[0] - rear_leaders[0].position[0] > 30
                    or
                    preceding_vs[1] is not None and rear_leaders[0].position[0] > preceding_vs[1].position[0]
                    or
                    rear_leaders[0].position[0] - rear_leaders[1].position[0] > 30):
                front_speed_control_type = "slightly slow down"
        elif merge_num == 2:
            if (preceding_vs[0] is not None and front_leader.position[0] > preceding_vs[0].position[0]
                    or
                    front_leader.position[0] - rear_leaders[0].position[0] > 30):
                front_speed_control_type = "slightly slow down"

        # TODO: Front leader planning
        # 横向
        [front_leader_action, front_leader_target_lane_index] = front_leader_v.change_lane_policy(
            controlled_vehicle=front_leader, group=group, target_lane_index=ref_lane_index[lead_idx]
        )
        # front_leader_action = ref_actions[lead_idx]
        # front_leader_target_lane_index = ref_lane_index[lead_idx]

        front_leader.timer = front_leader_v.timer
        target_actions[lead_idx] = front_leader_action
        target_lane_index[lead_idx] = front_leader_target_lane_index
        # 纵向
        front_leader_acceleration = front_leader_v.integrated_longitudinal_control(
            speed_control_type=front_speed_control_type, controlled_vehicles=self.env.controlled_vehicles
        )
        target_acceleration[lead_idx] = front_leader_acceleration

        # TODO: Front followers planning
        if isinstance(index_group[0], list):
            for i, follow_idx in zip(index_group[0][1:], range(1, len(index_group[0]))):
                vehicle = self.env.controlled_vehicles[i]
                # 横向
                [lateral_action, lane_index] = vehicle.change_lane_policy(
                    ref_lane_index=front_leader_target_lane_index, group=group
                )
                target_actions[i] = lateral_action
                target_lane_index[i] = lane_index
                # 纵向
                rear_leader_acceleration = vehicle.integrated_longitudinal_control(
                    lead_vehicle=front_leader, follow_index=follow_idx, group=group
                )
                target_acceleration[i] = rear_leader_acceleration

        # TODO: Rear leader planning
        for v_list in index_group[1:]:
            group = []
            if isinstance(v_list, list):
                rear_leader = self.env.controlled_vehicles[v_list[0]]
                lead_idx = v_list[0]
                for i in v_list:
                    group.append(self.env.controlled_vehicles[i])
            else:
                rear_leader = self.env.controlled_vehicles[v_list]
                group = None
                lead_idx = v_list

            rear_leader_v = LEADVehicle(
                road=self.env.road,
                position=rear_leader.position,
                speed=rear_leader.speed,
                target_speed=33,
                timer=rear_leader.timer,
                target_lane_index=rear_leader.target_lane_index,
            )
            bind_longitudinal_source(rear_leader_v, rear_leader)

            # TODO: 判断合并状态
            preceding_v, _ = self.env.road.neighbour_vehicles(rear_leader_v, front_leader_v.lane_index)
            rear_speed_control_type = "no control"
            target_merge_lane = front_leader_v.lane_index

            # Are they in near lanes?
            if abs(front_leader_v.lane_index[2] - rear_leader_v.lane_index[2]) == 1:
                # If: front_follower在rear_leader之前1辆车, mobile条件满足，直接合并
                if preceding_v is front_leader:
                    if not rear_leader.mobil(lane_index=target_merge_lane, forced=True, group=group):
                        new_preceding, _ = self.env.road.neighbour_vehicles(rear_leader, target_merge_lane, group)
                        if new_preceding is not None:
                            self_pred_a = rear_leader.acceleration(
                                ego_vehicle=rear_leader,
                                front_vehicle=new_preceding,
                                desired_gap=
                                rear_leader.desired_gap(ego_vehicle=rear_leader, front_vehicle=new_preceding)[1]
                            )
                            ttc_new_preceding = utils.ttc(new_preceding, rear_leader)
                            if self_pred_a < -rear_leader.ACC_MAX \
                                    or new_preceding.position[0] - rear_leader.position[
                                0] <= 1.5 * rear_leader.LENGTH \
                                    or ttc_new_preceding <= rear_leader.TTC_MIN:
                                rear_speed_control_type = "slightly slow down"
                # If: front_follower在rear_leader之后，或mobile条件不满足，减速
                elif front_leader.position[0] - rear_leader.position[0] <= 0:
                    rear_speed_control_type = "slightly slow down"

            # Are they in far lanes?
            elif abs(front_leader_v.lane_index[2] - rear_leader_v.lane_index[2]) > 1:
                # front_follower在rear_leader之后，减速，向merge_lane换道
                if front_leader.position[0] - rear_leader.position[0] <= 0:
                    rear_speed_control_type = "slightly slow down"

            # 横向：
            [rear_leader_action, rear_leader_target_lane_index] = rear_leader_v.change_lane_policy(
                controlled_vehicle=rear_leader, group=group, target_lane_index=ref_lane_index[lead_idx]
            )
            # rear_leader_action = ref_actions[lead_idx]
            # rear_leader_target_lane_index = ref_lane_index[lead_idx]

            target_actions[lead_idx] = rear_leader_action
            target_lane_index[lead_idx] = rear_leader_target_lane_index

            # 纵向：
            rear_leader_acceleration = rear_leader_v.integrated_longitudinal_control(
                ego_vehicle=rear_leader,
                speed_control_type=rear_speed_control_type,
                controlled_vehicles=self.env.controlled_vehicles
            )
            rear_leader.action["acceleration"] = rear_leader_acceleration
            target_acceleration[lead_idx] = rear_leader_acceleration

            # TODO: Rear followers planning
            if isinstance(v_list, list):
                for follow_idx, i in zip(range(1, len(v_list)), v_list[1:]):
                    rear_follower = self.env.controlled_vehicles[i]
                    # 横向
                    [rear_follower_action, rear_follower_target_lane_index] = rear_follower.change_lane_policy(
                        ref_lane_index=rear_leader_target_lane_index, group=group
                    )

                    target_actions[i] = rear_follower_action
                    target_lane_index[i] = rear_follower_target_lane_index
                    # 纵向
                    rear_leader_acceleration = rear_follower.integrated_longitudinal_control(
                        lead_vehicle=rear_leader, follow_index=follow_idx, group=group
                    )
                    target_acceleration[i] = rear_leader_acceleration

        return target_actions, target_lane_index, target_acceleration

    def split(self, action):
        """
        Make decision during splitting
        """

        """Update current GROUP"""
        self.group_update()
        index_group = self.GROUPS[action]

        weights = self.env.config["Split weights"]

        """If not decision frequency"""
        if self.ref_lane_index is not None and not utils.do_every(self.env.controlled_vehicles[0].LANE_CHANGE_DELAY,
                                                     self.env.controlled_vehicles[0].timer):
            ref_actions = self.ref_actions
            ref_lane_index = self.ref_lane_index
        else:
            ref_actions, ref_lane_index = self.coalition_game_decision(index_group, weights, platoon_state="split")
            self.ref_actions = ref_actions
            self.ref_lane_index = ref_lane_index
            self.env.controlled_vehicles[0].timer = 0  # Use car0's timer

        target_actions, target_lane_index, target_acceleration = self.decision_execution(
            index_group, ref_lane_index, ref_actions
        )

        return {
            "lateral action": target_actions,
            "lane index": target_lane_index,
            "acceleration": target_acceleration,
        }

    def keep(self, action):
        return self.split(action)

    def merge(self, env_state, action, split_and_merge):
        """
        Make decisions during merging
        """

        """Update Current GROUP"""
        self.group_update()
        ideal_index_group = self.GROUPS[action]
        origin_index_group = self.GROUPS[env_state]

        weights = self.env.config["Merge weights"]

        if split_and_merge:
            origin_index_group = self.GROUPS[2]

        """If not decision frequency or in space tuning state"""
        if self.ref_lane_index is not None and not utils.do_every(self.env.controlled_vehicles[0].LANE_CHANGE_DELAY,
                              self.env.controlled_vehicles[0].timer) or self.merge_state == "Space Tuning":
            ref_actions = self.ref_actions
            ref_lane_index = self.ref_lane_index
        else:
            ref_actions, ref_lane_index = self.coalition_game_decision(
                origin_index_group, weights, platoon_state="merge", ideal_index_group=ideal_index_group
            )
            self.ref_actions = ref_actions
            self.ref_lane_index = ref_lane_index
            self.env.controlled_vehicles[0].timer = 0

        vehicle_num = len(self.env.controlled_vehicles)
        target_actions = [None] * vehicle_num
        target_lane_index = [None] * vehicle_num
        target_acceleration = [None] * vehicle_num

        if len(ideal_index_group) == 1:
            target_actions, target_lane_index, target_acceleration = self.merge_decision_execution(
                origin_index_group, ref_lane_index, ref_actions, target_actions, target_lane_index, target_acceleration
            )
        else:
            for ideal_group in ideal_index_group:
                target_actions, target_lane_index, target_acceleration = self.merge_decision_execution(
                    ideal_group, ref_lane_index, ref_actions, target_actions, target_lane_index, target_acceleration
                )

        return {
            "lateral action": target_actions,
            "lane index": target_lane_index,
            "acceleration": target_acceleration,
        }

    def group_update(self):
        controlled_vehicles = self.env.controlled_vehicles
        position_x = []
        v_index = range(self.env.config["controlled_vehicles"])
        for v in self.env.controlled_vehicles:
            position_x.append(v.position[0])
        sorted_x, sorted_vehicles, sorted_v_index = zip(
            *sorted(zip(position_x, controlled_vehicles, v_index), reverse=True))

        # 计算当前index_group:
        self.GROUPS = {
            0: [sorted_v_index[0], [sorted_v_index[1], sorted_v_index[2]]],
            1: [[sorted_v_index[0], sorted_v_index[1]], sorted_v_index[2]],
            2: [sorted_v_index[0], sorted_v_index[1], sorted_v_index[2]],
            3: [[sorted_v_index[0], sorted_v_index[1], sorted_v_index[2]]]
        }
        return

    def game_optimization(self, cost_list, action_list, index_group):
        model = gp.Model("coalition_game")
        action_vars = model.addVars(range(len(cost_list)), vtype=GRB.BINARY, name="action")

        model.addConstr(
            gp.quicksum(action_vars) == 1, name="action_cons"
        )

        obj = gp.quicksum(action_vars[i] * cost for i, cost in zip(range(len(action_vars)), cost_list))
        model.setObjective(obj, GRB.MAXIMIZE)
        model.optimize()

        best_cost = -1e5
        if len(index_group) == 1:
            best_action = ["keep"]
        elif len(index_group) == 2:
            best_action = ["keep", "keep"]
        else:
            best_action = ["keep", "keep", "keep"]

        if model.status == GRB.OPTIMAL:
            action_values = [action_vars[i].X for i in range(len(action_vars))]
            best_action = action_list[action_values.index(1)]
            best_cost = cost_list[action_values.index(1)]
        else:
            model.computeIIS()
            model.write("model.ilp")
            model.write("model.mps")
        return best_action, best_cost

    # def lc_condition_check(self, ego_v, lane_index) -> bool:
    #     """
    #     Set three conditions to check lane-change safety
    #         Condition 1: safe distance
    #         Condition 2: new following car minimum acceleration
    #         Condition 3: ego car minimum acceleration
    #     """
    #     LANE_CHANGE_MAX_BRAKING_IMPOSED = 3.0
    #     ACC_MAX = 6.0
    #     TTC_MIN = 3.0
    #
    #     new_preceding, new_following = self.env.road.neighbour_vehicles(ego_v, lane_index)
    #
    #     # Is the distance unsafe?
    #     if new_preceding is not None:
    #         ttc_new_preceding = utils.ttc(new_preceding, ego_v)
    #         if (new_preceding.position[0] - ego_v.position[0] <= 1.5 * ego_v.LENGTH
    #                 or ttc_new_preceding <= TTC_MIN):
    #             return False
    #     if new_following is not None:
    #         ttc_new_following = utils.ttc(ego_v, new_following)
    #         if (ego_v.position[0] - new_following.position[0] <= 1.5 * ego_v.LENGTH
    #                 or ttc_new_following <= TTC_MIN):
    #             return False
    #
    #     # Is the maneuver unsafe for the new following vehicle?
    #     new_following_pred_a = ego_v.acceleration(
    #         ego_vehicle=new_following,
    #         front_vehicle=ego_v,
    #         desired_gap=ego_v.desired_gap(ego_vehicle=new_following, front_vehicle=ego_v)[1]
    #     )
    #     if new_following_pred_a < -LANE_CHANGE_MAX_BRAKING_IMPOSED:
    #         # if new_following_pred_a < -ACC_MAX:
    #         return False
    #
    #     # Is the maneuver unsafe for the ego vehicle?
    #     self_pred_a = ego_v.acceleration(
    #         ego_vehicle=ego_v,
    #         front_vehicle=new_preceding,
    #         desired_gap=ego_v.desired_gap(ego_vehicle=ego_v, front_vehicle=new_preceding)[1]
    #     )
    #     # if self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
    #     if self_pred_a < -ACC_MAX:
    #         return False
    #
    #     return True

    # def generate_connectivity(
    #         self, predict_vehicles, ideal_index_group, copy_background_vehicles, predict_background_vehicles
    # ) -> tuple:
    #     """
    #     Calculate connectivity of platoon
    #     """
    #     con = 0
    #     cons = []
    #     con_dis = 0
    #     con_y_err = 0
    #     if ideal_index_group is None:
    #         return con, con_dis, con_y_err
    #
    #     con_value = {
    #         "Connected": 1,
    #         "Weakly Connected": 0.4,
    #         "Disconnected": 0,
    #     }
    #     connected_list = []
    #     group = []
    #     x_list = []
    #     y_list = []
    #     for v in predict_vehicles:
    #         group.append(v)
    #
    #     for index_group in ideal_index_group:
    #         if isinstance(index_group, list):
    #             # con
    #             for i in index_group:
    #                 vehicle = predict_vehicles[i]
    #                 x_list.append(vehicle.position[0])
    #                 y_list.append(vehicle.position[1])
    #                 front_v, rear_v = self.env.road.predict_neighbour_vehicles(
    #                     vehicle=vehicle, lane_index=vehicle.lane_index,
    #                     predict_vehicles=predict_vehicles, background_vehicles=copy_background_vehicles,
    #                     pre_background_vehicles=predict_background_vehicles
    #                 )
    #
    #                 # Is front/rear vehicle connected?
    #                 if (([vehicle, front_v] not in connected_list) and ([front_v, vehicle] not in connected_list)
    #                         and front_v in group):
    #                     cons.append(con_value["Connected"])
    #                     connected_list.append([vehicle, front_v])
    #
    #                 if (([vehicle, rear_v] not in connected_list) and ([rear_v, vehicle] not in connected_list)
    #                         and rear_v in group):
    #                     cons.append(con_value["Connected"])
    #                     connected_list.append([vehicle, rear_v])
    #
    #                 # Is near front/rear vehicle connected?
    #                 for lane_index in self.env.road.network.side_lanes(vehicle.lane_index):
    #                     if self.env.road.network.get_lane(lane_index).is_reachable_from(vehicle.position):
    #                         front_v, rear_v = self.env.road.predict_neighbour_vehicles(
    #                             vehicle=vehicle, lane_index=lane_index,
    #                             predict_vehicles=predict_vehicles, background_vehicles=copy_background_vehicles,
    #                             pre_background_vehicles=predict_background_vehicles
    #                         )
    #                         if (([vehicle, front_v] not in connected_list) and (
    #                                 [front_v, vehicle] not in connected_list)
    #                                 and type(front_v).__name__ == "FOLLOWVehicle"):
    #                             cons.append(con_value["Weakly Connected"])
    #                             connected_list.append([vehicle, front_v])
    #                         if (([vehicle, rear_v] not in connected_list) and (
    #                                 [rear_v, vehicle] not in connected_list)
    #                                 and type(rear_v).__name__ == "FOLLOWVehicle"):
    #                             cons.append(con_value["Weakly Connected"])
    #                             connected_list.append([vehicle, rear_v])
    #             # con_dis, con_y_err
    #             ref_distance = 10
    #             max_distance = 50
    #             max_y_err = 8
    #             if len(index_group) == 2:
    #                 con_dis = max_distance / (abs(
    #                     predict_vehicles[index_group[0]].position[0] - predict_vehicles[index_group[1]].position[
    #                         0] - ref_distance) + 1)
    #                 con_y_err = max_y_err / (abs(
    #                     predict_vehicles[index_group[0]].position[1] - predict_vehicles[index_group[1]].position[
    #                         1]) + 1)
    #             if len(index_group) == 3:
    #                 distances = utils.diff(x_list)
    #                 y_errs = utils.diff(y_list)
    #                 y_errs = [abs(y_err) for y_err in y_errs]
    #                 con_dis = 1 - (np.mean(np.array(distances) - ref_distance) + 1) / max_distance
    #                 con_y_err = 1 - (np.mean(y_errs) + 1) / max_y_err
    #     if len(cons) > 0:
    #         con = sum(cons) / len(self.env.controlled_vehicles)
    #     return con, con_dis, con_y_err

    def generate_acc_reward(
            self, index, action, copy_controlled_vehicles, copy_background_vehicles,
            predict_background_vehicles, target_lane_index
    ) -> tuple:
        """
        Using prediction acceleration for accelerate cost
        """
        acc_min = -100
        acc_max = 100

        if isinstance(index, int):
            vehicle = self.env.controlled_vehicles[index]
        else:
            vehicle = self.env.controlled_vehicles[index[0]]
        """计算 acc cost"""
        leader_v = LEADVehicle(
            road=self.env.road,
            position=vehicle.position,
            speed=vehicle.speed,
            target_speed=33,
            timer=vehicle.timer,
            target_lane_index=vehicle.target_lane_index,
        )
        bind_longitudinal_source(leader_v, vehicle)
        front_v, rear_v = self.env.road.predict_neighbour_vehicles(
            vehicle=vehicle, lane_index=target_lane_index,
            predict_vehicles=copy_controlled_vehicles,
            background_vehicles=copy_background_vehicles,
            pre_background_vehicles=predict_background_vehicles
        )

        pre_acc = leader_v.longitudinal_control(ego_vehicle=leader_v, front_vehicle=front_v, is_limited=False)
        acc_reward = utils.normalization([acc_min, acc_max], pre_acc)
        # lat_reward = 1 - abs(self.LANE_CENTER_Y[target_lane_index[2]] - vehicle.position[1]) / self.LANE_WIDTH
        lat_reward = np.sign(vehicle.heading) * (
                    self.LANE_CENTER_Y[target_lane_index[2]] - vehicle.position[1]) / self.LANE_WIDTH

        return acc_reward, lat_reward

    def generate_safe_reward(
            self, index, target_lane_index, copy_controlled_vehicles, copy_background_vehicles,
            predict_background_vehicles
    ):
        """
        TTC
        """
        if isinstance(index, int):
            first_vehicle = copy_controlled_vehicles[index]
            end_vehicle = copy_controlled_vehicles[index]
            group = [copy_background_vehicles[index]]
        else:
            first_vehicle = copy_controlled_vehicles[index[0]]
            end_vehicle = copy_controlled_vehicles[index[-1]]
            group = [copy_background_vehicles[i] for i in index]

        front_v, _ = self.env.road.predict_neighbour_vehicles(
            vehicle=first_vehicle, lane_index=target_lane_index, group=group,
            predict_vehicles=copy_controlled_vehicles, background_vehicles=copy_background_vehicles,
            pre_background_vehicles=predict_background_vehicles
        )
        _, rear_v = self.env.road.predict_neighbour_vehicles(
            vehicle=end_vehicle, lane_index=target_lane_index, group=group,
            predict_vehicles=copy_controlled_vehicles, background_vehicles=copy_background_vehicles,
            pre_background_vehicles=predict_background_vehicles
        )

        max_ttc = 20
        if front_v is not None:
            front_ttc = utils.ttc(front_vehicle=front_v, rear_vehicle=first_vehicle)
        else:
            front_ttc = np.inf
        if rear_v is not None:
            rear_ttc = utils.ttc(front_vehicle=end_vehicle, rear_vehicle=rear_v)
        else:
            rear_ttc = np.inf

        safe_reward = np.clip(min(front_ttc, rear_ttc), 0, max_ttc) / max_ttc
        return safe_reward

    def action_predict(self, copy_controlled_vehicles, copy_background_vehicles, pre_background_vehicles, index,
                       action, pre_time) -> list:
        """
        Predict position of controlled vehicles
        """
        lane_center_y = self.LANE_CENTER_Y

        if isinstance(index, int):
            vehicle = copy_controlled_vehicles[index]
            # 车道预测
            target_lane_index = list(copy_controlled_vehicles[index].lane_index)
            if action == "left":
                if target_lane_index[2] >= 1:
                    target_lane_index[2] += -1
            elif action == "right":
                if target_lane_index[2] <= 1:
                    target_lane_index[2] += 1

            target_lane_index = tuple(target_lane_index)

            if self.env.road.network.get_lane(target_lane_index).is_reachable_from(
                    vehicle.position
            ):
                # position_y 预测
                vehicle.lane_index = target_lane_index
                vehicle.position[1] = lane_center_y[target_lane_index[2]]

                # position_x 预测
                front_v, rear_v = self.env.road.predict_neighbour_vehicles(
                    vehicle=vehicle, lane_index=target_lane_index,
                    predict_vehicles=copy_controlled_vehicles, background_vehicles=copy_background_vehicles,
                    pre_background_vehicles=pre_background_vehicles
                )
                lead_vehicle = LEADVehicle(
                    road=self.env.road,
                    position=vehicle.position,
                    speed=vehicle.speed,
                    target_speed=33,
                    timer=vehicle.timer,
                    target_lane_index=vehicle.target_lane_index,
                )
                bind_longitudinal_source(lead_vehicle, vehicle)
                pre_a = lead_vehicle.integrated_longitudinal_control(
                    ego_vehicle=vehicle, front_vehicle=front_v, controlled_vehicles=copy_controlled_vehicles
                )
                pre_x = vehicle.position[0] + vehicle.speed * pre_time + 0.5 + pre_a * pre_time ** 2
                if front_v is not None and pre_x > front_v.position[0]:
                    pre_x = front_v.position[0] - 10

                # Is the rear_vehicle very fast?
                if action != "keep" and rear_v is not None and pre_x - rear_v.position[0] < 5:
                    pre_a = lead_vehicle.integrated_longitudinal_control(
                        ego_vehicle=vehicle, front_vehicle=rear_v, controlled_vehicles=copy_controlled_vehicles
                    )
                    pre_x = vehicle.position[0] + vehicle.speed * pre_time + 0.5 + pre_a * pre_time ** 2
                    if pre_x > rear_v.position[0]:
                        pre_x = rear_v.position[0] - 10

                vehicle.position[0] = pre_x

        else:
            first_vehicle = copy_controlled_vehicles[index[0]]
            # 车道预测
            target_lane_index = list(first_vehicle.lane_index)
            if action == "left":
                if target_lane_index[2] >= 1:
                    target_lane_index[2] += -1
            elif action == "right":
                if target_lane_index[2] <= 1:
                    target_lane_index[2] += 1

            target_lane_index = tuple(target_lane_index)

            front_v, rear_v = self.env.road.predict_neighbour_vehicles(
                vehicle=first_vehicle, lane_index=target_lane_index,
                predict_vehicles=copy_controlled_vehicles, background_vehicles=copy_background_vehicles,
                pre_background_vehicles=pre_background_vehicles
            )

            pre_x_list = []

            # lead_v position_x 预测
            lead_vehicle = LEADVehicle(
                road=self.env.road,
                position=first_vehicle.position,
                speed=first_vehicle.speed,
                target_speed=33,
                timer=first_vehicle.timer,
                target_lane_index=first_vehicle.target_lane_index,
            )
            bind_longitudinal_source(lead_vehicle, first_vehicle)

            pre_a = lead_vehicle.integrated_longitudinal_control(
                ego_vehicle=first_vehicle, front_vehicle=front_v, controlled_vehicles=copy_controlled_vehicles
            )
            lead_pre_x = first_vehicle.position[0] + first_vehicle.speed * pre_time + 0.5 + pre_a * pre_time ** 2
            if front_v is not None and lead_pre_x > front_v.position[0]:
                lead_pre_x = front_v.position[0] - 10
            pre_x_list.append(lead_pre_x)

            # follow_v
            for i in range(1, len(index)):
                pre_x = lead_pre_x - 10 * i
                pre_x_list.append(pre_x)

            # Safety check
            if action != "keep" and rear_v is not None and any(x - rear_v.position[0] < 5 for x in pre_x_list):
                front_v = rear_v
                for i in range(len(pre_x_list)):
                    pre_x_list[i] = front_v.position[0] - 10 * (i + 1)

            # position_y 预测
            for idx, i in zip(index, range(len(index))):
                vehicle = copy_controlled_vehicles[idx]
                if self.env.road.network.get_lane(target_lane_index).is_reachable_from(
                        vehicle.position
                ):
                    vehicle.lane_index = target_lane_index
                    vehicle.position[1] = lane_center_y[target_lane_index[2]]
                    vehicle.position[0] = pre_x_list[i]

        return copy_controlled_vehicles

    def new_generate_rewards(self, action_list, index_group, weights):
        # Prediction of background vehicles
        pre_time = 0.1
        copy_all_vehicles = copy.deepcopy(self.env.road.vehicles)
        copy_background_vehicles, predict_background_vehicles = self.background_vehicles_predict(
            copy_all_vehicles, pre_time
        )

        reward_list = []
        new_action_list = []
        acc_rewards_list = []
        for actions in action_list:
            reward, acc_rewards = self._reward(
                actions, index_group, copy_background_vehicles, predict_background_vehicles, pre_time, weights
            )
            if reward is None:
                continue
            reward_list.append(reward)
            new_action_list.append(actions)
            acc_rewards_list.append(acc_rewards)

        return reward_list, new_action_list

    def _reward(
            self, actions, index_group, copy_background_vehicles, predict_background_vehicles, pre_time, weights
    ):
        copy_controlled_vehicles = copy.deepcopy(self.env.controlled_vehicles)
        acc_rewards = []

        """acc_reward & lat_reward & safe_reward"""
        if len(index_group) == 1:
            action = actions
            index = index_group[0]
            """If action is infeasible, cut."""
            infeasible, target_lane_index = self.action_cut(index, action)
            if infeasible:
                return None, None

            acceleration_reward, lateral_reward = self.generate_acc_reward(
                index, action, copy_controlled_vehicles, copy_background_vehicles,
                predict_background_vehicles, target_lane_index
            )
            acc_rewards.append(acceleration_reward)

            safe_reward = self.generate_safe_reward(
                index, target_lane_index, copy_controlled_vehicles, copy_background_vehicles,
                predict_background_vehicles
            )

            """Update action prediction"""
            copy_controlled_vehicles = self.action_predict(
                copy_controlled_vehicles, copy_background_vehicles, predict_background_vehicles,
                index, action, pre_time
            )
        else:
            acceleration_reward, lateral_reward, safe_reward = 0, 0, 0
            for action, index in zip(actions, index_group):
                """If action is infeasible, cut."""
                infeasible, target_lane_index = self.action_cut(index, action)
                if infeasible:
                    return None, None

                acc_reward, lat_reward = self.generate_acc_reward(
                    index, action, copy_controlled_vehicles, copy_background_vehicles,
                    predict_background_vehicles, target_lane_index
                )
                acceleration_reward += acc_reward
                lateral_reward += lat_reward
                acc_rewards.append(acc_reward)

                s_reward = self.generate_safe_reward(
                    index, target_lane_index, copy_controlled_vehicles, copy_background_vehicles,
                    predict_background_vehicles
                )
                safe_reward += s_reward

                """Update action prediction"""
                copy_controlled_vehicles = self.action_predict(
                    copy_controlled_vehicles, copy_background_vehicles, predict_background_vehicles,
                    index, action, pre_time
                )

        """connect_reward & x_reward & y_reward"""
        connect_reward = self.connect_nodes(
            copy_controlled_vehicles, self.env.controlled_vehicles
        )

        max_distance = 50
        max_y_err = 8
        pre_x = [v.position[0] for v in copy_controlled_vehicles]
        pre_y = [v.position[1] for v in copy_controlled_vehicles]
        x_error = sum(np.array(utils.diff(sorted(pre_x, reverse=True))) - 10) / (len(pre_x) - 1)
        x_reward = 1 - x_error / max_distance
        y_error = sum(utils.diff(sorted(pre_y, reverse=True))) / (len(pre_y) - 1)
        y_reward = 1 - y_error / max_y_err

        reward = sum([w * r for w, r in zip(
            weights, [acceleration_reward, connect_reward, x_reward, y_reward, safe_reward, lateral_reward]
        )])
        return reward, acc_rewards

class RULE_MAKER(COALITION_GAME_MAKER):
    """
    Decision made by rule
    """
    def __init__(self, env):
        self.env = env
        self.ref_actions = None
        self.ref_lane_index = None
        self.action_list = None
        self.merge_state = "Done"
        self.groups = {
            0: [0, [1, 2]],
            1: [[0, 1], 2],
            2: [0, 1, 2],
            3: [[0, 1, 2]],
        }

    def _scenario_target_lane_index(self, vehicle):
        """Read an optional scenario lane goal without coupling Rule to scenarios.

        Legacy environments do not provide ``scenario_target_lane_index`` and
        therefore keep the original MOBIL autonomous lane-selection behavior.
        """
        resolver = getattr(self.env, "scenario_target_lane_index", None)
        if not callable(resolver):
            return None
        return resolver(vehicle)

    def group_action_to_vehicle_action(self, action) -> dict:

        """
        根据分组动作计算单车横向动作，通过mobile模型计算横向动作，通过lqr模型计算加速度
        param: action: 分组动作
        return: (target_lateral_action, target_acceleration): 单车动作列表
        """
        # TODO: Is current action split or merge?
        env_state = self.env.controlled_vehicles[0].env_state
        if self.SPLIT_A_MATRIX[env_state, action]:
            # Update GROUPS
            # self.update_group()

            context = getattr(self.env.road, "longitudinal_control", None)
            if context is not None and context.kind == "lmpc":
                context.set_groups(self.groups[action])
            higher_action = self.split(action)
            # update env_state
            for v in self.env.controlled_vehicles:
                v.env_state = action

        elif self.MERGE_A_MATRIX[env_state, action]:
            # Update GROUPS
            self.update_group()

            context = getattr(self.env.road, "longitudinal_control", None)
            if context is not None and context.kind == "lmpc":
                context.set_groups(self.groups[action])
            higher_action = self.merge(action)
            merge_success, env_state = self.merge_success_update(merge_group=self.groups[action])
            if merge_success:
                # update env_state
                for v in self.env.controlled_vehicles:
                    v.env_state = action
            # if self.merge_success_update(merge_group=self.groups[action]):
            #     # update env_state
            #     for v in self.env.controlled_vehicles:
            #         v.env_state = action

        else:
            # Update GROUPS
            # self.update_group()

            context = getattr(self.env.road, "longitudinal_control", None)
            if context is not None and context.kind == "lmpc":
                context.set_groups(self.groups[action])
            higher_action = self.keep(action)
            # update env_state
            for v in self.env.controlled_vehicles:
                v.env_state = action

        # update group_action
        for v in self.env.controlled_vehicles:
            v.group_action = action

        return higher_action

    def update_group(self):
        controlled_vehicles = self.env.controlled_vehicles
        position_x = []
        v_index = range(self.env.config["controlled_vehicles"])
        for v in self.env.controlled_vehicles:
            position_x.append(v.position[0])
        sorted_x, sorted_vehicles, sorted_v_index = zip(
            *sorted(zip(position_x, controlled_vehicles, v_index), reverse=True))
        # 计算index_group:
        self.groups = {
            0: [sorted_v_index[0], [sorted_v_index[1], sorted_v_index[2]]],
            1: [[sorted_v_index[0], sorted_v_index[1]], sorted_v_index[2]],
            2: [sorted_v_index[0], sorted_v_index[1], sorted_v_index[2]],
            3: [[sorted_v_index[0], sorted_v_index[1], sorted_v_index[2]]]
        }
        return

    def merge_success_update(self, merge_group, env_state: int = None) -> tuple:
        controlled_vehicles = self.env.controlled_vehicles
        x_positions = []
        y_positions = []
        lane_indexes = []
        all_merge = True

        # Do all vehicles successfully merge?
        for group in merge_group:
            if isinstance(group, list):
                x_positions = []
                lane_indexes = []
                for idx in group:
                    x_positions.append(controlled_vehicles[idx].position[0])
                    lane_indexes.append(controlled_vehicles[idx].lane_index)
            else:
                continue

            # If group's vehicles are in the same lane?
            if not all(x == lane_indexes[0] for x in lane_indexes):
                all_merge = False

            # If there is no other vehicles inside group?
            for other_v in self.env.road.vehicles:
                if other_v not in controlled_vehicles:
                    # If other_v position_x inside platoon and other_v lane_index equal to platoon
                    if utils.inside_list(other_v.position[0], x_positions, reverse=True) and other_v.lane_index is lane_indexes[0]:
                        all_merge = False

            # If the distance between vehicles are close enough?
            distances = utils.diff(x_positions)
            ref_distance = 10
            if max(abs(np.array(distances) - ref_distance)) > 10:
                all_merge = False
        if all_merge:
            # All is done!
            return True, 3

        # Do partial vehicles successfully merge?
        if env_state is not None:
            if len(merge_group) == 1:
                for idx in merge_group[0]:
                    x_positions.append(controlled_vehicles[idx].position[0])
                    y_positions.append(controlled_vehicles[idx].position[1])
                    lane_indexes.append(controlled_vehicles[idx].lane_index)
                sorted_x_positions, sorted_y_positions, sorted_lane_index, sorted_index = zip(
                    *sorted(zip(x_positions, y_positions, lane_indexes, list(range(len(x_positions)))),
                            key=lambda x: x[0], reverse=True)
                )
                # prob : [[0, 1], 2], [0, [1, 2]]
                for index_group, i in zip([[0, 1], [1, 2]], [0, 1]):
                    # same lane?
                    if (sorted_lane_index[index_group[0]] == sorted_lane_index[index_group[1]]
                            and abs(sorted_y_positions[index_group[0]] - sorted_y_positions[
                                index_group[1]] <= 0.5)):
                        for other_v in self.env.road.vehicles:
                            if other_v not in controlled_vehicles:
                                # If other_v position_x inside platoon and other_v lane_index equal to platoon
                                x_list = [sorted_x_positions[index_group[0]],
                                          sorted_x_positions[index_group[1]]]
                                if utils.inside_list(other_v.position[0], x_list,
                                                     reverse=True) and other_v.lane_index == \
                                        sorted_lane_index[index_group[0]]:
                                    continue
                                # close enough?
                                distance = sorted_x_positions[index_group[0]] - sorted_x_positions[
                                    index_group[1]]
                                ref_distance = 10
                                if abs(distance - ref_distance) <= 10:
                                    if i == 0:
                                        return True, 1
                                    else:
                                        return True, 0
        return False, env_state

    def lc_condition_check(self, ego_v, lane_index) -> bool:
        """Check lane-change safety on the target lane's Frenet s axis."""
        # === FRENET RULE SAFETY V1 ===
        LANE_CHANGE_MAX_BRAKING_IMPOSED = 3.0
        ACC_MAX = 6.0
        TTC_MIN = 3.0

        new_preceding, new_following = (
            self.env.road.neighbour_vehicles(
                ego_v,
                lane_index,
            )
        )

        if new_preceding is not None:
            front_gap = self.env.road.longitudinal_gap(
                front_vehicle=new_preceding,
                rear_vehicle=ego_v,
                lane_index=lane_index,
            )
            front_ttc = self.env.road.longitudinal_ttc(
                front_vehicle=new_preceding,
                rear_vehicle=ego_v,
                lane_index=lane_index,
            )
            if (
                front_gap <= 1.5 * ego_v.LENGTH
                or front_ttc <= TTC_MIN
            ):
                return False

        if new_following is not None:
            rear_gap = self.env.road.longitudinal_gap(
                front_vehicle=ego_v,
                rear_vehicle=new_following,
                lane_index=lane_index,
            )
            rear_ttc = self.env.road.longitudinal_ttc(
                front_vehicle=ego_v,
                rear_vehicle=new_following,
                lane_index=lane_index,
            )
            if (
                rear_gap <= 1.5 * ego_v.LENGTH
                or rear_ttc <= TTC_MIN
            ):
                return False

        new_following_pred_a = ego_v.acceleration(
            ego_vehicle=new_following,
            front_vehicle=ego_v,
            desired_gap=ego_v.desired_gap(
                ego_vehicle=new_following,
                front_vehicle=ego_v,
            )[1],
        )
        if (
            new_following_pred_a
            < -LANE_CHANGE_MAX_BRAKING_IMPOSED
        ):
            return False

        self_pred_a = ego_v.acceleration(
            ego_vehicle=ego_v,
            front_vehicle=new_preceding,
            desired_gap=ego_v.desired_gap(
                ego_vehicle=ego_v,
                front_vehicle=new_preceding,
            )[1],
        )
        if self_pred_a < -ACC_MAX:
            return False

        return True
    def lane_change_safety_check(self, index, lane_index) -> bool:
        """
        Check if lane change is safe
        """
        if isinstance(index, int):
            ego_v = self.env.controlled_vehicles[index]
            if not self.lc_condition_check(ego_v, lane_index):
                return False
        else:
            for idx in index:
                ego_v = self.env.controlled_vehicles[idx]
                if not self.lc_condition_check(ego_v, lane_index):
                    return False
        return True

    def keep(self, action) -> dict:
        return self.split(action)

    def split(self, action) -> dict:

        controlled_num = self.env.config["controlled_vehicles"]
        target_actions = [None] * controlled_num
        target_lane_index = [None] * controlled_num
        target_acceleration = [None] * controlled_num

        index_group = self.groups[action]

        for v_list in index_group:
            # TODO: Leader planning
            if isinstance(v_list, int):
                front_id = v_list
                group = None
            else:
                front_id = v_list[0]
                group = []
                for i in v_list:
                    group.append(self.env.controlled_vehicles[i])

            first_v = self.env.controlled_vehicles[front_id]
            leader_v = LEADVehicle(
                road=self.env.road,
                position=first_v.position,
                speed=first_v.speed,
                target_speed=33,
                timer=first_v.timer,
                target_lane_index=first_v.target_lane_index,
            )
            bind_longitudinal_source(leader_v, first_v)

            # SCENARIO TARGET LANE V1: scenario decides where; MOBIL decides when.
            scenario_target_lane_index = self._scenario_target_lane_index(first_v)
            [lead_action, lead_target_lane_index] = leader_v.change_lane_policy(
                controlled_vehicle=first_v,
                group=group,
                target_lane_index=scenario_target_lane_index,
            )
            first_v.timer = leader_v.timer
            target_actions[front_id] = lead_action
            target_lane_index[front_id] = lead_target_lane_index

            # LQR计算leader车纵向加速度：
            lead_acceleration = leader_v.integrated_longitudinal_control()
            first_v.action["acceleration"] = lead_acceleration  # update acceleration to let the followers know
            target_acceleration[front_id] = lead_acceleration

            # TODO: Followers planning
            if isinstance(v_list, list):
                follow_list = v_list[1:]
                follow_index = 1
                for follow_id in follow_list:
                    v = self.env.controlled_vehicles[follow_id]
                    follower_v = FOLLOWVehicle(
                        road=self.env.road,
                        position=v.position,
                        speed=v.speed,
                        target_speed=33,
                        timer=v.timer,
                        target_lane_index=v.target_lane_index,
                    )
                    bind_longitudinal_source(follower_v, v)

                    # 计算follower车横向动作：
                    [follow_action, follow_target_lane_index] = follower_v.change_lane_policy(
                        lead_target_lane_index, group
                    )
                    target_actions[follow_id] = follow_action
                    target_lane_index[follow_id] = follow_target_lane_index
                    # 计算follower车纵向加速度：
                    follow_acceleration = follower_v.integrated_longitudinal_control(first_v, follow_index, group)
                    v.action["acceleration"] = follow_acceleration
                    target_acceleration[follow_id] = follow_acceleration
                    follow_index += 1

        return {
            "lateral action": target_actions,
            "lane index": target_lane_index,
            "acceleration": target_acceleration,
        }

    def merge(self, action) -> dict:
        """
        :param action
        """
        index_group = self.groups[action]
        controlled_num = self.env.config["controlled_vehicles"]
        target_actions = [None] * controlled_num
        target_lane_index = [None] * controlled_num
        target_acceleration = [None] * controlled_num

        if len(index_group) == 1:
            self.merge_planning(index_group[0], len(index_group[0]), target_actions, target_lane_index, target_acceleration)
        else:
            for index in index_group:
                if isinstance(index, list):
                    target_actions, target_lane_index, target_acceleration = self.merge_planning(
                        index, len(index), target_actions, target_lane_index, target_acceleration
                    )
                else:
                    target_actions, target_lane_index, target_acceleration = self.merge_planning(
                        index, 1, target_actions, target_lane_index, target_acceleration
                    )

        return {
            "lateral action": target_actions,
            "lane index": target_lane_index,
            "acceleration": target_acceleration,
        }

    def merge_planning(self, index_group, merge_num, target_actions, target_lane_index, target_acceleration):
        # TODO: 规划第一辆车(车道保持，自由行驶或缓慢减速)
        # Leader:
        group = None
        if isinstance(index_group, list):
            front_vehicle = self.env.controlled_vehicles[index_group[0]]
            front_vehicle_idx = index_group[0]
        else:
            front_vehicle = self.env.controlled_vehicles[index_group]
            front_vehicle_idx = index_group

        front_leader_v = LEADVehicle(
            road=self.env.road,
            position=front_vehicle.position,
            speed=front_vehicle.speed,
            target_speed=33,
            timer=front_vehicle.timer,
            target_lane_index=front_vehicle.target_lane_index,
        )
        bind_longitudinal_source(front_leader_v, front_vehicle)

        if merge_num == 1:
            # SCENARIO TARGET LANE V1: keep scenario intent during regrouping.
            scenario_target_lane_index = self._scenario_target_lane_index(front_vehicle)
            [front_leader_action, front_leader_target_lane_index] = front_leader_v.change_lane_policy(
                controlled_vehicle=front_vehicle,
                group=group,
                target_lane_index=scenario_target_lane_index,
            )
            front_vehicle.timer = front_leader_v.timer
            target_actions[front_vehicle_idx] = front_leader_action
            target_lane_index[front_vehicle_idx] = front_leader_target_lane_index

            front_leader_acceleration = front_leader_v.integrated_longitudinal_control(
                controlled_vehicles=self.env.controlled_vehicles
            )
            target_acceleration[front_vehicle_idx] = front_leader_acceleration
            return target_actions, target_lane_index, target_acceleration

        # TODO: 判断合并状态
        preceding_vs = []
        rear_leaders = []
        for v_list in index_group[1:]:
            rear_leader = self.env.controlled_vehicles[v_list]
            rear_leaders.append(rear_leader)
            preceding_v, _ = self.env.road.neighbour_vehicles(rear_leader, front_vehicle.lane_index)
            preceding_vs.append(preceding_v)

        front_speed_control_type = "no control"
        # If: front_follower在rear_leader_1之前n辆车或rear_leader_1在rear_leader_2之前n辆车,或间距 > 30m，front_leader减速
        if merge_num == 3:
            if (preceding_vs[0] is not None and front_vehicle.position[0] > preceding_vs[0].position[0]
                    or
                    front_vehicle.position[0] - rear_leaders[0].position[0] > 30
                    or
                    preceding_vs[1] is not None and rear_leaders[0].position[0] > preceding_vs[1].position[0]
                    or
                    rear_leaders[0].position[0] - rear_leaders[1].position[0] > 30):
                front_speed_control_type = "slightly slow down"
        elif merge_num == 2:
            if (preceding_vs[0] is not None and front_vehicle.position[0] > preceding_vs[0].position[0]
                    or
                    front_vehicle.position[0] - rear_leaders[0].position[0] > 30):
                front_speed_control_type = "slightly slow down"

        # TODO: Leader planning
        # SCENARIO TARGET LANE V1: front leader carries the scenario lane goal.
        scenario_target_lane_index = self._scenario_target_lane_index(front_vehicle)
        [front_leader_action, front_leader_target_lane_index] = front_leader_v.change_lane_policy(
            controlled_vehicle=front_vehicle,
            group=group,
            target_lane_index=scenario_target_lane_index,
        )
        front_vehicle.timer = front_leader_v.timer
        target_actions[front_vehicle_idx] = front_leader_action
        target_lane_index[front_vehicle_idx] = front_leader_target_lane_index

        front_leader_acceleration = front_leader_v.integrated_longitudinal_control(
            speed_control_type=front_speed_control_type, controlled_vehicles=self.env.controlled_vehicles
        )
        target_acceleration[front_vehicle_idx] = front_leader_acceleration

        # TODO: 规划后续车辆
        # How should follower groups act?
        for follow_index in index_group[1:]:
            group = None
            rear_leader = self.env.controlled_vehicles[follow_index]
            rear_leader_idx = follow_index
            rear_leader_v = LEADVehicle(
                road=self.env.road,
                position=rear_leader.position,
                speed=rear_leader.speed,
                target_speed=33,
                timer=rear_leader.timer,
                target_lane_index=rear_leader.target_lane_index,
            )
            bind_longitudinal_source(rear_leader_v, rear_leader)

            # TODO: 判断合并状态
            preceding_v, _ = self.env.road.neighbour_vehicles(rear_leader_v, front_leader_v.lane_index)
            rear_speed_control_type = "no control"
            target_merge_lane = front_leader_v.lane_index

            # Are they in near lanes?
            ideal_lane_index = rear_leader.lane_index
            if abs(front_leader_v.lane_index[2] - rear_leader_v.lane_index[2]) == 1:
                # If: front_follower在rear_leader之前1辆车, mobile条件满足，直接合并
                if preceding_v is front_vehicle:
                    ideal_lane_index = target_merge_lane
                    rear_leader.waite_timer = 0

                    if not rear_leader.mobil(lane_index=target_merge_lane, forced=True, group=group):
                        new_preceding, _ = self.env.road.neighbour_vehicles(rear_leader, target_merge_lane, group)
                        if new_preceding is not None:
                            self_pred_a = rear_leader.acceleration(
                                ego_vehicle=rear_leader,
                                front_vehicle=new_preceding,
                                desired_gap=rear_leader.desired_gap(ego_vehicle=rear_leader, front_vehicle=new_preceding)[1]
                            )
                            ttc_new_preceding = utils.ttc(new_preceding, rear_leader)
                            if self_pred_a < -rear_leader.ACC_MAX \
                                or new_preceding.position[0] - rear_leader.position[0] <= 1.5 * rear_leader.LENGTH \
                                or ttc_new_preceding <= rear_leader.TTC_MIN:
                                rear_speed_control_type = "slightly slow down"
                                rear_leader.waite_timer += 0.1
                # If: front_follower在rear_leader之后，或mobile条件不满足，减速
                elif front_vehicle.position[0] - rear_leader.position[0] <= 0:
                    ideal_lane_index = rear_leader.lane_index
                    rear_speed_control_type = "slightly slow down"
                    rear_leader.waite_timer += 0.1
                else:
                    ideal_lane_index = None
                    rear_leader.waite_timer += 0.1
            # Are they in same lane?
            elif abs(front_leader_v.lane_index[2] - rear_leader_v.lane_index[2]) == 0:
                # If: front_follower在rear_leader之前1辆车
                if preceding_v is front_vehicle:
                    ideal_lane_index = target_merge_lane
                    rear_leader.waite_timer = 0
                elif front_vehicle.position[0] > preceding_v.position[0]:
                    ideal_lane_index = None
                    rear_leader.waite_timer += 0.1
                else:
                    # TODO：choose one lane to change lanes
                    rear_leader.waite_timer += 0.1
                    for lane_index in self.env.road.network.side_lanes(rear_leader.lane_index):
                        # Is the candidate lane close enough?
                        if not self.env.road.network.get_lane(lane_index).is_reachable_from(rear_leader.position):
                            continue
                        # Only change lane when the vehicle is moving
                        if np.abs(rear_leader.speed) < 1:
                            continue
                        # Does the MOBIL model recommend a lane change?
                        if rear_leader_v.mobil(lane_index, group, forced=True):
                            ideal_lane_index = lane_index

            # else, in far lanes, try to move to the near lane
            else:
                if preceding_v is front_vehicle:
                    # 向merge_lane换道
                    ideal_lane_index = self.env.road.network.side_lanes(rear_leader.lane_index)[0]
                # front_follower在rear_leader之后，减速，向merge_lane换道
                elif front_vehicle.position[0] - rear_leader.position[0] <= 0:
                    rear_speed_control_type = "slightly slow down"
                    # 向merge_lane换道
                    ideal_lane_index = self.env.road.network.side_lanes(rear_leader.lane_index)[0]
                else:
                    ideal_lane_index = None
                rear_leader.waite_timer += 0.1

            # TODO: Leader planning
            # 横向规划
            if ideal_lane_index is None:
                [rear_leader_action, rear_leader_target_lane_index] = rear_leader_v.change_lane_policy(
                    controlled_vehicle=rear_leader, group=group
                )
            else:
                [rear_leader_action, rear_leader_target_lane_index] = rear_leader.change_lane_policy(
                    ref_lane_index=ideal_lane_index, group=group
                )
            rear_leader.timer = rear_leader_v.timer
            target_actions[rear_leader_idx] = rear_leader_action
            target_lane_index[rear_leader_idx] = rear_leader_target_lane_index
            # 纵向规划
            rear_leader_v.target_lane_index = rear_leader_target_lane_index
            rear_leader_acceleration = rear_leader_v.integrated_longitudinal_control(
                vehicle_group=self.env.controlled_vehicles,
                speed_control_type=rear_speed_control_type,
                controlled_vehicles=self.env.controlled_vehicles
            )
            target_acceleration[rear_leader_idx] = rear_leader_acceleration

            # TODO： front_leader更新
            front_leader_v = rear_leader_v
            front_vehicle = self.env.controlled_vehicles[follow_index]

        return target_actions, target_lane_index, target_acceleration

