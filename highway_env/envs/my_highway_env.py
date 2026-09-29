from typing import Dict, Text, Optional

import numpy as np

import highway_env.vehicle.behavior
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.kinematics import Vehicle
from itertools import groupby, count

Observation = np.ndarray


class MyHighwayEnv(AbstractEnv):
    """
    An aggressive highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
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

                # "Planner": {
                #     "state": True,
                #     "type": "Polynomial",  # Polynomial / Diffusion
                # },


                "Planner": {
                    "state": False,

                    # ========================================================
                    # Stage 1:
                    # Polynomial 仍负责真实车辆控制
                    # Diffusion 只进行 shadow inference
                    # ========================================================
                    "type": "Polynomial",


                    # Polynomial benchmark / expert shared mode.
                    "Polynomial": {
                        "mode": "aligned",
                        "aligned_horizon_s": 4.0,
                    },

                    # One-frame expert / Dynamic-Anchor alignment.
                    "ExpertAlignment": {
                        "enabled": True,
                        "require_aligned_polynomial": True,
                    },

                    # ========================================================
                    # AllMerge structured planner features
                    # ========================================================
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

                        # ====================================================
                        # Dynamic Anchor
                        # ====================================================
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

                            # 正常网络运行时关闭 collision debug
                            "diagnostics_enabled": False,
                        },
                    },

                    # ========================================================
                    # Structured Diffusion Planner
                    # ========================================================
                    "Diffusion": {

                        # Stage 1 核心开关
                        "shadow_enabled": True,

                        # STAGE_A_SPLINE_RUNTIME_V1
                        # Selected [8,2] -> clamped cubic [41,2] @ 10 Hz.
                        "diffusion_spline_enabled": True,

                        # Keep OFF by default: Stage A must not silently
                        # replace the existing Polynomial controller path.
                        "diffusion_execution_enabled": False,
                        "spline_dense_dt": 0.1,
                        "spline_tracking_index": 10,

                        # GPU
                        "device": "cuda:0",

                        # 当前只做接口测试，没有训练好的 checkpoint
                        "allow_random_weights": True,

                        "checkpoint": None,

                        "strict_checkpoint": False,

                        # 保证 debug 时噪声可复现
                        "deterministic_seed": 0,

                        # ====================================================
                        # Network
                        # ====================================================
                        "model": {
                            "d_model": 128,
                            "d_ffn": 512,
                            "num_heads": 4,

                            "num_scene_layers": 1,
                            "num_denoiser_layers": 2,

                            "num_modes": 10,

                            "horizon_steps": 8,
                            "trajectory_dt": 0.5,

                            # truncated diffusion
                            "inference_start_timestep": 8,

                            "inference_timesteps": (
                                8,
                                0,
                            ),
                        },
                    },
                },




                "Decision maker": "Rule",  # Game or Rule

                "Controller": {
                    "type": "lqr", # lqr / lmpc (longitudinal only)
                    "vehicle_model": "kinematics",  # kinematics / trucksim
                },

                "TruckSim": {
                    "simfiles": [
                        f"D:/Users/Public/Documents/TruckSim2016.1_Data_now_using/truck_{i}_lon_lat.sim"
                        for i in range(1, 4)
                    ],
                    "dlls": [
                        f"D:/Users/Public/Documents/TruckSim2016.1_Data_now_using/Extensions/Multi_vehicle_lon_lat/s_s_{i}.dll"
                        for i in range(1, 4)
                    ],
                    "steering_ratio": 25.0,  # road wheel rad -> steering wheel deg
                    "max_rpm": 3000,
                    "engine_map_path": None,  # bundled reference 4455kg map
                },

                "forward_speed_reward": -600,  # The vehicle was punished when speed < 19
                "on_road_reward": 1,
                "distance_reward": -300,
                "same_lane_reward": 150,  # The reward received when vehicles in the same lane
                'far_reward': 600,
                "collision_reward": -800,  # The reward received when colliding with a vehicle.
                "action_infeasible_reward": 0,  # The reward received when got infeasible action.
                "split_reward": -1200,  # -900 The vehicle was punished when splitting for long time
                "merge_reward": -30,  # -30 The vehicle was punished when merging slowly
                "high_speed_reward": 300,  # The reward received when driving at full speed, linearly mapped to zero
                "game_decision_reward": 0,  # reward of game process

                "Split weights": [x / sum_split_weights for x in config["w_game split weights"]],  # 使reward weights总和为1
                "Merge weights": [x / sum_merge_weights for x in config["w_game merge weights"]],

                "reward_speed_range": [18, 33],
                "lanes_count": 3,
                "vehicles_count": 30,  # 30
                "controlled_vehicles": 3,
                "initial_lane_id": 1,
                "duration": 25,  # [s] The time for each training episode
                "ego_spacing": 0.3,
                "max_vehicles_density": 0.5,
                "min_vehicles_density": 1,
                "initial_controlled_vehicle_speed": [25, 25, 25],
                "initial_controlled_vehicle_acc": [0, 0, 0],
                "lane_change_reward": 0,  # The reward received at each lane change action_num.
                "normalize_reward": True,
                "offroad_terminal": False,

                "screen_width": 800,  # [px]
                "screen_height": 150,  # [px]
                "centering_position": [0.7, 0.5],
                "show_trajectories": False,  # Show history
                "show_future_trajectories": True,
            }
        )
        return config

    def reset(self, *, seed=None, options=None):
        try:
            return super().reset(seed=seed, options=options)
        except Exception as error:
            self._terminate_trucksim(original_error=error)
            raise

    def _reset(self) -> None:
        self._terminate_trucksim()
        model = self.config.get("Controller", {}).get("vehicle_model", "kinematics")
        if model == "8-DoF":
            raise NotImplementedError("8-DoF vehicle state integration is not implemented")
        if model not in {"kinematics", "trucksim"}:
            raise ValueError(f"Unknown Controller.vehicle_model: {model!r}")
        self._create_road()
        self._create_vehicles()
        if model == "trucksim":
            self._create_trucksim_vehicles()

    def _create_trucksim_vehicles(self):
        # Lazy imports keep the kinematics branch independent of native DLLs.
        from pathlib import Path
        from highway_env.vehicle.trucksim_dynamics import TrucksimVehicle, load_engine_map
        from highway_env.vehicle.trucksim_simulation.Truck_Simulation import TrucksimSimulation

        config = self.config["TruckSim"]
        count = len(self.controlled_vehicles)
        simfiles, dlls = config["simfiles"], config["dlls"]
        if len(simfiles) != count or len(dlls) != count:
            raise ValueError("TruckSim needs one simfile and one distinct DLL per controlled vehicle")
        if len({str(Path(p).resolve()).casefold() for p in dlls}) != count:
            raise ValueError("Each TruckSim vehicle must use a distinct DLL")
        for path in [*simfiles, *dlls]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        engine_map = load_engine_map(config)
        signature = tuple(simfiles), tuple(dlls)
        if getattr(self, "_trucksim_signature", None) != signature:
            self.trucksim_models = []
            for simfile, dll in zip(simfiles, dlls):
                self.trucksim_models.append(TrucksimSimulation(simfile, dll))
            self._trucksim_signature = signature
        for i, vehicle in enumerate(self.controlled_vehicles):
            model = self.trucksim_models[i]
            model.reset({"vx_init": vehicle.speed * 3.6})
            truck = TrucksimVehicle.from_vehicle(vehicle, model, config, engine_map)
            road_index = self.road.vehicles.index(vehicle)
            self.road.vehicles[road_index] = truck
            self.controlled_vehicles[i] = truck

    def _terminate_trucksim(self, original_error=None):
        errors = []
        for index, model in enumerate(getattr(self, "trucksim_models", [])):
            try:
                model.stop()
            except Exception as error:
                errors.append((index, error))
        if errors:
            # Always attempt every solver; preserve the original failure when
            # cleanup is running inside an exception handler.
            primary = original_error if original_error is not None else errors[0][1]
            for index, error in errors:
                primary.add_note(f"TruckSim model {index} cleanup failed: {error!r}")
            if original_error is None:
                raise primary

    def step(self, action):
        try:
            result = super().step(action)
        except Exception as error:
            self._terminate_trucksim(original_error=error)
            raise
        if result[2] or result[3]:
            self._terminate_trucksim()
        return result

    def close(self):
        try:
            self._terminate_trucksim()
        finally:
            super().close()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
            show_future_trajectory=self.config["show_future_trajectories"],
        )

    """Scene 1-1 (Slow down: Fixed-position background vehicles)"""
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

    """Scene 1-2 (Slow down: Fixed-position background vehicles)"""
    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        # TODO：Scene 1
        # Scene 1: add obstacle vehicle
        obstacle_vehicle_position = [210, 4]
        obstacle_lane_id = 1
        obstacle_vehicle = other_vehicles_type.create_settled(
            road=self.road,
            speed=self.np_random.uniform(low=14, high=16),
            lane_id=obstacle_lane_id,
            spacing=self.config["ego_spacing"],
            lead_x=obstacle_vehicle_position[0] + 10,
        )
        obstacle_vehicle.target_speed = 16
        obstacle_vehicle.randomize_behavior()
        self.road.vehicles.append(obstacle_vehicle)

        # Scene 1：add jam vehicles
        jam_vehicles = [
            [230.35664167, 0],
            [253.79996468, 4],
            [298.02136631, 4],
            [329.03959963, 4],
            [359.32869456, 4],
            [383.76141537, 4],
            [271.15326486, 0],
            [410.44734299, 4],
            [308.47574134, 0],
            [252.83981489, 8],
            [293.4677917, 8],
            [344.46854993, 8],
            [384.63167393, 8],
            [451.5127433, 4],
            [493.86185227, 4],
            [431.55870084, 8],
            [521.96026237, 4],
            [345.7334867, 0],
            [460.66786908, 8],
            [371.11260416, 0],
            [491.89443077, 8],
            [541.45421275, 8],
            [399.63639464, 0],
            [432.95533198, 0],
            [551.46348314, 4],
            [584.37071895, 8],
            [575.52711016, 4],
            [624.55901334, 8],
            [-57.9194108, 0],
            [-25.56594757, 0],
            [12.43527285, 0],
            [51.65014295, 0],
            [78.11379769, 0],
            [118.80088958, 0],
            [150.43627267, 0],
            [185.23316511, 0],
            [-102.522636, 4],
            [-67.30251631, 4],
            [-25.4446913, 4],
            [9.4499791, 4],
            [39.2724661, 4],
            [77.18559684, 4],
            [118.63763485, 4],
            [154.88658971, 4],
            [-60.47759545, 8],
            [-27.97705679, 8],
            [14.58957211, 8],
            [48.88857299, 8],
            [86.73683446, 8],
            [118.63074592, 8],
            [155.23914675, 8],
            [189.13301876, 8],
        ]
        for jam_position in jam_vehicles:
            jam_lane_id = int(jam_position[1] / 4)
            jam_speed = self.np_random.uniform(low=22, high=25)
            jam_vehicle = other_vehicles_type.create_settled(
                road=self.road,
                speed=jam_speed,
                lane_id=jam_lane_id,
                spacing=self.config["ego_spacing"],
                lead_x=jam_position[0] + 10,
            )
            jam_vehicle.target_speed = 27
            jam_vehicle.randomize_behavior()
            self.road.vehicles.append(jam_vehicle)

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

    """Scene 2 (Slow down: Random-position background vehicles)"""
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
    #         # TODO: Generate settled platoon
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
    #     # TODO: Add front obstacle/slow vehicle
    #     obs_lane_id = 1
    #     obstacle_vehicle_position = [220, 4]
    #     obs_vehicle = other_vehicles_type.create_settled(
    #         road=self.road,
    #         speed=self.np_random.uniform(low=14, high=16),
    #         lane_id=obs_lane_id,
    #         spacing=1 / self.config["max_vehicles_density"],
    #         lead_x=obstacle_vehicle_position[0] + 10,
    #     )
    #     obs_vehicle.target_speed = self.np_random.uniform(low=15, high=17)
    #     obs_vehicle.set_lane_index = ["0", "1", obs_lane_id]
    #     obs_vehicle.color = (250, 71, 84)
    #     obs_vehicle.randomize_behavior()
    #     self.road.vehicles.append(obs_vehicle)
    #
    #     # TODO: Add front vehicles
    #     for _ in range(self.config["vehicles_count"]):
    #         vehicle = other_vehicles_type.create_random(
    #             road=self.road,
    #             speed=self.np_random.uniform(low=22, high=25),
    #             shortest_spacing=1 / self.config["max_vehicles_density"],
    #             longest_spacing=1 / self.config["min_vehicles_density"],
    #             controlled_vehicles=self.controlled_vehicles,
    #         )
    #         vehicle.target_speed = self.np_random.uniform(low=25, high=30)
    #         vehicle.randomize_behavior()
    #         self.road.vehicles.append(vehicle)
    #
    #     # TODO: Add rear aggressive vehicles
    #     rear_vehicles_type = highway_env.vehicle.behavior.AggressiveIDMVehicle
    #     rear_vehicle_count = 8
    #     for lane_id in [0, 1, 2]:
    #         if lane_id == 1:
    #             x_position = utils.generate_random_numbers(
    #                 -100, 150, rear_vehicle_count, 30, 5
    #             )
    #         else:
    #             x_position = utils.generate_random_numbers(
    #                 -60, 190, rear_vehicle_count, 30, 5
    #             )
    #         for x in x_position:
    #             vehicle = other_vehicles_type.create_settled(
    #                 road=self.road,
    #                 speed=self.np_random.uniform(low=22, high=25),
    #                 lane_id=lane_id,
    #                 lead_x=x + 10,
    #             )
    #             vehicle.target_speed = self.np_random.uniform(low=25, high=30)
    #             vehicle.randomize_behavior()
    #             self.road.vehicles.append(vehicle)

    """Scene 3 (Cut in: Random-position background vehicles)"""
    # def _create_vehicles(self) -> None:
    #     """Create some new random vehicles of a given type, and add them on the road."""
    #     other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     other_per_controlled = near_split(
    #         self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
    #     )
    #
    #     self.controlled_vehicles = []
    #     idx = 0
    #     lead_x_position = [204, 189, 174]
    #     for others in other_per_controlled:
    #         #TODO: Generate settled platoon
    #         vehicle = Vehicle.create_settled(
    #             self.road,
    #             speed=self.config["initial_controlled_vehicle_speed"][idx],
    #             lane_id=self.config["initial_lane_id"],
    #             spacing=self.config["ego_spacing"],
    #             lead_x=lead_x_position[idx],
    #         )
    #         vehicle = self.action_type.vehicle_class(
    #             self.road, vehicle.position, vehicle.heading, vehicle.speed
    #         )
    #         self.controlled_vehicles.append(vehicle)
    #         self.road.vehicles.append(vehicle)
    #         idx += 1
    #
    #     # TODO: Add cut-in vehicle
    #     cut_in_vehicle_type = utils.class_from_path("highway_env.vehicle.behavior.SpecialControlledVehicle")
    #     obs_lane_id = 2
    #     obstacle_vehicle_position = [188, 4]
    #     obs_vehicle = cut_in_vehicle_type.create_settled(
    #         road=self.road,
    #         speed=self.np_random.uniform(low=27, high=29),
    #         lane_id=obs_lane_id,
    #         spacing=1 / self.config["max_vehicles_density"],
    #         lead_x=obstacle_vehicle_position[0] + 10,
    #     )
    #     obs_vehicle.task = {
    #         "trigger time": 0.5,
    #         "type": "left lane change",
    #     }
    #     obs_vehicle.env = self
    #     obs_vehicle.target_speed = self.np_random.uniform(low=15, high=17)
    #     obs_vehicle.set_lane_index = ["0", "1", obs_lane_id]
    #     obs_vehicle.randomize_behavior()
    #     self.road.vehicles.append(obs_vehicle)
    #
    #     # TODO: Add front vehicles
    #     for _ in range(self.config["vehicles_count"]):
    #         vehicle = other_vehicles_type.create_random(
    #             road=self.road,
    #             speed=self.np_random.uniform(low=22, high=25),
    #             shortest_spacing=1 / self.config["max_vehicles_density"],
    #             longest_spacing=1 / self.config["min_vehicles_density"],
    #             controlled_vehicles=self.controlled_vehicles,
    #         )
    #         vehicle.target_speed = self.np_random.uniform(low=25, high=30)
    #         vehicle.randomize_behavior()
    #         self.road.vehicles.append(vehicle)
    #
    #     # TODO: Add rear aggressive vehicles
    #     rear_vehicles_type = highway_env.vehicle.behavior.AggressiveIDMVehicle
    #     rear_vehicle_count = 6
    #     for lane_id in [0, 1, 2]:
    #         if lane_id == 1:
    #             x_position = utils.generate_random_numbers(
    #                 -100, 150, rear_vehicle_count, 30, 5
    #             )
    #         elif lane_id == 0:
    #             x_position = utils.generate_random_numbers(
    #                 -60, 190, rear_vehicle_count, 30, 5
    #             )
    #         else:
    #             x_position = utils.generate_random_numbers(
    #                 -40, 170, rear_vehicle_count, 30, 5
    #             )
    #         for x in x_position:
    #             vehicle = other_vehicles_type.create_settled(
    #                 road=self.road,
    #                 speed=self.np_random.uniform(low=22, high=25),
    #                 lane_id=lane_id,
    #                 lead_x=x + 10,
    #             )
    #             vehicle.target_speed = self.np_random.uniform(low=25, high=30)
    #             vehicle.randomize_behavior()
    #             self.road.vehicles.append(vehicle)

    """Scene 4 (Drop: Random-position background vehicles)"""
    # def _create_vehicles(self) -> None:
    #     """Create some new random vehicles of a given type, and add them on the road."""
    #     other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     other_per_controlled = near_split(
    #         self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
    #     )
    #
    #     self.controlled_vehicles = []
    #     idx = 0
    #     lead_x_position = [204, 189, 174]
    #     for others in other_per_controlled:
    #         #TODO: Generate settled platoon
    #         vehicle = Vehicle.create_settled(
    #             self.road,
    #             speed=self.config["initial_controlled_vehicle_speed"][idx],
    #             lane_id=self.config["initial_lane_id"],
    #             spacing=self.config["ego_spacing"],
    #             lead_x=lead_x_position[idx],
    #         )
    #         vehicle = self.action_type.vehicle_class(
    #             self.road, vehicle.position, vehicle.heading, vehicle.speed
    #         )
    #         self.controlled_vehicles.append(vehicle)
    #         self.road.vehicles.append(vehicle)
    #         idx += 1
    #
    #     # TODO: Add drop vehicle
    #     drop_vehicle_type = utils.class_from_path("highway_env.vehicle.behavior.DropVehicle")
    #     obs_lane_id = 1
    #     obstacle_vehicle_position = [220, 4]
    #     obs_vehicle = drop_vehicle_type.create_settled(
    #         road=self.road,
    #         speed=self.np_random.uniform(low=23, high=26),
    #         lane_id=obs_lane_id,
    #         spacing=1 / self.config["max_vehicles_density"],
    #         lead_x=obstacle_vehicle_position[0] + 10,
    #     )
    #     obs_vehicle.task = {
    #         "trigger time": 0.5,
    #         "type": "drop",
    #     }
    #     obs_vehicle.env = self
    #     obs_vehicle.randomize_behavior()
    #     self.road.vehicles.append(obs_vehicle)
    #
    #     # TODO: Add front vehicles
    #     for _ in range(self.config["vehicles_count"]):
    #         vehicle = other_vehicles_type.create_random(
    #             road=self.road,
    #             speed=self.np_random.uniform(low=22, high=25),
    #             shortest_spacing=1 / self.config["max_vehicles_density"],
    #             longest_spacing=1 / self.config["min_vehicles_density"],
    #             controlled_vehicles=self.controlled_vehicles,
    #         )
    #         vehicle.target_speed = self.np_random.uniform(low=25, high=30)
    #         vehicle.randomize_behavior()
    #         self.road.vehicles.append(vehicle)
    #
    #     # TODO: Add rear aggressive vehicles
    #     rear_vehicles_type = highway_env.vehicle.behavior.AggressiveIDMVehicle
    #     rear_vehicle_count = 8
    #     for lane_id in [0, 1, 2]:
    #         if lane_id == 1:
    #             x_position = utils.generate_random_numbers(
    #                 -100, 150, rear_vehicle_count, 30, 5
    #             )
    #         elif lane_id == 0:
    #             x_position = utils.generate_random_numbers(
    #                 -60, 190, rear_vehicle_count, 30, 5
    #             )
    #         else:
    #             x_position = utils.generate_random_numbers(
    #                 -40, 170, rear_vehicle_count, 30, 5
    #             )
    #         for x in x_position:
    #             vehicle = other_vehicles_type.create_settled(
    #                 road=self.road,
    #                 speed=self.np_random.uniform(low=22, high=25),
    #                 lane_id=lane_id,
    #                 lead_x=x + 10,
    #             )
    #             vehicle.target_speed = self.np_random.uniform(low=25, high=30)
    #             vehicle.randomize_behavior()
    #             self.road.vehicles.append(vehicle)

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
                                [self.config["collision_reward"] +
                                    self.config["collision_reward"] +
                                    self.config["distance_reward"] +
                                    self.config["split_reward"] +
                                    self.config["merge_reward"],

                                    self.config["forward_speed_reward"] +
                                    self.config["on_road_reward"] +
                                    self.config["same_lane_reward"] +
                                    self.config["far_reward"] +
                                    self.config["action_infeasible_reward"] +
                                    self.config["high_speed_reward"] +
                                    self.config["game_decision_reward"]
                                 ],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

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
            "distance_reward": float(self._distance_reward(x_position, lane_index)),
            "same_lane_reward": self._samelane_reward(lane_index, forward_speed),
            # "equal_speed_reward": float(1 / (np.var(speeds) + 1)),
            "forward_speed_reward": self._forward_speed_reward(forward_speed),
            "far_reward": self._far(),
            "action_infeasible_reward": max(action_feasible_reward),
            "split_reward": self._split_reward(group_action_list),
            "merge_reward": self._merge_reward(env_state_list, group_action_list),
            # "action_reward": self._action_reward(action),
            # "same_action_reward": self._same_action_reward(group_action_list),
            "game_decision_reward": self._game_decision_reward(),
        }

    def _game_decision_reward(self) -> float:
        best_cost = self.controlled_vehicles[0].best_cost
        if best_cost is not None:
            reward = best_cost
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

    def _split_reward(self, group_action_list) -> float:
        split_action = [0, 1, 2]
        array = np.array(group_action_list[0])
        if 3 in array:
            last_merge_index = np.where(array == 3)[0][-1]
            if last_merge_index < len(array) - 1:
                array = array[last_merge_index + 1:]
            else:
                array = []

        fre = self.config["simulation_frequency"]

        max_num = 0
        for i in split_action:
            if i in group_action_list[0]:
                max_num += self.max_consecutive_count(array, i)
        if 5 < max_num / fre <= 10:
            return (max_num / fre - 5) / 5 * 0.8
        if 10 < max_num / fre <= 15:
            return 0.8 + (max_num / fre - 10) / 5 * 0.2
        if max_num / fre > 15:
            return 1
        return 0

    def _merge_reward(self, env_state_list, group_action_list) -> float:
        array = np.array(env_state_list[0]) - np.array(group_action_list[0])
        if len(array) > 0:
            last_zero_index = np.where(array == 0)[0][-1]
            count = self.max_consecutive_non_zero_count(array[last_zero_index+1:]) \
                if last_zero_index < len(array) - 1 else 0
        else:
            count = 0

        fre = self.config["simulation_frequency"]

        if 5 < count / fre <= 10:
            return (count / fre - 5) / 5 * 0.1
        if 10 < count / fre <= 15:
            return 0.3 + (count / fre - 5) / 5 * 0.7
        if count / fre > 15:
            return 1
        return 0

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
        min_speed = 17
        ave_speed = np.mean(forward_speed)
        if min_speed < ave_speed < 19:
            return (ave_speed - min_speed) / (19-17) * 1
        elif ave_speed <= min_speed:
            return 1
        else:
            return 0


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
            reward = 1
        else:
            reward = 0
        return reward

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
        if self.config.get("Controller", {}).get("vehicle_model") == "trucksim":
            info["trucksim"] = [
                {"measured_acceleration": v.measured_acceleration.copy(),
                 "inputs": v.trucksim_inputs.copy(),
                 "solver_time": v.trucksim_model.current_time}
                for v in self.controlled_vehicles
            ]
        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        context = getattr(self.road, "longitudinal_control", None)
        if context is not None:
            info["longitudinal_control"] = context.diagnostics()
        return info

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


class MyHighwayEnvFast(MyHighwayEnv):
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
