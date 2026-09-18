import copy
import select
from tokenize import group
from typing import List, Optional, Tuple, Union
from unittest.mock import patch

import numpy as np
from prompt_toolkit.buffer import indent
from scipy.linalg import solve_discrete_are
from scipy.signal import ellip

from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.planner import PolyPlanner
from highway_env.vehicle.longitudinal import bind_longitudinal_source, publish_longitudinal_result
import math
import matplotlib.pyplot as plt


class ControlledVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    """Characteristic time"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.4  # 0.2 0.4  # [s]
    TAU_LATERAL = 0.6  # 0.8 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    MAX_DELTA_STEERING_ANGLE = np.pi / 180  # [rad]
    DELTA_SPEED = 5  # [m/s]

    # Lateral policy parameters
    POLITENESS = 0.0  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    # LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 3
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: LaneIndex = None,
        target_speed: float = None,
        route: Route = None,
        waite_timer: float = 0
    ):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route
        # Kong add
        self.action_infeasible = False
        self.waite_timer = waite_timer
        self.success_merge = False

    @classmethod
    def create_from(cls, vehicle: "ControlledVehicle") -> "ControlledVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            route=vehicle.route,
        )
        bind_longitudinal_source(v, vehicle)
        return v

    def plan_route_to(self, destination: str) -> "ControlledVehicle":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        try:
            path = self.road.network.shortest_path(self.lane_index[1], destination)
        except KeyError:
            path = []
        if path:
            self.route = [self.lane_index] + [
                (path[i], path[i + 1], None) for i in range(len(path) - 1)
            ]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action_num to change the desired lane or speed.

        - If a high-level action_num is provided, update the target speed and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action_num
        """
        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = (
                _from,
                _to,
                np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1),
            )
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = target_lane_index

        action = {
            "steering": self.steering_control(self.target_lane_index),
            "acceleration": self.speed_control(self.target_speed),
        }
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        super().act(action)

    # def act_discrete_and_continuous(self, action: Union[dict, str] = None) -> None:
    #     """
    #     Perform a high-level action_num to change the desired lane or speed.
    #
    #     - If a high-level action_num is provided, update the target speed and lane;
    #     - then, perform longitudinal and lateral control.
    #
    #     :param action: a high-level action_num
    #     """
    #     self.follow_road()
    #     t_acc = 0.1
    #     if action == "LANE_RIGHT":
    #         _from, _to, _id = self.target_lane_index
    #         target_lane_index = (
    #             _from,
    #             _to,
    #             np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1),
    #         )
    #         if self.road.network.get_lane(target_lane_index).is_reachable_from(
    #             self.position
    #         ):
    #             self.target_lane_index = target_lane_index
    #         action = {
    #             "steering": self.steering_control(self.target_lane_index),
    #             "acceleration": self.speed_control_acc(target_speed=self.target_speed),
    #         }
    #     elif action == "LANE_LEFT":
    #         _from, _to, _id = self.target_lane_index
    #         target_lane_index = (
    #             _from,
    #             _to,
    #             np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1),
    #         )
    #         if self.road.network.get_lane(target_lane_index).is_reachable_from(
    #             self.position
    #         ):
    #             self.target_lane_index = target_lane_index
    #         action = {
    #             "steering": self.steering_control(self.target_lane_index),
    #             "acceleration": self.speed_control_acc(target_speed=self.target_speed),
    #         }
    #     else:
    #         # Use IDM to generate target_acc
    #         # TODO: Add IDM Method
    #         target_acc = self.idm_control()
    #         self.target_speed += target_acc * t_acc
    #         action = {
    #             "steering": self.steering_control(self.target_lane_index),
    #             "acceleration": self.speed_control_acc(target_speed=self.target_speed, sample_time=t_acc),
    #         }
    #     action["steering"] = np.clip(
    #         action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
    #     )
    #     super().act(action)

    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(
                self.target_lane_index,
                route=self.route,
                position=self.position,
                np_random=self.road.np_random,
            )

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = -self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(
            np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1)
        )
        heading_ref = lane_future_heading + np.clip(
            heading_command, -np.pi / 4, np.pi / 4
        )
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(
            heading_ref - self.heading
        )
        # Heading rate to steering angle
        slip_angle = np.arcsin(
            np.clip(
                self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                -1,
                1,
            )
        )
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(
            steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_speed - self.speed)

    def speed_control_acc(self, **kwargs) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :param sample_time: the sample time to calculate acceleration
        :return: an acceleration command [m/s2]
        """
        if len(kwargs) > 1:
            return (kwargs["target_speed"] - self.speed) / kwargs["sample_time"]
        else:
            return self.KP_A * (kwargs["target_speed"] - self.speed)

    def get_routes_at_intersection(self) -> List[Route]:
        """Get the list of routes that can be followed at the next intersection."""
        if not self.route:
            return []
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return [self.route]
        next_destinations_from = list(next_destinations.keys())
        routes = [
            self.route[0 : index + 1]
            + [(self.route[index][1], destination, self.route[index][2])]
            for destination in next_destinations_from
        ]
        return routes

    def set_route_at_intersection(self, _to: int) -> None:
        """
        Set the road to be followed at the next intersection.

        Erase current planned route.

        :param _to: index of the road to follow at next intersection, in the road network
        """

        routes = self.get_routes_at_intersection()
        if routes:
            if _to == "random":
                _to = self.road.np_random.integers(len(routes))
            self.route = routes[_to % len(routes)]

    def predict_trajectory_constant_speed(
        self, times: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predict the future positions of the vehicle along its planned route, under constant speed

        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        pos_heads = [
            self.road.network.position_heading_along_route(
                route, coordinates[0] + self.speed * t, 0, self.lane_index
            )
            for t in times
        ]
        return tuple(zip(*pos_heads))


class MDPVehicle(ControlledVehicle):

    """A controlled vehicle with a specified discrete range of allowed target speeds."""

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    DEFAULT_TARGET_SPEEDS = np.linspace(20, 30, 3)
    LANE_CHANGE_DELAY = 1.0  # [s]

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    def __init__(
        self,
        road: Road,
        position: List[float],
        heading: float = 0,
        speed: float = 0,
        target_lane_index: Optional[LaneIndex] = None,
        target_speed: Optional[float] = None,
        target_speeds: Optional[Vector] = None,
        route: Optional[Route] = None,
    ) -> None:
        """
        Initializes an MDPVehicle

        :param road: the road on which the vehicle is driving
        :param position: its position
        :param heading: its heading angle
        :param speed: its speed
        :param target_lane_index: the index of the lane it is following
        :param target_speed: the speed it is tracking
        :param target_speeds: the discrete list of speeds the vehicle is able to track, through faster/slower actions
        :param route: the planned route of the vehicle, to handle intersections
        """
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route
        )
        self.target_speeds = (
            np.array(target_speeds)
            if target_speeds is not None
            else self.DEFAULT_TARGET_SPEEDS
        )
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)
        self.timer = (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action_num.

        - If the action_num is a speed change, choose speed from the allowed discrete range.
        - Else, forward action_num to the ControlledVehicle handler.

        :param action: a high-level action_num
        """
        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
        else:
            super().act(action)
            return
        self.speed_index = int(
            np.clip(self.speed_index, 0, self.target_speeds.size - 1)
        )
        self.target_speed = self.index_to_speed(self.speed_index)
        super().act()

    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        return self.target_speeds[index]

    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.target_speeds[0]) / (
            self.target_speeds[-1] - self.target_speeds[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (self.target_speeds.size - 1)),
                0,
                self.target_speeds.size - 1,
            )
        )

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.DEFAULT_TARGET_SPEEDS[0]) / (
            cls.DEFAULT_TARGET_SPEEDS[-1] - cls.DEFAULT_TARGET_SPEEDS[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (cls.DEFAULT_TARGET_SPEEDS.size - 1)),
                0,
                cls.DEFAULT_TARGET_SPEEDS.size - 1,
            )
        )

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(
            vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed)
        )

    def predict_trajectory(
        self,
        actions: List,
        action_duration: float,
        trajectory_timestep: float,
        dt: float,
    ) -> List[ControlledVehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action_num.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action_num
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states


# Kong add
class LEADVehicle(ControlledVehicle):

    """A leader vehicle in a platoon or a group"""

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    DEFAULT_TARGET_SPEEDS = np.linspace(20, 30, 3)
    LANE_CHANGE_DELAY = 1  # 1 [s]

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5  # [m/s2]
    """Desired maximum deceleration."""

    TTC_MIN = 2.5  # s

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    Q = [1, 0.1, 0.01]  # d_d, d_v, d_a 5 0.1 0.01
    """LQR parameter"""

    R = 1
    """LQR parameter"""

    def __init__(
        self,
        road: Road,
        position: List[float],
        heading: float = 0,
        speed: float = 0,
        target_lane_index: Optional[LaneIndex] = None,
        target_speed: Optional[float] = None,
        target_speeds: Optional[Vector] = None,
        route: Optional[Route] = None,
        length: float = 5,
        width: float = 2,
        timer: float = None,
        group_action: int = None,
    ) -> None:
        """
        Initializes an LEADERVehicle

        :param road: the road on which the vehicle is driving
        :param position: its position
        :param heading: its heading angle
        :param speed: its speed
        :param target_lane_index: the index of the lane it is following
        :param target_speed: the speed it is tracking
        :param target_speeds: the discrete list of speeds the vehicle is able to track, through faster/slower actions
        :param route: the planned route of the vehicle, to handle intersections
        """
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route
        )
        self.target_speeds = (
            np.array(target_speeds)
            if target_speeds is not None
            else self.DEFAULT_TARGET_SPEEDS
        )
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)
        self.timer = timer
        self.LENGTH = length
        self.WIDTH = width
        self.group_action = group_action

    @classmethod
    def create_from(cls, vehicle: "ControlledVehicle") -> "ControlledVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            route=vehicle.route,
        )
        bind_longitudinal_source(v, vehicle)
        return v

    # def act(self, action: Union[dict, str] = None) -> None:
    #     """
    #     Perform a high-level action_num.
    #
    #     - If the action_num is a speed change, choose speed from the allowed discrete range.
    #     - Else, forward action_num to the ControlledVehicle handler.
    #
    #     :param action: a high-level action_num
    #     """
    #     if action == "FASTER":
    #         self.speed_index = self.speed_to_index(self.speed) + 1
    #     elif action == "SLOWER":
    #         self.speed_index = self.speed_to_index(self.speed) - 1
    #     else:
    #         super().act(action)
    #         return
    #     self.speed_index = int(
    #         np.clip(self.speed_index, 0, self.target_speeds.size - 1)
    #     )
    #     self.target_speed = self.index_to_speed(self.speed_index)
    #     super().act()
    def act(
            self,
            lateral_action: Union[dict, str] = None,
            lane_index: LaneIndex = None,
            acceleration: float = 0,
    ) -> None:
        """
        Perform a high-level action_num.

        - If the action_num is a speed change, choose speed from the allowed discrete range.
        - Else, forward action_num to the ControlledVehicle handler.

        :param lateral_action: a high-level lateral action
        :param lane_index: the target lane index
        :param acceleration: the continuous acceleration
        """
        self.follow_road()
        if lateral_action is None:
            return
        else:
            target_lane_index = lane_index
            if self.road.network.get_lane(target_lane_index).is_reachable_from(
                    self.position
            ):
                self.target_lane_index = target_lane_index
            action = {
                "steering": self.steering_control(self.target_lane_index),
                "acceleration": acceleration,
            }
            action["steering"] = np.clip(
                action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
            )
            self.action = action

    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        return self.target_speeds[index]

    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.target_speeds[0]) / (
            self.target_speeds[-1] - self.target_speeds[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (self.target_speeds.size - 1)),
                0,
                self.target_speeds.size - 1,
            )
        )

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.DEFAULT_TARGET_SPEEDS[0]) / (
            cls.DEFAULT_TARGET_SPEEDS[-1] - cls.DEFAULT_TARGET_SPEEDS[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (cls.DEFAULT_TARGET_SPEEDS.size - 1)),
                0,
                cls.DEFAULT_TARGET_SPEEDS.size - 1,
            )
        )

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(
            vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed)
        )

    def predict_trajectory(
        self,
        actions: List,
        action_duration: float,
        trajectory_timestep: float,
        dt: float,
    ) -> List[ControlledVehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action_num.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action_num
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states

    def change_lane_policy(
        self,
        controlled_vehicle,
        group: list = None,
        target_lane_index: LaneIndex = None,
    ) -> list:
        """Select/continue a lane change.

        Scenario target lanes still pass through MOBIL before the maneuver
        starts. Once accepted, the target becomes committed until the vehicle
        reaches that lane. Ordinary MOBIL is therefore not re-run every policy
        step during the same maneuver.
        """
        # === LANE CHANGE COMMITMENT V1 START ===
        current_lane_index = controlled_vehicle.lane_index
        committed_target = getattr(
            controlled_vehicle,
            "target_lane_index",
            current_lane_index,
        )

        if (
            committed_target is not None
            and current_lane_index != committed_target
            and current_lane_index[:2] == committed_target[:2]
        ):
            # Keep only the existing emergency conflict check. Do not re-run
            # ordinary MOBIL and oscillate between current/target lanes.
            emergency_abort = False
            for other in self.road.vehicles:
                if (
                    other is controlled_vehicle
                    or not isinstance(other, ControlledVehicle)
                ):
                    continue
                if (
                    other.lane_index != committed_target
                    and getattr(other, "target_lane_index", None)
                    == committed_target
                ):
                    target_lane = self.road.network.get_lane(
                        committed_target
                    )
                    ego_s, _ = target_lane.local_coordinates(
                        controlled_vehicle.position
                    )
                    other_s, _ = target_lane.local_coordinates(
                        other.position
                    )
                    d_star, _ = self.desired_gap(
                        controlled_vehicle,
                        other,
                    )
                    if 0.0 < other_s - ego_s < d_star:
                        emergency_abort = True
                        break

            if not emergency_abort:
                if current_lane_index[2] > committed_target[2]:
                    return [0, committed_target]
                if current_lane_index[2] < committed_target[2]:
                    return [1, committed_target]
                return [2, committed_target]

            return [2, current_lane_index]
        # === LANE CHANGE COMMITMENT V1 END ===

        if target_lane_index is None:
            for lane_index in self.road.network.side_lanes(
                current_lane_index
            ):
                if not self.road.network.get_lane(
                    lane_index
                ).is_reachable_from(controlled_vehicle.position):
                    continue
                if np.abs(controlled_vehicle.speed) < 1:
                    continue
                if self.mobil(lane_index, group):
                    if current_lane_index[2] > lane_index[2]:
                        return [0, lane_index]
                    if current_lane_index[2] < lane_index[2]:
                        return [1, lane_index]
        else:
            if self.road.network.get_lane(
                target_lane_index
            ).is_reachable_from(controlled_vehicle.position):
                if np.abs(controlled_vehicle.speed) >= 1:
                    if self.mobil(
                        target_lane_index,
                        group,
                        forced=True,
                    ):
                        if current_lane_index[2] > target_lane_index[2]:
                            return [0, target_lane_index]
                        if current_lane_index[2] < target_lane_index[2]:
                            return [1, target_lane_index]

        return [2, current_lane_index]
    def desired_gap(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = True,
    ) -> list:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau_star = self.TIME_WANTED
        tau_follow = self.TIME_WANTED / 10
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN

        if ego_vehicle is None:
            return [0, 0]
        elif front_vehicle is None:
            dv = np.dot(ego_vehicle.velocity, ego_vehicle.direction)
        else:
            dv = (
                np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
                if projected
                else ego_vehicle.speed - front_vehicle.speed
            )
        d_star = (
            d0 + ego_vehicle.speed * tau_star + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        d_follow = (
            d0 + ego_vehicle.speed * tau_follow + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        return [d_star, d_follow]

    def mobil(
        self,
        lane_index: LaneIndex,
        group: list = None,
        forced: bool = False,
    ) -> bool:
        """MOBIL safety/utility check using target-lane Frenet geometry."""
        if group is None:
            new_preceding, new_following = self.road.neighbour_vehicles(
                self,
                lane_index,
            )

            new_following_a = self.acceleration(
                ego_vehicle=new_following,
                front_vehicle=new_preceding,
                desired_gap=self.desired_gap(
                    ego_vehicle=new_following,
                    front_vehicle=new_preceding,
                )[1],
            )
            new_following_pred_a = self.acceleration(
                ego_vehicle=new_following,
                front_vehicle=self,
                desired_gap=self.desired_gap(
                    ego_vehicle=new_following,
                    front_vehicle=self,
                )[1],
            )

            old_preceding, old_following = self.road.neighbour_vehicles(
                self
            )

            if self.route and self.route[0][2] is not None:
                if np.sign(
                    lane_index[2] - self.target_lane_index[2]
                ) != np.sign(
                    self.route[0][2] - self.target_lane_index[2]
                ):
                    return False

            self_pred_a = self.acceleration(
                ego_vehicle=self,
                front_vehicle=new_preceding,
                desired_gap=self.desired_gap(
                    ego_vehicle=self,
                    front_vehicle=new_preceding,
                )[1],
            )
            if self_pred_a < -self.ACC_MAX:
                return False

            if new_preceding is not None:
                front_gap = self.road.longitudinal_gap(
                    front_vehicle=new_preceding,
                    rear_vehicle=self,
                    lane_index=lane_index,
                )
                front_ttc = self.road.longitudinal_ttc(
                    front_vehicle=new_preceding,
                    rear_vehicle=self,
                    lane_index=lane_index,
                )
                if (
                    front_gap <= 1.5 * self.LENGTH
                    or front_ttc <= self.TTC_MIN
                ):
                    return False

            if new_following is not None:
                rear_gap = self.road.longitudinal_gap(
                    front_vehicle=self,
                    rear_vehicle=new_following,
                    lane_index=lane_index,
                )
                rear_ttc = self.road.longitudinal_ttc(
                    front_vehicle=self,
                    rear_vehicle=new_following,
                    lane_index=lane_index,
                )
                if (
                    rear_gap <= 1.5 * self.LENGTH
                    or rear_ttc <= self.TTC_MIN
                ):
                    return False

            if forced:
                return True

            self_a = self.integrated_longitudinal_control(
                ego_vehicle=self,
                front_vehicle=old_preceding,
            )
            old_following_a = self.acceleration(
                ego_vehicle=old_following,
                front_vehicle=self,
                desired_gap=self.desired_gap(
                    ego_vehicle=old_following,
                    front_vehicle=self,
                )[1],
            )
            old_following_pred_a = self.acceleration(
                ego_vehicle=old_following,
                front_vehicle=old_preceding,
                desired_gap=self.desired_gap(
                    ego_vehicle=old_following,
                    front_vehicle=old_preceding,
                )[1],
            )
            jerk = (
                self_pred_a
                - self_a
                + self.POLITENESS
                * (
                    new_following_pred_a
                    - new_following_a
                    + old_following_pred_a
                    - old_following_a
                )
            )
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False
            return True

        new_precedings, new_followings = (
            self.road.group_neighbour_vehicles(
                group=group,
                lane_index=lane_index,
            )
        )
        old_precedings, old_followings = (
            self.road.group_neighbour_vehicles(group=group)
        )

        for idx, (
            new_preceding,
            new_following,
            old_preceding,
            old_following,
        ) in enumerate(
            zip(
                new_precedings,
                new_followings,
                old_precedings,
                old_followings,
            )
        ):
            # === GROUP MOBIL EGO REFERENCE V1 ===
            ego_vehicle = group[idx]

            if new_preceding is not None:
                front_gap = self.road.longitudinal_gap(
                    front_vehicle=new_preceding,
                    rear_vehicle=ego_vehicle,
                    lane_index=lane_index,
                )
                front_ttc = self.road.longitudinal_ttc(
                    front_vehicle=new_preceding,
                    rear_vehicle=ego_vehicle,
                    lane_index=lane_index,
                )
                if (
                    front_gap <= 1.5 * ego_vehicle.LENGTH
                    or front_ttc <= self.TTC_MIN
                ):
                    return False

            if new_following is not None:
                rear_gap = self.road.longitudinal_gap(
                    front_vehicle=ego_vehicle,
                    rear_vehicle=new_following,
                    lane_index=lane_index,
                )
                rear_ttc = self.road.longitudinal_ttc(
                    front_vehicle=ego_vehicle,
                    rear_vehicle=new_following,
                    lane_index=lane_index,
                )
                if (
                    rear_gap <= 1.5 * ego_vehicle.LENGTH
                    or rear_ttc <= self.TTC_MIN
                ):
                    return False

            new_following_a = self.acceleration(
                ego_vehicle=new_following,
                front_vehicle=new_preceding,
                desired_gap=self.desired_gap(
                    ego_vehicle=new_following,
                    front_vehicle=new_preceding,
                )[1],
            )
            new_following_pred_a = self.acceleration(
                ego_vehicle=new_following,
                front_vehicle=ego_vehicle,
                desired_gap=self.desired_gap(
                    ego_vehicle=new_following,
                    front_vehicle=ego_vehicle,
                )[1],
            )
            if new_following_pred_a < -self.ACC_MAX:
                return False

            ego_route = getattr(ego_vehicle, "route", None)
            ego_target_lane_index = getattr(
                ego_vehicle,
                "target_lane_index",
                ego_vehicle.lane_index,
            )
            if ego_route and ego_route[0][2] is not None:
                if np.sign(
                    lane_index[2] - ego_target_lane_index[2]
                ) != np.sign(
                    ego_route[0][2] - ego_target_lane_index[2]
                ):
                    return False

            ego_pred_a = self.acceleration(
                ego_vehicle=ego_vehicle,
                front_vehicle=new_preceding,
                desired_gap=self.desired_gap(
                    ego_vehicle=ego_vehicle,
                    front_vehicle=new_preceding,
                )[1],
            )
            if ego_pred_a < -self.ACC_MAX:
                return False

            if forced:
                continue

            ego_current_a = self.integrated_longitudinal_control(
                ego_vehicle=ego_vehicle,
                front_vehicle=old_preceding,
            )
            old_following_a = self.acceleration(
                ego_vehicle=old_following,
                front_vehicle=ego_vehicle,
                desired_gap=self.desired_gap(
                    ego_vehicle=old_following,
                    front_vehicle=ego_vehicle,
                )[1],
            )
            old_following_pred_a = self.acceleration(
                ego_vehicle=old_following,
                front_vehicle=old_preceding,
                desired_gap=self.desired_gap(
                    ego_vehicle=old_following,
                    front_vehicle=old_preceding,
                )[1],
            )
            jerk = (
                ego_pred_a
                - ego_current_a
                + self.POLITENESS
                * (
                    new_following_pred_a
                    - new_following_a
                    + old_following_pred_a
                    - old_following_a
                )
            )
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        return True
    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        desired_gap: float = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :param desired_gap
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(
                ego_target_speed, 0, ego_vehicle.lane.speed_limit
            )
        acceleration = self.COMFORT_ACC_MAX * (
            1
            - np.power(
                max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)),
                self.DELTA,
            )
        )

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            if desired_gap is None:
                acceleration -= self.COMFORT_ACC_MAX * np.power(
                    self.desired_gap(ego_vehicle, front_vehicle)[0] / utils.not_zero(d), 2
                )
            else:
                acceleration -= self.COMFORT_ACC_MAX * np.power(
                    desired_gap / utils.not_zero(d), 2
                )
        return acceleration

    # def idm(self, target_lane_index) -> float:
    #     # Longitudinal: IDM
    #     """
    #     IDM Module: generate proper acceleration
    #     :param group: the sub-platoon
    #     :param target_lane_index: the target lane index for leader
    #     :return acceleration
    #     """
    #     front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
    #         self, self.lane_index
    #     )
    #     target_acceleration = self.acceleration(
    #         ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
    #     )
    #     # When changing lane, check both current and target lanes
    #     if self.lane_index != target_lane_index:
    #         front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
    #             self, target_lane_index
    #         )
    #         target_lane_acceleration = self.acceleration(
    #             ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
    #         )
    #         target_acceleration = min(
    #             target_acceleration, target_lane_acceleration
    #         )
    #     # action_num['acceleration'] = self.recover_from_stop(action_num['acceleration'])
    #     target_acceleration = np.clip(
    #         target_acceleration, -self.ACC_MAX, self.ACC_MAX
    #     )
    #     return target_acceleration

    @publish_longitudinal_result
    def integrated_longitudinal_control(
            self,
            ego_vehicle: Vehicle = None,
            front_vehicle: Vehicle = None,
            vehicle_group: list = None,
            speed_control_type: str = None,
            controlled_vehicles: list = None,
    ) -> float:
        """
        Longitudinal: LQR and IDM Switched Control

        :return optimal acceleration
        """
        if front_vehicle is None:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
                self, self.lane_index
            )
        if ego_vehicle is None:
            ego_vehicle = self

        ref_dis = self.desired_gap(self, front_vehicle)[1]

        a_follow = self.longitudinal_control(
            ego_vehicle=ego_vehicle,
            front_vehicle=front_vehicle,
            reference_distance=ref_dis,
        )

        # When changing lane
        if ego_vehicle.lane_index != ego_vehicle.target_lane_index:
            target_front_vehicle, _ = self.road.neighbour_vehicles(self, self.target_lane_index)
            ref_dis = self.desired_gap(self, target_front_vehicle)[1]
            a_change_lane = self.longitudinal_control(
                ego_vehicle=ego_vehicle,
                front_vehicle=target_front_vehicle,
                reference_distance=ref_dis,
            )
            a_follow = min(a_follow, a_change_lane)

        if speed_control_type is None or speed_control_type == "no control":
            return a_follow
        elif speed_control_type == "slightly slow down":
            _, following_v = self.road.neighbour_vehicles(self, self.lane_index, None)
            if following_v is not None:
                min_speed = following_v.speed
            else:
                controlled_v_speeds = []
                for v in controlled_vehicles:
                    controlled_v_speeds.append(v.speed)
                min_speed = min(controlled_v_speeds) - 3
            # controlled_v_speeds = []
            # for v in controlled_vehicles:
            #     controlled_v_speeds.append(v.speed)
            # min_speed = min(controlled_v_speeds) - 3

            a_slow = self.longitudinal_control(
                    ego_vehicle=ego_vehicle,
                    front_vehicle=front_vehicle,
                    reference_distance=ref_dis,
                    speed_control=True,
                    reference_v=max(ego_vehicle.speed - 3, min_speed),
                )
            return min(a_follow, a_slow)
        # Else: speed_control_type = "keep speed"
        else:
            a_keep = self.longitudinal_control(
                    ego_vehicle=ego_vehicle,
                    front_vehicle=front_vehicle,
                    reference_distance=ref_dis,
                    speed_control=True,
                    reference_v=ego_vehicle.speed,
                )
            return min(a_follow, a_keep)


    def longitudinal_control(
            self, ego_vehicle=None, front_vehicle=None, reference_distance=10,
            speed_control=False, reference_v=33, is_limited=True):
        """Select only the longitudinal algorithm; steering is independent."""
        context = getattr(self.road, "longitudinal_control", None)
        fallback = lambda: self.lqr_control(
            ego_vehicle, front_vehicle, reference_distance,
            speed_control, reference_v, is_limited)
        if context is None or context.kind == "lqr":
            return fallback()
        speed_only = speed_control or front_vehicle is None
        upper = self.ACC_MAX if speed_only else self.COMFORT_ACC_MAX
        bounds = (-self.ACC_MAX, upper) if is_limited else (-1e6, 1e6)
        return context.compute(
            self, front_vehicle, reference_distance, reference_v, bounds,
            fallback, speed_only=speed_only, state_vehicle=ego_vehicle)


    def lqr_control(
            self,
            ego_vehicle: Vehicle = None,
            front_vehicle: Vehicle = None,
            reference_distance: float = 10,
            speed_control: bool = False,
            reference_v: float = 33,
            is_limited: bool = True,
    ) -> float:
        """
        LQR Longitudinal Control
        """
        if ego_vehicle is None:
            ego_x = self.position[0]
            ego_v = self.speed
            ego_a = self.action["acceleration"]
        else:
            ego_x = ego_vehicle.position[0]
            ego_v = ego_vehicle.speed
            ego_a = ego_vehicle.action["acceleration"]

        T = 0.05  # Sampling time
        Ts = 0.1  # Inertia time constant

        if not speed_control and front_vehicle is not None:
            # System matrix
            A = np.array([[1, T, 0],
                          [0, 1, T],
                          [0, 0, 1 - T / Ts]])

            B = np.array([[0],
                          [0],
                          [T / Ts]])

            # Weight matrix
            Q = np.diag(self.Q)  # d_d, d_v, d_a
            R = np.array(self.R)

            # Solve Riccati function
            P = solve_discrete_are(A, B, Q, R)

            # gain
            K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

            if type(front_vehicle).__name__ != "Obstacle":
                front_x = front_vehicle.position[0]
                front_v = front_vehicle.speed
                front_a = front_vehicle.action["acceleration"]
            else:
                front_x = front_vehicle.position[0]
                front_v = 0
                front_a = 0

            # State vector
            X_front = np.array([
                reference_distance - (front_x - ego_x),
                ego_v - front_v,
                ego_a - front_a
            ])

            # Optimal result
            optimal_acceleration_front = (-K @ X_front).item()

            if not is_limited:
                return optimal_acceleration_front
            else:
                return np.clip(
                    optimal_acceleration_front, -self.ACC_MAX, self.COMFORT_ACC_MAX
                )

        else:
            # System matrix
            A = np.array([[1, T],
                          [0, 1 - T / Ts]])

            B = np.array([[0],
                          [T / Ts]])

            # Weight matrix
            Q = np.diag([1, 0.1])  # d_v, d_a
            R = np.array([1])

            # Solve Riccati function
            P = solve_discrete_are(A, B, Q, R)

            # gain
            K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

            X = [ego_v - reference_v,
                 ego_a]

            # Optimal result
            optimal_acceleration = (-K @ X).item()

            if not is_limited:
                return optimal_acceleration
            else:
                return np.clip(
                optimal_acceleration, -self.ACC_MAX, self.ACC_MAX
            )


# Kong add
class FOLLOWVehicle(ControlledVehicle):

    """A leader vehicle in a platoon or a group"""

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    DEFAULT_TARGET_SPEEDS = np.linspace(20, 30, 3)
    LANE_CHANGE_DELAY = 1.0  # [s]

    COMFORT_ACC_MAX = 4.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5  # [m/s2]
    """Desired maximum deceleration."""

    TTC_MIN = 2.5  # [s]

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    Q = [1, 0.1, 0.01]  # d_d, d_v, d_a 5 0.1 0.01
    """LQR parameter"""

    R = 1
    """LQR parameter"""

    TTC_MAX = 1e3
    """risk-aware"""

    def __init__(
        self,
        road: Road,
        position: List[float],
        heading: float = 0,
        speed: float = 0,
        target_lane_index: Optional[LaneIndex] = None,
        target_speed: Optional[float] = None,
        target_speeds: Optional[Vector] = None,
        route: Optional[Route] = None,
        length: float = 5,
        width: float = 2,
        timer: float = None,
        group_action: int = 3,
    ) -> None:
        super().__init__(
            road,
            position,
            heading,
            speed,
            target_lane_index,
            target_speed,
            route,
        )

        self.target_speeds = (
            np.array(target_speeds)
            if target_speeds is not None
            else self.DEFAULT_TARGET_SPEEDS
        )

        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)
        self.timer = (
            timer
            or (np.sum(self.position) * np.pi)
            % self.LANE_CHANGE_DELAY
        )

        self.action_list = []
        self.env_state_list = []
        self.group_action = group_action
        self.env_state = group_action
        self.LENGTH = length
        self.WIDTH = width

        self.info = {
            "x": [],
            "y": [],
            "v": [],
            "a": [],
            "delta": [],
        }

        self.path = None
        self.latest_planner_trajectory = None

        self.best_cost = None
        self.jerk = 0
        self.ttc_f = self.TTC_MAX
        self.ttc_r = self.TTC_MAX
        self.ttc_lat = self.TTC_MAX
        self.ttc_lon = self.TTC_MAX
        self.ttc = self.TTC_MAX
        self.risk_value = 0
        self.risk_gradient = 0
        self.max_risk = 0
    def trajectory_steering_control(
            self,
            reference_position,
            reference_heading,
    ):
        reference_position = np.asarray(
            reference_position,
            dtype=float,
        ).reshape(2)

        delta = reference_position - np.asarray(self.position, dtype=float)
        c = math.cos(float(self.heading))
        s = math.sin(float(self.heading))

        local_x = c * delta[0] + s * delta[1]
        local_y = -s * delta[0] + c * delta[1]

        lookahead_sq = max(
            float(local_x ** 2 + local_y ** 2),
            1.0,
        )
        curvature = 2.0 * float(local_y) / lookahead_sq
        wheelbase = float(getattr(self, "LENGTH", 5.0))
        pure_pursuit = math.atan(wheelbase * curvature)

        heading_error = math.atan2(
            math.sin(float(reference_heading) - float(self.heading)),
            math.cos(float(reference_heading) - float(self.heading)),
        )

        # return pure_pursuit + 0.25 * heading_error
        return pure_pursuit

    def polynomial_planning(
            self,
            index_group,
            car_id,
            controller_vehicles,
            lane_index,
            polynomial_config=None,
    ) -> tuple:
        polynomial_config = dict(polynomial_config or {})

        planner_mode = polynomial_config.get("mode", "legacy")
        aligned_horizon_s = float(
            polynomial_config.get("aligned_horizon_s", 4.0)
        )

        leader_v = LEADVehicle(
            road=self.road,
            position=self.position,
            speed=self.speed,
            target_speed=33,
            timer=self.timer,
            target_lane_index=self.target_lane_index,
        )

        bind_longitudinal_source(leader_v, self)

        vehicle = leader_v
        vehicle_type = "Leader"
        lead_v = None
        follow_idx = None
        group = None

        for index in index_group:
            if isinstance(index, list):
                for i in index:
                    if i == car_id and index.index(i) != 0:
                        vehicle = self
                        vehicle_type = "Follower"
                        lead_v = controller_vehicles[index[0]]
                        follow_idx = index.index(i)

                        group = []
                        for j in index:
                            group.append(controller_vehicles[j])

        poly_planner = PolyPlanner(
            controlled_vehicle=vehicle,
            vehicle_type=vehicle_type,
            road=self.road,
            target_lane_index=lane_index,
            lead_vehicle=lead_v,
            follow_index=follow_idx,
            group=group,
            planner_mode=planner_mode,
            aligned_horizon_s=aligned_horizon_s,
        )

        (
            ref_headings,
            ref_speeds,
            ref_steering_angles,
            x_positions,
            y_positions,
            LTR_values,
        ) = poly_planner.planning()

        path = {
            "x": x_positions,
            "y": y_positions,
            "steering": ref_steering_angles,
            "heading": ref_headings,
            "speed": ref_speeds,
            "time_s": (
                None
                if poly_planner.last_time_s is None
                else np.asarray(
                    poly_planner.last_time_s,
                    dtype=np.float32,
                )
            ),
            "polynomial_mode": planner_mode,
        }

        return ref_steering_angles, path
    def act(
            self,
            lateral_action: Union[dict, str] = None,
            lane_index: LaneIndex = None,
            acceleration: float = 0,
            planner: dict = None,
            index_group: list = None,
            car_id: int = None,
            controlled_vehicles: list = None,
    ) -> None:
        self.follow_road()

        if lateral_action is None:
            return

        last_acc = self.action["acceleration"]
        now_acc = acceleration
        self.jerk = (now_acc - last_acc) / 0.1

        target_lane_index = lane_index

        if self.road.network.get_lane(
            target_lane_index
        ).is_reachable_from(
            self.position
        ):
            self.target_lane_index = target_lane_index

        if planner is not None and planner["state"] is True:
            if planner["type"] == "Polynomial":
                planning_origin_position = np.asarray(
                    self.position,
                    dtype=np.float32,
                ).copy()

                planning_origin_heading = float(self.heading)

                polynomial_config = planner.get(
                    "Polynomial",
                    {},
                )

                ref_steering_angles, path = self.polynomial_planning(
                    index_group=index_group,
                    car_id=car_id,
                    controller_vehicles=controlled_vehicles,
                    lane_index=lane_index,
                    polynomial_config=polynomial_config,
                )

                tracking_index = min(
                    10,
                    len(path["x"]) - 1,
                )

                reference_position = np.asarray(
                    [
                        path["x"][tracking_index],
                        path["y"][tracking_index],
                    ],
                    dtype=float,
                )

                ref_heading = path["heading"][tracking_index]

                action = {
                    "steering": self.trajectory_steering_control(
                        reference_position,
                        ref_heading,
                    ),
                    "acceleration": acceleration,
                }

                self.latest_planner_trajectory = None

                if polynomial_config.get(
                    "mode",
                    "legacy",
                ) == "aligned":
                    from highway_env.planner.trajectory import (
                        PlannerTrajectory,
                    )

                    anchor_config = (
                        planner
                        .get("features", {})
                        .get("anchors", {})
                    )

                    source_time_s = path.get("time_s")

                    if source_time_s is None:
                        raise RuntimeError(
                            "Polynomial aligned path has no explicit time axis"
                        )

                    # Polynomial path values can contain a mixture of
                    # Python scalars and NumPy shape-(1,) values because the
                    # quintic coefficient solve uses a column vector.
                    #
                    # Normalize each point to a scalar ONLY for the external
                    # expert trajectory representation. This does not change
                    # the Polynomial benchmark path/controller behavior.
                    path_x_scalar = np.asarray(
                        [
                            float(
                                np.asarray(value).reshape(-1)[0]
                            )
                            for value in path["x"]
                        ],
                        dtype=np.float32,
                    )

                    path_y_scalar = np.asarray(
                        [
                            float(
                                np.asarray(value).reshape(-1)[0]
                            )
                            for value in path["y"]
                        ],
                        dtype=np.float32,
                    )

                    if path_x_scalar.shape != path_y_scalar.shape:
                        raise RuntimeError(
                            "Polynomial x/y path length mismatch: "
                            f"x={path_x_scalar.shape}, "
                            f"y={path_y_scalar.shape}"
                        )

                    world_xy = np.stack(
                        [
                            path_x_scalar,
                            path_y_scalar,
                        ],
                        axis=-1,
                    )

                    planner_trajectory = PlannerTrajectory.from_world_path(
                        source_time_s=source_time_s,
                        world_xy=world_xy,
                        ego_position_world=planning_origin_position,
                        ego_heading_world=planning_origin_heading,
                        horizon_steps=int(
                            anchor_config.get("horizon_steps", 8)
                        ),
                        trajectory_dt=float(
                            anchor_config.get("trajectory_dt", 0.5)
                        ),
                        source="Polynomial-aligned",
                    )

                    self.latest_planner_trajectory = (
                        planner_trajectory.as_dict()
                    )

            else:
                path = None
                action = {
                    "steering": 0,
                    "acceleration": acceleration,
                }

            self.path = path

        else:
            target_lane = self.road.network.get_lane(
                target_lane_index
            )
            lane_coords = target_lane.local_coordinates(
                self.position
            )
            lateral_speed_command = (
                -self.KP_LATERAL * lane_coords[1]
            )
            heading_command = np.arcsin(
                np.clip(
                    lateral_speed_command
                    / utils.not_zero(self.speed),
                    -1,
                    1,
                )
            )

            action = {
                "steering": self.steering_control(
                    self.target_lane_index,
                    heading_command,
                ),
                "acceleration": acceleration,
            }

        action["steering"] = np.clip(
            action["steering"],
            -self.MAX_STEERING_ANGLE,
            self.MAX_STEERING_ANGLE,
        )

        self.action = action
        self.action_list.append(self.group_action)
        self.env_state_list.append(self.env_state)
        self.info["x"].append(self.position[0])
        self.info["y"].append(self.position[1])
        self.info["v"].append(self.speed)
        self.info["a"].append(self.action["acceleration"])
        self.info["delta"].append(self.action["steering"])
    def steering_pid_control(self, ref_y, ref_heading):
        Ky = 0.05
        Kh = 1.15
        steering_angle = Ky * (ref_y - self.position[1]) + Kh * (ref_heading - self.heading)
        # steering angle增量约束
        if len(self.info["delta"]) > 1:
            steering_angle = np.clip(
                steering_angle,
                self.info["delta"][-1] - self.MAX_DELTA_STEERING_ANGLE,
                self.info["delta"][-1] + self.MAX_DELTA_STEERING_ANGLE,
            )
        # steering angle约束
        steering_angle = np.clip(
            steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        return float(steering_angle)

    def steering_control(self, target_lane_index: LaneIndex, heading_command) -> float:
        """
        Steer the vehicle to follow the center of a given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        heading_ref = lane_future_heading + np.clip(
            heading_command, -np.pi / 4, np.pi / 4
        )
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(
            heading_ref - self.heading
        )
        # Heading rate to steering angle
        slip_angle = np.arcsin(
            np.clip(
                self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                -1,
                1,
            )
        )
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        # steering angle增量约束
        if len(self.info["delta"]) > 1:
            steering_angle = np.clip(
                steering_angle,
                self.info["delta"][-1] - self.MAX_DELTA_STEERING_ANGLE,
                self.info["delta"][-1] + self.MAX_DELTA_STEERING_ANGLE,
            )
        # steering angle约束
        steering_angle = np.clip(
            steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        return float(steering_angle)

    def step(self, dt: float) -> None:
        super().step(dt)
        self.timer += dt
        # self.action_list.append(self.group_action)

    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        return self.target_speeds[index]

    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.target_speeds[0]) / (
            self.target_speeds[-1] - self.target_speeds[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (self.target_speeds.size - 1)),
                0,
                self.target_speeds.size - 1,
            )
        )

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        Assumes a uniform list of target speeds to avoid searching for the closest target speed

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.DEFAULT_TARGET_SPEEDS[0]) / (
            cls.DEFAULT_TARGET_SPEEDS[-1] - cls.DEFAULT_TARGET_SPEEDS[0]
        )
        return np.int64(
            np.clip(
                np.round(x * (cls.DEFAULT_TARGET_SPEEDS.size - 1)),
                0,
                cls.DEFAULT_TARGET_SPEEDS.size - 1,
            )
        )

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(
            vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed)
        )

    def predict_trajectory(
        self,
        actions: List,
        action_duration: float,
        trajectory_timestep: float,
        dt: float,
    ) -> List[ControlledVehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action_num.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action_num
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states

    def change_lane_policy(
        self,
        ref_lane_index: LaneIndex,
        group: list = None,
    ) -> list:
        """Follow the leader's target lane with lane-change commitment."""
        current_lane_index = self.lane_index
        committed_target = self.target_lane_index

        # === LANE CHANGE COMMITMENT V1: follower ===
        if (
            committed_target is not None
            and current_lane_index != committed_target
            and current_lane_index[:2] == committed_target[:2]
        ):
            if current_lane_index[2] > committed_target[2]:
                return [0, committed_target]
            if current_lane_index[2] < committed_target[2]:
                return [1, committed_target]
            return [2, committed_target]

        if self.road.network.get_lane(
            ref_lane_index
        ).is_reachable_from(self.position):
            if np.abs(self.speed) >= 1:
                if self.mobil(
                    lane_index=ref_lane_index,
                    forced=True,
                    group=group,
                ):
                    if current_lane_index[2] > ref_lane_index[2]:
                        return [0, ref_lane_index]
                    if current_lane_index[2] < ref_lane_index[2]:
                        return [1, ref_lane_index]

        return [2, current_lane_index]
    def desired_gap(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = True,
    ) -> list:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau_star = self.TIME_WANTED
        tau_follow = self.TIME_WANTED / 10
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN

        if ego_vehicle is None:
            return [0, 0]
    
        elif front_vehicle is None:
            dv = np.dot(ego_vehicle.velocity, ego_vehicle.direction)

        else:
            dv = (
                np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
                if projected
                else ego_vehicle.speed - front_vehicle.speed
            )
        d_star = (
                d0 + ego_vehicle.speed * tau_star + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        d_follow = (
                d0 + ego_vehicle.speed * tau_follow + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        return [d_star, d_follow]

    def mobil(
        self,
        lane_index: LaneIndex,
        forced: bool = False,
        group: list = None,
    ) -> bool:
        """Follower MOBIL safety check using target-lane Frenet geometry."""
        new_preceding, new_following = self.road.neighbour_vehicles(
            self,
            lane_index,
            group,
        )

        new_following_a = self.acceleration(
            ego_vehicle=new_following,
            front_vehicle=new_preceding,
            desired_gap=self.desired_gap(
                ego_vehicle=new_following,
                front_vehicle=new_preceding,
            )[1],
        )
        new_following_pred_a = self.acceleration(
            ego_vehicle=new_following,
            front_vehicle=self,
            desired_gap=self.desired_gap(
                ego_vehicle=new_following,
                front_vehicle=self,
            )[1],
        )

        old_preceding, old_following = self.road.neighbour_vehicles(
            self
        )
        self_pred_a = self.acceleration(
            ego_vehicle=self,
            front_vehicle=new_preceding,
            desired_gap=self.desired_gap(
                ego_vehicle=self,
                front_vehicle=new_preceding,
            )[1],
        )

        if self.route and self.route[0][2] is not None:
            if np.sign(
                lane_index[2] - self.target_lane_index[2]
            ) != np.sign(
                self.route[0][2] - self.target_lane_index[2]
            ):
                return False

        if self_pred_a < -self.ACC_MAX:
            return False

        if new_preceding is not None:
            front_gap = self.road.longitudinal_gap(
                front_vehicle=new_preceding,
                rear_vehicle=self,
                lane_index=lane_index,
            )
            front_ttc = self.road.longitudinal_ttc(
                front_vehicle=new_preceding,
                rear_vehicle=self,
                lane_index=lane_index,
            )
            if (
                front_gap <= 1.5 * self.LENGTH
                or front_ttc <= self.TTC_MIN
            ):
                return False

        if new_following is not None:
            rear_gap = self.road.longitudinal_gap(
                front_vehicle=self,
                rear_vehicle=new_following,
                lane_index=lane_index,
            )
            rear_ttc = self.road.longitudinal_ttc(
                front_vehicle=self,
                rear_vehicle=new_following,
                lane_index=lane_index,
            )
            if (
                rear_gap <= 1.5 * self.LENGTH
                or rear_ttc <= self.TTC_MIN
            ):
                return False

        if forced:
            return True

        self_a = self.acceleration(
            ego_vehicle=self,
            front_vehicle=old_preceding,
            desired_gap=self.desired_gap(
                ego_vehicle=self,
                front_vehicle=old_preceding,
            )[1],
        )
        old_following_a = self.acceleration(
            ego_vehicle=old_following,
            front_vehicle=self,
            desired_gap=self.desired_gap(
                ego_vehicle=old_following,
                front_vehicle=self,
            )[1],
        )
        old_following_pred_a = self.acceleration(
            ego_vehicle=old_following,
            front_vehicle=old_preceding,
            desired_gap=self.desired_gap(
                ego_vehicle=old_following,
                front_vehicle=old_preceding,
            )[1],
        )
        jerk = (
            self_pred_a
            - self_a
            + self.POLITENESS
            * (
                new_following_pred_a
                - new_following_a
                + old_following_pred_a
                - old_following_a
            )
        )
        if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
            return False

        return True
    def acceleration(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        desired_gap: float = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param desired_gap
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(
                ego_target_speed, 0, ego_vehicle.lane.speed_limit
            )
        acceleration = self.COMFORT_ACC_MAX * (
            1
            - np.power(
                max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)),
                self.DELTA,
            )
        )

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            if desired_gap is None:
                acceleration -= self.COMFORT_ACC_MAX * np.power(
                    self.desired_gap(ego_vehicle, front_vehicle)[0] / utils.not_zero(d), 2
                )
            else:
                acceleration -= self.COMFORT_ACC_MAX * np.power(
                    desired_gap / utils.not_zero(d), 2
                )

        return acceleration

    @publish_longitudinal_result
    def integrated_longitudinal_control(
            self, lead_vehicle: Vehicle = None, follow_index: int = None, group: list = None
    ) -> float:
        """
        Longitudinal: LQR and IDM Switched Control

        :param lead_vehicle: lead vehicle in given group
        :param follow_index: the index number of given vehicle
        :param group: current vehicle group
        :return optimal acceleration
        """
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
            self, self.lane_index
        )

        # Use LQR Controller, if front vehicle is in the same group with current vehicle
        # if front_vehicle in group:
        #     return self.longitudinal_control(front_vehicle, lead_vehicle, follow_index)
        # return self.acceleration(ego_vehicle=self)
        return self.longitudinal_control(front_vehicle, lead_vehicle, follow_index)


    def longitudinal_control(
            self, front_vehicle=None, lead_vehicle=None, follow_index=None,
            reference_distance=10):
        """Track both the group leader and actual preceding vehicle."""
        context = getattr(self.road, "longitudinal_control", None)
        fallback = lambda: self.lqr_control(
            front_vehicle, lead_vehicle, follow_index, reference_distance)
        if context is None or context.kind == "lqr":
            return fallback()
        if lead_vehicle is None or follow_index is None:
            raise ValueError("Follower control requires a group leader and follow_index")
        bounds = (-self.ACC_MAX, self.COMFORT_ACC_MAX)
        leader_acc = context.compute(
            self, lead_vehicle, follow_index * reference_distance,
            lead_vehicle.speed, bounds, fallback)
        if front_vehicle is None:
            return leader_acc
        front_acc = context.compute(
            self, front_vehicle, reference_distance, front_vehicle.speed,
            bounds, fallback)
        return min(leader_acc, front_acc)


    def lqr_control(
            self,
            front_vehicle: Vehicle = None,
            lead_vehicle: Vehicle = None,
            follow_index: int = None,
            reference_distance: float = 10,
    ) -> float:
        """
        LQR Longitudinal Control
        """
        T = 0.05  # Sampling time
        Ts = 0.1  # Inertia time constant

        # System matrix
        A = np.array([[1, T, 0],
                      [0, 1, T],
                      [0, 0, 1 - T / Ts]])

        B = np.array([[0],
                      [0],
                      [T / Ts]])

        # Weight matrix
        Q = np.diag(self.Q)  # d_d, d_v, d_a
        R = np.array(self.R)

        # Solve Riccati function
        P = solve_discrete_are(A, B, Q, R)

        # gain
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        ego_x = self.position[0]
        ego_v = self.speed
        ego_a = self.action["acceleration"]

        lead_x = lead_vehicle.position[0]
        lead_v = lead_vehicle.speed
        lead_a = lead_vehicle.action["acceleration"]

        X_lead = np.array([
            follow_index * reference_distance - (lead_x - ego_x),
            ego_v - lead_v,
            ego_a - lead_a
        ])
        optimal_acceleration_lead = (-K @ X_lead).item()

        if front_vehicle is not None:
            front_x = front_vehicle.position[0]
            front_v = front_vehicle.speed
            front_a = front_vehicle.action["acceleration"]

            # State vector
            X_front = np.array([
                reference_distance - (front_x - ego_x),
                ego_v - front_v,
                ego_a - front_a
            ])
            optimal_acceleration_front = (-K @ X_front).item()
            # Optimal result
            optimal_acceleration = min(optimal_acceleration_front, optimal_acceleration_lead)
        else:
            optimal_acceleration = optimal_acceleration_lead

        return np.clip(
            optimal_acceleration, -self.ACC_MAX, self.COMFORT_ACC_MAX
        )

    def to_dict(
        self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True
    ) -> dict:
        self.update_observation_features_1()
        d = {
            "presence": 1,
            "x": self.position[0],
            "y": self.position[1],
            "vx": self.velocity[0],
            "vy": self.velocity[1],
            "heading": self.heading,
            "cos_h": self.direction[0],
            "sin_h": self.direction[1],
            "cos_d": self.destination_direction[0],
            "sin_d": self.destination_direction[1],
            "long_off": self.lane_offset[0],
            "lat_off": self.lane_offset[1],
            "ang_off": self.lane_offset[2],
            "a": self.action["acceleration"],
            "j": self.jerk,
            "ttc_f": self.ttc_f,
            "ttc_r": self.ttc_r,
            "ttc_lat": self.ttc_lat,  # 当自车/周车换道时，最小横/纵向ttc
            "ttc_lon": self.ttc_lon,
            "ttc": 0,  # *
            "risk_value": 0,  # *
            "risk_gradient": 0,  # *
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ["x", "y", "vx", "vy"]:
                d[key] -= origin_dict[key]
            # Update observation features
            self.update_observation_features_2(origin_vehicle)
            d["ttc"] = self.ttc
            d["risk_value"] = self.risk_value
            d["risk_gradient"] = self.risk_gradient

        return d

    def update_observation_features_1(self):
        """
        计算额外添加的observation值：
        ttc_f, ttc_r, ttc_lat. ttc_lon
        """
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(vehicle=self, lane_index=self.lane_index)
        self.ttc_f = utils.ttc(front_vehicle=front_vehicle,
                               rear_vehicle=self) if front_vehicle is not None else self.TTC_MAX
        self.ttc_r = utils.ttc(front_vehicle=self,
                               rear_vehicle=rear_vehicle) if rear_vehicle is not None else self.TTC_MAX

        self.ttc_lat = self.TTC_MAX
        self.ttc_lon = self.TTC_MAX
        risk_vehicles = [front_vehicle, rear_vehicle]
        for lane_index in self.road.network.side_lanes(self.lane_index):
            front_v, rear_v = self.road.neighbour_vehicles(self, lane_index)
            risk_vehicles = risk_vehicles + [front_v, rear_v]

            # 当周围车道的车与ego车的换道目标车道相同时，计算此时的横纵向ttc：ttc_lat & ttc_lon
            if front_v is not None and front_v.target_lane_index == self.target_lane_index:
                self.ttc_lat = min(utils.ttc_y(self, front_v), self.ttc_lat)
                self.ttc_lon = min(utils.ttc(front_v, self), self.ttc_lon)
            if rear_v is not None and rear_v.target_lane_index == self.target_lane_index:
                self.ttc_lat = min(utils.ttc_y(self, rear_v), self.ttc_lat)
                self.ttc_lon = min(utils.ttc(self, rear_v), self.ttc_lon)
        self.ttc_lat = np.clip(self.ttc_lat, 0, self.TTC_MAX)
        self.ttc_lon = np.clip(self.ttc_lon, 0, self.TTC_MAX)

    def update_observation_features_2(self, origin_vehicle):
        """
        计算额外添加的observation值：
        ttc, risk_value, risk_gradient
        """
        rf = RISK_FIELD()
        risk_value = rf.safety_risk(position=origin_vehicle.position, risk_vehicle=self)
        risk_gradient = risk_value - self.risk_value
        self.risk_value = risk_value
        self.risk_gradient = risk_gradient
        self.max_risk = rf.max_risk_value
        return


class RISK_FIELD:
    G = 4
    R = 1
    M = 1
    K1 = 2
    K2 = 0.1
    R_min = 1.5  # [m]
    V_MAX = 40  # [m/s]
    def __init__(self):
        self.max_risk_value = (self.G * self.R * self.M / self.R_min ** self.K1) ** (self.K2 * self.V_MAX)

    def safety_risk(self, position, risk_vehicle) -> float:
        """
        基于《基于人-车-路协同的行车风险场概念、原理及建模》建立风险场，并计算指定位置的风险值
        """
        if not risk_vehicle:
            return 0

        r = np.linalg.norm(np.array([risk_vehicle.position[0] - position[0], risk_vehicle.position[1] - position[1]]))
        v = risk_vehicle.speed
        gamma = math.atan((risk_vehicle.position[1] - position[1]) / (risk_vehicle.position[0] - position[0]))
        theta = gamma - risk_vehicle.heading

        risk_value = (self.G * self.R * self.M / (r ** self.K1)) ** (self.K2 * np.array(v) * np.cos(theta))

        return min(risk_value, self.max_risk_value)

    def average_safety_risk(self, target_position, risk_vehicles) -> float:
        position = []
        speed = []
        heading = []
        for vehicle in risk_vehicles:
            position.append([vehicle.position[0], vehicle.position[1]])
            speed.append(vehicle.speed)
            heading.append(vehicle.heading)
        position = np.array(position)
        r = np.linalg.norm(position, axis=1)
        gamma = np.arctan((position[:, 1] - target_position[1]) / (position[:, 0] - target_position[0]))
        theta = gamma - np.array(heading)

        risk_values = (self.G * self.R * self.M / (r ** self.K1)) ** (self.K2 * np.array(speed) * np.cos(theta))
        risk_value = sum(risk_values) / len(risk_values)

        return min(risk_value, self.max_risk_value)


    def visualize_risk_field(self, x_range, y_range, risk_vehicle, resolution=50):
        """
        可视化风险场：风险值和梯度分布
        参数:
            x_range, y_range: x 和 y 的取值范围 (min, max)
            resolution: 网格的分辨率
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # 计算网格点上的风险值
        risk_values = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                pos = [X[i, j], Y[i, j]]
                risk_values[i, j] = self.safety_risk(pos, risk_vehicle)

        # 绘制风险值的热力图
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, risk_values, levels=np.linspace(0, 50, 1000), cmap='viridis')
        plt.colorbar(label="Risk Value")
        plt.title("Risk Field Visualization with Risk Value and Gradient")
        plt.show()