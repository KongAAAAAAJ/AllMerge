from typing import List, Optional, Tuple, Union

import numpy as np
import random as rd
from PIL.ImagePalette import random
from docutils.nodes import target
from pygame import NOEVENT
from scipy.signal import ellip

from highway_env import utils
from highway_env.road.road import LaneIndex, Road, Route
from highway_env.utils import Vector
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import RISK_FIELD


class IDMVehicle(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

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

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    TTC_MAX = 1e3
    """risk aware"""

    # Lateral policy parameters
    POLITENESS = 0.0  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: int = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
    ):
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route
        )
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY
        self.set_lane_index = None
        self.jerk = 0
        self.ttc_f = self.TTC_MAX
        self.ttc_r = self.TTC_MAX
        self.ttc_lat = self.TTC_MAX
        self.ttc_lon = self.TTC_MAX
        self.ttc = self.TTC_MAX
        self.risk_value = 0
        self.risk_gradient = 0
        self.max_risk_value = 0

    def randomize_behavior(self):
        self.DELTA = self.road.np_random.uniform(
            low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1]
        )

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
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
            timer=getattr(vehicle, "timer", None),
        )
        return v

    @classmethod
    def create_settled(
            cls,
            road: Road,
            speed: float = None,
            lane_from: Optional[str] = None,
            lane_to: Optional[str] = None,
            lane_id: Optional[int] = None,
            spacing: float = 1,
            lead_x: float = None,
    ) -> "Vehicle":
        """
        Create a fixed vehicle according to lead vehicle on the road.

        The lane, speed and position are chosen firmly.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :param lead_x: x position of front controlled vehicle
        :return: A vehicle with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = (
            lane_id
            if lane_id is not None
            else road.np_random.choice(len(road.network.graph[_from][_to]))
        )
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(
                    0.7 * lane.speed_limit, 0.8 * lane.speed_limit
                )
            else:
                speed = road.np_random.uniform(
                    Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1]
                )
        default_spacing = 12 + 1.0 * speed
        offset = (
                spacing
                * default_spacing
                * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        )
        x0 = (
            np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles])
            if len(road.vehicles)
            else 3 * offset
        )

        x0 = x0 + offset * road.np_random.uniform(0.9, 1.1) if lead_x is None else lead_x - 10
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v

    @classmethod
    def create_random(
            cls,
            road: Road,
            speed: float = None,
            lane_from: Optional[str] = None,
            lane_to: Optional[str] = None,
            lane_id: Optional[int] = None,
            shortest_spacing: float = 1,
            longest_spacing: float = 3,
            controlled_vehicles: list = None,
            lead_x: float = None,
    ) -> "Vehicle":
        """
        Create a random vehicle.

        The lane, speed and position are chosen randomly.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param shortest_spacing: ratio of shortest spacing to the front vehicle, 1 being the default
        :param longest_spacing: ratio of longest spacing to the front vehicle, 3 being the default
        :param controlled_vehicles:
        :param lead_x: x position of front controlled vehicle
        :return: A vehicle with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = (
            lane_id
            if lane_id is not None
            else road.np_random.choice(len(road.network.graph[_from][_to]))
        )
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(
                    0.7 * lane.speed_limit, 0.8 * lane.speed_limit
                )
            else:
                speed = road.np_random.uniform(
                    Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1]
                )
        default_spacing = 12 + 1.0 * speed
        spacing = rd.uniform(shortest_spacing, longest_spacing)
        offset = (
                spacing
                * default_spacing
                * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        )
        # x0 = (
        #     np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles])
        #     if len(road.vehicles)
        #     else 3 * offset
        # )

        # Find the largest x of all vehicles in same lane
        vehicle_list = []
        for v in road.vehicles:
            if v.lane_index[2] == _id:
                vehicle_list.append(v)
        x0 = (
            np.max([lane.local_coordinates(v.position)[0] for v in vehicle_list])
            if vehicle_list
            else np.max([lane.local_coordinates(v.position)[0] for v in controlled_vehicles]) + 10
        )

        x0 = x0 + offset * road.np_random.uniform(0.9, 1.1) if lead_x is None else lead_x - 10
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v

    def act(self, action: Union[dict, str] = None):
        """
        Execute an action_num.

        For now, no action_num is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action_num
        """
        if self.crashed:
            return
        action = {}
        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action["steering"] = self.steering_control(self.target_lane_index)
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )

        # Longitudinal: IDM
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
            self, self.lane_index
        )

        """!!!!计算jerk by kong"""
        last_acceleration = self.action["acceleration"]

        action["acceleration"] = self.acceleration(
            ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
        )
        now_acceleration = action["acceleration"]
        self.jerk = (now_acceleration - last_acceleration) / 0.1

        # When changing lane, check both current and target lanes
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
                self, self.target_lane_index
            )
            target_idm_acceleration = self.acceleration(
                ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
            )
            action["acceleration"] = min(
                action["acceleration"], target_idm_acceleration
            )
        # action_num['acceleration'] = self.recover_from_stop(action_num['acceleration'])
        action["acceleration"] = np.clip(
            action["acceleration"], -self.ACC_MAX, self.ACC_MAX
        )
        Vehicle.act(
            self, action
        )  # Skip ControlledVehicle.act(), or the command will be overriden.

    def step(self, dt: float):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super().step(dt)

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
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
            acceleration -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2
            )
        return acceleration

    def desired_gap(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = True,
    ) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        # d0 = self.DISTANCE_WANTED
        d0 = self.DISTANCE_WANTED / 9
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = (
            np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
            if projected
            else ego_vehicle.speed - front_vehicle.speed
        )
        d_star = (
            d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        return d_star

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        if self.set_lane_index is None:
            # If a lane change is already ongoing
            if self.lane_index != self.target_lane_index:
                # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
                if self.lane_index[:2] == self.target_lane_index[:2]:
                    for v in self.road.vehicles:
                        if (
                            v is not self
                            and v.lane_index != self.target_lane_index
                            and isinstance(v, ControlledVehicle)
                            and v.target_lane_index == self.target_lane_index
                        ):
                            d = self.lane_distance_to(v)
                            d_star = self.desired_gap(self, v)
                            if 0 < d < d_star:
                                self.target_lane_index = self.lane_index
                                break
                return

            # else, at a given frequency,
            if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
                return
            self.timer = 0

            # decide to make a lane change
            for lane_index in self.road.network.side_lanes(self.lane_index):
                # Is the candidate lane close enough?
                if not self.road.network.get_lane(lane_index).is_reachable_from(
                    self.position
                ):
                    continue
                # Only change lane when the vehicle is moving
                if np.abs(self.speed) < 1:
                    continue
                # Does the MOBIL model recommend a lane change?
                if self.mobil(lane_index):
                    self.target_lane_index = lane_index
        else:
            self.target_lane_index = self.set_lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding
        )
        new_following_pred_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=self
        )
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2] is not None:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(
                self.route[0][2] - self.target_lane_index[2]
            ):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=self
            )
            old_following_pred_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=old_preceding
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

        # All clear, let's go!
        return True

    def recover_from_stop(self, acceleration: float) -> float:
        """
        If stopped on the wrong lane, try a reversing maneuver.

        :param acceleration: desired acceleration from IDM
        :return: suggested acceleration to recover from being stuck
        """
        stopped_speed = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.speed < stopped_speed:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(
                self, self.road.network.get_lane(self.target_lane_index)
            )
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and (
                not new_rear or new_rear.lane_distance_to(self) > safe_distance
            ):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2
        return acceleration

    # by kong
    def to_dict(
        self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True
    ) -> dict:
        # Update observation features
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


class AggressiveIDMVehicle(IDMVehicle):
    """
        A vehicle using both a longitudinal and a lateral decision policies.

        - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
        - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
        """

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 5.0  # [m/s2] 3 -> 5
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 3.0 + ControlledVehicle.LENGTH  # [m] 5.0 -> 3.0
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.2  # [s] 1.5 -> 1.2
    """Desired time gap to the front vehicle."""

    DELTA = 3.6  # [] 4.0 -> 3.6
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.1, 4.1]  # [3.5, 4.5] -> [3.1, 4.1]
    """Range of delta when chosen randomly."""

    # Lateral policy parameters
    POLITENESS = 0.0  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: int = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
    ):
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route, enable_lane_change, timer
        )

class LinearVehicle(IDMVehicle):

    """A Vehicle whose longitudinal and lateral controllers are linear with respect to parameters."""

    ACCELERATION_PARAMETERS = [0.3, 0.3, 2.0]
    STEERING_PARAMETERS = [
        ControlledVehicle.KP_HEADING,
        ControlledVehicle.KP_HEADING * ControlledVehicle.KP_LATERAL,
    ]

    ACCELERATION_RANGE = np.array(
        [
            0.5 * np.array(ACCELERATION_PARAMETERS),
            1.5 * np.array(ACCELERATION_PARAMETERS),
        ]
    )
    STEERING_RANGE = np.array(
        [
            np.array(STEERING_PARAMETERS) - np.array([0.07, 1.5]),
            np.array(STEERING_PARAMETERS) + np.array([0.07, 1.5]),
        ]
    )

    TIME_WANTED = 2.5

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: int = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
        data: dict = None,
    ):
        super().__init__(
            road,
            position,
            heading,
            speed,
            target_lane_index,
            target_speed,
            route,
            enable_lane_change,
            timer,
        )
        self.data = data if data is not None else {}
        self.collecting_data = True

    def act(self, action: Union[dict, str] = None):
        if self.collecting_data:
            self.collect_data()
        super().act(action)

    def randomize_behavior(self):
        ua = self.road.np_random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        self.ACCELERATION_PARAMETERS = self.ACCELERATION_RANGE[0] + ua * (
            self.ACCELERATION_RANGE[1] - self.ACCELERATION_RANGE[0]
        )
        ub = self.road.np_random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        self.STEERING_PARAMETERS = self.STEERING_RANGE[0] + ub * (
            self.STEERING_RANGE[1] - self.STEERING_RANGE[0]
        )

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        Compute an acceleration command with a Linear Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - reach the speed of the leading (resp following) vehicle, if it is lower (resp higher) than ego's;
        - maintain a minimum safety distance w.r.t the leading vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            Linear vehicle, which is why this method is a class method. This allows a Linear vehicle to
                            reason about other vehicles behaviors even though they may not Linear.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        return float(
            np.dot(
                self.ACCELERATION_PARAMETERS,
                self.acceleration_features(ego_vehicle, front_vehicle, rear_vehicle),
            )
        )

    def acceleration_features(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> np.ndarray:
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = (
                getattr(ego_vehicle, "target_speed", ego_vehicle.speed)
                - ego_vehicle.speed
            )
            d_safe = (
                self.DISTANCE_WANTED
                + np.maximum(ego_vehicle.speed, 0) * self.TIME_WANTED
            )
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.speed - ego_vehicle.speed, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Linear controller with respect to parameters.

        Overrides the non-linear controller ControlledVehicle.steering_control()

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        return float(
            np.dot(
                np.array(self.STEERING_PARAMETERS),
                self.steering_features(target_lane_index),
            )
        )

    def steering_features(self, target_lane_index: LaneIndex) -> np.ndarray:
        """
        A collection of features used to follow a lane

        :param target_lane_index: index of the lane to follow
        :return: a array of features
        """
        lane = self.road.network.get_lane(target_lane_index)
        lane_coords = lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = lane.heading_at(lane_next_coords)
        features = np.array(
            [
                utils.wrap_to_pi(lane_future_heading - self.heading)
                * self.LENGTH
                / utils.not_zero(self.speed),
                -lane_coords[1] * self.LENGTH / (utils.not_zero(self.speed) ** 2),
            ]
        )
        return features

    def longitudinal_structure(self):
        # Nominal dynamics: integrate speed
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        # Target speed dynamics
        phi0 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
        # Front speed control
        phi1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 1], [0, 0, 0, 0]])
        # Front position control
        phi2 = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [-1, 1, -self.TIME_WANTED, 0], [0, 0, 0, 0]]
        )
        # Disable speed control
        front_vehicle, _ = self.road.neighbour_vehicles(self)
        if not front_vehicle or self.speed < front_vehicle.speed:
            phi1 *= 0

        # Disable front position control
        if front_vehicle:
            d = self.lane_distance_to(front_vehicle)
            if d != self.DISTANCE_WANTED + self.TIME_WANTED * self.speed:
                phi2 *= 0
        else:
            phi2 *= 0

        phi = np.array([phi0, phi1, phi2])
        return A, phi

    def lateral_structure(self):
        A = np.array([[0, 1], [0, 0]])
        phi0 = np.array([[0, 0], [0, -1]])
        phi1 = np.array([[0, 0], [-1, 0]])
        phi = np.array([phi0, phi1])
        return A, phi

    def collect_data(self):
        """Store features and outputs for parameter regression."""
        self.add_features(self.data, self.target_lane_index)

    def add_features(self, data, lane_index, output_lane=None):
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        features = self.acceleration_features(self, front_vehicle, rear_vehicle)
        output = np.dot(self.ACCELERATION_PARAMETERS, features)
        if "longitudinal" not in data:
            data["longitudinal"] = {"features": [], "outputs": []}
        data["longitudinal"]["features"].append(features)
        data["longitudinal"]["outputs"].append(output)

        if output_lane is None:
            output_lane = lane_index
        features = self.steering_features(lane_index)
        out_features = self.steering_features(output_lane)
        output = np.dot(self.STEERING_PARAMETERS, out_features)
        if "lateral" not in data:
            data["lateral"] = {"features": [], "outputs": []}
        data["lateral"]["features"].append(features)
        data["lateral"]["outputs"].append(output)

class AggressiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [
        MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
        MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
        0.5,
    ]

class DefensiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [
        MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
        MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
        2.0,
    ]

class SpecialControlledVehicle(IDMVehicle):

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: int = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
        task: dict = None,
        env = None,
    ):
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route, enable_lane_change, timer
        )
        self.task = task
        self.env = env
        self.color = (250, 71, 84)

    def act(self, action: Union[dict, str] = None):
        """
        Execute a lane change at trigger time.
        """
        if self.crashed:
            return
        action = {}

        """Special mission"""
        if self.task is not None and self.env.time >= self.task["trigger time"]:

            if self.task["type"] == "left lane change":
                target_lane_index = self.target_lane_index
                if self.lane_index[2] == 2:
                    target_lane_index = list(target_lane_index)
                    target_lane_index[2] = 1
                    target_lane_index = tuple(target_lane_index)

                # Is lane change safe?
                front_v, rear_v = self.road.neighbour_vehicles(vehicle=self, lane_index=target_lane_index)
                if front_v is not None and type(front_v).__name__ == "FOLLOWVehicle" and front_v.position[0] - self.position[0] >= 5 \
                    and \
                    rear_v is not None and type(rear_v).__name__ == "FOLLOWVehicle" and self.position[0] - rear_v.position[0] >= 5 \
                    and \
                    utils.ttc(front_vehicle=front_v, rear_vehicle=self) > 1 and utils.ttc(front_vehicle=self, rear_vehicle=rear_v) > 1:
                    self.target_lane_index = target_lane_index
        if self.target_lane_index is None:
            self.target_lane_index = self.lane_index
        action["steering"] = self.steering_control(self.target_lane_index)
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )

        """Longitudinal: IDM"""
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
            self, self.lane_index
        )
        last_acc = self.action["acceleration"]
        action["acceleration"] = self.acceleration(
            ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
        )
        now_acc = action["acceleration"]
        self.jerk = (now_acc - last_acc) / 0.1

        # When changing lane, check both current and target lanes
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
                self, self.target_lane_index
            )
            target_idm_acceleration = self.acceleration(
                ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
            )
            action["acceleration"] = min(
                action["acceleration"], target_idm_acceleration
            )
        # action_num['acceleration'] = self.recover_from_stop(action_num['acceleration'])
        action["acceleration"] = np.clip(
            action["acceleration"], -self.ACC_MAX, self.ACC_MAX
        )

        """Update"""
        Vehicle.act(
            self, action
        )

class DropVehicle(SpecialControlledVehicle):

    def __init__(
            self,
            road: Road,
            position: Vector,
            heading: float = 0,
            speed: float = 0,
            target_lane_index: int = None,
            target_speed: float = None,
            route: Route = None,
            enable_lane_change: bool = True,
            timer: float = None,
            task: dict = None,
            env=None,
    ):
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route, enable_lane_change, timer, task, env
        )
        self.color = (250, 71, 84)

    def act(self, action: Union[dict, str] = None):
        if self.crashed:
            return
        action = {}

        """Special mission"""
        if self.task is not None and self.env.time >= self.task["trigger time"]:
            action["steering"] = 0
            action["acceleration"] = -20 if self.speed > 0 else 0

        """Update"""
        Vehicle.act(
            self, action
        )

class MergeIDMVehicle(IDMVehicle):
    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        Modified MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

            If vehicle in ramp merge area, do forced lane change

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding
        )
        new_following_pred_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=self
        )
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2] is not None:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(
                self.route[0][2] - self.target_lane_index[2]
            ):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            # Is the distance and TTC unsafe?
            TTC_MIN = 0.3
            if new_preceding is not None:
                ttc_new_preceding = utils.ttc(new_preceding, self)
                if new_preceding.position[0] - self.position[0] <= 1.5 * self.LENGTH or ttc_new_preceding <= TTC_MIN:
                    return False
            if new_following is not None:
                ttc_new_following = utils.ttc(self, new_following)
                if self.position[0] - new_following.position[0] <= 1.5 * self.LENGTH or ttc_new_following <= TTC_MIN:
                    return False

            # Is vehicle in merging lane?
            if self.lane_index == ("b", "c", 3):
                return True

            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=self
            )
            old_following_pred_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=old_preceding
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

        # All clear, let's go!
        return True
