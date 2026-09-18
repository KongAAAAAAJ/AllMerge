import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from highway_env.road.lane import AbstractLane, LineType, StraightLane, lane_from_config
from highway_env.vehicle.objects import Landmark

if TYPE_CHECKING:
    from highway_env.vehicle import kinematics, objects

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]


class RoadNetwork(object):
    graph: Dict[str, Dict[str, List[AbstractLane]]]

    def __init__(self):
        self.graph = {}

    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:
        """
        A lane is encoded as an edge in the road network.

        :param _from: the node at which the lane starts.
        :param _to: the node at which the lane ends.
        :param AbstractLane lane: the lane geometry.
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        Get the lane geometry corresponding to a given index in the road network.

        :param index: a tuple (origin node, destination node, lane id on the road).
        :return: the corresponding lane geometry.
        """
        _from, _to, _id = index
        if _id is None:
            pass
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        return self.graph[_from][_to][_id]

    def get_closest_lane_index(
        self, position: np.ndarray, heading: Optional[float] = None
    ) -> LaneIndex:
        """
        Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :param heading: a heading angle [rad].
        :return: the index of the closest lane.
        """
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def next_lane(
        self,
        current_index: LaneIndex,
        route: Route = None,
        position: np.ndarray = None,
        np_random: np.random.RandomState = np.random,
    ) -> LaneIndex:
        """
        Get the index of the next lane that should be followed after finishing the current lane.

        - If a plan is available and matches with current lane, follow it.
        - Else, pick next road randomly.
        - If it has the same number of lanes as current road, stay in the same lane.
        - Else, pick next road's closest lane.
        :param current_index: the index of the current target lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished.
        """
        _from, _to, _id = current_index
        next_to = next_id = None
        # Pick next road according to planned route
        if route:
            if (
                route[0][:2] == current_index[:2]
            ):  # We just finished the first step of the route, drop it.
                route.pop(0)
            if (
                route and route[0][0] == _to
            ):  # Next road in route is starting at the end of current road.
                _, next_to, next_id = route[0]
            elif route:
                logger.warning(
                    "Route {} does not start after current road {}.".format(
                        route[0], current_index
                    )
                )

        # Compute current projected (desired) position
        long, lat = self.get_lane(current_index).local_coordinates(position)
        projected_position = self.get_lane(current_index).position(long, lateral=0)
        # If next route is not known
        if not next_to:
            # Pick the one with the closest lane to projected target position
            try:
                lanes_dists = [
                    (
                        next_to,
                        *self.next_lane_given_next_road(
                            _from, _to, _id, next_to, next_id, projected_position
                        ),
                    )
                    for next_to in self.graph[_to].keys()
                ]  # (next_to, next_id, distance)
                next_to, next_id, _ = min(lanes_dists, key=lambda x: x[-1])
            except KeyError:
                return current_index
        else:
            # If it is known, follow it and get the closest lane
            next_id, _ = self.next_lane_given_next_road(
                _from, _to, _id, next_to, next_id, projected_position
            )
        return _to, next_to, next_id

    def next_lane_given_next_road(
        self,
        _from: str,
        _to: str,
        _id: int,
        next_to: str,
        next_id: int,
        position: np.ndarray,
    ) -> Tuple[int, float]:
        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            if next_id is None:
                next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(
                lanes, key=lambda l: self.get_lane((_to, next_to, l)).distance(position)
            )
        return next_id, self.get_lane((_to, next_to, next_id)).distance(position)

    def bfs_paths(self, start: str, goal: str) -> List[List[str]]:
        """
        Breadth-first search of all routes from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: list of paths from start to goal.
        """
        queue = [(start, [start])]
        while queue:
            (node, path) = queue.pop(0)
            if node not in self.graph:
                yield []
            for _next in sorted(
                [key for key in self.graph[node].keys() if key not in path]
            ):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    def shortest_path(self, start: str, goal: str) -> List[str]:
        """
        Breadth-first search of shortest path from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest path from start to goal.
        """
        return next(self.bfs_paths(start, goal), [])

    def all_side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
        :param lane_index: the index of a lane.
        :return: all lanes belonging to the same road.
        """
        return [
            (lane_index[0], lane_index[1], i)
            for i in range(len(self.graph[lane_index[0]][lane_index[1]]))
        ]

    def side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        """
        :param lane_index: the index of a lane.
        :return: indexes of lanes next to a an input lane, to its right or left.
        """
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes

    @staticmethod
    def is_same_road(
        lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False
    ) -> bool:
        """Is lane 1 in the same road as lane 2?"""
        return lane_index_1[:2] == lane_index_2[:2] and (
            not same_lane or lane_index_1[2] == lane_index_2[2]
        )

    @staticmethod
    def is_leading_to_road(
        lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False
    ) -> bool:
        """Is lane 1 leading to of lane 2?"""
        return lane_index_1[1] == lane_index_2[0] and (
            not same_lane or lane_index_1[2] == lane_index_2[2]
        )

    def is_connected_road(
        self,
        lane_index_1: LaneIndex,
        lane_index_2: LaneIndex,
        route: Route = None,
        same_lane: bool = False,
        depth: int = 0,
    ) -> bool:
        """
        Is the lane 2 leading to a road within lane 1's route?

        Vehicles on these lanes must be considered for collisions.
        :param lane_index_1: origin lane
        :param lane_index_2: target lane
        :param route: route from origin lane, if any
        :param same_lane: compare lane id
        :param depth: search depth from lane 1 along its route
        :return: whether the roads are connected
        """
        if RoadNetwork.is_same_road(
            lane_index_2, lane_index_1, same_lane
        ) or RoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                # Route is starting at current road, skip it
                return self.is_connected_road(
                    lane_index_1, lane_index_2, route[1:], same_lane, depth
                )
            elif route and route[0][0] == lane_index_1[1]:
                # Route is continuing from current road, follow it
                return self.is_connected_road(
                    route[0], lane_index_2, route[1:], same_lane, depth - 1
                )
            else:
                # Recursively search all roads at intersection
                _from, _to, _id = lane_index_1
                return any(
                    [
                        self.is_connected_road(
                            (_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1
                        )
                        for l1_to in self.graph.get(_to, {}).keys()
                    ]
                )
        return False

    def lanes_list(self) -> List[AbstractLane]:
        return [
            lane for to in self.graph.values() for ids in to.values() for lane in ids
        ]

    def lanes_dict(self) -> Dict[str, AbstractLane]:
        return {
            (from_, to_, i): lane
            for from_, tos in self.graph.items()
            for to_, ids in tos.items()
            for i, lane in enumerate(ids)
        }

    @staticmethod
    def straight_road_network(
        lanes: int = 4,
        start: float = 0,
        length: float = 10000,
        angle: float = 0,
        speed_limit: float = 30,
        nodes_str: Optional[Tuple[str, str]] = None,
        net: Optional["RoadNetwork"] = None,
    ) -> "RoadNetwork":
        net = net or RoadNetwork()
        nodes_str = nodes_str or ("0", "1")
        for lane in range(lanes):
            origin = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([start + length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array(
                [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
            )
            origin = rotation @ origin
            end = rotation @ end
            line_types = [
                LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE,
            ]
            net.add_lane(
                *nodes_str,
                StraightLane(
                    origin, end, line_types=line_types, speed_limit=speed_limit
                )
            )
        return net

    def position_heading_along_route(
        self,
        route: Route,
        longitudinal: float,
        lateral: float,
        current_lane_index: LaneIndex,
    ) -> Tuple[np.ndarray, float]:
        """
        Get the absolute position and heading along a route composed of several lanes at some local coordinates.

        :param route: a planned route, list of lane indexes
        :param longitudinal: longitudinal position
        :param lateral: : lateral position
        :param current_lane_index: current lane index of the vehicle
        :return: position, heading
        """

        def _get_route_head_with_id(route_):
            lane_index_ = route_[0]
            if lane_index_[2] is None:
                # We know which road segment will be followed by the vehicle, but not which lane.
                # Hypothesis: the vehicle will keep the same lane_id as the current one.
                id_ = (
                    current_lane_index[2]
                    if current_lane_index[2]
                    < len(self.graph[current_lane_index[0]][current_lane_index[1]])
                    else 0
                )
                lane_index_ = (lane_index_[0], lane_index_[1], id_)
            return lane_index_

        lane_index = _get_route_head_with_id(route)
        while len(route) > 1 and longitudinal > self.get_lane(lane_index).length:
            longitudinal -= self.get_lane(lane_index).length
            route = route[1:]
            lane_index = _get_route_head_with_id(route)

        return self.get_lane(lane_index).position(longitudinal, lateral), self.get_lane(
            lane_index
        ).heading_at(longitudinal)

    def random_lane_index(self, np_random: np.random.RandomState) -> LaneIndex:
        _from = np_random.choice(list(self.graph.keys()))
        _to = np_random.choice(list(self.graph[_from].keys()))
        _id = np_random.integers(len(self.graph[_from][_to]))
        return _from, _to, _id

    @classmethod
    def from_config(cls, config: dict) -> None:
        net = cls()
        for _from, to_dict in config.items():
            net.graph[_from] = {}
            for _to, lanes_dict in to_dict.items():
                net.graph[_from][_to] = []
                for lane_dict in lanes_dict:
                    net.graph[_from][_to].append(lane_from_config(lane_dict))
        return net

    def to_config(self) -> dict:
        graph_dict = {}
        for _from, to_dict in self.graph.items():
            graph_dict[_from] = {}
            for _to, lanes in to_dict.items():
                graph_dict[_from][_to] = []
                for lane in lanes:
                    graph_dict[_from][_to].append(lane.to_config())
        return graph_dict


class Road(object):

    """A road is a set of lanes, and a set of vehicles driving on these lanes."""

    def __init__(
        self,
        network: RoadNetwork = None,
        vehicles: List["kinematics.Vehicle"] = None,
        road_objects: List["objects.RoadObject"] = None,
        np_random: np.random.RandomState = None,
        record_history: bool = False,
        show_future_trajectory: bool = False,
    ) -> None:
        """
        New road.

        :param network: the road network describing the lanes
        :param vehicles: the vehicles driving on the road
        :param road_objects: the objects on the road including obstacles and landmarks
        :param np.random.RandomState np_random: a random number generator for vehicle behaviour
        :param record_history: whether the recent trajectories of vehicles should be recorded for display
        """
        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history
        self.show_future_trajectory = show_future_trajectory

    def close_objects_to(
        self,
        vehicle: "kinematics.Vehicle",
        distance: float,
        count: Optional[int] = None,
        see_behind: bool = True,
        sort: bool = True,
        vehicles_only: bool = False,
    ) -> object:
        vehicles = [
            v
            for v in self.vehicles
            if np.linalg.norm(v.position - vehicle.position) < distance
            and v is not vehicle
            and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))
        ]
        obstacles = [
            o
            for o in self.objects
            if np.linalg.norm(o.position - vehicle.position) < distance
            and -2 * vehicle.LENGTH < vehicle.lane_distance_to(o)
        ]

        objects_ = vehicles if vehicles_only else vehicles + obstacles

        if sort:
            objects_ = sorted(objects_, key=lambda o: abs(vehicle.lane_distance_to(o)))
        if count:
            objects_ = objects_[:count]
        return objects_

    def close_vehicles_to(
        self,
        vehicle: "kinematics.Vehicle",
        distance: float,
        count: Optional[int] = None,
        see_behind: bool = True,
        sort: bool = True,
    ) -> object:
        return self.close_objects_to(
            vehicle, distance, count, see_behind, sort, vehicles_only=True
        )

    def act(self) -> None:
        """Decide the actions of each entity on the road."""
        for vehicle in self.vehicles:
            vehicle.act()

    def step(self, dt: float) -> None:
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        for vehicle in self.vehicles:
            vehicle.step(dt)
        for i, vehicle in enumerate(self.vehicles):
            for other in self.vehicles[i + 1 :]:
                vehicle.handle_collisions(other, dt)
            for other in self.objects:
                vehicle.handle_collisions(other, dt)

    # === FRENET LANE SAFETY V1 START ===
    def lane_longitudinal_position(
        self,
        vehicle: "kinematics.Vehicle",
        lane_index: LaneIndex,
    ) -> float:
        """Project a vehicle onto ``lane_index`` and return lane-local s."""
        lane = self.network.get_lane(lane_index)
        s, _ = lane.local_coordinates(vehicle.position)
        return float(s)

    def longitudinal_gap(
        self,
        front_vehicle: "kinematics.Vehicle",
        rear_vehicle: "kinematics.Vehicle",
        lane_index: LaneIndex,
    ) -> float:
        """Signed front-to-rear longitudinal gap in lane-local Frenet s."""
        if front_vehicle is None or rear_vehicle is None:
            return float("inf")
        lane = self.network.get_lane(lane_index)
        front_s, _ = lane.local_coordinates(front_vehicle.position)
        rear_s, _ = lane.local_coordinates(rear_vehicle.position)
        return float(front_s - rear_s)

    def longitudinal_ttc(
        self,
        front_vehicle: "kinematics.Vehicle",
        rear_vehicle: "kinematics.Vehicle",
        lane_index: LaneIndex,
    ) -> float:
        """Longitudinal TTC computed on the queried lane's Frenet s axis."""
        if front_vehicle is None or rear_vehicle is None:
            return float("inf")

        closing_speed = float(rear_vehicle.speed - front_vehicle.speed)
        if closing_speed <= 0.0:
            return float("inf")

        gap = self.longitudinal_gap(
            front_vehicle=front_vehicle,
            rear_vehicle=rear_vehicle,
            lane_index=lane_index,
        )
        if gap <= 0.0:
            return 0.0

        return float(gap / closing_speed)

    def neighbour_vehicles(
        self,
        vehicle: "kinematics.Vehicle",
        lane_index: LaneIndex = None,
        group: list = None,
    ) -> Tuple[
        Optional["kinematics.Vehicle"],
        Optional["kinematics.Vehicle"],
    ]:
        """Find nearest front/rear objects using lane-local Frenet coordinates.

        The queried vehicle and all vehicles in ``group`` are excluded. This
        works for straight, curved and ramp lanes, including lane id 3.
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None

        lane = self.network.get_lane(lane_index)
        s, _ = lane.local_coordinates(vehicle.position)

        excluded = list(group or [])
        front_s = rear_s = None
        front_vehicle = rear_vehicle = None

        for candidate in self.vehicles + self.objects:
            if candidate is vehicle or candidate in excluded:
                continue

            candidate_s, candidate_lat = lane.local_coordinates(
                candidate.position
            )
            if not lane.on_lane(
                candidate.position,
                candidate_s,
                candidate_lat,
                margin=1,
            ):
                continue

            if s < candidate_s and (
                front_s is None or candidate_s < front_s
            ):
                front_s = candidate_s
                front_vehicle = candidate

            if candidate_s < s and (
                rear_s is None or candidate_s > rear_s
            ):
                rear_s = candidate_s
                rear_vehicle = candidate

        return front_vehicle, rear_vehicle
    # === FRENET LANE SAFETY V1 END ===
    def predict_neighbour_vehicles(
        self,
        vehicle: "kinematics.Vehicle",
        lane_index: LaneIndex = None,
        group: list = None,
        predict_vehicles: list = None,
        background_vehicles: list = None,
        pre_background_vehicles: list = None,
    ) -> Tuple[
        Optional["kinematics.Vehicle"],
        Optional["kinematics.Vehicle"],
    ]:
        """Prediction variant of neighbour lookup using lane-local Frenet s."""
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None

        lane = self.network.get_lane(lane_index)
        s, _ = lane.local_coordinates(vehicle.position)

        background_vehicles = list(background_vehicles or [])
        pre_background_vehicles = list(pre_background_vehicles or [])
        predict_vehicles = list(predict_vehicles or [])
        excluded = list(group or [])

        candidates = background_vehicles + self.objects + predict_vehicles
        previous_candidates = (
            pre_background_vehicles + self.objects + predict_vehicles
        )

        front_s = rear_s = None
        front_vehicle = rear_vehicle = None

        for candidate, previous_candidate in zip(
            candidates,
            previous_candidates,
        ):
            if candidate is vehicle or candidate in excluded:
                continue

            candidate_s, candidate_lat = lane.local_coordinates(
                candidate.position
            )
            if not lane.on_lane(
                candidate.position,
                candidate_s,
                candidate_lat,
                margin=1,
            ):
                continue

            if s < candidate_s and (
                front_s is None or candidate_s < front_s
            ):
                front_s = candidate_s
                front_vehicle = previous_candidate

            if candidate_s < s and (
                rear_s is None or candidate_s > rear_s
            ):
                rear_s = candidate_s
                rear_vehicle = previous_candidate

        return front_vehicle, rear_vehicle
    def predict_near_vehicles(self, first_vehicle, end_vehicle, lane_index, predict_vehicles):

        x_limit = [end_vehicle.position[0], first_vehicle.position[0]]
        near_vehicles = []
        back_vehicles = []

        for v in self.vehicles:
            if type(v).__name__ == 'IDMVehicle':
                back_vehicles.append(v)
        for v in back_vehicles + self.objects + predict_vehicles:
            if v.lane_index == lane_index and x_limit[0] <= v.position[0] <= x_limit[1]:
                near_vehicles.append(v)

        return near_vehicles

    # Kong add:
    def group_neighbour_vehicles(
        self,
        group: list,
        lane_index: LaneIndex = None,
    ) -> Tuple[Optional["list"], Optional["list"]]:
        """Find front/rear neighbours for every member of a vehicle group.

        Each member is projected onto the same queried target lane when one is
        supplied. Group members are excluded from one another's neighbour set.
        """
        front_vehicles = []
        rear_vehicles = []

        for current_vehicle in group:
            query_lane_index = lane_index or current_vehicle.lane_index
            if not query_lane_index:
                front_vehicles.append(None)
                rear_vehicles.append(None)
                continue

            front_vehicle, rear_vehicle = self.neighbour_vehicles(
                vehicle=current_vehicle,
                lane_index=query_lane_index,
                group=group,
            )
            front_vehicles.append(front_vehicle)
            rear_vehicles.append(rear_vehicle)

        return front_vehicles, rear_vehicles
    def __repr__(self):
        return self.vehicles.__repr__()

