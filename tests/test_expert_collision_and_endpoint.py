from types import SimpleNamespace

import numpy as np

from evaluation.ablation.anchor_ablation.paper_safety import intertruck_collision_free
from evaluation.ablation.anchor_ablation.runner import _apply_synchronized_intertruck
from highway_env.road.lane import CircularLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.planner import PolyPlanner


def _vehicle_config():
    return {'geometry': {'footprint': {'circle_offsets_m': [-3., -1.5, 0., 1.5, 3.],
                                     'circle_radius_m': 1.45}},
            'execution': {'dt_s': 0.1}}


def test_collision_marks_only_involved_trucks():
    paths = [np.zeros((41, 3)) for _ in range(3)]
    paths[1][:, 0] = 1.
    paths[2][:, 0] = 100.
    records = [{'raw_world_poses': path} for path in paths]
    assert intertruck_collision_free(records, _vehicle_config()) == [False, False, True]
    rows = [dict(method='Expert', seed=None, episode_index=1, frame_index=2,
                 dataset_seed=3, ego_index=i, external_collision_free=True,
                 external_min_ttc_s=float('inf'), _raw_world_poses=path)
            for i, path in enumerate(paths)]
    _apply_synchronized_intertruck(rows, _vehicle_config(), {'ttc_threshold_s': 1.0})
    assert [row['collision'] for row in rows] == [True, True, False]
    assert [row['ttc_compliant'] for row in rows] == [False, False, True]
    assert all('_raw_world_poses' not in row for row in rows)


def test_spatially_overlapping_paths_do_not_collide_at_different_times():
    first = np.zeros((41, 3))
    first[:, 0] = np.arange(41)
    second = first.copy()
    second[:, 0] += 15.
    assert intertruck_collision_free([{'raw_world_poses': p} for p in (first, second)],
                                     _vehicle_config()) == [True, True]


def _planner(network, route=()):
    return PolyPlanner(SimpleNamespace(route=list(route)), 'Leader', Road(network=network),
                       ('a', 'b', 0))


def test_terminal_lane_does_not_collapse_separated_expert_paths():
    network = RoadNetwork()
    lane = StraightLane([0., 0.], [10., 0.])
    network.add_lane('a', 'b', lane)
    planner = _planner(network)
    front = planner._frenet_to_world(lane, [8., 10., 12., 14.], np.zeros(4))
    rear = planner._frenet_to_world(lane, [3., 5., 7., 9.], np.zeros(4))
    np.testing.assert_allclose(front[:, 0] - rear[:, 0], 5.)
    np.testing.assert_allclose(front[-1], [14., 0.])
    np.testing.assert_allclose(np.diff(front[:, 0]), 2.)


def test_expert_prediction_follows_connected_curve_then_terminal_tangent():
    network = RoadNetwork()
    lane = StraightLane([0., 0.], [10., 0.])
    curve = CircularLane([10., 10.], 10., -np.pi/2, 0., clockwise=True)
    network.add_lane('a', 'b', lane)
    network.add_lane('b', 'c', curve)
    route = [('a', 'b', 0), ('b', 'c', 0)]
    planner = _planner(network, route)
    result = planner._frenet_to_world(lane, [8., 10., 12., 10. + curve.length + 3.], np.zeros(4))
    np.testing.assert_allclose(result[0], [8., 0.])
    np.testing.assert_allclose(result[1], [10., 0.], atol=1e-10)
    np.testing.assert_allclose(result[2], curve.position(2., 0.))
    np.testing.assert_allclose(result[3], [20., 13.], atol=1e-10)
    assert planner.controlled_vehicle.route == route


def test_expert_prediction_respects_route_at_a_fork():
    network = RoadNetwork()
    lane = StraightLane([0., 0.], [10., 0.])
    network.add_lane('a', 'b', lane)
    network.add_lane('b', 'straight', StraightLane([10., 0.], [30., 0.]))
    network.add_lane('b', 'turn', StraightLane([10., 0.], [10., 20.]))
    planner = _planner(network, [('a', 'b', 0), ('b', 'turn', 0)])
    np.testing.assert_allclose(planner._frenet_to_world(lane, [12.], [0.]), [[10., 2.]])


def test_expert_leader_proxy_keeps_observed_heading_and_dimensions(monkeypatch):
    from highway_env.vehicle import controller

    captured = {}

    def leader(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    class Planner:
        last_time_s = np.array([0., .1])

        def __init__(self, **kwargs):
            pass

        def planning(self):
            return [0., 0.], [20., 20.], [0.], [0., 2.], [0., 0.], [0.]

    monkeypatch.setattr(controller, 'LEADVehicle', leader)
    monkeypatch.setattr(controller, 'PolyPlanner', Planner)
    monkeypatch.setattr(controller, 'bind_longitudinal_source', lambda *args: None)
    source = SimpleNamespace(road=object(), position=np.zeros(2), speed=20., timer=0.,
                             target_lane_index=('a','b',0), heading=.4, LENGTH=8., WIDTH=2.6)
    controller.FOLLOWVehicle.polynomial_planning(source, [], 0, [], ('a','b',0))
    assert captured['heading'] == .4
    assert captured['length'] == 8. and captured['width'] == 2.6
