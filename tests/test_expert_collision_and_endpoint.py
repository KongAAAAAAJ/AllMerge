from types import SimpleNamespace

import numpy as np

from evaluation.ablation.anchor_ablation.paper_safety import (
    expert_cr_valid_pose_count, external_collision_free, intertruck_collision_free,
)
from evaluation.ablation.anchor_ablation.runner import _apply_synchronized_intertruck
from highway_env.road.lane import CircularLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.planner import PolyPlanner


def _vehicle_config():
    return {'geometry': {'footprint': {'circle_offsets_m': [-3., -1.5, 0., 1.5, 3.],
                                     'circle_radius_m': 1.45}},
            'execution': {'dt_s': 0.1}, 'body_dimensions_m': [8., 2.6]}


def test_collision_marks_only_involved_trucks():
    paths = [np.zeros((41, 3)) for _ in range(3)]
    paths[1][:, 0] = 1.
    paths[2][:, 0] = 100.
    records = [{'raw_world_poses': path} for path in paths]
    assert intertruck_collision_free(records, _vehicle_config()) == [False, False, True]
    rows = [dict(method='Expert', seed=None, episode_index=1, frame_index=2,
                 dataset_seed=3, ego_index=i, external_collision_free=True,
                 cr_valid_future_steps=40,
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


def test_cr_uses_zero_margin_body_rectangles_including_rotated_vehicles():
    paths = [np.zeros((41, 3)) for _ in range(2)]
    records = [{'raw_world_poses': p} for p in paths]
    for x in (8.2, 8.0):  # circle/margin overlap or boundary contact, not body overlap
        paths[1][:, 0] = x
        assert intertruck_collision_free(records, _vehicle_config()) == [True, True]
    paths[1][:, 0] = 7.9
    assert intertruck_collision_free(records, _vehicle_config()) == [False, False]
    paths[1][:] = [0., 5., np.pi / 2]
    assert intertruck_collision_free(records, _vehicle_config()) == [False, False]
    paths[1][:, 1] = 5.4
    assert intertruck_collision_free(records, _vehicle_config()) == [True, True]

    features = dict(ego_state=np.zeros(14), agent_states=np.zeros((1, 11)),
                    agent_valid_mask=np.ones(1, dtype=bool))
    features['agent_states'][0] = [6.6, 0., 0., 0., 0., 0., 0., 1., 5., 2., 0.]
    assert external_collision_free(paths[0], features, _vehicle_config())
    features['agent_states'][0, 0] = 6.4
    assert not external_collision_free(paths[0], features, _vehicle_config())


def _terminal_features():
    geometry = np.zeros((1, 2, 9))
    geometry[0, :, 0] = [0., 50.]
    geometry[0, :, 7] = 4.
    return dict(ego_state=np.zeros(14), road_lane_geometry=geometry,
                road_lane_point_mask=np.ones((1, 2), dtype=bool),
                road_lane_mask=np.ones(1, dtype=bool),
                road_lane_meta=np.array([[25., 50., 0., 0., 50., 50., 0.]]))


def test_terminal_clipping_does_not_remove_preview_ends_or_normal_stops():
    features = _terminal_features()
    xy = np.column_stack((np.minimum(np.arange(1, 41) * 2.5, 50.), np.zeros(40)))
    assert expert_cr_valid_pose_count(xy, features, .1) == 20
    features['road_lane_meta'][0, 1] = 100.  # end of preview, not actual lane
    assert expert_cr_valid_pose_count(xy, features, .1) == 41
    features['road_lane_meta'][0, 1] = 50.
    xy[:, 0] = np.minimum(np.arange(1, 41) * 2.5, 30.)
    assert expert_cr_valid_pose_count(xy, features, .1) == 41  # stop inside road
    xy[:, 0] = 50.
    xy[0, 0] = 49.95
    assert expert_cr_valid_pose_count(xy, features, .1) == 41  # slow stop at end


def test_shared_cr_window_keeps_early_collisions_and_excludes_invalid_denominator():
    import torch
    from torch.utils.data import DataLoader
    from evaluation.ablation.anchor_ablation.data import collate_anchor_batch
    from evaluation.ablation.anchor_ablation.reporting import summarize
    from evaluation.ablation.anchor_ablation.runner import _cr_frame_windows

    features = {k: torch.as_tensor(v) for k, v in _terminal_features().items()}
    dense = torch.column_stack((torch.arange(1, 41).clamp(max=20) * 2.5, torch.zeros(40)))
    samples = [(features, {'trajectory_dense': dense if ego == 0 else torch.zeros((40, 2))},
                dict(episode_index=1, frame_index=2, ego_index=ego, seed=3)) for ego in range(3)]
    loader = DataLoader(samples, batch_size=1, collate_fn=collate_anchor_batch)
    window = _cr_frame_windows(loader, .1)[(1, 2, 3)]
    assert window['cr_valid_future_steps'] == 19
    assert np.isclose(window['cr_valid_until_s'], 1.9)
    batched = DataLoader(samples, batch_size=3, collate_fn=collate_anchor_batch)
    assert _cr_frame_windows(batched, .1)[(1, 2, 3)] == window

    paths = [np.zeros((41, 3)) for _ in range(3)]
    paths[1][:, 0] = 20.
    paths[1][20:, 0] = 1.  # only overlaps after common cutoff
    paths[2][:, 0] = 100.
    for method in ('A0', 'A1', 'A2', 'Expert'):
        rows = [dict(method=method, seed=0, episode_index=1, frame_index=2,
                     dataset_seed=3, ego_index=i, external_collision_free=True,
                     external_min_ttc_s=float('inf'), _raw_world_poses=path, **window)
                for i, path in enumerate(paths)]
        _apply_synchronized_intertruck(rows, _vehicle_config(), {'ttc_threshold_s': 1.})
        assert [r['collision'] for r in rows] == [False, False, False]
        assert [r['ttc_compliant'] for r in rows] == [False, False, True]  # full TTC horizon retained
    paths[1][10, 0] = 1.
    rows = [dict(method='Expert', seed=None, episode_index=1, frame_index=2,
                 dataset_seed=3, ego_index=i, external_collision_free=True,
                 external_min_ttc_s=float('inf'), _raw_world_poses=path, **window)
            for i, path in enumerate(paths)]
    _apply_synchronized_intertruck(rows, _vehicle_config(), {'ttc_threshold_s': 1.})
    assert [r['collision'] for r in rows] == [True, True, False]
    rows[2].update(collision=None, cr_valid_future_steps=0)
    summary = summarize([], rows, methods=['Expert'])[0]
    assert summary['collision_rate_pct'] == 100.
    assert summary['num_cr_evaluated_runs'] == 2 and summary['num_cr_excluded_runs'] == 1
    for row, path in zip(rows, paths):
        row.update(cr_valid_future_steps=0, external_collision_free=None, _raw_world_poses=path)
    _apply_synchronized_intertruck(rows, _vehicle_config(), {'ttc_threshold_s': 1.})
    summary = summarize([], rows, methods=['Expert'])[0]
    assert summary['collision_rate_pct'] is None and summary['num_cr_excluded_runs'] == 3


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
