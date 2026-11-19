from __future__ import annotations

import json
import sys
import types
import importlib.machinery
import numpy as np
import torch

try:
    import gymnasium  # noqa: F401
except ImportError:
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.__spec__ = importlib.machinery.ModuleSpec("gymnasium", loader=None)
    envs = types.ModuleType("gymnasium.envs")
    envs.__spec__ = importlib.machinery.ModuleSpec("gymnasium.envs", loader=None)
    registration = types.ModuleType("gymnasium.envs.registration")
    registration.__spec__ = importlib.machinery.ModuleSpec("gymnasium.envs.registration", loader=None)
    registration.register = lambda *args, **kwargs: None
    envs.registration = registration
    gymnasium.envs = envs
    sys.modules["gymnasium"] = gymnasium
    sys.modules["gymnasium.envs"] = envs
    sys.modules["gymnasium.envs.registration"] = registration

from highway_env.planner.diffusion.common.safety_losses import (
    background_constant_velocity_poses,
    pairwise_vehicle_safety,
    road_boundary_safety_loss,
)
from data_pipeline.raw_expert_dataset import RawExpertDataset


def _straight_poses(batch=1, horizon=40, y=0.0):
    poses = torch.zeros(batch, horizon + 1, 3)
    poses[:, :, 0] = torch.arange(horizon + 1, dtype=torch.float32) * 0.5
    poses[:, :, 1] = y
    return poses


def test_pairwise_obb_collision_ttc_losses_backpropagate_to_ego():
    ego = _straight_poses(y=0.0).requires_grad_()
    other = torch.zeros(1, 1, 41, 3)
    other[:, :, :, 0] = 12.0
    dimensions = torch.tensor([[8.0, 2.6]])
    other_dimensions = torch.tensor([[[8.0, 2.6]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    result = pairwise_vehicle_safety(
        ego, dimensions, other, other_dimensions, valid,
        dt=0.1, collision_margin_m=0.2, ttc_threshold_s=1.0,
        softplus_temperature_m=0.2,
        collision_distance_scale_m=1.0, ttc_distance_scale_m=5.0,
    )
    assert result['collision_loss'].item() > 0.0
    assert result['ttc_loss'].item() > 0.0
    assert result['collision_rate'].item() == 1.0
    (result['collision_loss'] + result['ttc_loss']).backward()
    assert ego.grad is not None
    assert torch.isfinite(ego.grad).all()
    assert ego.grad.abs().sum().item() > 0.0


def test_obb_does_not_report_collision_from_overlapping_circumscribed_circles():
    ego = torch.zeros(1, 3, 3)
    other = torch.zeros(1, 1, 3, 3)
    other[..., 1] = 2.65
    dimensions = torch.tensor([[8.0, 2.6]])
    other_dimensions = torch.tensor([[[8.0, 2.6]]])
    result = pairwise_vehicle_safety(
        ego, dimensions, other, other_dimensions,
        torch.ones(1, 1, dtype=torch.bool),
        dt=0.1, collision_margin_m=0.0, ttc_threshold_s=1.0,
        softplus_temperature_m=0.2,
        collision_distance_scale_m=1.0, ttc_distance_scale_m=5.0,
    )
    assert result['collision_rate'].item() == 0.0
    assert result['collision_loss'].item() == 0.0
    torch.testing.assert_close(result['min_gap'], torch.tensor(0.05), atol=1e-5, rtol=0)


def test_obb_reports_actual_rectangle_overlap():
    ego = torch.zeros(1, 3, 3)
    other = torch.zeros(1, 1, 3, 3)
    other[..., 1] = 2.5
    dimensions = torch.tensor([[8.0, 2.6]])
    other_dimensions = torch.tensor([[[8.0, 2.6]]])
    result = pairwise_vehicle_safety(
        ego, dimensions, other, other_dimensions,
        torch.ones(1, 1, dtype=torch.bool),
        dt=0.1, collision_margin_m=0.0, ttc_threshold_s=1.0,
        softplus_temperature_m=0.2,
        collision_distance_scale_m=1.0, ttc_distance_scale_m=5.0,
    )
    assert result['collision_rate'].item() == 1.0
    assert result['collision_loss'].item() > 0.0

def test_ttc_does_not_flag_parallel_side_by_side_vehicles_without_lateral_closing():
    ego = torch.zeros(1, 5, 3)
    other = torch.zeros(1, 1, 5, 3)
    other[..., 1] = 2.65
    dimensions = torch.tensor([[8.0, 2.6]])
    other_dimensions = torch.tensor([[[8.0, 2.6]]])
    result = pairwise_vehicle_safety(
        ego, dimensions, other, other_dimensions,
        torch.ones(1, 1, dtype=torch.bool),
        dt=0.1, collision_margin_m=0.0, ttc_threshold_s=1.0,
        softplus_temperature_m=0.2,
        collision_distance_scale_m=1.0, ttc_distance_scale_m=5.0,
    )
    assert result['collision_rate'].item() == 0.0
    assert result['ttc_violation_rate'].item() == 0.0
    assert result['ttc_loss'].item() == 0.0


def test_background_prediction_excludes_controlled_trucks():
    agents = torch.zeros(1, 2, 11)
    agents[0, 0, 0:4] = torch.tensor([20.0, 0.0, -5.0, 0.0])
    agents[0, 0, 6:10] = torch.tensor([0.0, 1.0, 8.0, 2.6])
    agents[0, 1, 0:4] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    agents[0, 1, 6:10] = torch.tensor([0.0, 1.0, 8.0, 2.6])
    agents[0, 1, 10] = 1.0
    valid = torch.ones(1, 2, dtype=torch.bool)
    ego = torch.zeros(1, 14)
    ego[0, 0] = 20.0
    poses, dimensions, mask = background_constant_velocity_poses(
        agents, valid, ego, 40, 0.1
    )
    assert poses.shape == (1, 2, 41, 3)
    assert mask.tolist() == [[True, False]]
    torch.testing.assert_close(poses[0, 0, 1, 0], torch.tensor(21.5))


def _road_inputs(batch=1, lanes=1, points=50):
    geometry = torch.zeros(batch, lanes, points, 9)
    geometry[..., 0] = torch.linspace(-10.0, 100.0, points)
    geometry[..., 7] = 3.7
    if lanes > 1:
        for lane in range(lanes):
            geometry[:, lane, :, 1] = lane * 3.7
    point_mask = torch.ones(batch, lanes, points, dtype=torch.bool)
    lane_mask = torch.ones(batch, lanes, dtype=torch.bool)
    connections = torch.eye(lanes, dtype=torch.bool).unsqueeze(0).expand(batch, -1, -1).clone()
    if lanes > 1:
        for lane in range(lanes - 1):
            connections[:, lane, lane + 1] = True
            connections[:, lane + 1, lane] = True
    current_slot = torch.zeros(batch, dtype=torch.long)
    target_slot = current_slot.clone()
    return geometry, point_mask, lane_mask, connections, current_slot, target_slot


def _road_loss(poses, dimensions, road_inputs):
    geometry, point_mask, lane_mask, connections, current_slot, target_slot = road_inputs
    return road_boundary_safety_loss(
        poses, dimensions, geometry, point_mask, lane_mask, connections,
        current_slot, target_slot,
        softplus_temperature_m=0.2,
        distance_scale_m=1.0,
    )


def test_road_boundary_uses_rectangle_not_circumscribed_circle():
    # The truck rectangle fits with 0.17 m clearance, while the old 1.333 m
    # circumscribed circles would exceed the 0.15 m inset by about 0.013 m.
    ego = _straight_poses(y=0.38)
    dimensions = torch.tensor([[8.0, 2.6]])
    result = _road_loss(ego, dimensions, _road_inputs())
    assert result['loss'].item() == 0.0
    assert result['violation_rate'].item() == 0.0


def test_crossing_internal_lane_boundary_is_not_a_road_violation():
    ego = _straight_poses(y=1.85)
    dimensions = torch.tensor([[8.0, 2.6]])
    road_inputs = _road_inputs(lanes=2)
    result = _road_loss(ego, dimensions, road_inputs)
    assert result['loss'].item() == 0.0
    assert result['violation_rate'].item() == 0.0


def test_rectangle_crossing_outer_road_boundary_is_penalized_and_differentiable():
    ego = _straight_poses(y=1.0).requires_grad_()
    dimensions = torch.tensor([[8.0, 2.6]])
    result = _road_loss(ego, dimensions, _road_inputs())
    assert result['loss'].item() > 0.0
    assert result['violation_rate'].item() == 1.0
    result['loss'].backward()
    assert ego.grad is not None
    assert torch.isfinite(ego.grad).all()
    assert ego.grad.abs().sum().item() > 0.0


def test_disconnected_nearby_lane_is_not_part_of_current_drivable_road():
    ego = _straight_poses(y=10.0)
    dimensions = torch.tensor([[8.0, 2.6]])
    road_inputs = list(_road_inputs(lanes=2))
    road_inputs[0][:, 1, :, 1] = 10.0
    road_inputs[3][:, 0, 1] = False
    road_inputs[3][:, 1, 0] = False
    result = _road_loss(ego, dimensions, tuple(road_inputs))
    assert result['loss'].item() > 0.0
    assert result['violation_rate'].item() == 1.0

def _write_raw_dataset(root):
    (root / 'episodes').mkdir(parents=True)
    (root / 'reports').mkdir()
    n = 3
    arrays = {
        'ego_state': np.zeros((n, 14), np.float32),
        'agent_states': np.zeros((n, 4, 11), np.float32),
        'agent_valid_mask': np.zeros((n, 4), bool),
        'map_polylines': np.zeros((n, 1, 8, 10), np.float32),
        'map_valid_mask': np.ones((n, 1), bool),
        'target_point': np.zeros((n, 2), np.float32),
        'target_lane_polyline': np.zeros((n, 8, 10), np.float32),
        'road_lane_geometry': np.zeros((n, 1, 4, 9), np.float32),
        'road_lane_point_mask': np.ones((n, 1, 4), bool),
        'road_lane_mask': np.ones((n, 1), bool),
        'road_current_lane_slot': np.zeros(n, np.int16),
        'road_target_lane_slot': np.zeros(n, np.int16),
        'road_lane_connections': np.ones((n, 1, 1), bool),
        'road_lane_meta': np.zeros((n, 1, 7), np.float32),
        'koopman_state_hist_si': np.zeros((n, 5, 5), np.float32),
        'koopman_control_hist_si': np.zeros((n, 4, 2), np.float32),
        'koopman_pose0_si': np.asarray([[0., 0., 0.], [10., 0., np.pi/2], [20., 0., 0.]], np.float32),
        'koopman_payload_kg': np.zeros(n, np.float32),
        'koopman_road_mu': np.ones(n, np.float32),
        'expert_trajectory_xy': np.zeros((n, 8, 2), np.float32),
        'future_trajectory_dense': np.zeros((n, 40, 2), np.float32),
        'episode_index': np.zeros(n, np.int32),
        'frame_index': np.zeros(n, np.int32),
        'ego_index': np.asarray([0, 1, 2], np.int32),
        'seed': np.zeros(n, np.int32),
    }
    arrays['road_lane_geometry'][..., 0] = np.linspace(-10.0, 100.0, 4)
    arrays['road_lane_geometry'][..., 7] = 3.7
    arrays['future_trajectory_dense'][:, :, 0] = np.arange(1, 41, dtype=np.float32)
    arrays['agent_states'][:, :, 10] = 1.0
    name = 'episode_000000.npz'
    np.savez(root / 'episodes' / name, **arrays)
    manifest = {
        'schema_version': 'trace_expert_raw_k2_v2',
        'complete': True,
        'trajectory': {'dt_s': 0.1},
        'episode_files': [{'name': name, 'episode_index': 0, 'samples': n}],
        'split_ratios': [1.0, 0.0, 0.0],
        'split_seed': 0,
    }
    (root / 'reports' / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def test_dataset_builds_other_truck_expert_poses_in_ego_frame(tmp_path):
    _write_raw_dataset(tmp_path)
    dataset = RawExpertDataset(tmp_path, split='train', include_platoon_reference=True)
    _, targets, metadata = dataset[0]
    reference = targets['platoon_reference_poses']
    assert reference.shape == (2, 41, 3)
    # Truck 1 starts 10 m ahead and turns north; its future local x becomes ego-local +y.
    torch.testing.assert_close(reference[0, 0, :2], torch.tensor([10.0, 0.0]))
    torch.testing.assert_close(reference[0, 1, :2], torch.tensor([10.0, 1.0]), atol=1e-5, rtol=0)
    assert metadata['ego_index'] == 0


def test_dataset_does_not_build_platoon_references_for_evaluation(tmp_path):
    _write_raw_dataset(tmp_path)
    dataset = RawExpertDataset(tmp_path, split='train')
    _, targets, _ = dataset[0]
    assert 'platoon_reference_poses' not in targets
