import numpy as np
import pytest
import torch

from highway_env.planner.road_area import road_area, road_clearance
from highway_env.planner.anchor_scoring.safety import anchor_safety_cost
from evaluation.ablation.anchor_ablation.paper_safety import (
    drivable_area_compliance, local_to_world_poses, road_geometry_from_features,
)


def lane_features(edges):
    geometry = np.zeros((len(edges), len(edges[0][0]), 9))
    for i, (left, right) in enumerate(edges):
        geometry[i, :, 2:4] = left
        geometry[i, :, 4:6] = right
    return dict(road_lane_geometry=geometry,
                road_lane_point_mask=np.ones(geometry.shape[:2], bool),
                road_lane_mask=np.ones(len(edges), bool))


def strip(x0, x1, y0, y1):
    return ([[x0, y1], [x1, y1]], [[x0, y0], [x1, y0]])


VEHICLE = {'geometry': {'footprint': {
    'circle_offsets_m': [-3., -1.5, 0., 1.5, 3.], 'circle_radius_m': 1.45}}}


def dac(features, xy, pose0=np.zeros(3)):
    poses = np.zeros((len(xy) + 1, 3))
    poses[1:, :2] = xy
    return drivable_area_compliance(local_to_world_poses(poses, pose0),
                                    road_geometry_from_features(features, pose0), VEHICLE)


def test_lane_change_and_internal_line_are_inside_whole_road():
    f = lane_features([strip(-10, 60, -1.75, 1.75), strip(-10, 60, 1.75, 5.25)])
    assert dac(f, [[10, 1.75], [20, 1.75]]) == 100
    xy = np.column_stack((np.linspace(5, 45, 40), np.linspace(0, 3.5, 40)))
    assert dac(f, xy) == 100
    assert dac(f, xy, np.array([123., -40., 1.2])) == 100


def test_footprint_outer_edge_tolerance_and_road_ends():
    f = lane_features([strip(-10, 60, -4, 4)])
    assert dac(f, [[10, 2.55]]) == 100  # circle touches the outer edge
    assert dac(f, [[10, 2.5500005]]) == 100
    assert dac(f, [[10, 2.56]]) == 0  # center remains inside, body outside
    assert dac(f, [[59, 0]]) == 0  # finite road end


def test_curved_variable_width_adjacent_lanes_have_no_internal_boundary():
    x = np.linspace(-20, 70, 91)
    seam = np.column_stack((x, 0.001 * x**2))
    lower = seam - np.column_stack((np.zeros_like(x), 4 + x * .005))
    upper = seam + np.column_stack((np.zeros_like(x), 4 + x * .005))
    f = lane_features([(seam, lower), (upper, seam)])
    assert dac(f, seam[25:60]) == 100


def test_gaps_and_holes_remain_outside():
    disconnected = road_area(lane_features([strip(-10, 10, -4, 4), strip(20, 40, -4, 4)]))
    assert road_clearance(disconnected, [[15, 0]])[0] == -5
    ring = road_area(lane_features([strip(0, 10, 0, 2), strip(0, 10, 8, 10),
                                   strip(0, 2, 2, 8), strip(8, 10, 2, 8)]))
    assert road_clearance(ring, [[5, 5]])[0] == -3
    assert ring.area == 64


def test_anchor_center_cost_uses_raw_union_and_preserves_valid_mask():
    f = lane_features([strip(-10, 60, -1.75, 1.75), strip(-10, 60, 1.75, 5.25)])
    features = {key: torch.as_tensor(value)[None] for key, value in f.items()}
    features.update(ego_state=torch.zeros(1, 14), agent_states=torch.zeros(1, 1, 11),
                    agent_valid_mask=torch.zeros(1, 1, dtype=torch.bool))
    xy = torch.tensor([[[[10., 1.75], [20., 1.75]],
                        [[10., 6.25], [20., 6.25]],
                        [[10., 6.25], [20., 6.25]]]])
    result = anchor_safety_cost(features, xy, torch.tensor([[True, True, False]]))
    torch.testing.assert_close(result['road'], torch.tensor([[0., 1., 0.]]))
    assert result['hard_safe'].tolist() == [[True, False, False]]
    assert result['road_unsafe'].tolist() == [[False, True, False]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_teacher_returns_to_input_device():
    f = lane_features([strip(-10, 60, -4, 4)])
    features = {key: torch.as_tensor(value, device='cuda')[None] for key, value in f.items()}
    features.update(ego_state=torch.zeros(1, 14, device='cuda'),
                    agent_states=torch.zeros(1, 1, 11, device='cuda'),
                    agent_valid_mask=torch.zeros(1, 1, dtype=torch.bool, device='cuda'))
    result = anchor_safety_cost(features, torch.zeros(1, 1, 8, 2, device='cuda'))
    assert result['road'].is_cuda
    assert result['hard_safe'].item()
