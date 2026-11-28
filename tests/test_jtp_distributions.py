import json
import numpy as np
import pytest
import torch
import yaml

from evaluation.ablation.jtp.distributions import collect, DistributionRecorder, draw, KEYS
from evaluation.ablation.jtp.metrics import evaluate_group
from highway_env.planner.diffusion.projection import load_jtp_config, RoadGeometry
from evaluation.ablation.jtp.data import TrafficActor


def test_distribution_roundtrip_and_evaluator_parity(tmp_path):
    cfg = load_jtp_config()
    with open('configs/evaluation/jtp_ablation.yaml', encoding='utf-8') as f:
        ev = yaml.safe_load(f)
    n, h = 2, 4
    u = torch.zeros(n, h, 2)
    u[0, :, 0] = torch.tensor([1., -1., 1., 0.])
    u[0, :, 1] = torch.tensor([.04, -.04, .04, 0.])
    pose = torch.zeros(n, h+1, 3)
    pose[:, :, 0] = torch.arange(h+1)
    pose[1, :, 1] = 2.
    states = torch.zeros(n, h+1, 5)
    states[:, :, 0] = 10.
    states[0, 1:, 2] = 2.
    states[0, 1:, 1] = 3.
    states[0, 1:, 3] = .2
    ltr = torch.zeros(n, h+1)
    ltr[0, 1:] = .9
    context = {'koopman_payload_kg': torch.tensor([5000., 10000.]),
               'koopman_road_mu': torch.tensor([.6, .8]),
               'koopman_control_hist_si': torch.zeros(n, 2, 2)}
    context['koopman_control_hist_si'][0, -1, 0] = -.5
    road = RoadGeometry(torch.tensor([[-20., 0.], [100., 0.]]), 2., 2.)
    rollout = {'trajectory': pose, 'states': states, 'ltr': ltr}
    recorder = DistributionRecorder(cfg, ev)
    for method in ('Baseline', 'Ours'):
        expected = evaluate_group(u, pose, states, ltr, context, [road]*n, [], cfg, ev)
        result = collect(u, rollout, context, [road]*n, [], cfg, ev)
        assert result[0]['acceleration_rate'][0][0] == pytest.approx(-20.)
        assert result[0]['steering_rate'][0][0] == pytest.approx(-.8)
        assert len(result[0]['curvature'][0]) == h-1
        assert len(result[0]['external'][0]) == 0
        assert result[0]['intertruck'][0][0] == pytest.approx(2.-2*1.45)
        assert result[0]['road'][0][0] == pytest.approx(.2, abs=1e-6)
        assert result[1]['yaw'][0].all()
        recorder.add(method, result, 0, 1, 2, [0, 1], expected)
    path = recorder.save(tmp_path)
    with np.load(path, allow_pickle=False) as data:
        assert len(data.files) == 12*2*3
        assert data['Baseline_acceleration_rate_ids'][:, 4].tolist() == [2, 3, 4]*n
        assert metadata_version(path) == 3
        for k in KEYS:
            assert np.array_equal(data[f'Baseline_{k}_ids'], data[f'Ours_{k}_ids'])
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    for k in ('yaw', 'beta', 'roll'):
        assert metadata['constraints'][k]['thresholds'] == []
    draw(path, tmp_path)
    assert len(list(tmp_path.glob('*.png'))) == 2
    assert len(list(tmp_path.glob('*.pdf'))) == 2
    # External gap and its violation flags also match the independent evaluator.
    actor = TrafficActor(pose=pose[1].numpy(), length_m=4., width_m=2., source_ego=-1)
    result = collect(u, rollout, context, [road]*n, [actor], cfg, ev)
    expected = evaluate_group(u, pose, states, ltr, context, [road]*n, [actor], cfg, ev)
    recorder.add('Baseline', result, 0, 1, 3, [0, 1], expected)


def test_missing_data_explains_rerun(tmp_path):
    with pytest.raises(ValueError, match='rerun'):
        draw(tmp_path/'absent.npz', tmp_path)


def metadata_version(path):
    return json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['version']


@pytest.mark.parametrize('h,later', [(1, False), (3, False), (3, True)])
def test_rate_first_step_excluded(h, later, tmp_path):
    from highway_env.planner.diffusion.projection.constraints import control_rate_residual
    cfg = load_jtp_config()
    with open('configs/evaluation/jtp_ablation.yaml', encoding='utf-8') as f:
        ev = yaml.safe_load(f)
    # This fixture deliberately exceeds these limits, independent of user tuning.
    cfg['control']['steering_rate_max_rad_s'] = .12
    ev['limits']['steering_rate_max_rad_s'] = .12
    cfg['control']['acceleration_rate_max_mps3'] = 2.5
    ev['limits']['acceleration_rate_max_mps3'] = 2.5
    u = torch.zeros(1, h, 2)
    if later:
        u[0, 1:, :] = torch.tensor([1., .04])
    pose = torch.zeros(1, h+1, 3)
    pose[0, :, 0] = torch.arange(h+1)
    states = torch.zeros(1, h+1, 5)
    states[..., 0] = 10.
    ltr = torch.zeros(1, h+1)
    context = {'koopman_payload_kg': torch.tensor([5000.]),
               'koopman_road_mu': torch.tensor([.8]),
               'koopman_control_hist_si': torch.tensor([[[2., .08]]])}
    roads = [RoadGeometry(torch.tensor([[-20., 0.], [100., 0.]]), 20., 20.)]
    rollout = {'trajectory': pose, 'states': states, 'ltr': ltr}
    expected = evaluate_group(u, pose, states, ltr, context, roads, [], cfg, ev)
    for key in ('acceleration_rate_violation', 'steering_rate_violation', 'control_rate_violation'):
        assert expected[0][key] == later
    assert expected[0]['joint_feasible'] == (not later)
    if not later:
        assert expected[0]['control_rate_excess_mean_norm'] == 0.
        assert expected[0]['control_rate_excess_max_norm'] == 0.
    # Evaluation ignores historical control, but the unchanged projector does not.
    assert (control_rate_residual(u, context, cfg) > 0).any()
    collected = collect(u, rollout, context, roads, [], cfg, ev)
    recorder = DistributionRecorder(cfg, ev)
    for method in ('Baseline', 'Ours'):
        recorder.add(method, collected, 0, 0, 0, [0], expected)
    path = recorder.save(tmp_path)
    with np.load(path, allow_pickle=False) as data:
        assert data['Ours_acceleration_rate_ids'][:, 4].tolist() == list(range(2, h+1))
    if h == 1:
        draw(path, tmp_path)
