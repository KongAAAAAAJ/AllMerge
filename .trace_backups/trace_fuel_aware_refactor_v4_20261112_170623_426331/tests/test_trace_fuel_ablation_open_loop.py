"""TRACE fuel ablation regression tests. Run without TruckSim or real data."""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

if importlib.util.find_spec('gymnasium') is None:
    pkg = types.ModuleType('highway_env')
    pkg.__path__ = [str(Path(__file__).resolve().parents[1] / 'highway_env')]
    sys.modules.setdefault('highway_env', pkg)

from highway_env.vehicle.koopman.models.conditional_model import ConditionalKoopman
from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner
from highway_env.planner.diffusion.common.fuel import FuelParameters
from highway_env.planner.diffusion.evaluation.fuel_ablation import (
    aggregate, sample_metrics, evaluate, parse_checkpoint_specs, _latex_table,
)

SMALL = dict(d_model=32, d_ffn=64, num_heads=4, num_modes=3,
             num_scene_layers=0, num_denoiser_layers=1, dropout=0.0)
CTRL = dict(horizon_steps=40, dt=0.1, inference_steps=2,
            num_train_timesteps=12)
FUEL = dict(empty_mass_kg=8000., rolling_resistance=.007,
            drag_coefficient=.6, frontal_area_m2=9.,
            drivetrain_efficiency=.9, alpha0=.00156,
            alpha1=.000081, alpha2=1e-8)


def test_fuel_distance_progress_ade_definitions():
    params = FuelParameters(**{**FUEL, 'alpha0': .001, 'alpha1': 0., 'alpha2': 0.})
    poses = torch.zeros(2, 41, 3)
    poses[:, :, 0] = torch.arange(41).float() * 2.
    speeds = torch.full((2, 41), 20.)
    truth = poses[:, 1:, :2].clone()
    truth[1, :, 1] = 3.0
    lane = torch.zeros(2, 32, 10)
    lane[:, :, 0] = torch.linspace(-20., 120., 32)
    p = dict(pose=poses, trajectory_dense=poses[:, 1:, :2],
             speeds=speeds, trajectory_mode_idx=torch.tensor([2, 1]))
    metrics = sample_metrics(p, truth,
        {'target_lane_polyline': lane, 'target_point': torch.tensor([[80., 0.], [80., 0.]])},
        torch.zeros(2), params, dt=.1)
    torch.testing.assert_close(metrics['fuel_l'], torch.full((2,), .004))
    torch.testing.assert_close(metrics['fuel_l_per_100km'], torch.full((2,), 5.))
    torch.testing.assert_close(metrics['progress_m'], torch.full((2,), 80.))
    torch.testing.assert_close(metrics['ade_m'], torch.tensor([0., 3.]))
    torch.testing.assert_close(metrics['cumulative_fuel_l'][:, 0], torch.zeros(2))
    torch.testing.assert_close(metrics['cumulative_fuel_l'][:, -1], metrics['fuel_l'])


def test_weighted_fuel_ratio_and_pairwise_statistics():
    rows = [dict(fuel_l=.01, distance_m=100., progress_m=40., ade_m=2., goal_terminal_error_m=1.),
            dict(fuel_l=.09, distance_m=300., progress_m=80., ade_m=4., goal_terminal_error_m=3.)]
    stats = aggregate(rows)
    assert stats['fuel_l_per_100km'] == pytest.approx(25.)
    assert stats['progress_m'] == 60.
    assert stats['ade_m'] == 3.


def test_latex_success_removed_and_parsing():
    metrics = {'F0': dict(fuel_l_per_100km=42., progress_m=80., ade_m=1.3),
               'F3': dict(fuel_l_per_100km=40., progress_m=77., ade_m=1.5)}
    t = _latex_table(metrics, 'F3')
    assert '\\begin{tabular}{@{}lccc@{}}' in t
    assert 'Success' not in t
    assert r'\textbf{F3: Ours}' in t
    assert r'\textbf{40.00}' in t
    paths = parse_checkpoint_specs(['F1=C:\\my folder\\f1.pt', 'F0=f0.pt'])
    assert paths['F0'].name == 'f0.pt'
    assert paths['F1'].name.endswith('f1.pt')
    with pytest.raises(ValueError, match='Duplicate'):
        parse_checkpoint_specs(['F0=f0.pt', 'F0=f0.pt'])


def _k2_checkpoint(path):
    norm = dict(state_mean=[0.] * 5, state_std=[1.] * 5,
                control_mean=[0.] * 2, control_std=[1.] * 2)
    cond = dict(condition_mean=[3500., .6], condition_std=[1000., .1])
    kw = dict(history_steps=5, latent_dim=8, hidden_dim=16)
    model = ConditionalKoopman('K2', condition_hidden=12, **norm, **cond, **kw)
    torch.save(dict(model_type='K2', model_kwargs=kw, normalization=norm,
                    condition_normalization=cond, condition_hidden=12,
                    model_state=model.state_dict()), path)


def _dataset(root):
    shard_dir = root / 'shards'
    split_dir = root / 'splits'
    shard_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    n = 2
    fields = {
        'ego_state': np.zeros((n, 14), dtype=np.float32),
        'agent_states': np.zeros((n, 16, 11), dtype=np.float32),
        'agent_valid_mask': np.zeros((n, 16), dtype=bool),
        'map_polylines': np.zeros((n, 8, 32, 10), dtype=np.float32),
        'map_valid_mask': np.zeros((n, 8), dtype=bool),
        'target_point': np.tile([[80., 0.]], (n, 1)).astype(np.float32),
        'target_lane_polyline': np.zeros((n, 32, 10), dtype=np.float32),
        'coarse_trajectories': np.zeros((n, 3, 8, 2), dtype=np.float32),
        'mode_valid_mask': np.ones((n, 3), dtype=bool),
        'expert_trajectory_xy': np.zeros((n, 8, 2), dtype=np.float32),
        'future_trajectory_dense': np.zeros((n, 40, 2), dtype=np.float32),
        'target_mode': np.zeros(n, dtype=np.int64),
        'target_semantic': np.zeros(n, dtype=np.int64),
        'dense_dt': np.full(n, .1, dtype=np.float32),
        'trajectory_horizon_s': np.full(n, 4., dtype=np.float32),
        'koopman_state_hist_si': np.zeros((n, 5, 5), dtype=np.float32),
        'koopman_control_hist_si': np.zeros((n, 4, 2), dtype=np.float32),
        'koopman_pose0_si': np.zeros((n, 3), dtype=np.float32),
        'koopman_payload_kg': np.full(n, 3500., dtype=np.float32),
        'koopman_road_mu': np.full(n, .6, dtype=np.float32),
        'koopman_ltr0': np.zeros(n, dtype=np.float32),
        'koopman_vx_init_kmh': np.full(n, 72., dtype=np.float32),
    }
    fields['ego_state'][:, 0] = 20.
    fields['koopman_state_hist_si'][:, :, 0] = 20.
    fields['target_lane_polyline'][:, :, 0] = np.linspace(-20., 120., 32)
    fields['future_trajectory_dense'][:, :, 0] = np.arange(1, 41) * 2.
    fields['expert_trajectory_xy'][:, :, 0] = np.arange(1, 9) * 10.
    np.savez(shard_dir / 'shard_000.npz', **fields)
    (split_dir / 'test.txt').write_text('shard_000.npz\n', encoding='utf-8')


def _save_planner(path, k2_path, *, seed, weight):
    torch.manual_seed(seed)
    m = ControlDiffusionPlanner(SMALL, CTRL,
        execution_config=dict(model='koopman_k2', checkpoint=str(k2_path)),
        loss_config=dict(trajectory=dict(type='paper_mse', weight=1.),
                         fuel=dict(weight=weight, parameters=FUEL),
                         progress=dict(weight=.02 if weight else 0., minimum_progress_m=60.)))
    torch.save(dict(model_config=SMALL, control_config=CTRL,
                    execution_config=m.execution_config,
                    loss_config=m.loss_config,
                    koopman_metadata=m.vehicle_model.koopman_metadata,
                    model_state=m.state_dict(), epoch=1), path)


def test_end_to_end_real_k2_npz_checkpoint_pdf_and_table(tmp_path):
    root = tmp_path / 'expert'
    _dataset(root)
    k2 = tmp_path / 'k2.pt'
    _k2_checkpoint(k2)
    cps = {}
    for label, seed, w in [('F0', 0, 0.), ('F1', 2, 50.), ('F2', 3, 100.), ('F3', 4, 200.)]:
        cp = tmp_path / (label + '.pt')
        _save_planner(cp, k2, seed=seed, weight=w)
        cps[label] = cp
    import yaml
    yml = tmp_path / 'fuel_config.yaml'
    yml.write_text(yaml.safe_dump({'output_space': 'control',
                                   'control': {'training': {'dataset_root': str(root)},
                                               'dynamics': CTRL,
                                               'losses': {'fuel': {'parameters': FUEL}}}}), encoding='utf-8')
    out = tmp_path / 'result'
    values = evaluate(cps, yaml_config=yml, dataset_root=root, split='test',
                      num_samples=2, batch_size=1, device='cpu', seed=17,
                      output_dir=out, paper_ours='F3')
    assert sorted(values['summary']) == ['F0', 'F1', 'F2', 'F3']
    assert all(math.isfinite(v['fuel_l_per_100km']) for v in values['summary'].values())
    assert values['summary']['F0']['n_samples'] == 2
    assert values['manifest']['evaluation_type'] == 'open_loop'
    for name in ('metrics_summary.csv', 'metrics_per_sample.csv', 'metrics_summary.json',
                 'evaluation_manifest.json', 'fuel_ablation_table.tex', 'fuel_ablation.pdf',
                 'fuel_ablation.png'):
        assert (out / name).stat().st_size > 50
    assert 'Success' not in (out / 'fuel_ablation_table.tex').read_text(encoding='utf-8')
    import csv
    with (out / 'metrics_per_sample.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert len({(r['shard_name'], r['shard_sample_index']) for r in rows}) == 2


def test_fuel_ablation_rejects_inconsistent_K2(tmp_path):
    # Unit contract: F0 and another run must share exact surrogate weights.
    from highway_env.planner.diffusion.evaluation.fuel_ablation import check_checkpoint_compatibility
    ref = dict(model_config={'x': 1}, control_config={'t': 40},
               k2_signature='abc', mode_selection='goal_endpoint')
    with pytest.raises(ValueError, match='frozen K2'):
        check_checkpoint_compatibility(ref, {**ref, 'k2_signature': 'def'}, 'F3')
