"""Regression coverage for final Fuel-Aware evaluator (CPU / synthetic K2)."""
from __future__ import annotations
import importlib.util
import sys
import types
import math
from pathlib import Path
import numpy as np
import pytest
import torch

if importlib.util.find_spec('gymnasium') is None:
    pkg = types.ModuleType('highway_env')
    pkg.__path__ = [str(Path(__file__).resolve().parents[1] / 'highway_env')]
    sys.modules.setdefault('highway_env', pkg)

from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.vehicle.koopman.models.conditional_model import ConditionalKoopman
from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner
from highway_env.planner.diffusion.common.fuel import FuelParameters
from evaluation.ablation.fuel_aware.validation import check_checkpoint_compatibility
from evaluation.ablation.fuel_aware.metrics import sample_metrics,evaluate_sample_metrics
from evaluation.ablation.fuel_aware.statistics import (parse_multi_checkpoint_specs,
    parse_seeds,summarize_records,summarize_multiseed)
from evaluation.ablation.fuel_aware.sampling import (parse_payloads,
    select_payload_samples,select_figure_samples)
from evaluation.ablation.fuel_aware.reporting import latex_payload_table,_normal_sd_band
from evaluation.ablation.fuel_aware.runner import eval_payload_ablation

SMALL = dict(d_model=32, d_ffn=64, num_heads=4, num_modes=3,
             num_scene_layers=0, num_denoiser_layers=1, dropout=0.0)
CTRL = dict(horizon_steps=40, dt=0.1, inference_steps=2,
            num_train_timesteps=12)
FUEL = dict(empty_mass_kg=8000., rolling_resistance=.007,
            drag_coefficient=.6, frontal_area_m2=9.,
            drivetrain_efficiency=.9, alpha0=.00156,
            alpha1=.000081, alpha2=1e-8)


def _fixtures():
    return sys.modules[__name__]

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
    shard_dir = root / 'episodes'
    split_dir = root / 'reports'
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
    from tests.test_a1_online_runtime import _live_geometry
    fields.update(_live_geometry(batch=n))
    fields['episode_index'] = np.zeros(n, dtype=np.int32)
    fields['frame_index'] = np.arange(4,4+n)
    fields['ego_index'] = np.zeros(n, dtype=np.int32)
    fields['seed'] = np.zeros(n, dtype=np.int32)
    for key in ('coarse_trajectories','mode_valid_mask','target_mode','target_semantic'):
        fields.pop(key)
    np.savez(shard_dir / 'episode_000000.npz', **fields)
    import json
    (split_dir / 'manifest.json').write_text(json.dumps(dict(
        schema_version='trace_expert_raw_k2_v2', complete=True,
        trajectory={'dt_s': .1}, split_ratios=[0,0,1], split_seed=0,
        episode_files=[dict(name='episode_000000.npz', episode_index=0, samples=n)])))



def _save_planner(path, k2_path, *, seed, weight):
    torch.manual_seed(seed)
    m = ControlDiffusionPlanner(SMALL, CTRL,
        execution_config=dict(model='koopman_k2', checkpoint=str(k2_path)),
        loss_config=dict(trajectory=dict(type='paper_mse', weight=1.),
                         fuel=dict(weight=weight, parameters=FUEL),
                         progress=dict(weight=.02 if weight else 0., minimum_progress_m=60.)))
    torch.save(dict(model_config=SMALL, control_config=CTRL,
                    execution_config=m.execution_config,
                    loss_config=m.loss_config, anchors_config=m.anchors_config,
                    koopman_metadata=m.vehicle_model.koopman_metadata,
                    model_state=m.state_dict(), epoch=1), path)

def test_seed_parse_and_balanced_checkpoints():
    parsed = parse_multi_checkpoint_specs(['F0@0=tmp/f0_0.pt', 'F1@0=tmp/f1_0.pt',
                                           'F0@1=tmp/f0_1.pt', 'F1@1=tmp/f1_1.pt'])
    assert ('F1', '1') in parsed
    assert parse_seeds('0,1,2') == [0, 1, 2]
    with pytest.raises(ValueError, match='Duplicate'):
        parse_multi_checkpoint_specs(['F0=a.pt', 'F0=b.pt'])
    with pytest.raises(ValueError, match='Unpaired'):
        parse_multi_checkpoint_specs(['F0@0=a.pt', 'F0@1=b.pt', 'F1@0=c.pt'])
    with pytest.raises(ValueError, match='unique'):
        parse_seeds('1,1')


def _metric_row(fuel, distance, progress, ade, fde):
    return dict(fuel_l=fuel, distance_m=distance, progress_m=progress,
        pvr=float(progress < 60), ade_m=ade, fde_m=fde,
        goal_terminal_error_m=1., speed_mae_proxy_kmh=0.5,
        rms_longitudinal_jerk_m_s3=0.1)


def test_metrics_pooled_fuel_not_mean_of_ratios_and_std_basis():
    rows = [_metric_row(.01, 100., 50., 2., 3.),
            _metric_row(.09, 300., 80., 4., 5.)]
    x = summarize_records(rows)
    assert x['fuel_l_per_100km'] == pytest.approx(25.)
    assert x['fuel_l_per_4s'] == pytest.approx(.05)
    assert x['pvr_pct'] == pytest.approx(50.)
    assert x['fde_m'] == pytest.approx(4.)
    one_train = {'F0': {'0': {11: rows, 12: [rows[0]]}}}
    summary, details = summarize_multiseed(one_train)
    assert len(details) == 2
    assert summary['F0']['std_basis'] == 'inference_seeds'
    assert summary['F0']['fuel_l_per_100km_std'] > 0
    two_train = {'F0': {'0': {11: rows, 12: rows},
                        '1': {11: [rows[0]], 12: [rows[0]]}}}
    summary, details = summarize_multiseed(two_train)
    assert summary['F0']['std_basis'] == 'training_seeds'
    assert summary['F0']['fuel_l_per_100km_std_training'] > 0
    assert summary['F0']['fuel_l_per_100km_std_inference'] == 0

def _payload_dataset(tmp_path):
    fx = _fixtures()
    root = tmp_path/'dataset'
    fx._dataset(root)
    file = root/'episodes'/'episode_000000.npz'
    with np.load(file, allow_pickle=False) as z:
        values = {k:np.asarray(z[k]).copy() for k in z.files}
    # Non-uniform and non-sorted ordering: first few samples contain only
    # medium payload, so taking first N would be biased/invalid.
    loads = [3500.,3500.,3500.,0.,7000.,0.,7000.,0.,7000.]
    new={}
    for key,val in values.items():
        new[key] = np.stack([val[i % len(val)] for i in range(len(loads))])
    new['koopman_payload_kg'] = np.array(loads,dtype=np.float32)
    np.savez(file,**new)
    import json
    mp=root/'reports/manifest.json'
    manifest=json.loads(mp.read_text())
    manifest['episode_files'][0]['samples']=len(loads)
    mp.write_text(json.dumps(manifest))
    return fx, root


def test_stratified_selection_exact_random_and_missing_groups(tmp_path):
    _, root = _payload_dataset(tmp_path)
    source = RawExpertDataset(root, split='test')
    subset, groups, available = select_payload_samples(source,[0.,3500.,7000.],max_samples=6,sample_seed=81)
    assert available == {0.:3,3500.:3,7000.:3}
    assert [len(groups[x]) for x in [0.,3500.,7000.]] == [2,2,2]
    assert len(subset)==6
    fig = select_figure_samples(groups,[0.,3500.,7000.],figure_seed=18)
    assert set(fig)=={0.,3500.,7000.}
    for p,idx in fig.items():
        _, t, _=subset[idx]
        assert float(t['koopman_payload_kg'])==p
    _, groups2, _ = select_payload_samples(source,[0.,3500.,7000.],max_samples=6,sample_seed=81)
    assert groups==groups2
    with pytest.raises(ValueError,match='No samples'):
        select_payload_samples(source,[0.,3500.,8000.])
    with pytest.raises(ValueError,match='three'):
        parse_payloads('0,3500')


def test_normal_band_geometry():
    # Constant longitudinal path, variability is only in lateral Y.
    x = np.linspace(0,80,41)
    paths = np.stack([np.c_[x, np.full_like(x,v)] for v in [-2.,0.,2.]])
    mid, top, bottom = _normal_sd_band(paths)
    np.testing.assert_allclose(mid[:,1],0.,atol=1e-7)
    np.testing.assert_allclose(top[:,0],mid[:,0],atol=1e-7)
    np.testing.assert_allclose(bottom[:,0],mid[:,0],atol=1e-7)
    np.testing.assert_allclose(top[:,1],2.,atol=1e-6)
    np.testing.assert_allclose(bottom[:,1],-2.,atol=1e-6)


def test_paper_table_six_rows_only_five_mean_metrics():
    metrics=['fuel_l_per_100km','fuel_l_per_4s','progress_m','ade_m','fde_m']
    rows=[dict(payload_kg=p,method=method, **{x:10. for x in metrics},
               **{x+'_std':1. for x in metrics})
          for p in [0.,3500.,7000.] for method in ['F0','F3']]
    t=latex_payload_table(rows,[0.,3500.,7000.],'F3')
    assert t.count('F0: Imitation')==3 and t.count('F3: Fuel-aware')==3
    assert 'PVR' not in t and 'Success' not in t and 'pm' not in t
    assert r'\begin{tabular}{@{}llccccc@{}}' in t


def test_end_to_end_k2_three_payloads_two_seeds_3x2(tmp_path):
    fx, root = _payload_dataset(tmp_path)
    k2=tmp_path/'k2.pt'
    fx._k2_checkpoint(k2)
    cp={}
    for method,weight,seed in [('F0',0.,0),('F3',100.,4)]:
        f=tmp_path/f'{method}.pt'
        fx._save_planner(f,k2,seed=seed,weight=weight)
        cp[method]=f
    import yaml
    config=tmp_path/'diffusion.yaml'
    config.write_text(yaml.safe_dump({'output_space':'control','control':{
        'training':{'dataset_root':str(root)},
        'dynamics':fx.CTRL,
        'losses':{'fuel':{'parameters':fx.FUEL}}}}),encoding='utf-8')
    out=tmp_path/'out'
    result=eval_payload_ablation(cp,yaml_config=config,dataset_root=root,
         split='test',num_samples=6,batch_size=2,device='cpu',
         inference_seeds=[31,32],payloads=[0.,3500.,7000.],
         progress_threshold_m=60.,output_dir=out)
    assert len(result['summary_by_payload'])==6
    assert len(result['manifest']['figure_examples'])==3
    assert set(int(x) for x in result['manifest']['selected_counts_by_payload'].values())=={2}
    for name in ('metrics_summary.csv','metrics_summary_by_payload.csv',
                 'metrics_by_seed_payload.csv','metrics_per_sample.csv',
                 'fuel_ablation_table.tex','fuel_ablation.pdf','fuel_ablation.png',
                 'evaluation_manifest.json'):
        assert (out/name).stat().st_size > 50
    latex=(out/'fuel_ablation_table.tex').read_text(encoding='utf-8')
    assert 'Success' not in latex and 'PVR' not in latex
    assert latex.count('F0: Imitation')==3
    with (out/'metrics_per_sample.csv').open(encoding='utf-8') as fp:
        import csv
        rows=list(csv.DictReader(fp))
    assert len(rows)==2*2*6
    assert sorted(set(float(r['payload_kg']) for r in rows))==[0.,3500.,7000.]


def test_fuel_metric_and_objective_consistency():
    p = FuelParameters(**{**FUEL, 'alpha0': .001, 'alpha1': 0., 'alpha2': 0.})
    poses = torch.zeros(2,41,3)
    poses[:,:,0] = torch.arange(41).float()*2.
    truth = poses[:,1:,:2].clone()
    pred = {'pose':poses,'trajectory_dense':truth,'speeds':torch.full((2,41),20.),
            'trajectory_mode_idx':torch.tensor([0,1])}
    lane = torch.zeros(2,32,10)
    lane[:,:,0] = torch.linspace(-20,120,32)
    metrics = evaluate_sample_metrics(pred,{'trajectory_dense':truth},
        {'target_lane_polyline':lane,'target_point':torch.tensor([[80.,0.],[80.,0.]])},
        {'koopman_payload_kg':torch.zeros(2), 'koopman_pose0_si':torch.zeros(2,3)},
        fuel_params=p,dt=.1,progress_threshold_m=60.)
    torch.testing.assert_close(metrics['fuel_l_per_100km'],torch.full((2,),5.))
    torch.testing.assert_close(metrics['fuel_l'],torch.full((2,),.004))
    torch.testing.assert_close(metrics['fde_m'],torch.zeros(2))
    assert 'pvr' in metrics and not torch.isnan(metrics['pvr']).any()


def test_koopman_signature_mismatch_blocks_comparison():
    ref=dict(model_config={'x':1},control_config={'t':40},
             k2_signature='abc',mode_selection='goal_endpoint')
    with pytest.raises(ValueError,match='frozen K2'):
        check_checkpoint_compatibility(ref,{**ref,'k2_signature':'def'},'F3')
