"""Fuel-Aware multiseed open-loop V2: CPU unit and mini-end-to-end tests."""
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

from highway_env.planner.diffusion.evaluation.fuel_ablation_v2 import (
    parse_multi_checkpoint_specs, parse_seeds, summarize_records,
    summarize_multiseed, eval_multiseed, table_latex_v2,
    evaluate_sample_metrics,
)
from highway_env.planner.diffusion.common.fuel import FuelParameters


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


def test_extra_metrics_and_paper_latex_has_six_metrics_no_success():
    poses = torch.zeros(2, 41, 3)
    poses[:, :, 0] = torch.arange(41).float() * 2.
    truth = poses[:, 1:, :2].clone()
    truth[1, :, 1] = 3.
    lanes = torch.zeros(2, 32, 10)
    lanes[:, :, 0] = torch.linspace(-20, 120, 32)
    pred = dict(pose=poses, trajectory_dense=poses[:, 1:, :2].clone(),
                speeds=torch.full((2, 41), 20.),
                trajectory_mode_idx=torch.tensor([0, 1]))
    p = FuelParameters(empty_mass_kg=8000., rolling_resistance=.007,
        drag_coefficient=.6, frontal_area_m2=9.,drivetrain_efficiency=.9,
        alpha0=.001,alpha1=0.,alpha2=0.)
    extra = evaluate_sample_metrics(pred, {'trajectory_dense':truth},
          dict(target_lane_polyline=lanes, target_point=torch.tensor([[80.,0.],[80.,0.]])),
          dict(koopman_payload_kg=torch.zeros(2),koopman_pose0_si=torch.zeros(2,3)),
          fuel_params=p, dt=.1, progress_threshold_m=85.)
    torch.testing.assert_close(extra['fde_m'], torch.tensor([0., 3.]))
    torch.testing.assert_close(extra['pvr'], torch.ones(2))
    assert torch.isfinite(extra['rms_longitudinal_jerk_m_s3']).all()
    stub = {k:dict((m, 2.) for m in ('fuel_l_per_100km','fuel_l_per_4s','progress_m','pvr_pct','ade_m','fde_m'))
            for k in ('F0','F3')}
    for k in stub:
        stub[k].update({key+'_std':.1 for key in ('fuel_l_per_100km','fuel_l_per_4s','progress_m','pvr_pct','ade_m','fde_m')})
    tex = table_latex_v2(stub,'F3')
    assert 'Success' not in tex and 'PVR' in tex and 'FDE' in tex
    assert r'\begin{tabular}{@{}lcccccc@{}}' in tex


def test_k2_four_checkpoint_2_inference_seeds_full_outputs(tmp_path):
    # Load the V1 self-contained synthetic K2 fixture by explicit path; avoids
    # pytest depending on its own test-directory import mode on Windows.
    path = Path(__file__).with_name('test_trace_fuel_ablation_open_loop.py')
    spec = importlib.util.spec_from_file_location('_trace_fuel_v1_fixtures', path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    data = tmp_path/'data'
    fixtures._dataset(data)
    koopman = tmp_path/'koopman.pt'
    fixtures._k2_checkpoint(koopman)
    cp = {}
    for label, seed, weight in [('F0',0,0.),('F1',2,50.),('F2',3,100.),('F3',4,200.)]:
        path = tmp_path/(label+'.pt')
        fixtures._save_planner(path,koopman,seed=seed,weight=weight)
        cp[(label,'single')] = path
    import yaml
    config = tmp_path/'diffusion.yaml'
    config.write_text(yaml.safe_dump({'output_space':'control','control':{
       'training':{'dataset_root':str(data)},
       'dynamics':fixtures.CTRL,
       'losses':{'fuel':{'parameters':fixtures.FUEL}}}}), encoding='utf-8')
    output = tmp_path/'evaluation'
    got = eval_multiseed(cp, yaml_config=config, dataset_root=data,
        split='test', num_samples=2, batch_size=1, device='cpu',
        inference_seeds=[100,101], output_dir=output,
        paper_ours='F3', figure_count=2)
    assert got['summary']['F0']['n_inference_seeds']==2
    assert got['summary']['F0']['std_basis']=='inference_seeds'
    assert len(got['per_seed'])==8
    for filename in ('metrics_summary.csv','metrics_by_seed.csv','metrics_per_sample.csv',
      'paired_deltas.csv','metrics_per_sample_summary.csv','relative_to_F0.csv','fuel_ablation_table.tex','fuel_ablation.pdf',
      'fuel_ablation_cohort.pdf','fuel_ablation_metrics.pdf','evaluation_manifest.json'):
        assert (output/filename).stat().st_size>50
    assert 'Success' not in (output/'fuel_ablation_table.tex').read_text(encoding='utf-8')
    with (output/'metrics_per_sample.csv').open(encoding='utf-8') as f:
        import csv
        rows=list(csv.DictReader(f))
    assert len(rows)==16
    assert set(int(r['inference_seed']) for r in rows)=={100,101}
    assert got['manifest']['figure_cohort_indices']==[0,1]
