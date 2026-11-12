"""V3 payload table and 3x2 multi-seed curves: no TruckSim needed."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec('gymnasium') is None:
    pkg = types.ModuleType('highway_env')
    pkg.__path__ = [str(Path(__file__).resolve().parents[1] / 'highway_env')]
    sys.modules.setdefault('highway_env', pkg)

from data_pipeline.expert_dataset import AllMergeExpertDatasetConfig, AllMergeExpertShardDataset
from highway_env.planner.diffusion.evaluation.fuel_ablation_v3 import (
    parse_payloads, select_payload_samples, select_figure_samples,
    latex_payload_table, _normal_sd_band, eval_payload_ablation,
)


def _fixtures():
    fn = Path(__file__).with_name('test_trace_fuel_ablation_open_loop.py')
    spec = importlib.util.spec_from_file_location('_trace_fuel_v3_fixtures', fn)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload_dataset(tmp_path):
    fx = _fixtures()
    root = tmp_path/'dataset'
    fx._dataset(root)
    file = root/'shards'/'shard_000.npz'
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
    return fx, root


def test_stratified_selection_exact_random_and_missing_groups(tmp_path):
    _, root = _payload_dataset(tmp_path)
    source = AllMergeExpertShardDataset(AllMergeExpertDatasetConfig(dataset_root=root,split='test'))
    subset, groups, available = select_payload_samples(source,[0.,3500.,7000.],max_samples=6,sample_seed=81)
    assert available == {0.:3,3500.:3,7000.:3}
    assert [len(groups[x]) for x in [0.,3500.,7000.]] == [2,2,2]
    assert len(subset)==6
    fig = select_figure_samples(groups,[0.,3500.,7000.],figure_seed=18)
    assert set(fig)=={0.,3500.,7000.}
    for p,idx in fig.items():
        _, t=subset[idx]
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
