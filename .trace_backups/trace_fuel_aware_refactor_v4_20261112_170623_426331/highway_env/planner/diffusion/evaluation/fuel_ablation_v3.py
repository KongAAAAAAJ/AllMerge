"""TRACE V3 payload-stratified fuel-aware open-loop ablation.

The paper output is one 3-payload x 2-method mean-only table and one 3x2
figure: speed(t) and XY trajectory, F0 against the selected fuel-aware method.
Each figure row is an independently, reproducibly sampled scenario from the
specified payload. Rows do NOT claim to share an identical maneuver.

Inference seeds and independent training seeds are kept distinct. No success
rate is computed: the entire evaluation is open-loop, on frozen K2 rollouts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_pipeline.expert_dataset import AllMergeExpertDatasetConfig, AllMergeExpertShardDataset
from highway_env.planner.diffusion.common.execution import K2_CONTEXT
from highway_env.planner.diffusion.config.experiment import load_mode_yaml, resolve_repo_file
from highway_env.planner.diffusion.evaluation.fuel_ablation import (
    REQUIRED_LABELS, check_ablation_objectives, check_checkpoint_compatibility,
    checkpoint_hash, fuel_parameters_from_yaml, k2_signature, resolve_device,
)
from highway_env.planner.diffusion.evaluation.fuel_ablation_v2 import (
    METRICS, _write_csv, evaluate_sample_metrics, parse_multi_checkpoint_specs,
    parse_seeds, summarize_multiseed, summarize_per_sample_repeats,
)
from highway_env.planner.diffusion.training.control import load_control_checkpoint

TABLE_METRICS = ('fuel_l_per_100km', 'fuel_l_per_4s', 'progress_m', 'ade_m', 'fde_m')


def parse_payloads(value):
    result = [float(v.strip()) for v in value.split(',')] if isinstance(value, str) else [float(v) for v in value]
    if len(result) != 3 or any(not math.isfinite(x) or x < 0 for x in result) or len(set(result)) != 3:
        raise ValueError('Exactly three distinct nonnegative payloads required, e.g. 0,3500,7000')
    return result


class IndexedSubset(Dataset):
    """Preserve NPZ shard provenance and indices while stratifying the split."""
    def __init__(self, source, indices):
        self.source = source
        self.source_indices = [int(i) for i in indices]
        self._index = [source._index[i] for i in self.source_indices]
        self.shard_paths = source.shard_paths

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self, i):
        return self.source[self.source_indices[i]]


def select_payload_samples(ds, payloads, *, max_samples=0, sample_seed=2026, tolerance_kg=1.0):
    """Scan all split shards for actual payload values; select without replacement.

    max_samples=0 selects all matching samples. Otherwise sample as evenly as
    possible across the three payloads. Do not take the dataset's first N items:
    they may belong to only one loading condition.
    """
    if tolerance_kg < 0 or not math.isfinite(tolerance_kg):
        raise ValueError('Invalid payload tolerance')
    if max_samples < 0 or 0 < max_samples < 3:
        raise ValueError('--num-samples must be 0 (all) or >=3 for three payloads')
    if any(abs(a-b) <= 2*tolerance_kg for i,a in enumerate(payloads) for b in payloads[i+1:]):
        raise ValueError('Payload groups overlap at the requested tolerance')
    by_payload = {float(x): [] for x in payloads}
    # Dataset index is shard-major; avoid expensive __getitem__ and per-sample
    # decompression solely for payload discovery.
    by_shard = defaultdict(list)
    for global_idx, (shard_idx, row_idx) in enumerate(ds._index):
        by_shard[shard_idx].append((global_idx, row_idx))
    for shard_idx, entries in sorted(by_shard.items()):
        with np.load(ds.shard_paths[shard_idx], allow_pickle=False) as shard:
            if 'koopman_payload_kg' not in shard.files:
                raise KeyError(f'Missing koopman_payload_kg in {ds.shard_paths[shard_idx]}')
            values = np.asarray(shard['koopman_payload_kg']).reshape(-1)
        for global_idx, row_idx in entries:
            value = float(values[row_idx])
            matches = [p for p in payloads if abs(value-p) <= tolerance_kg]
            if len(matches) == 1:
                by_payload[matches[0]].append(global_idx)
    missing = [p for p in payloads if not by_payload[p]]
    if missing:
        raise ValueError(f'No samples for payload(s) {missing}; available groups: '
                         + str({k:len(v) for k,v in by_payload.items()}))
    rng = np.random.default_rng(int(sample_seed))
    if max_samples == 0:
        chosen = {p: list(by_payload[p]) for p in payloads}
    else:
        # Round-robin quota allocation, compensating for short groups.
        quota = {p:0 for p in payloads}
        available = sum(len(v) for v in by_payload.values())
        needed = min(max_samples, available)
        for _ in range(needed):
            eligible = [p for p in payloads if quota[p] < len(by_payload[p])]
            p = min(eligible, key=lambda v: (quota[v], payloads.index(v)))
            quota[p] += 1
        chosen = {p: rng.choice(by_payload[p], size=quota[p], replace=False).tolist()
                  for p in payloads}
    selected = sorted(i for ids in chosen.values() for i in ids)
    offset = {source_idx: local_idx for local_idx, source_idx in enumerate(selected)}
    groups = {p: sorted(offset[i] for i in chosen[p]) for p in payloads}
    return IndexedSubset(ds, selected), groups, {p: len(by_payload[p]) for p in payloads}


def select_figure_samples(groups, payloads, *, figure_seed=2026):
    rng = np.random.default_rng(int(figure_seed))
    return {p: int(rng.choice(groups[p])) for p in payloads}


def _infer_one(model, ds, *, seed, train_seed, method, device,
               batch_size, num_workers, params, threshold, fig_indices):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == 'cuda')
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    rows, figures = [], {}
    global_index = 0
    with torch.inference_mode():
        for f_cpu, t_cpu in loader:
            features = {k: v.to(device, non_blocking=True) for k,v in f_cpu.items()}
            missing = [k for k in K2_CONTEXT if k not in t_cpu]
            if missing:
                raise KeyError(f'Expert shard missing K2 context: {missing}')
            if 'trajectory_dense' not in t_cpu:
                raise KeyError('Expert shard missing 40-step trajectory_dense')
            context = {k: t_cpu[k].to(device, non_blocking=True) for k in K2_CONTEXT}
            truth = t_cpu['trajectory_dense'].to(device)
            prediction = model.infer_multimodal(features, generator=rng, context=context)
            metrics = evaluate_sample_metrics(prediction, {'trajectory_dense':truth},
                features, context, fuel_params=params, dt=model.ctrl_cfg.dt,
                progress_threshold_m=threshold)
            for j in range(len(truth)):
                idx = global_index
                shard_id, shard_sample_index = ds._index[idx]
                scalar = lambda key: float(metrics[key][j].detach().cpu().item())
                row = dict(method=method, train_seed=train_seed, inference_seed=int(seed),
                    sample_index=idx, original_sample_index=ds.source_indices[idx],
                    shard_name=ds.shard_paths[shard_id].name,
                    shard_sample_index=shard_sample_index,
                    selected_mode=int(metrics['selected_mode'][j].item()),
                    fuel_l=scalar('fuel_l'), distance_m=scalar('distance_m'),
                    fuel_l_per_100km=scalar('fuel_l_per_100km'),
                    progress_m=scalar('progress_m'), pvr=scalar('pvr'),
                    ade_m=scalar('ade_m'), fde_m=scalar('fde_m'),
                    goal_terminal_error_m=scalar('goal_terminal_error_m'),
                    speed_mae_proxy_kmh=scalar('speed_mae_proxy_kmh'),
                    rms_longitudinal_jerk_m_s3=scalar('rms_longitudinal_jerk_m_s3'),
                    initial_speed_kmh=float(context['koopman_state_hist_si'][j,-1,0].item())*3.6,
                    payload_kg=float(context['koopman_payload_kg'][j].item()),
                    road_mu=float(context['koopman_road_mu'][j].item()))
                rows.append(row)
                if idx in fig_indices:
                    xy = prediction['pose'][j, :, :2].detach().cpu().numpy().copy()
                    gt0 = context['koopman_pose0_si'][j, :2].detach().cpu().numpy()[None,:]
                    gtxy = np.concatenate((gt0, truth[j].detach().cpu().numpy()), axis=0)
                    figures[idx] = dict(speeds_kmh=metrics['speeds_kmh'][j].detach().cpu().numpy().copy(),
                                        xy=xy, gt_xy=gtxy)
                global_index += 1
    return rows, figures


def _grouped_summary(records, groups, payloads):
    indices_to_payload = {int(idx):float(p) for p,ids in groups.items() for idx in ids}
    nested = {p:defaultdict(lambda: defaultdict(dict)) for p in payloads}
    for method, trains in records.items():
        for train_seed, inference in trains.items():
            for seed, rows in inference.items():
                gr = {p:[] for p in payloads}
                for row in rows:
                    gr[indices_to_payload[row['sample_index']]].append(row)
                for p in payloads:
                    if not gr[p]:
                        raise RuntimeError(f'Empty payload group {p} for {method}@{train_seed}/{seed}')
                    nested[p][method][train_seed][seed] = gr[p]
    summary_rows, byseed_rows = [], []
    for p in payloads:
        summary, details = summarize_multiseed(nested[p])
        for method, values in summary.items():
            summary_rows.append({'payload_kg':p, **values})
        for row in details:
            byseed_rows.append({'payload_kg':p, **row})
    return summary_rows, byseed_rows


def _fmt_payload(p):
    return f'{p:g} kg'


def latex_payload_table(rows, payloads, ours='F3'):
    lookup = {(float(r['payload_kg']),r['method']):r for r in rows}
    lines = [
        r'\begin{table*}[!t]', r'\centering',
        r'\caption{Effects of Fuel-Aware Training under Different Payload Conditions}',
        r'\label{tab:fuel_ablation}', r'\scriptsize',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\setlength{\tabcolsep}{6pt}',
        r'\begin{tabular}{@{}llccccc@{}}', r'\hline',
        r'\textbf{Payload} & \textbf{Method} & '
        r'\shortstack{\textbf{Fuel}\\\textbf{(L/100 km)} $\downarrow$} & '
        r'\shortstack{\textbf{Total Fuel}\\\textbf{(L)} $\downarrow$} & '
        r'\shortstack{\textbf{Progress}\\\textbf{(m)} $\uparrow$} & '
        r'\shortstack{\textbf{ADE}\\\textbf{(m)} $\downarrow$} & '
        r'\shortstack{\textbf{FDE}\\\textbf{(m)} $\downarrow$} \\',
        r'\hline',
    ]
    for k,p in enumerate(payloads):
        for method in ('F0',ours):
            if (p,method) not in lookup:
                raise ValueError(f'Cannot create paper table: missing {method} at {p} kg')
            r = lookup[(p,method)]
            name = 'F0: Imitation' if method=='F0' else 'F3: Fuel-aware'
            if method != 'F0':
                name = r'\textbf{' + name + '}'
            cells = [_fmt_payload(p) if method=='F0' else '', name]
            for key in TABLE_METRICS:
                value = f'{r[key]:.2f}' if key != 'fuel_l_per_4s' else f'{r[key]:.4f}'
                cells.append(r'\textbf{' + value + '}' if method != 'F0' else value)
            lines.append(' & '.join(cells) + r' \\')
        lines.append(r'\hline')
    lines.extend([r'\end{tabular}',r'\end{table*}',''])
    return '\n'.join(lines)


def _normal_sd_band(xy):
    """Normal projection of repeat variation around the time-aligned mean path."""
    paths = np.asarray(xy, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[1:] != (41,2) or len(paths) < 1:
        raise ValueError('XY paths must be [repeat,41,2]')
    mean = paths.mean(axis=0)
    tangent = np.gradient(mean, axis=0)
    size = np.linalg.norm(tangent, axis=1, keepdims=True)
    unit = tangent / np.maximum(size, 1e-8)
    bad = size[:,0] < 1e-8
    unit[bad] = [1.,0.]
    normal = np.stack((-unit[:,1],unit[:,0]),axis=1)
    signed = np.einsum('ntc,tc->nt', paths-mean[None,:,:], normal)
    sd = signed.std(axis=0, ddof=1) if len(paths)>1 else np.zeros(len(mean))
    return mean, mean+sd[:,None]*normal, mean-sd[:,None]*normal


def draw_payload_figure(path, curves, payloads, selected, dt, *, fig_details):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,2,figsize=(10.8,10.0),constrained_layout=True)
    time = np.arange(41,dtype=np.float64)*dt
    palette = {'F0':'#2563eb', selected:'#ea580c'}
    for row,p in enumerate(payloads):
        ax_v,ax_xy = axes[row]
        if p not in curves or set(curves[p]) != set(palette):
            raise ValueError(f'Missing F0/selected method figure traces for payload {p}')
        truth_sets = []
        for method,color in palette.items():
            repeats = curves[p][method]
            vel = np.asarray([r['speeds_kmh'] for r in repeats],dtype=np.float64)
            xy = np.asarray([r['xy'] for r in repeats],dtype=np.float64)
            if vel.ndim!=2 or vel.shape[1]!=41:
                raise ValueError('Expected [n,41] velocity traces')
            avg = vel.mean(axis=0)
            std = vel.std(axis=0, ddof=1) if len(vel)>1 else np.zeros_like(avg)
            label = 'F0: Imitation' if method=='F0' else 'F3: Fuel-aware'
            ax_v.plot(time,avg,color=color,lw=1.8,label=label)
            ax_v.fill_between(time,avg-std,avg+std,color=color,alpha=.16,linewidth=0)
            mid,high,low=_normal_sd_band(xy)
            ax_xy.plot(mid[:,0],mid[:,1],color=color,lw=1.9,label=label)
            ax_xy.fill(np.r_[high[:,0], low[::-1,0]],
                       np.r_[high[:,1], low[::-1,1]],color=color,alpha=.16,linewidth=0)
            truth_sets.extend(r['gt_xy'] for r in repeats)
        first=np.asarray(truth_sets[0])
        if any(not np.allclose(first,r,atol=1e-5) for r in truth_sets[1:]):
            raise ValueError('Ground-truth path not identical across methods/seeds for one scenario')
        ax_xy.plot(first[:,0],first[:,1],ls='--',lw=1.4,color='#777777',label='Expert')
        ax_v.set_title(f'({chr(ord("a")+2*row)}) {_fmt_payload(p)}: speed',fontsize=10)
        ax_xy.set_title(f'({chr(ord("b")+2*row)}) {_fmt_payload(p)}: X-Y path',fontsize=10)
        ax_v.set(xlabel='Time (s)',ylabel='Longitudinal speed (km/h)')
        ax_xy.set(xlabel='X (m)',ylabel='Y (m)')
        ax_v.set_xlim(0,40*dt)
        # Longitudinal X spans tens of metres while lane changes span only
        # a few metres in Y; equal axes would visually flatten the maneuver.
        ax_xy.set_aspect('auto')
        ymin,ymax=ax_xy.get_ylim()
        if ymax-ymin < 2.:
            yc=.5*(ymin+ymax)
            ax_xy.set_ylim(yc-1.,yc+1.)
        ax_v.grid(alpha=.17)
        ax_xy.grid(alpha=.17)
        if row == 0:
            ax_v.legend(frameon=False,fontsize=8,loc='best')
            ax_xy.legend(frameon=False,fontsize=8,loc='best')
        # Different rows are independently sampled, so no claim of matched scenarios.
    fig.savefig(path,bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)


def eval_payload_ablation(checkpoints, *, yaml_config, dataset_root,
    split='val', num_samples=99, batch_size=2, num_workers=0, device='auto',
    inference_seeds=(0,1,2,3,4), output_dir, paper_ours='F3',
    progress_threshold_m=60., payloads=(0.,3500.,7000.),
    payload_tolerance_kg=1., sample_seed=2026, figure_seed=2027):
    payloads = parse_payloads(payloads)
    inferred = parse_seeds(inference_seeds)
    cps = checkpoints if all(isinstance(k,tuple) for k in checkpoints) else {
        (k,'single'):v for k,v in checkpoints.items()}
    methods = [m for m in REQUIRED_LABELS if any(k[0]==m for k in cps)]
    if 'F0' not in methods or paper_ours not in methods or paper_ours == 'F0':
        raise ValueError('Require baseline F0 and selected fuel-aware checkpoint')
    if num_samples < 0 or 0 < num_samples < 3 or batch_size < 1 or num_workers < 0:
        raise ValueError('Invalid evaluation size or batch setting')
    train_ids = sorted({s for _,s in cps})
    for tr in train_ids:
        if any((m,tr) not in cps for m in methods):
            raise ValueError(f'Missing paired training checkpoint for {tr}')
    params = fuel_parameters_from_yaml(yaml_config)
    dev = resolve_device(device)
    root = resolve_repo_file(dataset_root)
    source = AllMergeExpertShardDataset(AllMergeExpertDatasetConfig(
        dataset_root=root,split=split,max_samples=None))
    ds, groups, available = select_payload_samples(source,payloads,max_samples=num_samples,
        sample_seed=sample_seed,tolerance_kg=payload_tolerance_kg)
    selected = select_figure_samples(groups,payloads,figure_seed=figure_seed)
    selected_ids = set(selected.values())
    checkpoint_info, reference = {}, None
    # Strictly compare all checkpoints before the first evaluation.
    for tr in train_ids:
        for method in methods:
            cp=resolve_repo_file(cps[(method,tr)])
            if not cp.is_file():
                raise FileNotFoundError(cp)
            model,payload=load_control_checkpoint(cp,device='cpu')
            check_ablation_objectives(method,model.loss_config,params)
            meta=dict(model_config=payload['model_config'],
                      control_config=payload['control_config'],
                      k2_signature=k2_signature(model),
                      mode_selection=model.loss_config.get('mode_selection'))
            if reference is not None:
                check_checkpoint_compatibility(reference,meta,f'{method}@{tr}')
            else:
                reference=meta
                if meta['mode_selection']!='goal_endpoint':
                    raise ValueError('Expected goal_endpoint training')
            if method != 'F0':
                minimum=float(model.loss_config['progress']['minimum_progress_m'])
                if not math.isclose(minimum,progress_threshold_m,abs_tol=1e-7):
                    raise ValueError(f'{method}@{tr} progress threshold {minimum} differs from {progress_threshold_m}')
            checkpoint_info[f'{method}@{tr}']=dict(checkpoint=str(cp),sha256=checkpoint_hash(cp),
                epoch=payload.get('epoch'),training_loss_config=model.loss_config)
            del model
    out=resolve_repo_file(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    nested=defaultdict(lambda:defaultdict(dict))
    all_rows=[]
    traces={p:{'F0':[] ,paper_ours:[]} for p in payloads}
    for tr in train_ids:
        for method in methods:
            cp=resolve_repo_file(cps[(method,tr)])
            model,_=load_control_checkpoint(cp,device='cpu')
            model.to(dev).eval()
            for seed in inferred:
                print(f'[PAYLOAD FUEL EVAL] {method}@{tr} inference_seed={seed} n={len(ds)}',flush=True)
                rows, figs=_infer_one(model,ds,seed=seed,train_seed=tr,method=method,device=dev,
                    batch_size=batch_size,num_workers=num_workers,params=params,
                    threshold=progress_threshold_m,fig_indices=selected_ids)
                if len(rows)!=len(ds):
                    raise RuntimeError('Dataset item count mismatch')
                nested[method][tr][seed]=rows
                all_rows.extend(rows)
                if method in ('F0',paper_ours):
                    for p,idx in selected.items():
                        traces[p][method].append(figs[idx])
            del model
            if dev.type=='cuda':
                torch.cuda.empty_cache()
    expected=[(r['shard_name'],r['shard_sample_index']) for r in nested['F0'][train_ids[0]][inferred[0]]]
    for method in methods:
        for tr in train_ids:
            for seed in inferred:
                got=[(r['shard_name'],r['shard_sample_index']) for r in nested[method][tr][seed]]
                if got!=expected:
                    raise RuntimeError(f'Mismatched samples in {method}@{tr}/seed{seed}')
    overall_summary, byseed = summarize_multiseed(nested)
    payload_summary, byseed_payload = _grouped_summary(nested,groups,payloads)
    _write_csv(out/'metrics_per_sample.csv',all_rows)
    _write_csv(out/'metrics_per_sample_summary.csv',summarize_per_sample_repeats(all_rows))
    _write_csv(out/'metrics_summary.csv',list(overall_summary.values()))
    _write_csv(out/'metrics_by_seed.csv',byseed)
    _write_csv(out/'metrics_summary_by_payload.csv',payload_summary)
    _write_csv(out/'metrics_by_seed_payload.csv',byseed_payload)
    (out/'fuel_ablation_table.tex').write_text(
        latex_payload_table(payload_summary,payloads,paper_ours),encoding='utf-8')
    figure_details={}
    for p,idx in selected.items():
        s,row_id=ds._index[idx]
        example=next(r for r in all_rows if r['sample_index']==idx)
        figure_details[str(p)]=dict(local_sample_index=idx,
            original_sample_index=ds.source_indices[idx],
            shard=ds.shard_paths[s].name,shard_sample_index=row_id,
            actual_payload_kg=example['payload_kg'],road_mu=example['road_mu'],
            initial_speed_kmh=example['initial_speed_kmh'])
    draw_payload_figure(out/'fuel_ablation.pdf',traces,payloads,paper_ours,
                        reference['control_config']['dt'],fig_details=figure_details)
    manifest=dict(study='TRACE payload-stratified fuel-aware open-loop ablation V3',
        evaluation_type='open_loop',success_metric=False,
        paper_metrics=TABLE_METRICS,auxiliary_metrics=METRICS,
        methods=methods,paper_comparison=['F0',paper_ours],
        payloads_kg=payloads,payload_tolerance_kg=payload_tolerance_kg,
        full_split_counts_by_payload={str(k):v for k,v in available.items()},
        selected_counts_by_payload={str(k):len(v) for k,v in groups.items()},
        figure_selection='One reproducibly random, independent sample per payload; different rows need not share a maneuver',
        figure_examples=figure_details,sample_seed=sample_seed,figure_seed=figure_seed,
        training_seed_ids=train_ids,inference_seeds=inferred,
        std_basis='Time-aligned inference/training repetitions, mean +/- 1 SD; spatial SD projected onto mean path normal',
        sample_ids=[{'original_index':i,'shard':ds.shard_paths[ds._index[j][0]].name,
                     'shard_sample_index':ds._index[j][1]} for j,i in enumerate(ds.source_indices)],
        split=split,eval_samples=len(ds),dataset_root=str(root),
        fuel_parameters=vars(params),frozen_k2_signature=reference['k2_signature'],
        configuration=str(resolve_repo_file(yaml_config)),
        checkpoint_info=checkpoint_info,
        fuel_definition='100000 * sum(VT-CPFM fuel [L]) / sum(K2 executed XY distance [m]) for each run',
        fuel_note='Estimated fuel via K2+VT-CPFM, not measured by TruckSim',
        warning='Equal planner initialization and identical source episodes must be guaranteed externally; checkpoint hashes cannot prove them.')
    (out/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[PAYLOAD FUEL EVAL DONE]',out,flush=True)
    for r in payload_summary:
        print(f"[{r['payload_kg']:g} kg] {r['method']} " +
              ' '.join(f'{k}={r[k]:.3f}' for k in TABLE_METRICS),flush=True)
    return dict(summary_by_payload=payload_summary,summary=overall_summary,
                manifest=manifest,output_dir=out)


def main(argv=None):
    parser=argparse.ArgumentParser(description='TRACE fuel-aware payload-stratified open-loop evaluator V3; no Success')
    parser.add_argument('--checkpoint',action='append',required=True,
        help='F0=path and F3=path; independent training: F0@0=path, F3@0=path etc.')
    parser.add_argument('--config',default='configs/train/diffusion.yaml')
    parser.add_argument('--dataset-root')
    parser.add_argument('--split',choices=('train','val','test'),default='val')
    parser.add_argument('--num-samples',type=int,default=99,
        help='Total stratified sample count; 0 uses all matching samples')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--num-workers',type=int,default=0)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--seeds',default='0,1,2,3,4')
    parser.add_argument('--paper-ours',choices=('F1','F2','F3'),default='F3')
    parser.add_argument('--payloads',default='0,3500,7000')
    parser.add_argument('--payload-tolerance-kg',type=float,default=1.0)
    parser.add_argument('--sample-seed',type=int,default=2026)
    parser.add_argument('--figure-seed',type=int,default=2027)
    parser.add_argument('--progress-threshold-m',type=float,default=60.)
    parser.add_argument('--output-dir',default='outputs/trace_fuel_ablation/open_loop_eval_v3')
    args=parser.parse_args(argv)
    specs=parse_multi_checkpoint_specs(args.checkpoint)
    cfg=load_mode_yaml(args.config)
    dsroot=args.dataset_root or cfg['training']['dataset_root']
    eval_payload_ablation(specs,yaml_config=args.config,dataset_root=dsroot,
        split=args.split,num_samples=args.num_samples,batch_size=args.batch_size,
        num_workers=args.num_workers,device=args.device,
        inference_seeds=parse_seeds(args.seeds),output_dir=args.output_dir,
        paper_ours=args.paper_ours,progress_threshold_m=args.progress_threshold_m,
        payloads=parse_payloads(args.payloads),payload_tolerance_kg=args.payload_tolerance_kg,
        sample_seed=args.sample_seed,figure_seed=args.figure_seed)
    return 0
