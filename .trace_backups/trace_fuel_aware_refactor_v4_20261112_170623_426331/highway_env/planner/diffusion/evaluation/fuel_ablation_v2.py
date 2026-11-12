"""TRACE multi-seed fuel-aware open-loop ablation, project-native V2.

Paper metrics: VT-CPFM fuel intensity, 4-s fuel volume, road progress,
progress violation rate, ADE and FDE. Success is NOT an open-loop metric.

Training seed and inference seed are distinct: stochastic inference repeatability
cannot substitute for independent training runs. All comparisons are paired by
(train seed, inference seed, dataset sample identity).
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
from torch.utils.data import DataLoader

from data_pipeline.expert_dataset import AllMergeExpertDatasetConfig, AllMergeExpertShardDataset
from highway_env.planner.diffusion.common.execution import K2_CONTEXT
from highway_env.planner.diffusion.config.experiment import load_mode_yaml, resolve_repo_file
from highway_env.planner.diffusion.training.control import load_control_checkpoint
from highway_env.planner.diffusion.evaluation.fuel_ablation import (
    REQUIRED_LABELS, check_ablation_objectives, check_checkpoint_compatibility,
    checkpoint_hash, fuel_parameters_from_yaml, k2_signature, resolve_device,
    sample_metrics,
)

METRICS = ('fuel_l_per_100km', 'fuel_l_per_4s', 'progress_m',
           'pvr_pct', 'ade_m', 'fde_m')
METRIC_LABELS = ('Fuel (L/100 km)', 'Fuel (L/4 s)', 'Progress (m)',
                 'PVR (%)', 'ADE (m)', 'FDE (m)')


def parse_multi_checkpoint_specs(specs: list[str]):
    """Parse F0=path or F0@TRAIN_SEED=path; insist on matched training seeds."""
    result = {}
    for spec in specs:
        name, sep, value = spec.partition('=')
        if not sep or not value:
            raise ValueError(f'Expected --checkpoint F0[@train_seed]=PATH: {spec}')
        label, at, train_seed = name.partition('@')
        label = label.upper()
        if label not in REQUIRED_LABELS or (at and not train_seed.strip()):
            raise ValueError(f'Invalid method or train_seed: {name}')
        train_seed = train_seed if at else 'single'
        key = (label, train_seed)
        if key in result:
            raise ValueError(f'Duplicate checkpoint {key}')
        result[key] = Path(value)
    by_method = defaultdict(set)
    for method, tr_seed in result:
        by_method[method].add(tr_seed)
    if 'F0' not in by_method:
        raise ValueError('Must provide F0 baseline checkpoint')
    baseline_set = by_method['F0']
    for method, ids in by_method.items():
        if ids != baseline_set:
            raise ValueError(f'Unpaired training seeds: {method}={sorted(ids)}, F0={sorted(baseline_set)}')
    if len(baseline_set) > 1 and 'single' in baseline_set:
        raise ValueError('Use explicit train-seed IDs (F0@0 etc.) for repeated trainings')
    return result


def parse_seeds(value: str | list[int] | tuple[int, ...]):
    if isinstance(value, str):
        result = [int(s.strip()) for s in value.split(',') if s.strip()]
    else:
        result = [int(s) for s in value]
    if not result or any(s < 0 for s in result) or len(set(result)) != len(result):
        raise ValueError('Inference seeds must be unique nonnegative integers')
    return result


def evaluate_sample_metrics(prediction, targets, features, context, *, fuel_params, dt, progress_threshold_m):
    truth = targets['trajectory_dense']
    base = sample_metrics(prediction, truth, features, context['koopman_payload_kg'], fuel_params, dt)
    xy = prediction['trajectory_dense']
    fde = torch.linalg.vector_norm(xy[:, -1] - truth[:, -1], dim=-1)
    pvr = (base['progress_m'] < progress_threshold_m).to(torch.float32)
    # K2's predicted vx is longitudinal; finite-difference GT XY gives planar
    # speed magnitude, so this is an explicitly labelled auxiliary proxy only.
    gt_poses = torch.cat((context['koopman_pose0_si'][:, None, :2], truth), dim=1)
    gt_speed = torch.linalg.vector_norm(gt_poses[:, 1:] - gt_poses[:, :-1], dim=-1) / dt
    speed_mae_proxy = (prediction['speeds'][:, 1:] - gt_speed).abs().mean(dim=1) * 3.6
    accel = (prediction['speeds'][:, 1:] - prediction['speeds'][:, :-1]) / dt
    jerk = (accel[:, 1:] - accel[:, :-1]) / dt
    rms_jerk = jerk.square().mean(dim=1).sqrt()
    result = dict(base, fde_m=fde, pvr=pvr,
                  speed_mae_proxy_kmh=speed_mae_proxy,
                  rms_longitudinal_jerk_m_s3=rms_jerk)
    for name, value in result.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f'Non-finite fuel evaluation metric: {name}')
    return result


def summarize_records(rows):
    if not rows:
        raise ValueError('Empty matched evaluation sample list')
    total_fuel = math.fsum(r['fuel_l'] for r in rows)
    total_distance = math.fsum(r['distance_m'] for r in rows)
    if total_distance <= 1:
        raise ValueError('Insufficient executed distance for intensity metric')
    avg = lambda key: math.fsum(r[key] for r in rows) / len(rows)
    return {
        'n_samples': len(rows),
        'fuel_l_per_100km': 100000.0 * total_fuel / total_distance,
        'fuel_l_per_4s': avg('fuel_l'),
        'progress_m': avg('progress_m'),
        'pvr_pct': 100.0 * avg('pvr'),
        'ade_m': avg('ade_m'), 'fde_m': avg('fde_m'),
        'goal_terminal_error_m': avg('goal_terminal_error_m'),
        'speed_mae_proxy_kmh': avg('speed_mae_proxy_kmh'),
        'rms_longitudinal_jerk_m_s3': avg('rms_longitudinal_jerk_m_s3'),
        'total_fuel_l': total_fuel, 'total_executed_distance_m': total_distance,
    }


def _avg_sd(vals):
    x = np.asarray(vals, dtype=np.float64)
    if not np.isfinite(x).all() or x.size < 1:
        raise ValueError('Invalid or empty values for statistics')
    return float(np.mean(x)), float(np.std(x, ddof=1)) if x.size > 1 else 0.0


def summarize_multiseed(by_method_train_inference):
    """Each per-inference statistic is first pooled over identical dataset samples.

    If multiple independently trained checkpoints exist, the headline SD is
    across the training-seed averages; otherwise it is across inference seeds.
    This avoids treating 1000 correlated samples as 1000 independent runs.
    """
    result = {}
    details = []
    for method in sorted(by_method_train_inference, key=lambda m: REQUIRED_LABELS.index(m)):
        train_groups = by_method_train_inference[method]
        training_mean_stats = defaultdict(list)
        within_training_std = defaultdict(list)
        for train_seed, inference_groups in sorted(train_groups.items()):
            run_stats = []
            for inference_seed, samples in sorted(inference_groups.items()):
                stats = summarize_records(samples)
                details.append(dict(method=method, train_seed=train_seed,
                                    inference_seed=inference_seed, **stats))
                run_stats.append(stats)
            for metric in METRICS:
                m, sd = _avg_sd([s[metric] for s in run_stats])
                training_mean_stats[metric].append(m)
                within_training_std[metric].append(sd)
        n_train = len(train_groups)
        n_inference = len(next(iter(train_groups.values())))
        row = dict(method=method, n_train_seeds=n_train, n_inference_seeds=n_inference,
                   n_samples=len(next(iter(next(iter(train_groups.values())).values()))),
                   std_basis='training_seeds' if n_train > 1 else 'inference_seeds')
        for metric in METRICS:
            mean, std_train = _avg_sd(training_mean_stats[metric])
            std_infer = float(np.mean(within_training_std[metric]))
            row[metric] = mean
            row[metric + '_mean'] = mean
            row[metric + '_std'] = std_train if n_train > 1 else std_infer
            row[metric + '_std_training'] = std_train if n_train > 1 else 0.0
            row[metric + '_std_inference'] = std_infer
        result[method] = row
    return result, details




def summarize_per_sample_repeats(rows):
    """A seed-aware per-sample summary; avoid pseudo-independent sample replicates."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row['method'], row['sample_index'], row['shard_name'],
                 row['shard_sample_index'])][row['train_seed']].append(row)
    result = []
    fields = ('fuel_l', 'fuel_l_per_100km', 'distance_m', 'progress_m',
              'pvr', 'ade_m', 'fde_m', 'goal_terminal_error_m',
              'speed_mae_proxy_kmh', 'rms_longitudinal_jerk_m_s3')
    for (method, sample_index, shard_name, shard_idx), trainings in sorted(grouped.items()):
        record = dict(method=method, sample_index=sample_index, shard_name=shard_name,
                      shard_sample_index=shard_idx, n_train_seeds=len(trainings),
                      n_evaluation_runs=sum(len(x) for x in trainings.values()),
                      std_basis='training_seeds' if len(trainings)>1 else 'inference_seeds')
        for key in fields:
            train_means = [np.mean([r[key] for r in reps]) for reps in trainings.values()]
            mean, train_std = _avg_sd(train_means)
            infer_stds = [_avg_sd([r[key] for r in reps])[1] for reps in trainings.values()]
            record[key+'_mean'] = mean
            record[key+'_std'] = train_std if len(trainings)>1 else float(np.mean(infer_stds))
        result.append(record)
    return result


def _write_csv(path, rows):
    if not rows:
        raise ValueError('No CSV rows')
    fields = list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def _tex_number(v, bold=False):
    s = f'{v:.2f}'
    return r'\textbf{' + s + '}' if bold else s


def table_latex_v2(summary, ours):
    if ours not in summary or 'F0' not in summary:
        raise ValueError('Paper table requires F0 and chosen proposed method')
    def tr(label, label_text, bold):
        item = summary[label]
        cells = [label_text] + [
            f'{_tex_number(item[m], bold)} $\\pm$ {_tex_number(item[m+"_std"], bold)}'
            for m in METRICS
        ]
        if bold:
            cells[0] = r'\textbf{' + cells[0] + '}'
        return ' & '.join(cells) + r' \\'
    return '\n'.join([
        r'\begin{table*}[!t]', r'\centering',
        r'\caption{Fuel-Aware Training Ablation: Open-Loop Mean $\pm$ Standard Deviation}',
        r'\label{tab:fuel_ablation}', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\begin{tabular}{@{}lcccccc@{}}', r'\hline',
        r'\textbf{Method} & \shortstack{\textbf{Fuel}\\\textbf{(L/100 km)}} & '
        r'\shortstack{\textbf{Total Fuel}\\\textbf{(L/4 s)}} & '
        r'\shortstack{\textbf{Progress}\\\textbf{(m)}} & '
        r'\shortstack{\textbf{PVR}\\\textbf{(\%)}} & '
        r'\shortstack{\textbf{ADE}\\\textbf{(m)}} & '
        r'\shortstack{\textbf{FDE}\\\textbf{(m)}} \\',
        r'\hline',
        tr('F0', 'F0: Imitation', False),
        tr(ours, ours + ': Ours', True),
        r'\hline', r'\end{tabular}', r'\end{table*}', '',
    ])


def _fig_curves(path, curves, *, title, dt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    t = np.arange(41) * dt
    for method, data in curves.items():
        fuel = np.asarray([r['cumulative_fuel_l'] for r in data], dtype=float)
        speed = np.asarray([r['speeds_kmh'] for r in data], dtype=float)
        if not len(fuel) or fuel.shape[1] != 41 or speed.shape != fuel.shape:
            raise ValueError('Expected [repetitions,41] curve tensors')
        for a, values, label in ((ax[0], fuel, 'Accumulated fuel (L)'),
                                 (ax[1], speed, 'Longitudinal speed (km/h)')):
            mu = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mu)
            line, = a.plot(t, mu, lw=1.8, label=method)
            a.fill_between(t, mu-sd, mu+sd, color=line.get_color(), alpha=.20, linewidth=0)
            a.set_ylabel(label)
    for i, a in enumerate(ax):
        a.set_xlabel('Time (s)')
        a.set_xlim(0, 40*dt)
        a.grid(alpha=.2)
        a.legend(frameon=False)
        a.set_title('(a) Accumulated fuel' if i == 0 else '(b) Vehicle speed')
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def _fig_metrics(path, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = list(summary)
    fig, axs = plt.subplots(2, 3, figsize=(11.2, 5.8), constrained_layout=True)
    for ax, key, ylabel in zip(axs.ravel(), METRICS, METRIC_LABELS):
        center = [summary[name][key] for name in labels]
        spread = [summary[name][key+'_std'] for name in labels]
        ax.bar(labels, center, yerr=spread, capsize=3, alpha=.85)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.16)
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def _select_fig_indices(n, count, representative_index=None, user_indices=None):
    if user_indices is not None:
        indexes = sorted(set(int(i) for i in user_indices))
        if not indexes or any(i < 0 or i >= n for i in indexes):
            raise ValueError('figure sample index out of range')
    else:
        indexes = sorted(set(np.linspace(0, n-1, min(n, count)).round().astype(int).tolist()))
    representative = n // 2 if representative_index is None else int(representative_index)
    if not 0 <= representative < n:
        raise ValueError('representative index out of range')
    return representative, indexes


def _infer_one_run(model, ds, *, seed, train_seed, method, device,
                   batch_size, num_workers, params, threshold, plot_indices):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type=='cuda')
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    rows, figures = [], {}
    global_index = 0
    with torch.inference_mode():
        for features_cpu, targets_cpu in loader:
            features = {k: v.to(device, non_blocking=True) for k, v in features_cpu.items()}
            missing = [k for k in K2_CONTEXT if k not in targets_cpu]
            if missing:
                raise KeyError(f'Expert data lacks K2 context: {missing}')
            context = {k: targets_cpu[k].to(device, non_blocking=True) for k in K2_CONTEXT}
            targets = {'trajectory_dense': targets_cpu['trajectory_dense'].to(device)}
            pred = model.infer_multimodal(features, generator=rng, context=context)
            m = evaluate_sample_metrics(pred, targets, features, context,
                                        fuel_params=params, dt=model.ctrl_cfg.dt,
                                        progress_threshold_m=threshold)
            for j in range(len(targets['trajectory_dense'])):
                idx = global_index
                sid, row_id = ds._index[idx]
                scalar = lambda name: float(m[name][j].detach().cpu().item())
                rows.append(dict(method=method, train_seed=train_seed, inference_seed=seed,
                                 sample_index=idx, shard_name=ds.shard_paths[sid].name,
                                 shard_sample_index=row_id,
                                 selected_mode=int(m['selected_mode'][j]),
                                 fuel_l=scalar('fuel_l'), distance_m=scalar('distance_m'),
                                 fuel_l_per_100km=scalar('fuel_l_per_100km'),
                                 progress_m=scalar('progress_m'), pvr=scalar('pvr'),
                                 ade_m=scalar('ade_m'), fde_m=scalar('fde_m'),
                                 goal_terminal_error_m=scalar('goal_terminal_error_m'),
                                 speed_mae_proxy_kmh=scalar('speed_mae_proxy_kmh'),
                                 rms_longitudinal_jerk_m_s3=scalar('rms_longitudinal_jerk_m_s3'),
                                 initial_speed_kmh=float(context['koopman_vx_init_kmh'][j])
                                 if 'koopman_vx_init_kmh' in context else float(context['koopman_state_hist_si'][j,-1,0])*3.6,
                                 payload_kg=float(context['koopman_payload_kg'][j])))
                if idx in plot_indices:
                    figures[idx] = dict(
                        cumulative_fuel_l=m['cumulative_fuel_l'][j].detach().cpu().numpy().copy(),
                        speeds_kmh=m['speeds_kmh'][j].detach().cpu().numpy().copy())
                global_index += 1
    return rows, figures


def eval_multiseed(checkpoints, *, yaml_config, dataset_root,
                   split='val', num_samples=100, batch_size=2, num_workers=0,
                   device='auto', inference_seeds=(0,1,2,3,4), output_dir,
                   paper_ours='F3', progress_threshold_m=60.0,
                   figure_count=16, representative_index=None,
                   figure_indices=None):
    inferred = parse_seeds(inference_seeds)
    cps = checkpoints if all(isinstance(k, tuple) for k in checkpoints) else {
        (k, 'single'): v for k,v in checkpoints.items()}
    if not cps or ('F0','single') not in cps and not any(k[0]=='F0' for k in cps):
        raise ValueError('Missing F0 baseline')
    methods = [l for l in REQUIRED_LABELS if any(k[0]==l for k in cps)]
    if paper_ours not in methods:
        raise ValueError('paper_ours must be among provided fuel-aware methods')
    if figure_count < 1 or num_samples < 0 or batch_size < 1 or num_workers < 0:
        raise ValueError('Invalid evaluation size')
    if not (progress_threshold_m > 0):
        raise ValueError('Invalid progress threshold')
    train_ids = sorted(set(seed for _, seed in cps))
    for tr in train_ids:
        if any((m,tr) not in cps for m in methods):
            raise ValueError(f'Incomplete paired training checkpoints for train seed {tr}')
    params = fuel_parameters_from_yaml(yaml_config)
    dev = resolve_device(device)
    root = resolve_repo_file(dataset_root)
    ds = AllMergeExpertShardDataset(AllMergeExpertDatasetConfig(
        dataset_root=root, split=split, max_samples=num_samples or None))
    if not len(ds):
        raise ValueError('Empty evaluation split')
    rep, cohort = _select_fig_indices(len(ds), figure_count, representative_index, figure_indices)
    selection = set(cohort) | {rep}

    # Check all checkpoints before inference; compare frozen K2 globally.
    checkpoint_info = {}
    reference = None
    for tr in train_ids:
        for method in methods:
            cp = resolve_repo_file(cps[(method,tr)])
            if not cp.is_file():
                raise FileNotFoundError(cp)
            model, payload = load_control_checkpoint(cp, device='cpu')
            check_ablation_objectives(method, model.loss_config, params)
            meta = dict(model_config=payload['model_config'],
                        control_config=payload['control_config'],
                        k2_signature=k2_signature(model),
                        mode_selection=model.loss_config.get('mode_selection'))
            if reference is not None:
                check_checkpoint_compatibility(reference, meta, f'{method}@{tr}')
            else:
                reference = meta
                if meta['mode_selection'] != 'goal_endpoint':
                    raise ValueError('Required goal_endpoint training')
            if method != 'F0':
                minimum = float(model.loss_config['progress']['minimum_progress_m'])
                if not math.isclose(minimum, progress_threshold_m, abs_tol=1e-7):
                    raise ValueError(f'{method}@{tr}: progress threshold {minimum} != {progress_threshold_m}')
            checkpoint_info[f'{method}@{tr}'] = dict(checkpoint=str(cp), sha256=checkpoint_hash(cp),
                epoch=payload.get('epoch'), training_loss_config=model.loss_config)
            del model

    out = resolve_repo_file(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records_nested = defaultdict(lambda: defaultdict(dict))
    curve_representative = defaultdict(list)
    curve_cohort = defaultdict(list)
    all_rows = []
    for tr in train_ids:
        for method in methods:
            cp = resolve_repo_file(cps[(method,tr)])
            model, _ = load_control_checkpoint(cp, device='cpu')
            model.to(dev).eval()
            for seed in inferred:
                print(f'[FUEL EVAL] {method}@{tr} inference_seed={seed}, n={len(ds)}', flush=True)
                rows, curves = _infer_one_run(model, ds, seed=seed, train_seed=tr,
                    method=method, device=dev, batch_size=batch_size, num_workers=num_workers,
                    params=params, threshold=progress_threshold_m, plot_indices=selection)
                if len(rows) != len(ds):
                    raise RuntimeError('Dataset item count mismatch')
                records_nested[method][tr][seed] = rows
                all_rows.extend(rows)
                curve_representative[method].append(curves[rep])
                curve_cohort[method].extend(curves[idx] for idx in cohort)
            del model
            if dev.type == 'cuda':
                torch.cuda.empty_cache()

    ref_ids = [(r['shard_name'], r['shard_sample_index'])
               for r in records_nested['F0'][train_ids[0]][inferred[0]]]
    for method in methods:
        for tr in train_ids:
            for seed in inferred:
                rows = records_nested[method][tr][seed]
                ids = [(r['shard_name'], r['shard_sample_index']) for r in rows]
                if ids != ref_ids:
                    raise RuntimeError(f'Unpaired evaluation samples: {method}@{tr} seed={seed}')
    summary, per_seed = summarize_multiseed(records_nested)
    _write_csv(out/'metrics_per_sample.csv', all_rows)
    _write_csv(out/'metrics_per_sample_summary.csv', summarize_per_sample_repeats(all_rows))
    _write_csv(out/'metrics_by_seed.csv', per_seed)
    _write_csv(out/'metrics_summary.csv', list(summary.values()))
    (out/'metrics_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out/'fuel_ablation_table.tex').write_text(table_latex_v2(summary, paper_ours), encoding='utf-8')

    curves_for_paper = {'F0':curve_representative['F0'],paper_ours:curve_representative[paper_ours]}
    _fig_curves(out/'fuel_ablation.pdf', curves_for_paper,
        title=f'Fixed scenario #{rep}: mean ± SD across evaluation runs', dt=reference['control_config']['dt'])
    curves_for_cohort = {'F0':curve_cohort['F0'],paper_ours:curve_cohort[paper_ours]}
    _fig_curves(out/'fuel_ablation_cohort.pdf', curves_for_cohort,
        title=f'{len(cohort)} fixed scenarios: descriptive mean ± SD (seeds × scenarios)',
        dt=reference['control_config']['dt'])
    _fig_metrics(out/'fuel_ablation_metrics.pdf', summary)
    # Paired differences: same train seed, inference seed and scenario index.
    deltas = []
    for tr in train_ids:
        for seed in inferred:
            baseline = records_nested['F0'][tr][seed]
            for method in methods:
                if method == 'F0':
                    continue
                for r0, rm in zip(baseline, records_nested[method][tr][seed]):
                    deltas.append(dict(method=method, train_seed=tr, inference_seed=seed,
                        sample_index=r0['sample_index'],
                        fuel_l_delta=rm['fuel_l']-r0['fuel_l'],
                        progress_m_delta=rm['progress_m']-r0['progress_m'],
                        ade_m_delta=rm['ade_m']-r0['ade_m'],
                        fde_m_delta=rm['fde_m']-r0['fde_m']))
    if deltas:
        _write_csv(out/'paired_deltas.csv', deltas)
    relative = []
    for method in methods:
        if method == 'F0':
            continue
        row = dict(method=method, relative_to='F0')
        for metric in METRICS:
            base = summary['F0'][metric]
            new = summary[method][metric]
            row[metric+'_delta'] = new-base
            row[metric+'_relative_pct'] = 100.*(new-base)/base if abs(base)>1e-12 else float('nan')
        relative.append(row)
    if relative:
        _write_csv(out/'relative_to_F0.csv', relative)
    manifest = dict(study='TRACE fuel-aware open-loop multi-seed ablation V2',
       evaluation_type='open_loop', metrics=METRICS, methods=methods,
       training_seed_ids=train_ids, inference_seeds=inferred,
       total_evaluation_repeats=len(train_ids)*len(inferred),
       std_policy=('SD of independent training-seed means (averaged across inference seeds) '
                   'if multiple training seeds exist; otherwise SD of inference-seed metrics. '
                   'Not a confidence interval and not sample-level standard error.'),
       num_samples=len(ds), split=split, batch_size=batch_size,
       dataset_root=str(root), configuration=str(resolve_repo_file(yaml_config)),
       frozen_k2_signature=reference['k2_signature'], fuel_parameters=vars(params),
       progress_threshold_m=progress_threshold_m,
       mode_selection='goal_endpoint', paper_ours=paper_ours,
       figure_representative_index=rep, figure_cohort_indices=cohort,
       figure_cohort_sd_description='Descriptive dispersion across scenarios × training × inference seeds; not inferential uncertainty',
       fuel_definition='100000 * sum(VT-CPFM L) / sum(K2 executed XY path m) for each evaluation run',
       vtcpmf_note='Estimated model-based fuel on frozen Koopman K2, not measured TruckSim consumption',
       auxiliary_speed_metric='Planar expert finite-difference speed vs predicted longitudinal vx; diagnostic proxy only',
       checkpoint_info=checkpoint_info,
       warning='Same initial planner weights must be guaranteed by training setup; checkpoint compatibility alone cannot prove this.',
       closed_loop_evaluation=False)
    (out/'evaluation_manifest.json').write_text(json.dumps(manifest, indent=2,
        ensure_ascii=False), encoding='utf-8')
    print('[FUEL EVAL DONE]',out,flush=True)
    for method in methods:
        print(f'[{method}] '+', '.join(f'{m}={summary[method][m]:.3f}±{summary[method][m+"_std"]:.3f}' for m in METRICS), flush=True)
    return dict(summary=summary, per_seed=per_seed, manifest=manifest,
                output_dir=out)


def main(argv=None):
    parser = argparse.ArgumentParser(description='TRACE Fuel-Aware open-loop multiseed evaluation (no success rate)')
    parser.add_argument('--checkpoint', action='append', required=True,
        help='F0=path F1=path ...; for independent training runs F0@0=path, F0@1=path, etc.')
    parser.add_argument('--config', default='configs/train/diffusion.yaml')
    parser.add_argument('--dataset-root')
    parser.add_argument('--split', choices=('train','val','test'), default='val')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='auto')
    seeds_group = parser.add_mutually_exclusive_group()
    seeds_group.add_argument('--seeds', default=None,
        help='Comma-separated inference seeds, e.g. 0,1,2,3,4 (default)')
    seeds_group.add_argument('--seed', type=int, default=None,
        help='Single inference seed (backward compatible)')
    parser.add_argument('--paper-ours', choices=('F1','F2','F3'), default='F3')
    parser.add_argument('--progress-threshold-m', type=float, default=60.0)
    parser.add_argument('--figure-count', type=int, default=16)
    parser.add_argument('--figure-indices', default=None,
        help='Explicit comma-separated cohort sample indices, fixed before comparing methods')
    parser.add_argument('--representative-index', type=int, default=None)
    parser.add_argument('--output-dir', default='outputs/trace_fuel_ablation/open_loop_eval_v2')
    args = parser.parse_args(argv)
    checkpoints = parse_multi_checkpoint_specs(args.checkpoint)
    cfg = load_mode_yaml(args.config)
    dataset = args.dataset_root or cfg['training']['dataset_root']
    eval_multiseed(checkpoints, yaml_config=Path(args.config), dataset_root=Path(dataset),
        split=args.split, num_samples=args.num_samples, batch_size=args.batch_size,
        num_workers=args.num_workers, device=args.device,
        inference_seeds=[args.seed] if args.seed is not None else parse_seeds(args.seeds or '0,1,2,3,4'),
        output_dir=Path(args.output_dir), paper_ours=args.paper_ours,
        progress_threshold_m=args.progress_threshold_m, figure_count=args.figure_count,
        figure_indices=parse_seeds(args.figure_indices) if args.figure_indices is not None else None,
        representative_index=args.representative_index)
    return 0
