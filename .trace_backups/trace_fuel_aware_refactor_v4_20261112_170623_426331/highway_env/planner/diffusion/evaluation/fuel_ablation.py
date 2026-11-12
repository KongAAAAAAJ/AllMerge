"""Paired open-loop fuel-aware ablation for TRACE control-space diffusion.

The evaluation is strictly open loop. It calls infer_multimodal() and therefore
uses the same goal-endpoint selection as online planning. Reported fuel is an
estimate from frozen Koopman K2 states and a calibrated VT-CPFM power model;
not measured TruckSim fuel or closed-loop task completion.

The three paper metrics are:
  Fuel [L/100 km] = 100000 * sum(fuel_l) / sum(executed_path_length_m)
  Progress [m]   = mean(end-to-start target lane arc-length change)
  ADE [m]        = mean over all samples/timestamps of Euclidean XY error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline.expert_dataset import AllMergeExpertDatasetConfig, AllMergeExpertShardDataset
from highway_env.planner.diffusion.common.execution import K2_CONTEXT
from highway_env.planner.diffusion.common.fuel import FuelParameters, vt_cpfm_from_states
from highway_env.planner.diffusion.common.progress import centerline_progress
from highway_env.planner.diffusion.config.experiment import load_mode_yaml, resolve_repo_file
from highway_env.planner.diffusion.training.control import load_control_checkpoint

PAPER_COLUMNS = ('Method', 'Fuel (L/100 km)', 'Progress (m)', 'ADE (m)')
REQUIRED_LABELS = ('F0', 'F1', 'F2', 'F3')


def resolve_device(name: str) -> torch.device:
    return torch.device('cuda' if name == 'auto' and torch.cuda.is_available()
                        else 'cpu' if name == 'auto' else name)


def fuel_parameters_from_yaml(path: str | Path) -> FuelParameters:
    cfg = load_mode_yaml(path)
    if cfg['output_space'] != 'control':
        raise ValueError('Fuel evaluation requires output_space: control YAML')
    values = cfg.get('losses', {}).get('fuel', {}).get('parameters')
    if not isinstance(values, dict):
        raise ValueError('Set calibrated control.losses.fuel.parameters in YAML; '
                         'these must use litres/second at power in kW.')
    try:
        params = FuelParameters(**values)
        params.validate()
    except (TypeError, ValueError) as err:
        raise ValueError(f'Invalid calibrated fuel.parameters: {err}') from err
    return params


def checkpoint_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for part in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def k2_signature(model) -> str:
    if model.vehicle_model.kind != 'koopman_k2':
        raise ValueError('Fuel ablation requires koopman_k2 execution for every method')
    h = hashlib.sha256()
    for key, tensor in sorted(model.vehicle_model.koopman.state_dict().items()):
        h.update(key.encode('utf-8'))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_ablation_objectives(label: str, loss_cfg: dict, params: FuelParameters):
    """Guard against mislabeled checkpoints in the fuel-only ablation."""
    def w(name):
        return float(loss_cfg.get(name, {}).get('weight',
                     1.0 if name == 'trajectory' else 0.0))
    if not math.isclose(w('trajectory'), 1., abs_tol=1e-8):
        raise ValueError(f'{label}: expected trajectory.weight=1.0')
    if not math.isclose(w('ltr'), 0., abs_tol=1e-8):
        raise ValueError(f'{label}: expected ltr.weight=0 for this fuel-only ablation')
    if label == 'F0':
        if w('fuel') != 0. or w('progress') != 0.:
            raise ValueError('F0 must use trajectory imitation only (fuel=progress=0)')
    else:
        if w('fuel') <= 0 or w('progress') <= 0:
            raise ValueError(f'{label} needs both positive fuel and progress weights')
        used = loss_cfg.get('fuel', {}).get('parameters')
        if not isinstance(used, dict):
            raise ValueError(f'{label} must store fuel calibration in checkpoint')
        try:
            used_params = FuelParameters(**used)
        except TypeError as err:
            raise ValueError(f'{label} has invalid fuel calibration: {err}') from err
        for field, value in vars(params).items():
            if not math.isclose(float(getattr(used_params, field)), float(value), rel_tol=1e-7, abs_tol=1e-10):
                raise ValueError(f'{label} fuel calibration {field} differs from evaluation YAML')


def check_checkpoint_compatibility(reference, current, label):
    """Fail before evaluating incomparable planner structures/K2 surrogates."""
    if current['model_config'] != reference['model_config']:
        raise ValueError(f'{label}: model_config differs from F0')
    if current['control_config'] != reference['control_config']:
        raise ValueError(f'{label}: control_config differs from F0')
    if current['k2_signature'] != reference['k2_signature']:
        raise ValueError(f'{label}: frozen K2 weights differ from F0')
    if current['mode_selection'] != 'goal_endpoint':
        raise ValueError(f'{label}: expected goal_endpoint training mode')


def sample_metrics(prediction: dict, truth_xy: torch.Tensor, features: dict,
                   payload_kg: torch.Tensor, fuel_params: FuelParameters,
                   dt: float) -> dict[str, torch.Tensor]:
    """Compute batch metrics, no oracle mode selection or GT future used in selection."""
    poses = prediction['pose']
    speeds = prediction['speeds']
    xy = prediction['trajectory_dense']
    if poses.ndim != 3 or poses.shape[-1] != 3 or poses.shape[1] != 41:
        raise ValueError('Expected selected K2 poses [B,41,3]')
    if xy.shape != truth_xy.shape or xy.shape[1:] != (40, 2):
        raise ValueError('Expected predicted and expert dense XY [B,40,2]')
    if speeds.shape != (len(xy), 41):
        raise ValueError('Expected selected K2 longitudinal speeds [B,41]')
    if payload_kg.reshape(-1).shape[0] != len(xy):
        raise ValueError('Missing/invalid per-sample physical payload_kg')

    fuel = vt_cpfm_from_states(speeds, payload_kg.reshape(-1), fuel_params, dt=dt)
    path_segments = torch.linalg.vector_norm(poses[:, 1:, :2] - poses[:, :-1, :2], dim=-1)
    distance_m = path_segments.sum(dim=1)
    fuel_l = fuel['fuel_total']
    if bool((distance_m <= 1.0).any()):
        raise ValueError('Predicted executed distance <=1m; fuel per 100km is undefined')
    ade_m = torch.linalg.vector_norm(xy - truth_xy, dim=-1).mean(dim=1)
    progress_m = centerline_progress(poses[:, -1], features['target_lane_polyline'])
    fuel_rate = fuel['fuel_rate']
    cumulative = torch.cat((torch.zeros_like(fuel_l[:, None]),
                            torch.cumsum(fuel_rate, dim=1) * dt), dim=1)
    values = dict(fuel_l=fuel_l, distance_m=distance_m,
                  fuel_l_per_100km=100000.0 * fuel_l / distance_m,
                  progress_m=progress_m, ade_m=ade_m,
                  goal_terminal_error_m=torch.linalg.vector_norm(
                      poses[:, -1, :2] - features['target_point'], dim=-1),
                  selected_mode=prediction['trajectory_mode_idx'],
                  speeds_kmh=speeds * 3.6,
                  cumulative_fuel_l=cumulative)
    for key, value in values.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f'Non-finite open-loop evaluation values: {key}')
    return values


def aggregate(records: list[dict]) -> dict:
    if not records:
        raise ValueError('No evaluation records')
    total_fuel = math.fsum(x['fuel_l'] for x in records)
    total_distance = math.fsum(x['distance_m'] for x in records)
    if total_distance <= 1.0:
        raise ValueError('Insufficient travelled distance for L/100 km')
    return {
        'n_samples': len(records),
        'fuel_l_per_100km': 100000.0 * total_fuel / total_distance,
        'progress_m': math.fsum(x['progress_m'] for x in records) / len(records),
        'ade_m': math.fsum(x['ade_m'] for x in records) / len(records),
        'total_fuel_l': total_fuel,
        'total_executed_distance_m': total_distance,
        'mean_terminal_goal_error_m': math.fsum(x['goal_terminal_error_m'] for x in records) / len(records),
    }


def _format_scalar(v: torch.Tensor, k: int):
    return float(v[k].detach().cpu().item())


def _run_model(model, dataset, *, device, batch_size, num_workers, seed, params,
               method, representative_data):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == 'cuda')
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    records: list[dict] = []
    global_index = 0
    with torch.inference_mode():
        for features_cpu, targets_cpu in loader:
            features = {k: v.to(device, non_blocking=True) for k, v in features_cpu.items()}
            missing = [k for k in K2_CONTEXT if k not in targets_cpu]
            if missing:
                raise KeyError(f'Dataset missing synchronized K2 context: {missing}')
            if 'trajectory_dense' not in targets_cpu:
                raise KeyError('Dataset missing future_trajectory_dense [40,2]')
            context = {k: targets_cpu[k].to(device, non_blocking=True) for k in K2_CONTEXT}
            future = targets_cpu['trajectory_dense'].to(device)
            pred = model.infer_multimodal(features, generator=rng, context=context)
            m = sample_metrics(pred, future, features, context['koopman_payload_kg'],
                               params, dt=model.ctrl_cfg.dt)
            b = future.shape[0]
            for j in range(b):
                shard_id, shard_sample_index = dataset._index[global_index]
                row = {
                    'method': method, 'sample_index': global_index,
                    'shard_name': dataset.shard_paths[shard_id].name,
                    'shard_sample_index': shard_sample_index,
                    'selected_mode': int(m['selected_mode'][j]),
                    'fuel_l': _format_scalar(m['fuel_l'], j),
                    'distance_m': _format_scalar(m['distance_m'], j),
                    'fuel_l_per_100km': _format_scalar(m['fuel_l_per_100km'], j),
                    'progress_m': _format_scalar(m['progress_m'], j),
                    'ade_m': _format_scalar(m['ade_m'], j),
                    'goal_terminal_error_m': _format_scalar(m['goal_terminal_error_m'], j),
                }
                records.append(row)
                representative_data[method][global_index] = {
                    'speeds_kmh': m['speeds_kmh'][j].detach().cpu().numpy().copy(),
                    'cumulative_fuel_l': m['cumulative_fuel_l'][j].detach().cpu().numpy().copy(),
                }
                global_index += 1
    return records


def _write_csv(path: Path, records: list[dict], columns=None):
    if not records:
        raise ValueError('Refusing to write an empty CSV')
    with path.open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=columns or list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _latex_table(metrics: dict, ours_label: str) -> str:
    if 'F0' not in metrics or ours_label not in metrics:
        raise ValueError('Paper table needs F0 and selected paper ours checkpoint')
    def line(label, m, strong=False):
        cells = [label, f'{m["fuel_l_per_100km"]:.2f}',
                 f'{m["progress_m"]:.2f}', f'{m["ade_m"]:.2f}']
        if strong:
            cells = [r'\textbf{' + c + '}' for c in cells]
        return ' & '.join(cells) + r' \\'
    return '\n'.join([
        r'\begin{table}[!t]',
        r'\centering',
        r'\caption{Effects of Fuel-Aware Training on Open-Loop Planning Performance}',
        r'\label{tab:fuel_ablation}',
        r'\scriptsize',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{@{}lccc@{}}',
        r'\hline',
        r'\textbf{Method} & \shortstack{\textbf{Fuel}\\\textbf{(L/100 km)}} & '
        r'\shortstack{\textbf{Progress}\\\textbf{(m)}} & '
        r'\shortstack{\textbf{ADE}\\\textbf{(m)}} \\',
        r'\hline',
        line('F0: Imitation', metrics['F0']),
        line(ours_label + ': Ours', metrics[ours_label], strong=True),
        r'\hline',
        r'\end{tabular}',
        r'\end{table}',
        '',
    ])


def _figure(path: Path, reference: dict, proposed: dict, *, reference_label='F0', proposed_label='F3', dt=0.1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.arange(41) * dt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for label, item, style in ((reference_label, reference, '-'), (proposed_label, proposed, '--')):
        if not np.isfinite(item['cumulative_fuel_l']).all() or not np.isfinite(item['speeds_kmh']).all():
            raise FloatingPointError('Figure values nonfinite')
        axes[0].plot(t, item['cumulative_fuel_l'], style, linewidth=1.8, label=label)
        axes[1].plot(t, item['speeds_kmh'], style, linewidth=1.8, label=label)
    axes[0].set(xlabel='Time (s)', ylabel='Accumulated fuel (L)', title='(a) Fuel consumption')
    axes[1].set(xlabel='Time (s)', ylabel='Longitudinal speed (km/h)', title='(b) Vehicle speed')
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
        ax.set_xlim(0, t[-1])
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def evaluate(checkpoints: dict[str, Path], *, yaml_config: Path, dataset_root: Path,
             split='test', num_samples=1000, batch_size=2, num_workers=0,
             device='auto', seed=2048, output_dir: Path,
             paper_ours='F3', representative_index=None) -> dict:
    if 'F0' not in checkpoints:
        raise ValueError('F0 baseline checkpoint is mandatory')
    if paper_ours not in checkpoints:
        raise ValueError(f'Paper comparison checkpoint {paper_ours} missing')
    if batch_size < 1 or num_workers < 0 or num_samples < 0:
        raise ValueError('Invalid batch/worker/sample settings')
    params = fuel_parameters_from_yaml(yaml_config)
    selected_device = resolve_device(device)
    root = resolve_repo_file(dataset_root)
    ds = AllMergeExpertShardDataset(AllMergeExpertDatasetConfig(
        dataset_root=root, split=split, max_samples=num_samples or None))
    if not len(ds):
        raise ValueError('Dataset contains no evaluation samples')

    info = {}
    ordered_checkpoints = {k: checkpoints[k] for k in REQUIRED_LABELS if k in checkpoints}
    for label, path in ordered_checkpoints.items():
        cp = resolve_repo_file(path)
        if not cp.is_file():
            raise FileNotFoundError(f'Cannot find {label} checkpoint: {cp}')
        model, payload = load_control_checkpoint(cp, device='cpu')
        kind = model.vehicle_model.kind
        check_ablation_objectives(label, model.loss_config, params)
        meta = {
            'model_config': payload['model_config'],
            'control_config': payload['control_config'],
            'k2_signature': k2_signature(model),
            'mode_selection': model.loss_config.get('mode_selection'),
        }
        if label != 'F0':
            check_checkpoint_compatibility(info['F0']['compat'], meta, label)
        info[label] = {'checkpoint': str(cp), 'sha256': checkpoint_hash(cp),
                       'trained_epoch': payload.get('epoch'),
                       'loss_config': payload.get('loss_config', {}),
                       'compat': meta}
        if kind != 'koopman_k2':
            raise ValueError('Fuel-aware paper comparison must use K2 for all checkpoints')
        if meta['mode_selection'] != 'goal_endpoint':
            raise ValueError(f'{label} checkpoint not trained with goal_endpoint')
        del model

    # Evaluate exactly the same ordered samples using freshly seeded identical
    # random draws in each method. No GT-based mode oracle or cherry picking.
    records_by_method = {}
    trace_by_method: dict[str, dict[int, dict]] = {label: {} for label in ordered_checkpoints}
    for label, path in ordered_checkpoints.items():
        print(f'[EVAL] {label}: {len(ds)} samples, seed={seed}, device={selected_device}', flush=True)
        model, _ = load_control_checkpoint(resolve_repo_file(path), device='cpu')
        model.to(selected_device).eval()
        records_by_method[label] = _run_model(
            model, ds, device=selected_device, batch_size=batch_size,
            num_workers=num_workers, seed=seed, params=params, method=label,
            representative_data=trace_by_method)
        del model
        if selected_device.type == 'cuda':
            torch.cuda.empty_cache()

    summary = {label: aggregate(rows) for label, rows in records_by_method.items()}
    expected = [(x['shard_name'], x['shard_sample_index']) for x in records_by_method['F0']]
    for label, rows in records_by_method.items():
        actual = [(x['shard_name'], x['shard_sample_index']) for x in rows]
        if actual != expected:
            raise RuntimeError(f'Paired sample order mismatch: {label}')

    if representative_index is None:
        # Median-F0-fuel-intensity representative sample (not selected for Ours gain).
        base = records_by_method['F0']
        median_fuel = float(np.median([x['fuel_l_per_100km'] for x in base]))
        idx = min(range(len(base)), key=lambda i: (abs(base[i]['fuel_l_per_100km'] - median_fuel), i))
        representative_rule = 'nearest_to_F0_median_fuel_intensity'
    else:
        idx = int(representative_index)
        if not 0 <= idx < len(ds):
            raise ValueError(f'representative_index {idx} outside [0,{len(ds)-1}]')
        representative_rule = 'user_specified_index'

    out = resolve_repo_file(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for label in records_by_method for r in records_by_method[label]]
    _write_csv(out / 'metrics_per_sample.csv', rows)
    summary_rows = [{'method': k, **v} for k, v in summary.items()]
    _write_csv(out / 'metrics_summary.csv', summary_rows)
    (out / 'metrics_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'fuel_ablation_table.tex').write_text(_latex_table(summary, paper_ours), encoding='utf-8')
    _figure(out / 'fuel_ablation.pdf', trace_by_method['F0'][idx],
            trace_by_method[paper_ours][idx], proposed_label=paper_ours,
            dt=info['F0']['compat']['control_config']['dt'])
    record = records_by_method['F0'][idx]
    manifest = {
        'study': 'TRACE fuel-aware open-loop ablation',
        'metric_definitions': {
            'fuel_l_per_100km': '100000 * sum(VT-CPFM model fuel in L) / sum(K2 executed XY distance in m)',
            'progress_m': 'mean of K2 terminal minus initial target-lane arc-length coordinate',
            'ade_m': 'mean sample-wise average Euclidean error over 40 K2-executed positions vs expert XY',
        },
        'fuel_coefficient_units': {'fuel_rate': 'L/s', 'power': 'kW', 'fuel_integral': 'L'},
        'fuel_model': 'VT-CPFM on frozen Koopman K2 predicted vx, not TruckSim measured consumption',
        'evaluation_type': 'open_loop',
        'mode_selection': 'shared goal_endpoint inference',
        'seed': seed, 'split': split, 'samples': len(ds),
        'batch_size': batch_size, 'dataset_root': str(root),
        'yaml_config': str(resolve_repo_file(yaml_config)),
        'calibrated_fuel_parameters': vars(params),
        'paper_ours': paper_ours,
        'representative_sample': {**{key: record[key] for key in ('sample_index', 'shard_name', 'shard_sample_index')},
                                  'rule': representative_rule},
        'checkpoints': info,
        'training_initialization_warning': ('Matching network and K2 are enforced, but identical '
                                             'initial planner parameters must be guaranteed '
                                             'by the training protocol/checkpoint.'),
        'caveats': ['Open-loop estimated fuel, not measured fuel',
                    'Four-second predictions are not closed-loop success rates',
                    'All checkpoints must share network, diffusion configuration, and K2 parameters'],
    }
    (out / 'evaluation_manifest.json').write_text(json.dumps(manifest, indent=2,
                                                              ensure_ascii=False), encoding='utf-8')
    print('[EVAL DONE]', out, flush=True)
    for row in summary_rows:
        print(f'[{row["method"]}] Fuel={row["fuel_l_per_100km"]:.3f} L/100 km; '
              f'Progress={row["progress_m"]:.3f} m; ADE={row["ade_m"]:.3f} m', flush=True)
    return {'summary': summary, 'manifest': manifest, 'output_dir': out}


def parse_checkpoint_specs(items: list[str]) -> dict[str, Path]:
    paths = {}
    for item in items:
        label, sep, filename = item.partition('=')
        if not sep or not label or not filename:
            raise ValueError(f'--checkpoint expects LABEL=PATH; got {item!r}')
        label = label.upper()
        if label not in REQUIRED_LABELS:
            raise ValueError(f'Unknown checkpoint label: {label}; valid: {REQUIRED_LABELS}')
        if label in paths:
            raise ValueError(f'Duplicate checkpoint label: {label}')
        paths[label] = Path(filename)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description='TRACE paired fuel-aware open-loop evaluation; no success metric')
    parser.add_argument('--checkpoint', action='append', required=True, metavar='F0=PATH',
                        help='Repeat for F0, F1, F2, F3; F0 mandatory')
    parser.add_argument('--config', default='configs/train/diffusion.yaml',
                        help='Calibrated VT-CPFM parameters from current YAML')
    parser.add_argument('--dataset-root', default=None)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=2048,
                        help='Common candidate-generation noise across all checkpoints')
    parser.add_argument('--paper-ours', choices=REQUIRED_LABELS[1:], default='F3',
                        help='Which frozen/validation-selected method gets the paper comparison row')
    parser.add_argument('--representative-index', type=int, default=None)
    parser.add_argument('--output-dir', default='outputs/trace_fuel_ablation/open_loop_eval')
    args = parser.parse_args(argv)
    paths = parse_checkpoint_specs(args.checkpoint)
    cfg = load_mode_yaml(args.config)
    root = args.dataset_root or cfg['training']['dataset_root']
    evaluate(paths, yaml_config=Path(args.config), dataset_root=Path(root),
             split=args.split, num_samples=args.num_samples, batch_size=args.batch_size,
             num_workers=args.num_workers, device=args.device, seed=args.seed,
             paper_ours=args.paper_ours, representative_index=args.representative_index,
             output_dir=Path(args.output_dir))
    return 0
