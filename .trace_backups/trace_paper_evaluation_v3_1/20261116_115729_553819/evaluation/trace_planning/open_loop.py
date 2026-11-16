"""TRACE control-space Diffusion + frozen K2 expert-trajectory open-loop evaluation.

No TruckSim is stepped here. Results must never be copied into the paper's
closed-loop Success/Collision/Spacing table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import score_open_loop, summarize_open_loop_rows


def _parse_seeds(value: str) -> list[int]:
    items = [int(x.strip()) for x in value.split(',') if x.strip()]
    if not items or len(items) != len(set(items)):
        raise ValueError('Provide unique, comma-separated inference seeds')
    return items


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _preflight(ds, *, require_k2=True, strategy='dynamic'):
    if len(ds) < 1:
        raise ValueError('Empty evaluation split')
    # Validate EVERY shard, preventing a late failure halfway through an experiment.
    for path in ds.shard_paths:
        with np.load(path, allow_pickle=False) as content:
            required = ['future_trajectory_dense', 'coarse_trajectories', 'mode_valid_mask']
            if strategy == 'dynamic':
                required.append('coarse_controls')
            if require_k2:
                from highway_env.planner.diffusion.common.execution import K2_CONTEXT
                required.extend(K2_CONTEXT)
            missing = [x for x in required if x not in content.files]
            if missing:
                raise ValueError(f'{path.name} lacks required TRACE fields: {missing}. '
                                 'Use the synchronized paired-anchor TruckSim expert dataset.')
            if content['future_trajectory_dense'].shape[1:] != (40,2):
                raise ValueError(f'{path.name}: future_trajectory_dense must be [N,40,2]')
    f,t = ds[0]
    if strategy == 'dynamic' and 'coarse_controls' not in f:
        raise ValueError('Dynamic control anchors were not saved in the expert dataset')
    if require_k2 and t['koopman_state_hist_si'].shape != (5,5):
        raise ValueError('K2 state history must be [5,5]')


def evaluate(*, checkpoint, config, dataset_root=None, split='val', num_samples=0,
             batch_size=8, num_workers=0, device='auto', seeds='0,1,2,3,4',
             output_dir='outputs/evaluation/trace_open_loop',
             jtp_config='configs/projection/jtp.yaml', allow_kinematic=False):
    from data_pipeline.expert_dataset import AllMergeExpertDatasetConfig, AllMergeExpertShardDataset
    from highway_env.planner.diffusion.config.experiment import load_mode_yaml, resolve_repo_file
    from highway_env.planner.diffusion.training.control import load_control_checkpoint
    from highway_env.planner.diffusion.projection.config import load_jtp_config
    from highway_env.planner.diffusion.common.fuel import FuelParameters
    from highway_env.planner.diffusion.common.execution import K2_CONTEXT

    cfg = load_mode_yaml(config)
    if cfg['output_space'] != 'control':
        raise ValueError('This is a Control Diffusion evaluator: output_space must be control')
    cp = resolve_repo_file(checkpoint)
    if not cp.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {cp}')
    dev = _device(device)
    model, payload = load_control_checkpoint(cp, device='cpu')
    if model.vehicle_model.kind != 'koopman_k2' and not allow_kinematic:
        raise RuntimeError('TRACE paper evaluation requires Koopman K2 checkpoint. '
                           'For architecture smoke tests only, pass --allow-kinematic.')
    if model.vehicle_model.kind == 'koopman_k2':
        values = cfg['losses'].get('fuel', {}).get('parameters')
        if not isinstance(values, dict):
            raise ValueError('VT-CPFM calibration missing: control.losses.fuel.parameters')
        fuel = FuelParameters(**values)
        fuel.validate()
        # The checkpoint is the architectural source of truth; fail when its
        # stored fuel calibration disagrees with the evaluation calibration.
        saved_values = (payload.get('loss_config') or {}).get('fuel',{}).get('parameters')
        if saved_values is not None:
            for field, value in vars(fuel).items():
                if field in saved_values and not np.isclose(
                    float(saved_values[field]),float(value),rtol=1e-7,atol=1e-10):
                    raise ValueError(f'Fuel calibration mismatch for {field}: '
                                     f'checkpoint={saved_values[field]}, YAML={value}')
    else:
        fuel = None
    model.to(dev).eval()
    root = resolve_repo_file(dataset_root or cfg['training']['dataset_root'])
    ds = AllMergeExpertShardDataset(AllMergeExpertDatasetConfig(
        dataset_root=root, split=split,
        max_samples=(int(num_samples) or None)))
    _preflight(ds, require_k2=(model.vehicle_model.kind == 'koopman_k2'),
               strategy=model.anchors_config.get('strategy', 'dynamic'))
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False,
                        num_workers=int(num_workers),
                        pin_memory=dev.type == 'cuda',
                        persistent_workers=int(num_workers) > 0)
    jtp_cfg = load_jtp_config(resolve_repo_file(jtp_config))
    out = resolve_repo_file(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps = _parse_seeds(seeds)
    all_records, by_seed = [], []
    horizon = int(model.ctrl_cfg.horizon_steps)
    if horizon != 40:
        raise ValueError('Expected TRACE 40x0.1s control horizon')
    for seed in steps:
        generator = torch.Generator(device=dev).manual_seed(seed)
        run_rows, global_idx, elapsed_times = [], 0, []
        shard_metadata_cache = {}
        with torch.inference_mode():
            for features_cpu, targets_cpu in loader:
                feats = {k: v.to(dev, non_blocking=True) for k,v in features_cpu.items()}
                truth = targets_cpu['trajectory_dense'].to(dev)
                ctx = {k: targets_cpu[k].to(dev, non_blocking=True) for k in K2_CONTEXT} if fuel is not None else {}
                _sync(dev)
                started = time.perf_counter()
                pred = model.infer_multimodal(feats, generator=generator, context=ctx)
                _sync(dev)
                elapsed_ms = (time.perf_counter()-started)*1000.
                elapsed_times.append(elapsed_ms)
                metrics = score_open_loop(pred, truth, feats, ctx,
                    dt=model.ctrl_cfg.dt, fuel_parameters=fuel,
                    stability_config=jtp_cfg, require_k2=fuel is not None)
                batch = int(truth.shape[0])
                for j in range(batch):
                    shard_id, shard_idx = ds._index[global_idx+j]
                    if shard_id not in shard_metadata_cache:
                        with np.load(ds.shard_paths[shard_id],allow_pickle=False) as part:
                            shard_metadata_cache[shard_id] = {
                                k: np.asarray(part[k]) for k in (
                                    'episode_index','frame_index','ego_index','seed') if k in part.files}
                    ids = shard_metadata_cache[shard_id]
                    row = {
                        **{key:int(v[shard_idx]) for key,v in ids.items()},
                        'inference_seed':seed, 'sample_index':global_idx+j,
                        'shard':ds.shard_paths[shard_id].name,
                        'shard_sample_index':shard_idx,
                        'execution_model':model.vehicle_model.kind,
                        'selected_mode':int(metrics['selected_mode'][j].item()),
                        'inference_ms_per_sample_throughput':elapsed_ms/batch,
                    }
                    for key,val in metrics.items():
                        if key != 'selected_mode':
                            value=float(val[j].detach().cpu().item())
                            row[key] = value if np.isfinite(value) else None
                    run_rows.append(row)
                global_idx += batch
        summary = summarize_open_loop_rows(run_rows)
        summary.update(inference_seed=seed,
                       inference_batch_ms=float(np.mean(elapsed_times)),
                       inference_throughput_ms_per_sample=float(np.mean([r['inference_ms_per_sample_throughput'] for r in run_rows])))
        by_seed.append(summary)
        all_records.extend(run_rows)
        print(f'[TRACE OPEN LOOP] seed={seed} n={len(run_rows)} '
              f'ADE={summary["ade_m"]:.4f}m FDE={summary["fde_m"]:.4f}m '
              f'K2-violation={summary.get("k2_stability_horizon_rate_pct",float("nan")):.2f}%',flush=True)
    _write_csv(out/'samples.csv',all_records)
    _write_csv(out/'metrics_by_seed.csv',by_seed)
    macro = {}
    for key in by_seed[0]:
        if key in ('inference_seed','num_samples'):
            continue
        vals = [float(s[key]) for s in by_seed if key in s and np.isfinite(float(s[key]))]
        if vals:
            macro[key]={'mean':float(np.mean(vals)),
                        'sd':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0}
    # For efficiency, also report all-seed pooled trajectory samples; for fuel,
    # both this and per-seed use distance-weighted normalization.
    pooled = summarize_open_loop_rows(all_records)
    manifest = {
        'evaluation_type':'open_loop_expert_imitation',
        'model':'TRACE ControlDiffusionPlanner',
        'plant':'frozen_koopman_k2_prediction' if fuel is not None else 'kinematic_smoke_only',
        'important':'No TruckSim is stepped. Success, collision, executed spacing and true closed-loop stability are NOT measured.',
        'checkpoint':str(cp),'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
        'checkpoint_architecture':payload.get('architecture'),
        'checkpoint_control_config':payload.get('control_config'),
        'checkpoint_execution_config':payload.get('execution_config'),
        'checkpoint_anchors_config':payload.get('anchors_config'),
        'yaml':str(resolve_repo_file(config)), 'dataset':str(root),
        'split':split,'samples_per_seed':len(ds),'seeds':steps,
        'jtp_enabled':False,
        'jtp_note':'Open-loop evaluation is NOMINAL K2: JTP requires synchronized geometric road inputs; test full TRACE in TruckSim closed loop.',
        'stability_threshold_source':str(resolve_repo_file(jtp_config)),
        'stability_definition':'Any violation over 40 future K2 steps for horizon metric; step fraction also reported.',
        'fuel_definition':'VT-CPFM estimated from K2 predicted longitudinal speed. L/100km = 100000*sum(fuel L)/sum(K2 XY distance m). Check calibration before paper use.',
        'planning_time_note':'Batched model inference only, synchronized CUDA, excluding data loading. Not paper closed-loop planning time.',
    }
    report={'manifest':manifest,'mean_sd_over_seeds':macro,'pooled':pooled,'per_seed':by_seed}
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='TRACE control-space K2 diffusion open-loop evaluator')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--config',default='configs/train/diffusion.yaml')
    parser.add_argument('--dataset-root',default=None)
    parser.add_argument('--split',choices=('all','train','val','test'),default='val')
    parser.add_argument('--num-samples',type=int,default=0,help='0 = full split')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--num-workers',type=int,default=0)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--seeds',default='0,1,2,3,4')
    parser.add_argument('--output-dir',default='outputs/evaluation/trace_open_loop')
    parser.add_argument('--jtp-config',default='configs/projection/jtp.yaml')
    parser.add_argument('--allow-kinematic',action='store_true',help='Smoke test only; no fuel/stability metrics')
    parser.add_argument('--eval-config',default='configs/evaluation/trace_planning.yaml')
    probe,_=parser.parse_known_args(argv)
    from .eval_config import defaults
    parser.set_defaults(**defaults(probe.eval_config,'open_loop'))
    args=parser.parse_args(argv)
    if args.num_samples < 0 or args.batch_size < 1 or args.num_workers < 0:
        parser.error('num-samples>=0, batch-size>=1 and num-workers>=0 are required')
    options=vars(args).copy()
    options.pop("eval_config")
    evaluate(**options)
    return 0
