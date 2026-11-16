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



def _anchor_prediction(model, features, context):
    """Evaluate the exact prepared paired anchor bank through the same K2.

    Baseline selection uses only the current target_point and valid mask.
    The future expert trajectory is never used to select a deployment mode.
    """
    controls = features['coarse_controls']
    outputs = model._all_predictions(controls, features, context)
    poses = outputs['poses']
    mode, logits = model._select_goal_endpoint_mode(poses, features)
    idx = torch.arange(controls.shape[0], device=controls.device)
    selected_pose = poses[idx, mode]
    result = {
        'control': controls[idx, mode],
        'control_candidates': controls,
        'pose': selected_pose,
        'pose_candidates': poses,
        'trajectory_dense': selected_pose[:, 1:, :2],
        'trajectory': selected_pose[:, 5::5, :2],
        'trajectory_dense_candidates': poses[:, :, 1:, :2],
        'trajectory_candidates': poses[:, :, 5::5, :2],
        'trajectory_mode_idx': mode,
        'trajectory_mode_logits': logits,
        'trajectory_candidate_valid_mask': features['mode_valid_mask'],
        'speeds': outputs['speeds'][idx, mode],
    }
    if outputs['states'] is not None:
        result.update(states=outputs['states'][idx, mode],
                      ltr=outputs['ltr'][idx, mode],
                      states_candidates=outputs['states'],
                      ltr_candidates=outputs['ltr'])
    return result


def _representative(pred, anchor_pred, truth, features, context, selected, sample_index, seed):
    """Collect random figure inputs, including real map lane geometry and all paired anchor modes."""
    j = selected
    mode = int(pred['trajectory_mode_idx'][j].item())
    as_np = lambda x: x.detach().cpu().numpy().copy()
    result = {
        'sample_index': sample_index, 'seed': seed, 'selected_mode': mode,
        'gt': as_np(truth[j]),
        'candidate_xy': as_np(pred['trajectory_dense_candidates'][j]),
        'candidate_valid': as_np(pred['trajectory_candidate_valid_mask'][j]),
        'selected_xy': as_np(pred['trajectory_dense'][j]),
        'anchor_candidate_xy': as_np(anchor_pred['trajectory_dense_candidates'][j]),
        'anchor_selected_xy': as_np(anchor_pred['trajectory_dense_candidates'][j, mode]),
        'control': as_np(pred['control'][j]),
        'anchor_control': as_np(anchor_pred['control_candidates'][j, mode]),
        'anchor_controls_bank': as_np(anchor_pred['control_candidates'][j]),
        'anchor_goal_selected_xy': as_np(anchor_pred['trajectory_dense'][j]),
        'anchor_goal_selected_mode': int(anchor_pred['trajectory_mode_idx'][j].item()),
        'anchor_kinematic_xy': as_np(features['coarse_trajectories'][j]),
        'map_polylines': as_np(features['map_polylines'][j]),
        'map_valid_mask': as_np(features['map_valid_mask'][j]),
        'target_point': as_np(features['target_point'][j]),
        'initial_speed_mps': float(features['ego_state'][j,0].item()),
        'goal_lateral_m': float(features['target_point'][j,1].item()),
    }
    if 'states' in pred:
        result['states'] = as_np(pred['states'][j])
        result['ltr'] = as_np(pred['ltr'][j])
        result['payload'] = float(context['koopman_payload_kg'][j].item())
        result['mu'] = float(context['koopman_road_mu'][j].item())
    return result

def evaluate(*, checkpoint, config, dataset_root=None, split='val', num_samples=0,
             batch_size=8, num_workers=0, device='auto', seeds='0,1,2,3,4',
             output_dir='outputs/evaluation/trace_open_loop',
             jtp_config='configs/projection/jtp.yaml', allow_kinematic=False,
             visualization_seed=0, figure_samples=8, chassis_stat_samples=64):
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
    # Uniform random sampling, independent of predictions/GT errors.
    # Capture 64 per-sample K2 responses for Fig. 4, reusing the same pool
    # for 8 trajectory panels and 3 condition-stratified control columns.
    rng = np.random.default_rng(int(visualization_seed))
    pool_size = min(len(ds), max(int(figure_samples), int(chassis_stat_samples), 3))
    chosen_indices = set(map(int, rng.choice(len(ds), size=pool_size, replace=False)))
    representative=[]
    temporal_sum=np.zeros(40,dtype=np.float64)
    temporal_sum_sq=np.zeros(40,dtype=np.float64)
    temporal_count=0
    candidate_rows=[]
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
                # A single anchor selection is shared by the baseline and the
                # diffusion inference, including adaptive/fixed anchor strategies.
                prepared = model._prepare_anchors(feats)
                anchor_pred = _anchor_prediction(model, prepared, ctx)
                _sync(dev)
                started = time.perf_counter()
                pred = model.infer_multimodal(prepared, generator=generator,
                                             context=ctx, anchors_prepared=True)
                _sync(dev)
                elapsed_ms = (time.perf_counter()-started)*1000.
                elapsed_times.append(elapsed_ms)
                metrics = score_open_loop(pred, truth, feats, ctx,
                    dt=model.ctrl_cfg.dt, fuel_parameters=fuel,
                    stability_config=jtp_cfg, require_k2=fuel is not None,
                    control_limits=model.control_limits)
                anchor_metrics = score_open_loop(anchor_pred, truth, prepared, ctx,
                    dt=model.ctrl_cfg.dt, fuel_parameters=fuel,
                    stability_config=jtp_cfg, require_k2=fuel is not None,
                    control_limits=model.control_limits)
                # Error-by-horizon statistics include every evaluated sample/seed.
                displacement = torch.linalg.vector_norm(pred['trajectory_dense'] - truth,dim=-1)
                temporal_sum += displacement.sum(dim=0).cpu().numpy()
                temporal_sum_sq += displacement.square().sum(dim=0).cpu().numpy()
                temporal_count += int(displacement.shape[0])
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
                    for old,new in (('ade_m','anchor_ade_m'),('fde_m','anchor_fde_m'),
                                    ('minade_k_m','anchor_minade_k_m'),
                                    ('minfde_k_m','anchor_minfde_k_m'),
                                    ('k2_stability_violated','anchor_k2_stability_violated'),
                                    ('fuel_l_4s','anchor_fuel_l_4s'),
                                    ('distance_m','anchor_distance_m')):
                        if old in anchor_metrics:
                            row[new]=float(anchor_metrics[old][j].cpu().item())
                    for base,metric,name in (
                        ('anchor_ade_m','ade_m','delta_ade_m'),
                        ('anchor_fde_m','fde_m','delta_fde_m'),
                        ('anchor_minade_k_m','minade_k_m','delta_minade_m'),
                        ('anchor_minfde_k_m','minfde_k_m','delta_minfde_m')):
                        row[name] = row[base]-row[metric]
                    valid = pred['trajectory_candidate_valid_mask'][j].bool()
                    ce=torch.linalg.vector_norm(
                        pred['trajectory_dense_candidates'][j] - truth[j, None],dim=-1)
                    for mid in range(int(valid.numel())):
                        candidate_rows.append(dict(inference_seed=seed,
                            sample_index=global_idx+j, mode=mid,
                            valid=bool(valid[mid].item()),
                            selected=bool(int(metrics['selected_mode'][j].item())==mid),
                            ade_m=float(ce[mid].mean().cpu().item()),
                            fde_m=float(ce[mid,-1].cpu().item())))
                    # The same-mode paired anchor is NOT the independently
                    # goal-selected anchor used by anchor_ade_m baseline.
                    same_mode = int(metrics['selected_mode'][j].item())
                    paired_xy = anchor_pred['trajectory_dense_candidates'][j, same_mode]
                    paired_err = torch.linalg.vector_norm(paired_xy - truth[j], dim=-1)
                    row['paired_anchor_ade_m'] = float(paired_err.mean().cpu().item())
                    row['paired_anchor_fde_m'] = float(paired_err[-1].cpu().item())
                    row['delta_paired_ade_m'] = row['paired_anchor_ade_m'] - row['ade_m']
                    row['delta_paired_fde_m'] = row['paired_anchor_fde_m'] - row['fde_m']
                    anchor_controls = prepared['coarse_controls'][j]
                    valid_controls = prepared['mode_valid_mask'][j].bool()
                    if bool(valid_controls.any()):
                        anchor_accel = anchor_controls[valid_controls, :, 0]
                        const = (anchor_accel.amax(dim=-1) - anchor_accel.amin(dim=-1) < 1e-5)
                        row['anchor_constant_accel_mode_fraction'] = float(const.float().mean().cpu().item())
                        row['anchor_accel_abs_mean_mps2'] = float(anchor_accel.abs().mean().cpu().item())
                        row['anchor_steering_time_variation_rad'] = float(
                            torch.diff(anchor_controls[valid_controls,:,1],dim=1).abs().sum(dim=1).mean().cpu().item())
                    if seed==steps[0] and global_idx+j in chosen_indices:
                        representative.append(_representative(
                            pred, anchor_pred, truth, prepared, ctx, j, global_idx+j,seed))
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
    _write_csv(out/'candidate_metrics.csv',candidate_rows)
    _write_csv(out/'control_metrics.csv',[{k:v for k,v in r.items() if k in (
        'sample_index','inference_seed','selected_mode','mean_abs_jerk_mps3',
        'max_abs_jerk_mps3','mean_abs_steer_rate_radps',
        'max_abs_steer_rate_radps','control_total_variation',
        'control_bound_violation_fraction','control_saturation_fraction')} for r in all_records])
    _write_csv(out/'anchor_metrics.csv',[{k:v for k,v in r.items() if k in (
        'sample_index','inference_seed','anchor_ade_m','anchor_fde_m',
        'anchor_minade_k_m','anchor_minfde_k_m','delta_ade_m',
        'delta_fde_m','delta_minade_m','delta_minfde_m',
        'anchor_k2_stability_violated','anchor_fuel_l_4s',
        'anchor_distance_m','paired_anchor_ade_m','paired_anchor_fde_m',
        'delta_paired_ade_m','delta_paired_fde_m',
        'anchor_constant_accel_mode_fraction','anchor_accel_abs_mean_mps2',
        'anchor_steering_time_variation_rad')} for r in all_records])
    _write_csv(out/'dynamics_metrics.csv',[{k:v for k,v in r.items() if k in (
        'sample_index','inference_seed','peak_yaw_rate_deg_s',
        'peak_sideslip_deg','peak_roll_deg','peak_ltr',
        'k2_stability_violated','k2_stability_step_fraction',
        'k2_yaw_rate_step_fraction','k2_sideslip_step_fraction',
        'k2_roll_step_fraction','k2_ltr_step_fraction','fuel_l_per_100km')} for r in all_records])
    _write_csv(out/'anchor_control_audit.csv',[{k:v for k,v in r.items() if k in (
        'sample_index','inference_seed','selected_mode',
        'anchor_constant_accel_mode_fraction','anchor_accel_abs_mean_mps2',
        'anchor_steering_time_variation_rad',
        'paired_anchor_ade_m','anchor_ade_m','ade_m',
        'delta_paired_ade_m','delta_ade_m')} for r in all_records])
    time_grid=np.arange(1,41,dtype=np.float64)*model.ctrl_cfg.dt
    error_series=[dict(time_s=float(t),mean_error_m=float(s/temporal_count),
        std_error_m=float(np.sqrt(max(0.,q/temporal_count-(s/temporal_count)**2))),
        observations=temporal_count) for t,s,q in zip(time_grid,temporal_sum,temporal_sum_sq)]
    _write_csv(out/'error_by_horizon.csv',error_series)
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
        'planning_time_note':'Batched model inference only, synchronized CUDA, excluding anchor-only rollout and data loading. Not paper closed-loop planning time.',
        'candidate_definition':'K valid modes, all evaluated at the same 40-step K2 prediction times; oracle minADE and minFDE are independently selected.',
        'anchor_definition':'Same prepared control bank, same K2, goal-endpoint model selection without using GT; figure shows the diffusion-selected mode paired anchor.',
        'plot_samples':f'Uniform random sample pool up to {pool_size}, visualization_seed={visualization_seed}; 8 trajectory and 3 varied-condition control profiles; no error ranking.',
        'anchor_metric_note':'anchor_ade_m = goal-selected independent anchor K2 ADE; paired_anchor_ade_m = same-mode anchor paired with diffusion selected mode. Never conflate these.',
        'anchor_control_note':'The A1 AllMergeAnchorBuilder longitudinal acceleration is piecewise constant by construction, not caused by plotting.',
        'paper_visualization_sources':'Representative arrays and CSV; paper graphics are generated from actual predictions.',
    }
    report={'manifest':manifest,'mean_sd_over_seeds':macro,'pooled':pooled,'per_seed':by_seed}
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    if representative:
        from .open_loop_visualization import plot_open_loop_figures
        plot_open_loop_figures(representative, all_records, error_series,
                               out/'figures',dt=model.ctrl_cfg.dt,
                               stability_config=jtp_cfg,
                               visualization_seed=int(visualization_seed),
                               trajectory_count=int(figure_samples),
                               chassis_count=int(chassis_stat_samples),
                               control_limits=model.control_limits)
    from .paper_reporting import write_open_loop_panel
    write_open_loop_panel(out)
    report_text=[
        '# TRACE Open-loop Evaluation (Nominal K2)',
        '',f'- Checkpoint: `{cp}`',f'- Dataset: `{root}`',
        f'- Split: `{split}`',f'- Samples per inference seed: {len(ds)}',
        f'- Inference seeds: {steps}',
        '- K2 predicts trajectories and chassis responses; no TruckSim is stepped.',
        '- JTP is disabled. These are not closed-loop success/collision/spacing metrics.',
        '- Paper minADE_K/minFDE_K use the 40-step **valid** candidate set.',
        '', '## Pooled metrics', '', '| Metric | Value |', '|---|---:|'
    ]
    report_text += [f'| {k} | {v:.6g} |' for k,v in pooled.items() if k!='num_samples']
    report_text += ['', '## Manuscript figure outputs', '']
    for name in ('open_loop_trajectory_distribution','open_loop_error_statistics',
                 'open_loop_control_profiles','open_loop_chassis_responses','open_loop_anchor_distribution'):
        report_text.append(f'- `figures/{name}.pdf`')
    (out/'report.md').write_text('\n'.join(report_text)+'\n',encoding='utf-8')
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
    parser.add_argument('--visualization-seed',type=int,default=0,
                        help='Random seed for reproducible figure sample selection')
    parser.add_argument('--figure-samples',type=int,default=8,
                        help='Number of random trajectory/anchor panels (paper uses 8)')
    parser.add_argument('--chassis-stat-samples',type=int,default=64,
                        help='Number of random K2 samples used in response percentile curves')
    parser.add_argument('--jtp-config',default='configs/projection/jtp.yaml')
    parser.add_argument('--allow-kinematic',action='store_true',help='Smoke test only; no fuel/stability metrics')
    parser.add_argument('--eval-config',default='configs/evaluation/trace_planning.yaml')
    probe,_=parser.parse_known_args(argv)
    from .eval_config import defaults
    parser.set_defaults(**defaults(probe.eval_config,'open_loop'))
    args=parser.parse_args(argv)
    if args.num_samples < 0 or args.batch_size < 1 or args.num_workers < 0 or args.figure_samples < 1 or args.chassis_stat_samples < 1:
        parser.error('num-samples>=0, batch-size>=1 and num-workers>=0 are required')
    options=vars(args).copy()
    options.pop("eval_config")
    evaluate(**options)
    return 0
