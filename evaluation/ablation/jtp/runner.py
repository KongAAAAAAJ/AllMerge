"""TRACE third paper ablation: paired raw Diffusion K2 versus JTP projection.

Every selected synchronized frame is evaluated with the controls and execution
trajectory returned by JTP. Full paper feasibility is measured independently
from the complete constraint set; no nominal-output fallback is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from highway_env.planner.diffusion.config.experiment import resolve_repo_file, load_mode_yaml
from highway_env.planner.diffusion.training.control import load_control_checkpoint
from highway_env.planner.diffusion.projection import JointTrajectoryProjector, load_jtp_config
from highway_env.planner.diffusion.projection.constraints import stability_limits
from .data import (selected_frames, collate_frame, build_road_geometry,
                   build_external_traffic, local_to_world)
from .metrics import evaluate_group, trajectory_deviation_m, validate_evaluation_config
from .reporting import write_json
from .visualization import draw_projection_figures


def parse_seeds(value):
    seeds = [int(x) for x in str(value).split(',') if x.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Provide distinct comma-separated inference seeds')
    return seeds


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):
            h.update(data)
    return h.hexdigest()


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _seed_for_frame(seed, episode, frame):
    payload = f'JTP:{seed}:{episode}:{frame}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],'big') % (2**63-1)




def _road_safety_edges_world(road_meta, pose0, safety_margin_m):
    """Inset the measured road edges by the configured footprint safety margin."""
    center = np.asarray(road_meta['center_local'], dtype=np.float64)
    half_width = np.asarray(road_meta['half_width_m'], dtype=np.float64)
    left = np.asarray(road_meta['left_edge_local'], dtype=np.float64)
    right = np.asarray(road_meta['right_edge_local'], dtype=np.float64)
    ratio = np.divide(float(safety_margin_m), half_width,
                      out=np.full_like(half_width, np.nan), where=half_width > 1.0e-9)
    # A non-positive residual corridor is represented as NaN and is not drawn
    # as a feasible safety envelope. This avoids visually inventing safe space.
    valid = ratio < 1.0
    ratio = np.where(valid, ratio, np.nan)
    safe_left = left + ratio[:, None] * (center - left)
    safe_right = right + ratio[:, None] * (center - right)
    zeros = np.zeros((len(center), 1), dtype=np.float64)
    safe_left_world = local_to_world(np.column_stack((safe_left, zeros[:, 0])), pose0)[:, :2]
    safe_right_world = local_to_world(np.column_stack((safe_right, zeros[:, 0])), pose0)[:, :2]
    return safe_left_world, safe_right_world, bool(np.all(valid))


def _summarize_solver_diagnostics(frame_diagnostics):
    """Aggregate termination, active-set truncation, and group-wise progress."""
    from collections import Counter

    status_counts = Counter()
    group_values = {name: {key: [] for key in (
        'initial_max_violation', 'final_max_violation', 'max_violation_reduction',
        'initial_violation_count', 'final_violation_count', 'violation_count_reduction',
        'initial_violation_l2', 'final_violation_l2', 'violation_l2_reduction')}
        for name in ('stability', 'road', 'curvature', 'control', 'intertruck_collision', 'external_collision', 'control_rate')}
    active_total = Counter(); active_selected = Counter(); active_truncated = Counter()
    iterations = []; accepted_iterations = []
    linearization_ms = []; solver_ms = []; validation_ms = []; group_steps = []
    cache_builds = []; linearization_counts = []; validation_counts = []; methods = Counter()
    internal_feasible = 0
    for frame in frame_diagnostics:
        diag = frame.get('active_set_diagnostics') or {}
        status = str(frame.get('projection_status', 'unknown'))
        status_counts[status] += 1
        internal_feasible += int(bool(frame.get('projector_constraints_satisfied', False)))
        iterations.append(int(frame.get('iterations', 0)))
        accepted_iterations.append(int(diag.get('accepted_iterations', 0)))
        for name, values in (diag.get('group_progress') or {}).items():
            if name in group_values:
                for key in group_values[name]:
                    if key in values:
                        group_values[name][key].append(float(values[key]))
        cache_builds.append(int(diag.get('static_cache_builds', 0)))
        linearization_counts.append(int(diag.get('linearization_count', 0)))
        validation_counts.append(int(diag.get('nonlinear_validation_count', 0)))
        methods[str(diag.get('solver_method', 'unknown'))] += 1
        for step in diag.get('projection', []):
            linearization_ms.append(float(step.get('linearization_ms', 0)))
            solver_ms.append(float(step.get('grouped_solver_ms', 0)))
            validation_ms.append(float(step.get('nonlinear_validation_ms', 0)))
            group_steps.extend(step.get('groups', []))
            for name, value in (step.get('active_total_counts') or {}).items():
                active_total[name] += int(value)
            for name, value in (step.get('active_counts') or {}).items():
                active_selected[name] += int(value)
            for name, value in (step.get('active_truncated_counts') or {}).items():
                active_truncated[name] += int(value)

    mean_progress = {name: {key: (float(np.mean(values)) if values else None)
                             for key, values in metrics.items()}
                     for name, metrics in group_values.items()}
    truncation = {}
    for name in ('stability', 'road', 'curvature', 'control', 'intertruck_collision', 'external_collision', 'control_rate'):
        total = int(active_total[name]); selected = int(active_selected[name])
        truncation[name] = {
            'violated_constraints_considered': total,
            'active_constraints_selected': selected,
            'active_constraints_truncated': int(active_truncated[name]),
            'truncation_rate_pct': (100.0 * active_truncated[name] / total) if total else 0.0,
        }
    count = len(frame_diagnostics)
    return {
        'num_frame_seed_projections': count,
        'termination_status_counts': dict(status_counts),
        'internal_projector_feasible_pct': 100.0 * internal_feasible / count if count else None,
        'mean_iterations': float(np.mean(iterations)) if iterations else None,
        'mean_accepted_iterations': float(np.mean(accepted_iterations)) if accepted_iterations else None,
        'solver_method_counts': dict(methods),
        'mean_static_cache_builds': float(np.mean(cache_builds)) if cache_builds else None,
        'mean_linearization_count': float(np.mean(linearization_counts)) if linearization_counts else None,
        'mean_nonlinear_validation_count': float(np.mean(validation_counts)) if validation_counts else None,
        'mean_linearization_ms_per_outer': float(np.mean(linearization_ms)) if linearization_ms else None,
        'mean_grouped_solver_ms_per_outer': float(np.mean(solver_ms)) if solver_ms else None,
        'mean_nonlinear_validation_ms_per_outer': float(np.mean(validation_ms)) if validation_ms else None,
        'num_group_solves': len(group_steps),
        'num_group_protection_or_own_group_rejections': sum(not g.get('protection_ok', False) for g in group_steps),
        'max_applied_linear_protection_excess': max((g.get('protection_peak', 0.) for g in group_steps), default=0.),
        'mean_group_progress': mean_progress,
        'active_set_truncation': truncation,
    }


def _projection_pair(model, projector, features, context, roads, actors, cfg, ev_cfg,
                     *, seed, episode, frame, consistency_tol):
    device = next(model.parameters()).device
    rng = torch.Generator(device=device)
    rng.manual_seed(_seed_for_frame(seed, episode, frame))
    with torch.no_grad():
        prediction = model.infer_multimodal(features, generator=rng, context=context)
        nominal_u = prediction['control'].detach().clone()
        if 'states' not in prediction or 'ltr' not in prediction:
            raise RuntimeError('K2 inference did not return chassis states and LTR')
        nominal_rollout = projector.rollout_fn(nominal_u, context)
        nominal_world = nominal_rollout['trajectory'].detach().cpu().numpy()
        poses_local = prediction['pose'].detach().cpu().numpy()
        pose0 = context['koopman_pose0_si'].detach().cpu().numpy()
        from_local = np.stack([local_to_world(poses_local[i], pose0[i]) for i in range(len(pose0))])
        difference = from_local - nominal_world
        difference[..., 2] = np.arctan2(np.sin(difference[..., 2]), np.cos(difference[..., 2]))
        difference = max(
            np.abs(difference).max().item(),
            (prediction['states'] - nominal_rollout['states']).abs().max().item(),
            (prediction['ltr'] - nominal_rollout['ltr']).abs().max().item(),
        )
        if difference > consistency_tol:
            raise RuntimeError(
                f'Selected diffusion K2 and projector K2 differ by {difference:.4g}, '
                f'tolerance {consistency_tol:.4g}; paired evaluation invalid')
        before = evaluate_group(
            nominal_u, nominal_rollout['trajectory'], nominal_rollout['states'],
            nominal_rollout['ltr'], context, roads, actors, cfg, ev_cfg)

    _sync(device)
    start = time.perf_counter()
    # A returned JTP iterate is always evaluated, regardless of whether all
    # projector-internal constraints reached their tolerances. Exceptions are
    # surfaced and abort the evaluation; there is no nominal-trajectory fallback.
    result = projector.project(nominal_u, context, roads, external_actors=actors)
    _sync(device)
    latency_ms = (time.perf_counter() - start) * 1000.0

    projected_u = result.controls.detach()
    candidate = {
        'trajectory': result.trajectories.detach(),
        'states': result.states.detach(),
        'ltr': result.ltr.detach(),
    }
    with torch.no_grad():
        after = evaluate_group(
            projected_u, candidate['trajectory'], candidate['states'], candidate['ltr'],
            context, roads, actors, cfg, ev_cfg)

    control_scale = projected_u.new_tensor(cfg['control']['trust_scale']).view(1, 1, 2)
    delta_norm = torch.linalg.vector_norm(
        (projected_u - nominal_u) / control_scale, dim=(1, 2))
    change_tolerance = float(cfg['projection']['change_tolerance'])
    projection_applied = delta_norm > change_tolerance
    deviation = trajectory_deviation_m(
        nominal_world, candidate['trajectory'].detach().cpu().numpy())

    return dict(
        before=before,
        after=after,
        nominal_u=nominal_u,
        projected_u=projected_u,
        nominal=nominal_rollout,
        projected=candidate,
        deviation=deviation,
        projector_constraints_satisfied=bool(result.projector_constraints_satisfied),
        projection_status=str(result.status),
        projection_applied=projection_applied.detach().cpu().numpy(),
        projection_control_delta_norm=delta_norm.detach().cpu().numpy(),
        iterations=int(result.iterations),
        diagnostics=result.diagnostics or {},
        latency_ms=latency_ms,
        consistency_max_abs=difference,
    )


def _evaluate_frame(job, model, projector, jtp_cfg, ev, device):
    seed, frame_index, key, entries = job
    episode, frame = key
    features,context,meta=collate_frame(entries,device)
    if not {'agent_states','agent_valid_mask','map_polylines','map_valid_mask'}.issubset(features):
        raise KeyError('Missing world traffic/map features for paper JTP evaluation')
    pose0=context['koopman_pose0_si'].detach().cpu().numpy()
    roads=[]; roads_meta=[]
    try:
        for i in range(len(entries)):
            road,rmeta=build_road_geometry({k:v[i] for k,v in features.items()},pose0[i])
            roads.append(road); roads_meta.append(rmeta)
    except Exception as exc:
        error={'episode_index':episode,'frame_index':frame,'seed':seed,'reason':f'{type(exc).__name__}: {exc}'}
        raise RuntimeError(
            f'Invalid road frame {key}; every selected synchronized frame must produce a JTP result. '
            f'{error}') from exc
    collision_runtime = jtp_cfg.get('runtime', {}).get('collision', {})
    actors=build_external_traffic(features,context,int(jtp_cfg['execution']['horizon_steps']),
                                  float(jtp_cfg['execution']['dt_s']),
                                  max_actors=int(collision_runtime['max_external_actors']),
                                  max_distance_m=float(collision_runtime['max_external_distance_m']))
    data=_projection_pair(model,projector,features,context,roads,actors,jtp_cfg,ev,
                         seed=seed,episode=episode,frame=frame,
                         consistency_tol=float(ev.get('k2_consistency_tolerance',1.e-4)))
    return context, meta, pose0, roads, roads_meta, actors, data


def evaluate(*, checkpoint: str, diffusion_config: str, evaluation_config: str,
             jtp_config: str, dataset_root: str | None, split: str | None,
             num_frames: int | None, seeds: str | None, device_name: str | None,
             output_dir: str | None, constraint_distributions: bool = False,
             statistics_unit: str = 'both', workers: int | None = None,
             worker_threads: int | None = None):
    if statistics_unit not in ('both', 'trajectory', 'step'):
        raise ValueError('statistics_unit must be both, trajectory or step')
    units = ('trajectory', 'step') if statistics_unit == 'both' else (statistics_unit,)
    from .distributions import collect, KEYS, FLAGS
    from .statistics import step_records, write_reports
    from .cases import KINDS, frame_candidates, candidate_key, save_case
    config_path=resolve_repo_file(evaluation_config)
    ev=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(ev,dict): raise ValueError('JTP ablation config must be a YAML mapping')
    workers = int(workers if workers is not None else ev.get('workers', 1))
    worker_threads = int(worker_threads if worker_threads is not None else ev.get('worker_threads', 1))
    if workers < 1 or worker_threads < 1:
        raise ValueError('workers and worker_threads must be positive')
    diffusion_path=resolve_repo_file(diffusion_config)
    diffusion_cfg=load_mode_yaml(diffusion_path)
    if diffusion_cfg.get('output_space')!='control':
        raise ValueError('JTP ablation requires control-space diffusion')
    jtp_path=resolve_repo_file(jtp_config or ev.get('jtp_config','configs/projection/jtp.yaml'))
    jtp_cfg=load_jtp_config(jtp_path)
    validate_evaluation_config(ev,jtp_cfg)
    recorder = None
    if constraint_distributions:
        from .distributions import DistributionRecorder, collect, draw
        recorder = DistributionRecorder(jtp_cfg, ev)
    checkpoint_path=resolve_repo_file(checkpoint)
    if not checkpoint_path.is_file():raise FileNotFoundError(checkpoint_path)
    root=resolve_repo_file(dataset_root or ev.get('dataset_root') or diffusion_cfg['training']['dataset_root'])
    if not root.is_dir():raise FileNotFoundError(root)
    split=split or ev.get('split','test')
    num_frames=int(num_frames if num_frames is not None else ev.get('num_frames',100))
    if num_frames < 0:raise ValueError('num_frames must be nonnegative (0 means all)')
    seed_values=parse_seeds(seeds if seeds is not None else ev.get('seeds','0,1,2'))
    sample_seed=int(ev.get('sample_seed',2026))
    device_name=device_name or ev.get('device','auto')
    device=torch.device('cuda' if device_name=='auto' and torch.cuda.is_available() else
                        'cpu' if device_name=='auto' else device_name)
    output=resolve_repo_file(output_dir or ev.get('output_dir','outputs/jtp_ablation/evaluation'))
    model,payload=load_control_checkpoint(checkpoint_path,device='cpu')
    if model.vehicle_model.kind!='koopman_k2':
        raise ValueError('Paper JTP requires a frozen K2 execution model')
    if model.ctrl_cfg.horizon_steps!=int(jtp_cfg['execution']['horizon_steps']):
        raise ValueError('Diffusion/JTP horizon mismatch')
    if not np.isclose(model.ctrl_cfg.dt,float(jtp_cfg['execution']['dt_s']),atol=1.e-10):
        raise ValueError('Diffusion/JTP dt mismatch')
    model.to(device if workers == 1 else 'cpu').eval()
    projector=JointTrajectoryProjector(model.vehicle_model,config=jtp_path) if workers == 1 else None
    frames,data_manifest=selected_frames(root,split,num_frames=num_frames,sample_seed=sample_seed)
    if not frames:raise RuntimeError('No complete synchronized frames to evaluate')
    print(f'[JTP] device={device} workers={workers} frame/seed pairs={len(frames)*len(seed_values)}', flush=True)
    best_cases = {}
    case_figures = {}
    rows=[]; step_rows=[]; errors=[]; frame_diagnostics=[]
    from .parallel import frame_results
    jobs = ((seed, index, key, entries) for seed in seed_values
            for index, (key, entries) in enumerate(frames))
    evaluation_start = time.perf_counter()
    def serial_frame(job):
        return _evaluate_frame(job, model, projector, jtp_cfg, ev, device)
    worker_args = (str(checkpoint_path), str(jtp_path), jtp_cfg, ev, str(device), worker_threads)
    with frame_results(jobs, serial_frame, workers, worker_args) as results:
        for job, result in results:
            seed, frame_index, key, entries = job
            episode, frame = key
            context, meta, pose0, roads, roads_meta, actors, data = result
            collected_pair = {}
            step_metadata = [{'seed': seed, 'episode_index': episode, 'frame_index': frame,
                              'ego_index': int(item['ego_index'])} for item in meta]
            deviation_xy = np.linalg.norm(
                data['projected']['trajectory'].detach().cpu().numpy()[:, 1:, :2] -
                data['nominal']['trajectory'].detach().cpu().numpy()[:, 1:, :2], axis=-1)
            for method, control_key, rollout_key, expected_key in (
                    ('Baseline', 'nominal_u', 'nominal', 'before'),
                    ('Ours', 'projected_u', 'projected', 'after')):
                collected = collect(data[control_key], data[rollout_key], context,
                                    roads, actors, jtp_cfg, ev)
                collected_pair[method] = collected
                for k, flag in zip(KEYS, FLAGS):
                    for i, evaluated in enumerate(data[expected_key]):
                        if bool(np.any(collected[1][k][i])) != bool(evaluated[flag]):
                            raise ValueError(f'Per-step/evaluator mismatch: {method}/{k}/{i}')
                if recorder is not None:
                    recorder.add(method, collected, seed, episode, frame,
                                 [int(item['ego_index']) for item in meta], data[expected_key])
                if 'step' in units:
                    step_rows.extend(step_records(collected, step_metadata, method, deviation_xy))
            for i, item in enumerate(meta):
                base={'seed':seed,'episode_index':episode,'frame_index':frame,
                      'ego_index':int(item['ego_index']),
                      'dataset_seed':int(item.get('seed',-1)),
                      'payload_kg':float(context['koopman_payload_kg'][i].item()),
                      'road_mu':float(context['koopman_road_mu'][i].item()),
                      'group_size':len(entries),'external_actors':len(actors)}
                for method,values in [('Baseline',data['before'][i]),('Ours',data['after'][i])]:
                    frame_joint=all(r['joint_feasible'] for r in (data['before'] if method=='Baseline' else data['after']))
                    record={**base,'method':method,**values,'joint_frame_feasible':frame_joint,
                            'deviation_m':None if method=='Baseline' else float(data['deviation'][i]),
                            'projector_constraints_satisfied':None if method=='Baseline' else bool(data['projector_constraints_satisfied']),
                            'projection_applied':False if method=='Baseline' else bool(data['projection_applied'][i]),
                            'projection_control_delta_norm':0. if method=='Baseline' else float(data['projection_control_delta_norm'][i]),
                            'projection_status':'not_run' if method=='Baseline' else data['projection_status'],
                            'projection_iterations':0 if method=='Baseline' else data['iterations'],
                            'projection_latency_ms':0. if method=='Baseline' else data['latency_ms'],
                            'k2_consistency_max_abs':data['consistency_max_abs']}
                    if 'trajectory' in units:
                        rows.append(record)
            frame_diagnostics.append({
                'seed': seed, 'episode_index': episode, 'frame_index': frame,
                'projection_status': data['projection_status'],
                'projector_constraints_satisfied': data['projector_constraints_satisfied'],
                'projection_applied_by_vehicle': data['projection_applied'].tolist(),
                'control_delta_norm_by_vehicle': data['projection_control_delta_norm'].tolist(),
                'iterations': data['iterations'],
                'latency_ms': data['latency_ms'],
                'rollout_consistency': data['consistency_max_abs'],
                'active_set_diagnostics': data['diagnostics'],
            })
            candidates = frame_candidates(data, collected_pair, meta, seed, episode, frame, jtp_cfg)
            improved = {kind: c for kind, c in candidates.items()
                        if kind not in best_cases or candidate_key(c) > candidate_key(best_cases[kind])}
            if improved:
                nom=data['nominal'];post=data['projected'];bounds_a=[];bounds_b=[]
                for i in range(len(entries)):
                    bounds_a.append(stability_limits(nom['states'][i,1:,0].detach(),
                        context['koopman_payload_kg'][i],context['koopman_road_mu'][i],jtp_cfg).detach().cpu().numpy())
                    bounds_b.append(stability_limits(post['states'][i,1:,0].detach(),
                        context['koopman_payload_kg'][i],context['koopman_road_mu'][i],jtp_cfg).detach().cpu().numpy())
                safety_margin=float(jtp_cfg['geometry']['road_clearance_m']) + float(
                    jtp_cfg['geometry']['footprint']['circle_radius_m'])
                safety_edge_pairs=[_road_safety_edges_world(m, pose0[i], safety_margin)
                                   for i,m in enumerate(roads_meta)]
                frame_figure={'seed':seed,'episode_index':episode,'frame_index':frame,
                    'ego_ids':np.asarray([int(item['ego_index']) for item in meta]),
                    'dt_s':np.asarray(jtp_cfg['execution']['dt_s']),
                    'nominal_controls':data['nominal_u'].detach().cpu().numpy(),
                    'projected_controls':data['projected_u'].detach().cpu().numpy(),
                    'control_lower':np.asarray([jtp_cfg['control']['acceleration_min_mps2'], jtp_cfg['control']['steering_min_rad']]),
                    'control_upper':np.asarray([jtp_cfg['control']['acceleration_max_mps2'], jtp_cfg['control']['steering_max_rad']]),
                    'control_rate_limits':np.asarray([ev['limits']['acceleration_rate_max_mps3'], ev['limits']['steering_rate_max_rad_s']]),
                    'nominal_world':nom['trajectory'].detach().cpu().numpy(),
                    'projected_world':post['trajectory'].detach().cpu().numpy(),
                    'nominal_states':nom['states'].detach().cpu().numpy(),
                    'projected_states':post['states'].detach().cpu().numpy(),
                    'nominal_ltr':nom['ltr'].detach().cpu().numpy(),
                    'projected_ltr':post['ltr'].detach().cpu().numpy(),
                    'nominal_stability_bounds':np.stack(bounds_a),
                    'projected_stability_bounds':np.stack(bounds_b),
                    'road_edge_world':[local_to_world(np.column_stack((m['left_edge_local'],
                                                np.zeros(len(m['left_edge_local'])))),pose0[i])[:,:2]
                                       for i,m in enumerate(roads_meta)]+
                                      [local_to_world(np.column_stack((m['right_edge_local'],
                                                np.zeros(len(m['right_edge_local'])))),pose0[i])[:,:2]
                                       for i,m in enumerate(roads_meta)],
                    'road_safety_edge_world':[pair[0] for pair in safety_edge_pairs]+
                                             [pair[1] for pair in safety_edge_pairs],
                    'road_safety_envelope_valid_by_vehicle':[pair[2] for pair in safety_edge_pairs],
                    'road_safety_margin_m':safety_margin,
                    'external_actors':[a.pose for a in actors]}
                for kind, candidate in improved.items():
                    i = candidate['exemplar_index']
                    chosen = {**frame_figure, 'case_kind': kind, 'exemplar_index': i,
                              'ego_index': int(meta[i]['ego_index']),
                              'selection_grade': candidate['grade'], 'selection_score': candidate['rank'],
                              'before_violation_steps': candidate['before_count'],
                              'after_violation_steps': candidate['after_count']}
                    if kind == 'geometry':
                        road_before = -collected_pair['Baseline'][0]['road'][i]
                        road_after = -collected_pair['Ours'][0]['road'][i]
                        gain = np.maximum(road_before-ev['tolerances']['road_m'], 0)-np.maximum(road_after-ev['tolerances']['road_m'], 0)
                        focus = max(range(len(gain)), key=lambda t: (gain[t], abs(road_before[t]-road_after[t]), -t))
                        chosen.update(focus_step=focus+1, nominal_road_residual=road_before,
                                      projected_road_residual=road_after)
                    best_cases[kind] = candidate
                    case_figures[kind] = chosen
            print(f'[JTP] seed={seed} frame={episode}/{frame} cars={len(entries)} '
                  f'projection={data["projection_status"]} before={sum(x["joint_feasible"] for x in data["before"])}/{len(entries)} '
                  f'after={sum(x["joint_feasible"] for x in data["after"])}/{len(entries)}',flush=True)
    evaluation_wall_s = time.perf_counter() - evaluation_start
    if not frame_diagnostics:raise RuntimeError('No valid paired JTP observations')
    output.mkdir(parents=True,exist_ok=True)
    if recorder is not None:
        distribution_path = recorder.save(output)
        draw(distribution_path, output)
    statistics = {unit: write_reports(output, rows if unit == 'trajectory' else step_rows,
                  unit, seed_values, int(ev.get('bootstrap_repeats', 2000)),
                  int(ev.get('bootstrap_seed', 2026))) for unit in units}
    write_json(output/'jtp_projection_diagnostics.json',frame_diagnostics)
    solver_summary = _summarize_solver_diagnostics(frame_diagnostics)
    write_json(output/'jtp_solver_summary.json',solver_summary)
    write_json(output/'jtp_excluded_frames.json',errors)
    draw_projection_figures(output, case_figures)
    for kind in KINDS:
        save_case(output/f'projection_ablation_{kind}_data.npz', case_figures[kind])
    manifest={'execution': {'workers': workers, 'worker_threads': worker_threads if workers > 1 else torch.get_num_threads(),
                           'device': str(device), 'evaluation_wall_s': evaluation_wall_s,
                           'frame_seed_pairs_per_second': len(frame_diagnostics)/evaluation_wall_s,
                           'projection_latency_scope': 'per-worker wall time; concurrent resource contention included' if workers > 1 else 'serial wall time'},
              'study':'TRACE Ablation 3: Joint Trajectory Projection',
              'projection_method':'Priority-grouped non-degradation JTP (structured K2)',
              'projection_solver_method':jtp_cfg['projection']['solver_method'],
              'projection_output_policy':'Always retain the returned JTP controls and Koopman rollout; no nominal-output fallback. Projection exceptions abort the evaluation.',
                            'projector_constraint_families':['full-horizon yaw-rate, sideslip, roll and LTR stability',
                                               'road-boundary clearance','trajectory curvature',
                                               'acceleration and steering magnitude',
                                               'inter-truck and external-actor collision clearance',
                                               'acceleration and steering control rates'],
              'collision_handling':'inter-truck and constant-velocity external-actor collision constraints are included in the JTP active-set projection and independently re-evaluated over the full horizon',
              'control_rate_handling':'JTP projection and solver diagnostics include previous-applied-control to step 1; independent evaluation and distributions use only prediction-internal differences at steps 2..H',
              'control_rate_evaluation_scope':'prediction_steps_2_to_H',
              'control_rate_first_step_excluded':True,
              'projector_constraints_satisfied_definition':'diagnostic flag for only the constraint families implemented inside JTP; not used as paper joint feasibility',
              'projection_applied_definition':'per-vehicle normalized control correction L2 norm exceeds configs/projection/jtp.yaml projection.change_tolerance',
              'reference_trajectories':'single shared frozen K2 inference per synchronized frame and seed',
              'deviation_definition':'per-vehicle mean pointwise Euclidean XY distance t=1..H from the paired nominal K2 execution trajectory, metres',
              'violation_denominator':'see statistics entries for each selected unit',
              'uncertainty_unit':'cluster bootstrap resampling physical synchronized frames, preserving every inference seed and truck',
              'paper_metrics':['joint feasibility rate','single-vehicle feasibility rate','geometric violation rate','stability violation rate','mean trajectory deviation'],
              'supplementary_metrics':['collision violation rates','control magnitude/rate violation rates',
                                       'normalized constraint-excess magnitudes','projection applied rate',
                                       'normalized control correction','projection latency','projection iterations'],
              'evaluation_scope':'offline expert-data open loop, not TruckSim closed loop',
              'dataset':data_manifest,'evaluated_seed_count':len(seed_values),
              'included_frames_across_seeds':len(frame_diagnostics),
              'excluded_frames':len(errors),'plot_selection':{
                   'method':'independent ranked repair examples; not representative aggregate performance',
                   'cases':{kind:{'seed':c['identity'][0], 'episode_index':c['identity'][1],
                                  'frame_index':c['identity'][2], 'ego_index':c['identity'][3],
                                  'grade':c['grade'], 'score':c['rank'],
                                  'before_violation_steps':c['before_count'],
                                  'after_violation_steps':c['after_count']}
                            for kind,c in best_cases.items()}},
              'checkpoint':{'path':str(checkpoint_path),'sha256':_sha256(checkpoint_path),
                  'model_execution':model.vehicle_model.kind,'anchor_strategy':model.anchors_config.get('strategy')},
              'configs':{'diffusion':str(diffusion_path),'jtp':str(jtp_path),
                         'jtp_sha256':_sha256(jtp_path),'evaluation':str(config_path),
                         'evaluation_sha256':_sha256(config_path)},
              'statistics_unit':statistics_unit,'statistics':statistics,
              'solver_summary':solver_summary}
    write_json(output/'jtp_manifest.json',manifest)
    print(f'[COMPLETE] {output}; statistics units: {", ".join(units)}', flush=True)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description='TRACE third ablation: raw Diffusion K2 versus JTP')
    parser.add_argument('--config',default='configs/evaluation/jtp_ablation.yaml')
    parser.add_argument('--diffusion-config',default='configs/train/diffusion.yaml')
    parser.add_argument('--checkpoint',required=True,help='The SAME control diffusion checkpoint for both conditions')
    parser.add_argument('--jtp-config',default='configs/projection/jtp.yaml')
    parser.add_argument('--dataset-root',default=None)
    parser.add_argument('--split',choices=('train','val','test'),default=None)
    parser.add_argument('--num-frames',type=int,default=None,help='0=all complete synchronized frames')
    parser.add_argument('--seeds',default=None,help='Comma-separated diffusion noise seeds')
    parser.add_argument('--device',default=None)
    parser.add_argument('--output-dir',default=None)
    parser.add_argument('--constraint-distributions',action='store_true',help='Save full twelve-constraint distributions and plot two 3x2 figures')
    parser.add_argument('--workers', type=int, default=None, help='Parallel frame processes; default 1 (serial)')
    parser.add_argument('--worker-threads', type=int, default=None, help='CPU threads per parallel worker; default 1')
    parser.add_argument('--statistics-unit', choices=('both', 'trajectory', 'step'), default='both')
    args=parser.parse_args(argv)
    evaluate(checkpoint=args.checkpoint,diffusion_config=args.diffusion_config,
             evaluation_config=args.config,jtp_config=args.jtp_config,
             dataset_root=args.dataset_root,split=args.split,
             num_frames=args.num_frames,seeds=args.seeds,device_name=args.device,
             output_dir=args.output_dir,
             constraint_distributions=args.constraint_distributions,
             statistics_unit=args.statistics_unit,
             workers=args.workers,worker_threads=args.worker_threads,
)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
