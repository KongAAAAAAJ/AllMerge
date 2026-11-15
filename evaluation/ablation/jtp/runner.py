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
from .reporting import paired_report, paper_table, write_csv, write_json
from .visualization import draw_projection_figure


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
    result = projector.project(nominal_u, context, roads)
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


def evaluate(*, checkpoint: str, diffusion_config: str, evaluation_config: str,
             jtp_config: str, dataset_root: str | None, split: str | None,
             num_frames: int | None, seeds: str | None, device_name: str | None,
             output_dir: str | None, figure_seed: int | None):
    config_path=resolve_repo_file(evaluation_config)
    ev=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(ev,dict): raise ValueError('JTP ablation config must be a YAML mapping')
    diffusion_path=resolve_repo_file(diffusion_config)
    diffusion_cfg=load_mode_yaml(diffusion_path)
    if diffusion_cfg.get('output_space')!='control':
        raise ValueError('JTP ablation requires control-space diffusion')
    jtp_path=resolve_repo_file(jtp_config or ev.get('jtp_config','configs/projection/jtp.yaml'))
    jtp_cfg=load_jtp_config(jtp_path)
    validate_evaluation_config(ev,jtp_cfg)
    checkpoint_path=resolve_repo_file(checkpoint)
    if not checkpoint_path.is_file():raise FileNotFoundError(checkpoint_path)
    root=resolve_repo_file(dataset_root or ev.get('dataset_root') or diffusion_cfg['training']['dataset_root'])
    if not root.is_dir():raise FileNotFoundError(root)
    split=split or ev.get('split','test')
    num_frames=int(num_frames if num_frames is not None else ev.get('num_frames',100))
    if num_frames < 0:raise ValueError('num_frames must be nonnegative (0 means all)')
    seed_values=parse_seeds(seeds if seeds is not None else ev.get('seeds','0,1,2'))
    sample_seed=int(ev.get('sample_seed',2026))
    figure_seed=int(figure_seed if figure_seed is not None else ev.get('figure_seed',2027))
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
    model.to(device).eval()
    projector=JointTrajectoryProjector(model.vehicle_model,config=jtp_path)
    frames,data_manifest=selected_frames(root,split,num_frames=num_frames,sample_seed=sample_seed)
    if not frames:raise RuntimeError('No complete synchronized frames to evaluate')
    pick=np.random.default_rng(figure_seed).integers(0,len(frames))
    figure_key=(seed_values[0],*frames[int(pick)][0]); figure_data=None
    rows=[]; errors=[]; frame_diagnostics=[]
    for seed in seed_values:
        for frame_index,(key,entries) in enumerate(frames):
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
                errors.append(error)
                raise RuntimeError(
                    f'Invalid road frame {key}; every selected synchronized frame must produce a JTP result. '
                    f'{error}') from exc
            actors=build_external_traffic(features,context,int(jtp_cfg['execution']['horizon_steps']),
                                          float(jtp_cfg['execution']['dt_s']),
                                          max_actors=int(ev.get('max_external_actors',12)),
                                          max_distance_m=float(ev.get('max_external_distance_m',150.)))
            data=_projection_pair(model,projector,features,context,roads,actors,jtp_cfg,ev,
                                 seed=seed,episode=episode,frame=frame,
                                 consistency_tol=float(ev.get('k2_consistency_tolerance',1.e-4)))
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
            if figure_data is None and ((seed,episode,frame)==figure_key or
                (seed==seed_values[-1] and frame_index==len(frames)-1)):
                nom=data['nominal'];post=data['projected'];bounds_a=[];bounds_b=[]
                for i in range(len(entries)):
                    bounds_a.append(stability_limits(nom['states'][i,1:,0].detach(),
                        context['koopman_payload_kg'][i],context['koopman_road_mu'][i],jtp_cfg).detach().cpu().numpy())
                    bounds_b.append(stability_limits(post['states'][i,1:,0].detach(),
                        context['koopman_payload_kg'][i],context['koopman_road_mu'][i],jtp_cfg).detach().cpu().numpy())
                figure_data={'seed':seed,'episode_index':episode,'frame_index':frame,
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
                    'external_actors':[a.pose for a in actors]}
            print(f'[JTP] seed={seed} frame={episode}/{frame} cars={len(entries)} '
                  f'projection={data["projection_status"]} before={sum(x["joint_feasible"] for x in data["before"])}/{len(entries)} '
                  f'after={sum(x["joint_feasible"] for x in data["after"])}/{len(entries)}',flush=True)
    if not rows:raise RuntimeError('No valid paired JTP observations')
    by_method,deltas,confidence=paired_report(rows,
        bootstrap_repeats=int(ev.get('bootstrap_repeats',2000)),bootstrap_seed=int(ev.get('bootstrap_seed',2026)))
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'jtp_per_vehicle.csv',rows)
    write_csv(output/'jtp_per_seed.csv',
        [{**{'seed':seed,'method':method},**paired_report([r for r in rows if r['seed']==seed],
         bootstrap_repeats=0)[0][method]} for seed in seed_values for method in ('Baseline','Ours')])
    write_csv(output/'jtp_summary.csv',[{'method':m,**v} for m,v in by_method.items()])
    write_json(output/'jtp_paired_statistics.json',{'difference_ours_minus_baseline_pp':deltas,
                                                    'paired_cluster_bootstrap':confidence})
    write_json(output/'jtp_projection_diagnostics.json',frame_diagnostics)
    write_json(output/'jtp_excluded_frames.json',errors)
    (output/'projection_ablation_table.tex').write_text(
        paper_table(by_method),encoding='utf-8')
    if figure_data:
        draw_projection_figure(output/'projection_ablation.pdf', figure_data,
             dt=float(jtp_cfg['execution']['dt_s']),
             kappa_lim=float(jtp_cfg['geometry']['curvature_limit_per_m']))
        np.savez_compressed(output/'projection_ablation_figure_data.npz',
            **{k:v for k,v in figure_data.items() if isinstance(v,np.ndarray)})
    manifest={'study':'TRACE Ablation 3: Joint Trajectory Projection',
              'projection_method':'Active-set Weighted Joint Jacobian Projection',
              'projection_output_policy':'Always retain the returned JTP controls and Koopman rollout; no nominal-output fallback. Projection exceptions abort the evaluation.',
              'projector_constraint_families':['full-horizon yaw-rate, sideslip, roll and LTR stability',
                                               'road-boundary clearance','trajectory curvature',
                                               'acceleration and steering magnitude'],
              'collision_handling':'external-actor and inter-truck collision are handled upstream by Adaptive Anchor and independently re-evaluated over the full horizon',
              'control_rate_handling':'not part of the lightweight JTP projection update; independently re-evaluated over the full horizon',
              'projector_constraints_satisfied_definition':'diagnostic flag for only the constraint families implemented inside JTP; not used as paper joint feasibility',
              'projection_applied_definition':'per-vehicle normalized control correction L2 norm exceeds configs/projection/jtp.yaml projection.change_tolerance',
              'reference_trajectories':'single shared frozen K2 inference per synchronized frame and seed',
              'deviation_definition':'per-vehicle mean pointwise Euclidean XY distance t=1..H from the paired nominal K2 execution trajectory, metres',
              'violation_denominator':'joint feasibility per synchronized N-truck frame/seed; geometry/stability/collision/control violation rates per vehicle trajectory',
              'uncertainty_unit':'cluster bootstrap resampling physical synchronized frames, preserving every inference seed and truck',
              'paper_metrics':['joint feasibility rate','geometric violation rate','stability violation rate','mean trajectory deviation'],
              'supplementary_metrics':['collision violation rates','control magnitude/rate violation rates',
                                       'normalized constraint-excess magnitudes','projection applied rate',
                                       'normalized control correction','projection latency','projection iterations'],
              'evaluation_scope':'offline expert-data open loop, not TruckSim closed loop',
              'dataset':data_manifest,'evaluated_seed_count':len(seed_values),
              'included_frames_across_seeds':len({(r['seed'],r['episode_index'],r['frame_index']) for r in rows}),
              'excluded_frames':len(errors),'plot_selection':{
                   'method':'preselected fixed random synchronized frame, not largest improvement',
                   'figure_seed':figure_seed,
                   'selected':{k:figure_data[k] for k in ('seed','episode_index','frame_index')} if figure_data else None},
              'checkpoint':{'path':str(checkpoint_path),'sha256':_sha256(checkpoint_path),
                  'model_execution':model.vehicle_model.kind,'anchor_strategy':model.anchors_config.get('strategy')},
              'configs':{'diffusion':str(diffusion_path),'jtp':str(jtp_path),
                         'jtp_sha256':_sha256(jtp_path),'evaluation':str(config_path),
                         'evaluation_sha256':_sha256(config_path)},
              'summary':by_method,'paired_deltas':deltas}
    write_json(output/'jtp_manifest.json',manifest)
    print(f'[COMPLETE] {output}; '
          f'joint feasibility {by_method["Baseline"]["joint_feasibility_pct"]:.2f}% -> '
          f'{by_method["Ours"]["joint_feasibility_pct"]:.2f}%', flush=True)
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
    parser.add_argument('--figure-seed',type=int,default=None)
    args=parser.parse_args(argv)
    evaluate(checkpoint=args.checkpoint,diffusion_config=args.diffusion_config,
             evaluation_config=args.config,jtp_config=args.jtp_config,
             dataset_root=args.dataset_root,split=args.split,
             num_frames=args.num_frames,seeds=args.seeds,device_name=args.device,
             output_dir=args.output_dir,figure_seed=args.figure_seed,
)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
