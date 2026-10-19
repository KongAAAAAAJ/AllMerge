"""Reproduce EXACT paired-t0 terminal denoising; evaluate V6 reward on reconstructed road.

Source bank must have been collected from initial frozen pretrain. This is a
single-t0-input absorption study, not independent full DDIM rollout performance.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def _build_model(checkpoint,device):
    from pretraining.checkpoint_io import load_checkpoint_file
    from highway_env.planner.diffusion.config import build_structured_diffusion_config
    from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
    from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter
    import torch
    state,stored=load_checkpoint_file(checkpoint,map_location='cpu')
    cfg=build_structured_diffusion_config(**stored)
    model=StructuredDiffusionPlanner(cfg,PlannerTensorAdapter(cfg,device)).to(device)
    model.load_state_dict(state,strict=True)
    model.eval();model.requires_grad_(False)
    return model


def decode_paired(model,feature,t0,mode,device):
    """Same mode-locked terminal regression path as paired_replay_loss()."""
    import torch
    features={key:val.unsqueeze(0).to(device) for key,val in feature.items()}
    anchor=model.adapter.normalize_trajectory(features['coarse_trajectories'])
    expected=tuple(anchor.shape[1:])
    t0=t0.to(device=device,dtype=anchor.dtype)
    if tuple(t0.shape)!=expected:
        raise RuntimeError(f'Paired-t0 shape {tuple(t0.shape)} != {expected}')
    if not(0<=mode<anchor.shape[1]):raise RuntimeError('Teacher mode not valid for checkpoint')
    with torch.no_grad():
        scene=model.scene_encoder(features)
        pred,_=model.denoiser(t0[None],torch.zeros((1,),dtype=torch.long,device=device),
                              scene,navigation_point_norm=anchor[:,:,-1,:])
        xy=model.adapter.denormalize_trajectory(pred)[0,mode].detach().cpu()
    if tuple(xy.shape)!=(8,2) or not bool(torch.isfinite(xy).all()):
        raise RuntimeError('Paired decoder produced invalid [8,2] trajectory')
    return xy


def dense(xy,vel,cfg):
    import torch
    from highway_env.planner.diffusion.trajectory_mode_reward.spline_dense import evaluate_dense_spline
    with torch.no_grad():
        result=evaluate_dense_spline(xy[None].double(),cfg,initial_velocity_xy=vel.double())[0][0]
    if tuple(result.shape)!=(41,2):raise RuntimeError('Expected 41 dense points')
    return result.detach().cpu().numpy()


def _road_env(entry):
    """Reconstruct planning-time road only; saved features remain authoritative."""
    from scripts.run_grpo_noise_probe import SCENARIOS
    scene=entry['scenario']
    if scene not in SCENARIOS:raise RuntimeError('Unknown teacher scenario: '+str(scene))
    env=SCENARIOS[scene](config={'show_trajectories':False,'show_future_trajectories':False},render_mode=None)
    try:
        env.reset(seed=int(entry['env_seed']))
        _,_,terminated,truncated,_=env.step(0)
        if terminated or truncated:raise RuntimeError('Recreated teacher state ended early')
        return env
    except Exception:
        env.close();raise


def _compare_features(saved,actual,role):
    from scripts.run_grpo_noise_probe import FEATURE_KEYS
    import torch
    for key in FEATURE_KEYS:
        s=saved.get(key);a=actual.get(key)
        if a is not None:a=a[role]
        if s is None or a is None or tuple(s.shape)!=tuple(a.shape):
            raise RuntimeError(f'Cannot reproduce teacher scene feature {key}: '
                               f'{None if s is None else tuple(s.shape)} vs '
                               f'{None if a is None else tuple(a.shape)}')
        sa=s.detach().cpu();aa=a.detach().cpu()
        if s.dtype==torch.bool:
            if not torch.equal(sa,aa):raise RuntimeError(f'Teacher scene boolean feature changed: {key}')
        elif not torch.allclose(sa,aa,atol=1e-3,rtol=1e-4):
            raise RuntimeError(f'Teacher scene numeric feature changed: {key}')


def score_triplet(entry,model,pre,post,target,cfg,device,*,verify_scene=True):
    """Compute the same differentiable unified V6 task reward used for PGD teachers.

    Safety fields are partial diagnostics. Hard collision/TTC flags are deliberately
    absent: the teacher bank has only cached scene/features, not full traffic
    counterfactuals for each decoded candidate.
    """
    import torch
    from highway_env.planner.diffusion.guide_distill.reward_gradient import DifferentiableTaskReward
    from highway_env.planner.diffusion.guide_distill.road_boundary import RoadBoundaryField
    from highway_env.planner.diffusion.trajectory_mode_reward.state_adapter import _planning_state_snapshot,_vehicle_pose
    from scripts.run_grpo_noise_probe import _scenario_features
    env=_road_env(entry)
    try:
        if verify_scene:
            current=_scenario_features(env,model.adapter)
            _compare_features(entry['features'],current,int(entry['_teacher_role']))
        role=int(entry['_teacher_role'])
        snap=_planning_state_snapshot(env)
        pose=(np.asarray(snap[0][role],dtype=np.float64) if snap is not None
              else _vehicle_pose(env.controlled_vehicles[role]))
        proxy=DifferentiableTaskReward(cfg,device,high_precision=True)
        proxy.bind_road(RoadBoundaryField.from_road(env.road),pose)
        features=entry['features']
        proxy.bind_initial_velocity(features['ego_state'][:2])
        proxy.bind_centerline(features['target_lane_polyline'][:,:2].detach().cpu().numpy())
        xyz=torch.stack((pre,post,target),dim=0).to(device)
        with torch.no_grad():
            result=proxy.components(xyz)
            reward=result['balanced'].detach().cpu().numpy()
            road=proxy.road_field.components(xyz,pose,cfg,initial_velocity_xy=proxy.initial_velocity_xy)
            margin=road.get('road_min_margin_m')
            if margin is None:raise RuntimeError('Road components missing road_min_margin_m')
            margin=margin.detach().cpu().numpy().reshape(-1)
            curvature=result.get('curvature_max_abs')
            if curvature is not None:curvature=curvature.detach().cpu().numpy().reshape(-1)
        values={key:float(reward[i]) for i,key in enumerate(('pre','post','teacher'))}
        safety={key:{'minimum_road_margin_m':float(margin[i]),
                     **({'curvature_max_abs':float(curvature[i])} if curvature is not None else {}),
                     'road_pass':int(margin[i]>=0),
                     'curvature_pass':int(curvature is not None and curvature[i]<=float(cfg.curvature_limit_m_inv)+1e-6)}
                for i,key in enumerate(('pre','post','teacher'))}
        return values,safety
    finally:
        env.close()


def run_collect(teachers,pretrain_ckpt,posttrain_ckpt,cfg,device,replay_tol=0.002,limit=0,verify_scene=True):
    import torch
    from .metrics import metric_row,attach_safety
    pre_model=_build_model(pretrain_ckpt,device)
    post_model=_build_model(posttrain_ckpt,device)
    rows=[];samples={};skipped=[]
    accepted=[(j,s) for j,s in enumerate(teachers['states']) if s.get('teachers')]
    print(f'[collect] eligible states={len(accepted)} (teacher bank selection already safety-filtered)',flush=True)
    if limit>0:accepted=accepted[:limit]
    for idx,entry in accepted:
        try:
            # Only the best *already accepted* teacher, preserving native previous convention.
            teacher=max(entry['teachers'],key=lambda t:float(t['reward_after']))
            rp=teacher['paired_replay']
            if rp.get('schema')!='frozen_ddim_v1' or tuple(rp.get('timesteps',()))!=(8,0):
                raise RuntimeError('Missing valid frozen paired replay')
            if int(rp.get('source_group_id',-1))!=int(teacher['group_id']):
                raise RuntimeError('Teacher/source group mismatch')
            role=int(teacher['role']);mode=int(teacher['mode'])
            ref=decode_paired(pre_model,entry['features'],rp['t0'],mode,device)
            pre_saved=teacher['pretrain_xy']
            err=float((ref-pre_saved).abs().max())
            if not np.isfinite(err) or err>replay_tol:
                raise RuntimeError(f'Pretrain paired-replay mismatch: {err:.6g} m > tolerance {replay_tol}; '
                                   'teacher source is likely not the requested pretrain')
            post=decode_paired(post_model,entry['features'],rp['t0'],mode,device)
            target=teacher['target'].detach().cpu().double()
            vel=entry['features']['ego_state'][:2].detach().cpu().double()
            pd=dense(ref.double(),vel,cfg);qd=dense(post.double(),vel,cfg);td=dense(target,vel,cfg)
            # Read-only temporary field, no mutation of source entry persisted to disk.
            scratch=dict(entry);scratch['_teacher_role']=role
            score,safety=score_triplet(scratch,pre_model,ref,post,target,cfg,device,verify_scene=verify_scene)
            info={'state_index':int(idx),'scenario':str(entry['scenario']),'env_seed':int(entry['env_seed']),
                  'noise_seed':int(entry.get('noise_seed',-1)),'group_id':int(teacher['group_id']),
                  'mode':mode,'role':role,'teacher_count':len(entry['teachers']),
                  'paired_pretrain_max_abs_err_m':err,
                  'recorded_teacher_reward_gain':float(teacher['reward_gain'])}
            r=metric_row(pd,qd,td,score,info)
            for name in ('pre','post','teacher'):
                for k,v in safety[name].items():r[f'{name}_{k}']=v
            rows.append(r);samples[int(idx)]={'pre_dense':pd,'post_dense':qd,'teacher_dense':td,
                     'pre_sparse':ref.numpy(),'post_sparse':post.numpy(),'teacher_sparse':target.numpy()}
            print(f"[state] {idx} {entry['scenario']} ΔR={r['train_reward_gain']:+.4f} ΔteacherDist={r['teacher_distance_improve_m']:+.3f}m",flush=True)
        except Exception as exc:
            # Fail closed on provenance or replay error: do not cherry-pick silently.
            if any(k in str(exc).lower() for k in ('replay', 'teacher scene', 'cannot reproduce teacher', 'feature changed')):raise
            skipped.append({'state_index':idx,'reason':str(exc)})
            print(f'[skip] state={idx}: {exc}',flush=True)
    return rows,samples,skipped
