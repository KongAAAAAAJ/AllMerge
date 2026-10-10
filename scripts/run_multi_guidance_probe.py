#!/usr/bin/env python3
"""Read-only UNIFIED Reward Guidance exploration probe (V5).

Legacy filename intentionally retained for existing Windows launchers.
Only the current `progress_comfort` task objective guides the search.
No independent comfort, road, curvature, or progress guidance branches.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_grpo_noise_probe import (SCENARIOS,CheckpointSpec,_build_model,
    _frozen_reward_context,_scenario_features,_state_seed,_sample_noise_seed)
from highway_env.planner.diffusion.grpo.sampling import GroupDiffusionSampler
from highway_env.planner.diffusion.grpo.reward_adapter import CandidateRewardAdapter,resolve_reward_evaluator
from highway_env.planner.diffusion.grpo.task_reward import task_reward_from_w4_result
from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
from highway_env.planner.diffusion.guide_distill.reward_gradient import DifferentiableTaskReward,DIRECTIONS
from highway_env.planner.diffusion.guide_distill.exploration import guide_all
from highway_env.planner.diffusion.guide_distill.matp_adapter import load_matp,project_w1
from highway_env.planner.diffusion.guide_distill.road_boundary import RoadBoundaryField,_polygon_components
from highway_env.planner.diffusion.trajectory_mode_reward.centerline_reward import centerline_reward_components
from highway_env.planner.diffusion.trajectory_mode_reward.state_adapter import _planning_state_snapshot,_vehicle_pose
from highway_env.planner.diffusion.trajectory_mode_reward.spline_dense import evaluate_dense_spline


def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--scenario',choices=('all',*SCENARIOS.keys()),default='all')
    p.add_argument('--samples-per-scenario',type=int,default=9)
    p.add_argument('--group-size',type=int,default=48)
    p.add_argument('--guide-count',type=int,default=12)
    p.add_argument('--role',type=int,choices=(0,1,2),default=0)
    p.add_argument('--seed',type=int,default=20261008)
    p.add_argument('--noise-seed',type=int,default=70261008)
    p.add_argument('--eta',type=float,default=.02)
    p.add_argument('--guidance-iters',type=int,default=8)
    p.add_argument('--guidance-step-m',type=float,default=.12)
    p.add_argument('--guidance-trust-rms-m',type=float,default=.75)
    p.add_argument('--guidance-max-point-move-m',type=float,default=1.25)
    p.add_argument('--reward-type',choices=('progress_comfort',),default='progress_comfort')
    # V5: same task reward for production, Guidance and MATP post-score.
    p.add_argument('--centerline-weight',type=float,default=0.15)
    p.add_argument('--centerline-scale-m',type=float,default=2.0)
    p.add_argument('--centerline-late-power',type=float,default=2.0)
    p.add_argument('--curvature-weight',type=float,default=0.05,
                   help='0=V4; positive=V5 full-reward curvature term')
    p.add_argument('--curvature-peak-weight',type=float,default=0.5)
    p.add_argument('--curvature-margin-weight',type=float,default=0.05,
                   help='0=V5-A violation+peak; default=V5-B')
    p.add_argument('--road-sampling-m',type=float,default=.4)
    p.add_argument('--road-inside-scale-m',type=float,default=2.0)
    p.add_argument('--road-inside-bonus',type=float,default=1.)
    p.add_argument('--road-outside-slope',type=float,default=.5)
    p.add_argument('--reward-parity-tol',type=float,default=.002)
    p.add_argument('--guidance-root',type=Path,default=Path('all_merge-guidance'))
    p.add_argument('--no-matp',action='store_true')
    p.add_argument('--group-action',type=int,default=0)
    p.add_argument('--max-seed-attempts',type=int,default=1000)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir',type=Path,default=Path('outputs/guide_distill_reward_v5'))
    p.add_argument('--no-plots',action='store_true')
    return p.parse_args()


def paired_score(adapter,features,context,candidates,*,role,mode):
    result=adapter.evaluate_result(candidates,features=features,context=context)
    task=task_reward_from_w4_result(result,context=context,device=candidates.device,
                                     dtype=candidates.dtype,reward_type='progress_comfort')
    if tuple(task.shape)!=(3,candidates.shape[1],10):
        raise RuntimeError('Unexpected current task reward shape '+str(tuple(task.shape)))
    values=task[role,:,mode].detach().cpu().numpy().copy()
    def field(key):
        if hasattr(result,key):arr=np.asarray(getattr(result,key))
        elif key in result.components:arr=np.asarray(result.components[key])
        else:return np.full(len(values),np.nan)
        return np.asarray(arr[role,mode,:]).copy()
    info={k:field(k) for k in ('unsafe','collision','out_of_drivable',
        'minimum_road_margin_m','comfort_penalty','progress_score','road_boundary_reward',
        'curvature_penalty','curvature_max_abs','curvature_valid_points',
        'centerline_penalty','centerline_mean_error_m','centerline_late_error_m','centerline_valid',
        'curvature_violation_penalty','curvature_peak_penalty','curvature_margin_penalty')}
    info['legacy_w4_reward']=np.asarray(result.rewards)[role,mode,:].copy()
    return values,info,result


def save_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for r in rows for key in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def avg(rows,key):
    v=np.asarray([r.get(key,float('nan')) for r in rows],dtype=float)
    return float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else None


def create_summary(rows,semantics,matp):
    base=np.asarray([r['reward_before'] for r in rows],dtype=float)
    guided=np.asarray([r['reward_guided'] for r in rows],dtype=float)
    entry={'N':len(rows),'reward_before_mean':float(base.mean()),
           'guided_reward_mean':float(guided.mean()),
           'guided_delta_mean':float((guided-base).mean()),
           'guided_positive_fraction':float(((guided-base)>1e-6).mean()),
           'guided_exceeds_noise48_best_fraction':avg(rows,'guided_exceeds_noise48_best'),
           'guided_ade_m':avg(rows,'guided_ade_m'),  # 10Hz dense, t=0.1..4.0s
           'guided_lateral_ade_m':avg(rows,'guided_lateral_ade_m'),
           'sparse_control_ade_m':avg(rows,'sparse_control_ade_m'),
           'sparse_control_lateral_ade_m':avg(rows,'sparse_control_lateral_ade_m'),
           'road_guided_gain_mean':avg(rows,'road_reward_guided_gain'),
           'road_min_margin_guided_mean':avg(rows,'road_union_min_margin_guided_m'),
           'road_feasible_guided':avg(rows,'road_feasible_guided'),
            'curvature_feasible_guided':avg(rows,'guided_curvature_feasible'),
           'reward_without_curvature_gain_mean':avg(rows,'reward_without_curvature_gain'),
           'centerline_penalty_guided_mean':avg(rows,'centerline_penalty_guided'),
           'centerline_late_error_before_m':avg(rows,'centerline_late_error_before_m'),
           'centerline_late_error_guided_m':avg(rows,'centerline_late_error_guided_m'),
           'curvature_penalty_before_mean':avg(rows,'curvature_penalty_before'),
           'curvature_penalty_guided_mean':avg(rows,'curvature_penalty_guided'),
           'curvature_max_abs_guided_mean':avg(rows,'curvature_max_abs_guided'),
           'unsafe_guided':avg(rows,'guided_unsafe'),
           'gradient_norm_mean':avg(rows,'initial_gradient_norm')}
    if matp:
        post=np.asarray([r['reward_matp'] for r in rows],dtype=float)
        entry.update({'matp_reward_mean':float(post.mean()),
                      'matp_vs_guided_reward_mean':float((post-guided).mean()),
                      'matp_vs_original_reward_mean':float((post-base).mean()),
                      'matp_ade_m':avg(rows,'matp_ade_m'),
                      'centerline_late_error_matp_m':avg(rows,'centerline_late_error_matp_m'),
                      'matp_lateral_ade_m':avg(rows,'matp_lateral_ade_m'),
                      'road_feasible_matp':avg(rows,'road_feasible_matp'),
                      'curvature_feasible_matp':avg(rows,'matp_curvature_feasible'),
                      'curvature_penalty_matp_mean':avg(rows,'curvature_penalty_matp'),
                      'curvature_max_abs_matp_mean':avg(rows,'curvature_max_abs_matp'),
                      'curvature_valid_points_matp_mean':avg(rows,'curvature_valid_points_matp'),
                      'unsafe_matp':avg(rows,'matp_unsafe'),
                      'teacher_candidate_rate':avg(rows,'teacher_candidate')})
    summary={'trajectory_comparison':'All geometry and ADE use shared ClampedCubicTrajectorySpline 10Hz, 41 samples; ADE excludes fixed t=0',
             'reward_source':'current progress_comfort (V5 curvature, cubic dense, 100m)',
             'guidance_directions':list(DIRECTIONS),'reward_semantics':semantics,
             'matp_enabled':matp,'unified':entry,'by_scenario':{}}
    for scenario in sorted(set(r['scenario'] for r in rows)):
        sub=[r for r in rows if r['scenario']==scenario]
        summary['by_scenario'][scenario]={
            'N':len(sub),'reward_guided_gain':avg(sub,'guided_delta_reward'),
            'reward_matp_gain_vs_original':avg(sub,'matp_vs_original_reward'),
            'road_feasible_guided':avg(sub,'road_feasible_guided'),
            'road_feasible_matp':avg(sub,'road_feasible_matp'),
            'curvature_feasible_matp':avg(sub,'matp_curvature_feasible'),
            'curvature_penalty_before':avg(sub,'curvature_penalty_before'),
            'curvature_penalty_guided':avg(sub,'curvature_penalty_guided'),
            'curvature_penalty_matp':avg(sub,'curvature_penalty_matp'),
            'teacher_candidate_rate':avg(sub,'teacher_candidate'),
        }
    return summary


def create_plots(rows,gallery,out,has_matp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    gains=[r['guided_delta_reward'] for r in rows]
    ax.hist(gains,bins=35);ax.axvline(0,c='gray');ax.set_title('Unified Reward Guidance: paired gain')
    ax.set_xlabel('Guided reward - original reward')
    fig.tight_layout();fig.savefig(out/'01_unified_reward_gain.png',dpi=180);plt.close(fig)
    if has_matp:
        fig,ax=plt.subplots(figsize=(7,4))
        gains=[r['matp_vs_guided_reward'] for r in rows]
        ax.hist(gains,bins=35);ax.axvline(0,c='gray');ax.set_title('Frozen MATP W1: reward impact')
        ax.set_xlabel('MATP reward - Guided reward')
        fig.tight_layout();fig.savefig(out/'02_matp_reward_retention.png',dpi=180);plt.close(fig)
    scenarios=sorted(set(r['scenario'] for r in rows))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(len(scenarios));width=.28
    for ax,keypairs,title in [(axes[0],('road_feasible_guided','road_feasible_matp'),'Road feasible'),
                               (axes[1],('guided_curvature_feasible','matp_curvature_feasible'),'Curvature feasible')]:
        ax.bar(x-width/2,[avg([r for r in rows if r['scenario']==s],keypairs[0]) or 0 for s in scenarios],width,label='Guided')
        if has_matp:
            ax.bar(x+width/2,[avg([r for r in rows if r['scenario']==s],keypairs[1]) or 0 for s in scenarios],width,label='MATP')
        ax.set_xticks(x,scenarios,rotation=20);ax.set_ylim(0,1);ax.set_title(title);ax.legend()
    fig.tight_layout();fig.savefig(out/'03_feasibility_by_scenario.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    before=np.asarray([r['curvature_max_abs_before'] for r in rows])
    guided=np.asarray([r['curvature_max_abs_guided'] for r in rows])
    ax.hist(before,bins=35,alpha=.5,label='Original')
    ax.hist(guided,bins=35,alpha=.5,label='Unified guidance')
    ax.axvline(.02,c='black',ls='--',label='MATP limit')
    ax.set_xlabel('Max absolute curvature (1/m)');ax.set_title('V5 Curvature before/after guidance');ax.legend()
    fig.tight_layout();fig.savefig(out/'06_curvature_before_after_guidance.png',dpi=180);plt.close(fig)
    if gallery:
        rng=np.random.default_rng(123)
        ids=np.arange(len(gallery));rng.shuffle(ids)
        picks=[gallery[i] for i in ids[:9]]
        fig,axes=plt.subplots(3,3,figsize=(13,12))
        for ax,e in zip(axes.flat,picks):
            def draw(dense_key,sparse_key,label):
                dense=e[dense_key]  # exact production 10Hz ClampedCubicTrajectorySpline
                line,=ax.plot(dense[:,0],dense[:,1],lw=1.5,label=label)
                ax.scatter(e[sparse_key][:,0],e[sparse_key][:,1],
                           s=10,marker='o',alpha=.5,color=line.get_color())  # control points only
            draw('base_dense','base','Original')
            draw('guided_dense','guided','Unified guidance')
            if e['matp_dense'] is not None:draw('matp_dense','matp','MATP W1')
            if 'target_lane_centerline' in e:
                ref=e['target_lane_centerline']
                ax.plot(ref[:,0],ref[:,1],c='purple',ls='--',lw=1,label='Target lane center')
            ax.grid(alpha=.25);ax.set_aspect('equal',adjustable='datalim')
            ax.set_title(f"{e['scenario']} state {e['state_id']}")
        axes.flat[0].legend(fontsize=8)
        for ax in axes.flat[len(picks):]:ax.axis('off')
        fig.tight_layout();fig.savefig(out/'04_unified_gallery_3x3.png',dpi=180);plt.close(fig)
        curves=[e for e in gallery if e['scenario']=='curved'][:9]
        if curves:
            fig,axes=plt.subplots(3,3,figsize=(13,12))
            for ax,e in zip(axes.flat,curves):
                pose=e['role_pose'];c=np.cos(pose[2]);s=np.sin(pose[2])
                def world(xy):
                    return np.column_stack((pose[0]+c*xy[:,0]-s*xy[:,1],pose[1]+s*xy[:,0]+c*xy[:,1]))
                for key,label in [('base_dense','Original'),('guided_dense','Unified'),('matp_dense','MATP')]:
                    if e[key] is not None:
                        pts=world(e[key])  # transform 41 dense samples, NOT 8 control points
                        ax.plot(pts[:,0],pts[:,1],lw=1.3,label=label)
                for bound in e['road_outline']:ax.plot(bound[:,0],bound[:,1],c='black',lw=.7)
                if 'target_lane_centerline' in e:
                    mid=world(e['target_lane_centerline'])
                    ax.plot(mid[:,0],mid[:,1],c='purple',ls='--',lw=1.3,label='Target lane center')
                center=world(e['base_dense'])
                ax.set_xlim(center[:,0].min()-12,center[:,0].max()+12)
                ax.set_ylim(center[:,1].min()-12,center[:,1].max()+12)
                ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
            axes.flat[0].legend(fontsize=8)
            for ax in axes.flat[len(curves):]:ax.axis('off')
            fig.tight_layout();fig.savefig(out/'05_curved_outer_boundary_3x3.png',dpi=180);plt.close(fig)
    # Explicit V5.2 metric: target-lane late (2-4s) offset, not road-union margin.
    fig,ax=plt.subplots(figsize=(9,4))
    xs=np.arange(len(scenarios));ww=.23
    for offset,key,label in [(-ww,'centerline_late_error_before_m','Original'),
                              (0,'centerline_late_error_guided_m','Guided'),
                              (ww,'centerline_late_error_matp_m','MATP')]:
        ax.bar(xs+offset,[avg([r for r in rows if r['scenario']==s],key) or 0.
                          for s in scenarios],width=ww,label=label)
    ax.set_xticks(xs,scenarios);ax.set_ylabel('Mean target-lane deviation, 2-4 s (m)')
    ax.set_title('Late-horizon dense trajectory vs FROZEN target-lane centerline')
    ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'07_target_lane_centerline_error.png',dpi=180);plt.close(fig)


def main():
    args=args_parser()
    if not 1<=args.guide_count<=args.group_size:raise ValueError('guide-count must be 1..group-size')
    if args.samples_per_scenario<1:raise ValueError('samples-per-scenario must be >= 1')
    if min(args.road_sampling_m,args.road_inside_scale_m,args.road_inside_bonus,args.road_outside_slope)<=0:
        raise ValueError('Road geometry configuration must be positive')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    guidance_root=args.guidance_root.expanduser().resolve()
    matp=None if args.no_matp else load_matp(guidance_root)
    cfg=replace(TrajectoryModeRewardConfig(), task_curvature_weight=args.curvature_weight,
                task_curvature_peak_weight=args.curvature_peak_weight,
                task_curvature_margin_weight=args.curvature_margin_weight,
                task_centerline_weight=args.centerline_weight,
                centerline_scale_m=args.centerline_scale_m,
                centerline_late_power=args.centerline_late_power)
    proxy=DifferentiableTaskReward(cfg,device)
    model_adapter,model=_build_model(CheckpointSpec(name='pretrained',path=args.checkpoint.resolve()),device)
    sampler=GroupDiffusionSampler(model,group_size=args.group_size,eta=args.eta)
    reward_adapter=CandidateRewardAdapter(resolve_reward_evaluator('auto'))
    rows=[];gallery=[];attempts_log=[]
    print('[reward] CURRENT BRANCH unified:',proxy.semantics)
    print('[mode] READ ONLY, ONE guidance (balanced == full task reward); no RL/distillation')
    print('[matp]',guidance_root if matp is not None else 'disabled explicitly')
    for scen_idx,name in enumerate(list(SCENARIOS.keys()) if args.scenario=='all' else [args.scenario]):
        count=attempts=0
        while count<args.samples_per_scenario:
            if attempts>=args.max_seed_attempts:raise RuntimeError(f'Could not collect {name}')
            env_seed=_state_seed(args.seed,scen_idx,attempts);attempts+=1
            env=SCENARIOS[name](config={'show_trajectories':False,'show_future_trajectories':False},render_mode=None)
            try:
                env.reset(seed=env_seed)
                _,_,terminated,truncated,_=env.step(int(args.group_action))
                if terminated or truncated:
                    attempts_log.append({'scenario':name,'seed':env_seed,'reason':'terminated'});continue
                features=_scenario_features(env,model_adapter)
                frozen,selected=_frozen_reward_context(model,features,env)
                context=replace(frozen,config=cfg)
                road=RoadBoundaryField.from_road(env.road,sampling_m=args.road_sampling_m,
                    inside_scale_m=args.road_inside_scale_m,inside_bonus=args.road_inside_bonus,
                    outside_slope=args.road_outside_slope)
                snapshot=_planning_state_snapshot(env)
                if snapshot is not None:
                    role_pose=np.asarray(snapshot[0][args.role],dtype=np.float64)
                    pose_source='planning_snapshot'
                else:
                    role_pose=_vehicle_pose(env.controlled_vehicles[args.role]);pose_source='live_fallback'
                proxy.bind_road(road,role_pose)
                lane_line=features['target_lane_polyline'][args.role,:,:2].detach().cpu().numpy()
                proxy.bind_centerline(lane_line)
                gen=torch.Generator(device=device.type).manual_seed(_sample_noise_seed(args.noise_seed,scen_idx,count))
                with torch.no_grad():trace=sampler.sample(features,generator=gen)
                full=trace.candidates.detach().clone()
                original=full[:,:args.guide_count].clone()
                if original.ndim!=5 or original.shape[:3]!=(3,args.guide_count,10):
                    raise RuntimeError('Unexpected sampler contract '+str(original.shape))
                mode=int(selected[args.role].item())
                if not bool(features['mode_valid_mask'][args.role,mode].item()):
                    raise RuntimeError('Frozen selected mode is invalid')
                xy=original[args.role,:,mode,:,:].detach().clone()
                r0,i0,res0=paired_score(reward_adapter,features,context,original,role=args.role,mode=mode)
                parity=proxy.validate_against_production(xy,res0,role=args.role,mode=mode,tol=args.reward_parity_tol)
                if args.guide_count==args.group_size:noise_best=float(r0.max())
                else:
                    all_reward,_,_=paired_score(reward_adapter,features,context,full,role=args.role,mode=mode)
                    noise_best=float(all_reward.max())
                with torch.no_grad():raw_road=road.components(xy,role_pose,cfg)
                g=proxy.gradient_of(xy)
                with torch.enable_grad():guided,hist=guide_all(xy,proxy,iters=args.guidance_iters,
                    step_m=args.guidance_step_m,trust_rms_m=args.guidance_trust_rms_m,
                    max_point_move_m=args.guidance_max_point_move_m)
                guided_xy=guided['balanced']
                candidates=original.clone();candidates[args.role,:,mode,:,:]=guided_xy
                with torch.no_grad():
                    r1,i1,_=paired_score(reward_adapter,features,context,candidates,role=args.role,mode=mode)
                    gd_road=road.components(guided_xy,role_pose,cfg)
                    if matp is not None:
                        physical,diag=project_w1(guided_xy.detach(),module=matp,reward_config=cfg)
                        if not all(k in diag for k in ('max_abs_curvature_before','max_abs_curvature_after')):
                            raise RuntimeError('MATP native curvature diagnostics absent')
                        k_before=np.asarray(diag['max_abs_curvature_before'].detach().cpu()).reshape(-1)
                        k_after=np.asarray(diag['max_abs_curvature_after'].detach().cpu()).reshape(-1)
                        candidates[args.role,:,mode,:,:]=physical
                        r2,i2,_=paired_score(reward_adapter,features,context,candidates,role=args.role,mode=mode)
                        post_road=road.components(physical,role_pose,cfg)
                    else:
                        physical=diag=None;r2=i2=post_road=None
                # Match the EXACT production Reward/Guidance/MATP dense decoder.
                # Decode the same sparse candidates with the same endpoint contract.
                with torch.no_grad():
                    original_dense=evaluate_dense_spline(xy,cfg)[0].detach().cpu().numpy()
                    guided_dense=evaluate_dense_spline(guided_xy,cfg)[0].detach().cpu().numpy()
                    matp_dense=(evaluate_dense_spline(physical,cfg)[0].detach().cpu().numpy()
                                if matp is not None else None)
                if original_dense.shape[1:]!=(41,2) or guided_dense.shape!=original_dense.shape:
                    raise RuntimeError('Unexpected 10Hz dense decoder output shape')
                grad_norm=torch.linalg.vector_norm(g.flatten(start_dim=1),dim=-1).cpu().numpy()
                for i in range(args.guide_count):
                    diff=guided_xy[i].cpu().numpy()-xy[i].cpu().numpy()
                    dense_diff=guided_dense[i,1:]-original_dense[i,1:]  # 40 future points
                    row={'scenario':name,'state_id':count,'env_seed':env_seed,'group_id':i,
                         'direction':'balanced','reward_before':float(r0[i]),'reward_guided':float(r1[i]),
                         'guided_delta_reward':float(r1[i]-r0[i]),
                         'noise48_best_reward':noise_best,'guided_exceeds_noise48_best':int(r1[i]>noise_best+1e-6),
                         'initial_gradient_norm':float(grad_norm[i]),
                         'guided_ade_m':float(np.linalg.norm(dense_diff,axis=-1).mean()),
                         'guided_lateral_ade_m':float(np.abs(dense_diff[:,1]).mean()),
                         'sparse_control_ade_m':float(np.linalg.norm(diff,axis=-1).mean()),
                         'sparse_control_lateral_ade_m':float(np.abs(diff[:,1]).mean()),
                         'guided_unsafe':float(i1['unsafe'][i]),
                         'guided_curvature_feasible':int(k_before[i]<=.020001) if matp else float('nan'),
                         'road_reward_before':float(raw_road['road'][i]),
                         'road_reward_guided':float(gd_road['road'][i]),
                         'road_reward_guided_gain':float(gd_road['road'][i]-raw_road['road'][i]),
                         'road_union_min_margin_before_m':float(raw_road['road_min_margin_m'][i]),
                         'road_union_min_margin_guided_m':float(gd_road['road_min_margin_m'][i]),
                         'road_feasible_guided':int(float(gd_road['road_min_margin_m'][i])>=0),
                         'original_progress_score':float(i0['progress_score'][i]),
                         'guided_progress_score':float(i1['progress_score'][i]),
                         'reward_without_curvature_before':float(r0[i]+cfg.task_curvature_weight*i0['curvature_penalty'][i]),
                         'reward_base3_before':float(r0[i]+cfg.task_curvature_weight*i0['curvature_penalty'][i]
                                                     +cfg.task_centerline_weight*i0['centerline_penalty'][i]),
                         'reward_base3_guided':float(r1[i]+cfg.task_curvature_weight*i1['curvature_penalty'][i]
                                                     +cfg.task_centerline_weight*i1['centerline_penalty'][i]),
                         'reward_without_curvature_guided':float(r1[i]+cfg.task_curvature_weight*i1['curvature_penalty'][i]),
                         'reward_without_curvature_gain':float((r1[i]-r0[i])+cfg.task_curvature_weight*(i1['curvature_penalty'][i]-i0['curvature_penalty'][i])),
                         'centerline_penalty_before':float(i0['centerline_penalty'][i]),
                         'centerline_penalty_guided':float(i1['centerline_penalty'][i]),
                         'centerline_late_error_before_m':float(i0['centerline_late_error_m'][i]),
                         'centerline_late_error_guided_m':float(i1['centerline_late_error_m'][i]),
                         'centerline_valid_guided':float(i1['centerline_valid'][i]),
                         'curvature_penalty_before':float(i0['curvature_penalty'][i]),
                         'curvature_penalty_guided':float(i1['curvature_penalty'][i]),
                         'curvature_max_abs_before':float(i0['curvature_max_abs'][i]),
                         'curvature_max_abs_guided':float(i1['curvature_max_abs'][i]),
                         'curvature_valid_points_guided':float(i1['curvature_valid_points'][i]),
                         'curvature_violation_guided':float(i1['curvature_violation_penalty'][i]),
                         'curvature_peak_guided':float(i1['curvature_peak_penalty'][i]),
                         'curvature_margin_guided':float(i1['curvature_margin_penalty'][i]),
                         'original_comfort_penalty':float(i0['comfort_penalty'][i]),
                         'guided_comfort_penalty':float(i1['comfort_penalty'][i]),
                         'road_pose_source':pose_source}
                    if matp:
                        row.update({'reward_matp':float(r2[i]),
                            'matp_vs_guided_reward':float(r2[i]-r1[i]),
                            'matp_vs_original_reward':float(r2[i]-r0[i]),
                            'matp_ade_m':float(np.linalg.norm(matp_dense[i,1:]-guided_dense[i,1:],axis=-1).mean()),
                            'matp_lateral_ade_m':float(np.abs(matp_dense[i,1:,1]-guided_dense[i,1:,1]).mean()),
                            'matp_unsafe':float(i2['unsafe'][i]),
                            'matp_collision':float(i2['collision'][i]),
                            'matp_curvature_feasible':int(k_after[i]<=.020001),
                            'matp_max_curvature':float(k_after[i]),
                            'road_feasible_matp':int(float(post_road['road_min_margin_m'][i])>=0),
                            'road_union_min_margin_matp_m':float(post_road['road_min_margin_m'][i]),
                            'matp_progress_score':float(i2['progress_score'][i]),
                            'matp_comfort_penalty':float(i2['comfort_penalty'][i]),
                            'reward_without_curvature_matp':float(r2[i]+cfg.task_curvature_weight*i2['curvature_penalty'][i]),
                            'reward_base3_matp':float(r2[i]+cfg.task_curvature_weight*i2['curvature_penalty'][i]
                                                    +cfg.task_centerline_weight*i2['centerline_penalty'][i]),
                            'centerline_penalty_matp':float(i2['centerline_penalty'][i]),
                            'centerline_late_error_matp_m':float(i2['centerline_late_error_m'][i]),
                            'curvature_penalty_matp':float(i2['curvature_penalty'][i]),
                            'curvature_max_abs_matp':float(i2['curvature_max_abs'][i]),
                            'curvature_valid_points_matp':float(i2['curvature_valid_points'][i]),
                            'curvature_valid_fraction_matp':float(i2['curvature_valid_points'][i])/39.0,
                            'teacher_candidate':int((r2[i]>r0[i]+1e-6)
                                and (k_after[i]<=.020001)
                                and (float(i2['curvature_valid_points'][i])>=1.0)
                                and (float(post_road['road_min_margin_m'][i])>=0)
                                and (float(i2['collision'][i])<.5)
                                and (float(i2['unsafe'][i])<.5))})
                    rows.append(row)
                if len(gallery)<100:
                    gallery.append({'scenario':name,'state_id':count,
                        'base':xy[0].cpu().numpy().copy(),
                        'guided':guided_xy[0].cpu().numpy().copy(),
                        'matp':physical[0].cpu().numpy().copy() if matp else None,
                        'base_dense':original_dense[0].copy(),
                        'guided_dense':guided_dense[0].copy(),
                        'matp_dense':matp_dense[0].copy() if matp_dense is not None else None,
                        'role_pose':role_pose.copy(),
                        'road_outline':[np.asarray(p.exterior.coords) for p in _polygon_components(road.geometry)],
                        'target_lane_centerline':lane_line.copy()})
                print(f'[state] {name}:{count+1}/{args.samples_per_scenario} '
                      f'parity={parity} unified_gain={float(np.mean(r1-r0)):+.6f}',flush=True)
                count+=1
            finally:
                if hasattr(env,'close'):env.close()
    save_csv(out/'paired_trajectories.csv',rows)
    save_csv(out/'skipped_states.csv',attempts_log)
    summary=create_summary(rows,proxy.semantics,matp is not None)
    summary['gradient_fallback_counts']=dict(proxy.gradient_fallback_counts)
    summary['config']={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    summary['source_provenance']={
        'task_reward_sha256':hashlib.sha256((ROOT/'highway_env/planner/diffusion/grpo/task_reward.py').read_bytes()).hexdigest(),
        'checkpoint_sha256':hashlib.sha256(args.checkpoint.resolve().read_bytes()).hexdigest(),
        'matp_sha256':hashlib.sha256((guidance_root/'highway_env/planner/diffusion/guidance.py').read_bytes()).hexdigest() if matp else None}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    if not args.no_plots:create_plots(rows,gallery,out,matp is not None)
    print('[PASS] unified reward guidance probe complete:',out)
    print('summary:',summary['unified'])
    return 0

if __name__=='__main__':raise SystemExit(main())
