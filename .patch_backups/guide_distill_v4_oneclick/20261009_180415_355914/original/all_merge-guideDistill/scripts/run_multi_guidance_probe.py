#!/usr/bin/env python3
"""AllMerge guideDistill V1: read-only four-direction exploration + true MATP-W1.

Strictly evaluates the task reward and safety outputs from THIS repository's
trajectory_mode_reward implementation. No GRPO updates, distillation, or teacher
filtering. Candidate groups and environment seeds are paired for every direction.
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts.run_grpo_noise_probe import (
    SCENARIOS, CheckpointSpec, _build_model, _frozen_reward_context,
    _scenario_features, _state_seed, _sample_noise_seed,
)
from highway_env.planner.diffusion.grpo.sampling import GroupDiffusionSampler
from highway_env.planner.diffusion.grpo.reward_adapter import CandidateRewardAdapter, resolve_reward_evaluator
from highway_env.planner.diffusion.grpo.task_reward import task_reward_from_w4_result
from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
from highway_env.planner.diffusion.guide_distill.reward_gradient import DifferentiableTaskReward,DIRECTIONS
from highway_env.planner.diffusion.guide_distill.exploration import gradient_cosine,guide_all
from highway_env.planner.diffusion.guide_distill.matp_adapter import load_matp,project_w1
from highway_env.planner.diffusion.guide_distill.road_boundary import RoadBoundaryField
from highway_env.planner.diffusion.trajectory_mode_reward.state_adapter import _planning_state_snapshot, _vehicle_pose
from highway_env.planner.diffusion.trajectory_mode_reward.geometry import (
    road_margin_series_batch as old_road_margin_series_batch,
    _dense_local_trajectories_batch, local_to_world,
)


def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--scenario',choices=('all',*SCENARIOS.keys()),default='all')
    p.add_argument('--samples-per-scenario',type=int,default=9)
    p.add_argument('--group-size',type=int,default=48)
    p.add_argument('--guide-count',type=int,default=12,help='Number of paired initial trajectories per state')
    p.add_argument('--role',type=int,choices=(0,1,2),default=0)
    p.add_argument('--seed',type=int,default=20261008)
    p.add_argument('--noise-seed',type=int,default=70261008)
    p.add_argument('--eta',type=float,default=.02)
    p.add_argument('--guidance-iters',type=int,default=8)
    p.add_argument('--guidance-step-m',type=float,default=.12)
    p.add_argument('--guidance-trust-rms-m',type=float,default=.75)
    p.add_argument('--guidance-max-point-move-m',type=float,default=1.25)
    p.add_argument('--reward-type',choices=('progress_comfort',),default='progress_comfort')
    p.add_argument('--road-balanced-weight',type=float,default=0.25,
                   help='Probe-only additional road term in Balanced; production Reward unchanged')
    p.add_argument('--road-sampling-m',type=float,default=0.4,
                   help='Lane boundary polygon resolution (m)')
    p.add_argument('--road-inside-scale-m',type=float,default=2.0)
    p.add_argument('--road-inside-bonus',type=float,default=1.0)
    p.add_argument('--road-outside-slope',type=float,default=0.5)
    p.add_argument('--reward-parity-tol',type=float,default=.002)
    p.add_argument('--guidance-root',type=Path,default=Path('all_merge-guidance'),
                   help='Sibling worktree containing original frozen MATP V4.3 W1 guidance.py')
    p.add_argument('--no-matp',action='store_true',help='Stage A only; skip MATP comparison explicitly')
    p.add_argument('--group-action',type=int,default=0)
    p.add_argument('--max-seed-attempts',type=int,default=1000)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir',type=Path,default=Path('outputs/guide_distill_probe_v1'))
    p.add_argument('--no-plots',action='store_true')
    return p.parse_args()


def paired_score(adapter,features,context,candidates,*,role,mode,task):
    # Critical: production W4 result from this worktree, never guidance branch.
    result=adapter.evaluate_result(candidates,features=features,context=context)
    reward=task_reward_from_w4_result(result,context=context,
        device=candidates.device,dtype=candidates.dtype,reward_type=task)
    if tuple(reward.shape)!=(3,candidates.shape[1],10):
        raise RuntimeError('Unexpected task reward shape '+str(tuple(reward.shape)))
    rr=reward[role,:,mode].detach().cpu().numpy().copy()
    def field(name):
        if hasattr(result,name): arr=np.asarray(getattr(result,name))
        elif name in result.components: arr=np.asarray(result.components[name])
        else:return np.full(len(rr),np.nan,dtype=float)
        return np.asarray(arr[role,mode,:]).copy()
    info={name:field(name) for name in
          ('unsafe','collision','out_of_drivable','clearance_violation',
           'minimum_road_margin_m','comfort_penalty','progress_score')}
    info['legacy_w4_reward']=np.asarray(result.rewards)[role,mode,:].copy()
    return rr,info,result


def metrics_xy(a,b):
    diff=b-a
    d=np.linalg.norm(diff,axis=-1)
    return dict(ade_m=float(d.mean()),fde_m=float(d[-1]),
                lateral_ade_m=float(np.abs(diff[:,1]).mean()),
                longitudinal_ade_m=float(np.abs(diff[:,0]).mean()),
                final_dx_m=float(diff[-1,0]),final_dy_m=float(diff[-1,1]))


def save_csv(path,rows):
    if not rows:return
    keys=[]
    for r in rows:
        for key in r:
            if key not in keys:keys.append(key)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def safe_mean(values):
    x=np.asarray(values,dtype=float)
    return float(np.nanmean(x)) if x.size and np.isfinite(x).any() else None


def create_summary(rows,gradient_rows,semantics,matp):
    summary={'reward_source':'task_reward_from_w4_result(current worktree, progress_comfort)',
             'reward_semantics':semantics,'matp_enabled':matp,'directions':{}}
    for name in DIRECTIONS:
        sub=[r for r in rows if r['direction']==name]
        if not sub:continue
        base=np.array([r['reward_before'] for r in sub],dtype=float)
        guided=np.array([r['reward_guided'] for r in sub],dtype=float)
        delta=guided-base
        entry={'N':len(sub),'reward_before_mean':float(base.mean()),
               'guided_reward_mean':float(guided.mean()),
               'road_guided_gain_mean':safe_mean([r['road_reward_guided_gain'] for r in sub]),
               'road_outer_union_feasible_rate_guided':safe_mean([1-r['road_union_offroad_guided'] for r in sub]),
               'road_margin_before_mean_m':safe_mean([r['road_union_min_margin_before_m'] for r in sub]),
               'road_margin_guided_mean_m':safe_mean([r['road_union_min_margin_guided_m'] for r in sub]),
               'guided_delta_mean':float(delta.mean()),
               'guided_positive_fraction':float(np.mean(delta>1e-6)),
               'guided_ade_m':safe_mean([r['guided_ade_m'] for r in sub]),
               'guided_lateral_ade_m':safe_mean([r['guided_lateral_ade_m'] for r in sub]),
               'guided_exceeds_noise48_best_fraction':safe_mean([r['guided_exceeds_noise48_best'] for r in sub]),
               'guided_max_curvature_mean':safe_mean([r['guided_max_curvature'] for r in sub]),
               'guided_curvature_feasible_rate':safe_mean([r['guided_curvature_feasible'] for r in sub]),
               'guided_unsafe_rate':safe_mean([r['guided_unsafe'] for r in sub]),
               'before_unsafe_rate':safe_mean([r['before_unsafe'] for r in sub]),
               'gradient_norm_mean':safe_mean([r['initial_gradient_norm'] for r in sub]),
               'gradient_degenerate_fraction':float(np.mean([r['initial_gradient_norm']<1e-8 for r in sub]))}
        if matp:
            post=np.array([r['reward_matp'] for r in sub],dtype=float)
            entry.update({'matp_reward_mean':float(post.mean()),
                          'matp_vs_guided_delta_mean':float((post-guided).mean()),
                          'matp_unsafe_rate':safe_mean([r['matp_unsafe'] for r in sub]),
                          'matp_max_curvature_mean':safe_mean([r['matp_max_curvature'] for r in sub]),
                          'matp_curvature_feasible_rate':safe_mean([r['matp_curvature_feasible'] for r in sub]),
                          'matp_curvature_fix_fraction':safe_mean([r['matp_curvature_fixed'] for r in sub
                              if r['guided_curvature_feasible'] < 0.5]),
                          'matp_reward_drop_gt_0_05':float(np.mean(post-guided<-.05))})
        summary['directions'][name]=entry
    summary['gradient_cosine']={}
    for i,name in enumerate(DIRECTIONS):
        for j,name2 in enumerate(DIRECTIONS):
            key=f'{name}__{name2}'
            summary['gradient_cosine'][key]=safe_mean([r[key] for r in gradient_rows])
    for scenario in sorted(set(r['scenario'] for r in rows)):
        summary.setdefault('by_scenario',{})[scenario]={}
        for name in DIRECTIONS:
            sub=[r for r in rows if r['scenario']==scenario and r['direction']==name]
            if not sub:continue
            summary['by_scenario'][scenario][name]={
                'N':len(sub),
                'guided_reward_gain':safe_mean([r['guided_delta_reward'] for r in sub]),
                'guided_curvature_feasible_rate':safe_mean([r['guided_curvature_feasible'] for r in sub]),
                'matp_curvature_feasible_rate':safe_mean([r.get('matp_curvature_feasible',float('nan')) for r in sub]),
                'matp_vs_guided_reward':safe_mean([r.get('matp_vs_guided_reward',float('nan')) for r in sub]),
            }
    return summary


def create_plots(rows,gradient_rows,gallery,out,do_matp,shape_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(exist_ok=True,parents=True)
    fig,ax=plt.subplots(figsize=(9,5))
    data=[[r['reward_guided']-r['reward_before'] for r in rows if r['direction']==n] for n in DIRECTIONS]
    ax.boxplot(data,tick_labels=list(DIRECTIONS),showmeans=True)
    ax.axhline(0,color='gray',lw=1);ax.set_title('Paired task reward gain: guided - original');ax.set_ylabel('Reward gain')
    fig.tight_layout();fig.savefig(out/'01_guidance_reward_gain.png',dpi=180);plt.close(fig)
    if do_matp:
        fig,axes=plt.subplots(1,2,figsize=(12,4.6))
        for stage,title in [('guided','Original to guidance'),('matp','Guidance to MATP')]:
            ax=axes[0] if stage=='guided' else axes[1]
            group=[r for r in rows if r['direction'] in DIRECTIONS]
            vals=[safe_mean([r['guided_unsafe'] if stage=='guided' else r['matp_unsafe'] for r in group if r['direction']==n]) for n in DIRECTIONS]
            ax.bar(DIRECTIONS,vals);ax.set_ylim(0,1);ax.set_title('Unsafe fraction: '+title);ax.tick_params(axis='x',rotation=15)
        fig.tight_layout();fig.savefig(out/'02_safety_comparison.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5))
        dr=[[r['reward_matp']-r['reward_guided'] for r in rows if r['direction']==n] for n in DIRECTIONS]
        ax.boxplot(dr,tick_labels=list(DIRECTIONS),showmeans=True);ax.axhline(0,color='gray',lw=1)
        ax.set_title('Paired task reward impact of frozen MATP W1');ax.set_ylabel('MATP - Guidance')
        fig.tight_layout();fig.savefig(out/'03_matp_reward_retention.png',dpi=180);plt.close(fig)
    if gradient_rows:
        matrix=np.array([[safe_mean([r[f'{a}__{b}'] for r in gradient_rows]) or float('nan')
                          for b in DIRECTIONS] for a in DIRECTIONS],dtype=float)
        fig,ax=plt.subplots(figsize=(6.5,5.5));im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(4),DIRECTIONS,rotation=20);ax.set_yticks(range(4),DIRECTIONS)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=10)
        fig.colorbar(im,ax=ax,label='Gradient cosine');fig.tight_layout()
        fig.savefig(out/'04_gradient_cosine.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    data=[[r['road_reward_guided_gain'] for r in rows if r['direction']==n] for n in DIRECTIONS]
    axes[0].boxplot(data,tick_labels=list(DIRECTIONS),showmeans=True)
    axes[0].axhline(0,color='gray');axes[0].set_title('Outermost-road reward gain')
    data=[[r['road_union_min_margin_guided_m'] for r in rows if r['direction']==n] for n in DIRECTIONS]
    axes[1].boxplot(data,tick_labels=list(DIRECTIONS),showmeans=True)
    axes[1].axhline(0,color='gray');axes[1].set_title('Union outer-road signed footprint margin (m)')
    fig.tight_layout();fig.savefig(out/'07_road_reward_boundary_margin.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    keys=sorted(set(r['scenario'] for r in rows))
    x=np.arange(len(keys));bar_width=.35
    old=[safe_mean([r['guided_out_of_drivable'] for r in rows if r['scenario']==q and r['direction']=='road']) for q in keys]
    union=[safe_mean([r['road_union_offroad_guided'] for r in rows if r['scenario']==q and r['direction']=='road']) for q in keys]
    ax.bar(x-bar_width/2,old,bar_width,label='W4 per-lane margin')
    ax.bar(x+bar_width/2,union,bar_width,label='Outermost lane union')
    ax.set_xticks(x,keys);ax.set_ylim(0,1);ax.set_ylabel('Offroad fraction')
    ax.set_title('Curved road-boundary diagnosis: legacy vs union');ax.legend()
    fig.tight_layout();fig.savefig(out/'08_legacy_vs_union_road.png',dpi=180);plt.close(fig)
    # Gallery: smooth 10 Hz spline line + unconnected sparse controls.
    if gallery:
        rng=np.random.default_rng(123)
        sample_ids=np.arange(len(gallery)); rng.shuffle(sample_ids)
        picks=[gallery[i] for i in sample_ids[:9]]
        for name in DIRECTIONS:
            fig,axes=plt.subplots(3,3,figsize=(13,12))
            for ax,entry in zip(axes.flat,picks):
                base=entry['base'];guide=entry[name]['guided'];post=entry[name]['matp']
                from scipy.interpolate import CubicSpline
                def spline_line(xy, label):
                    knots=np.concatenate((np.zeros((1,2)),xy),axis=0)
                    v0=np.array([max(float(xy[0,0]),0.)/.5,0.])
                    vf=(xy[-1]-xy[-2])/.5
                    smooth=CubicSpline(np.arange(9)*.5,knots,bc_type=((1,v0),(1,vf)))(np.arange(41)*.1)
                    line,=ax.plot(smooth[:,0],smooth[:,1],lw=1.4,label=label)
                    ax.scatter(xy[:,0],xy[:,1],s=11,color=line.get_color())
                spline_line(base,'Original')
                spline_line(guide,name)
                if post is not None:spline_line(post,'W1 MATP')
                ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.25)
                ax.set_title(f"{entry['scenario']} state {entry['sample_index']}")
            for ax in axes.flat[len(picks):]:ax.axis('off')
            axes.flat[0].legend(fontsize=8)
            fig.suptitle('Paired 8-point trajectory shape / '+name)
            fig.tight_layout();fig.savefig(out/f'05_gallery_{name}_3x3.png',dpi=180);plt.close(fig)


    curved_gallery=[e for e in gallery if e.get('scenario')=='curved' and 'road' in e]
    if curved_gallery:
        fig,axes=plt.subplots(3,3,figsize=(13,12))
        for ax,entry in zip(axes.flat,curved_gallery[:9]):
            pose=entry['role_pose'];c=np.cos(pose[2]);s=np.sin(pose[2])
            def to_world(xy):
                return np.column_stack((pose[0]+c*xy[:,0]-s*xy[:,1],
                                        pose[1]+s*xy[:,0]+c*xy[:,1]))
            names=['base','guided','matp']
            values=[entry['base'],entry['road']['guided'],entry['road']['matp']]
            for n,v in zip(names,values):
                if v is not None:
                    w=to_world(v)
                    ax.plot(w[:,0],w[:,1],marker='.',ms=3,lw=1.3,label=n)
            for boundary in entry['road_outline']:
                ax.plot(boundary[:,0],boundary[:,1],c='black',lw=.8,alpha=.6)
            center=to_world(entry['base'])
            ax.set_xlim(float(center[:,0].min()-12),float(center[:,0].max()+12))
            ax.set_ylim(float(center[:,1].min()-12),float(center[:,1].max()+12))
            ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
            ax.set_title(f"Curved state {entry['sample_index']}: outermost road union")
        for ax in axes.flat[len(curved_gallery[:9]):]:ax.axis('off')
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/'09_curved_outer_boundary_bev_3x3.png',dpi=180);plt.close(fig)
    if do_matp:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for scen in sorted(set(x['scenario'] for x in rows)):
            a=[x for x in rows if x['scenario']==scen and x['direction']=='road']
            if a:
                axes[0].scatter([x['guided_analytic_max_curvature'] for x in a],
                    [x['guided_matp_native_max_curvature'] for x in a],s=10,alpha=.4,label=scen)
                axes[1].scatter([x.get('matp_trust_saturation_fraction',float('nan')) for x in a],
                    [x['matp_max_curvature'] for x in a],s=10,alpha=.4,label=scen)
        axes[0].set_xlabel('Spline analytic max curvature');axes[0].set_ylabel('MATP native curvature')
        axes[0].set_title('Curvature definition mismatch');axes[0].legend()
        axes[1].axhline(.02,c='gray',linestyle='--')
        axes[1].set_xlabel('MATP trust saturation');axes[1].set_ylabel('MATP final max curvature')
        axes[1].set_title('Curvature feasibility vs MATP trust')
        fig.tight_layout();fig.savefig(out/'10_curvature_diagnostics.png',dpi=180);plt.close(fig)
    if shape_rows:
        fig,ax=plt.subplots(figsize=(9,5))
        labels=[f'{a}/{b}' for i,a in enumerate(DIRECTIONS) for b in DIRECTIONS[i+1:]]
        data=[[r['pair_ade_m'] for r in shape_rows if r['pair']==label] for label in labels]
        ax.boxplot(data,tick_labels=labels,showmeans=True)
        ax.tick_params(axis='x',rotation=25);ax.set_ylabel('Guided-to-guided ADE (m)')
        ax.set_title('Same initial trajectory, different guidance directions')
        fig.tight_layout();fig.savefig(out/'06_pairwise_guided_shape.png',dpi=180);plt.close(fig)


def main():
    args=args_parser()
    if not 1<=args.guide_count<=args.group_size:raise ValueError('1 <= guide-count <= group-size required')
    if args.samples_per_scenario<1:raise ValueError('samples-per-scenario must be >= 1')
    if min(args.road_sampling_m,args.road_inside_scale_m,args.road_inside_bonus,args.road_outside_slope)<=0:
        raise ValueError('Road guidance geometric and reward parameters must be positive')
    if args.road_balanced_weight<0:raise ValueError('Balanced road weight must be nonnegative')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    guidance_root=args.guidance_root.expanduser().resolve()
    matp=None if args.no_matp else load_matp(guidance_root)
    cfg=TrajectoryModeRewardConfig()
    proxy=DifferentiableTaskReward(cfg,device)
    model_adapter,model=_build_model(CheckpointSpec(name='pretrained',path=args.checkpoint.resolve()),device)
    sampler=GroupDiffusionSampler(model,group_size=args.group_size,eta=args.eta)
    reward_adapter=CandidateRewardAdapter(resolve_reward_evaluator('auto'))
    rows=[];grad_rows=[];shape_rows=[];gallery=[];attempt_log=[];road_diag=[]
    scenarios=list(SCENARIOS.keys()) if args.scenario=='all' else [args.scenario]
    print('[reward] CURRENT BRANCH config:',proxy.semantics)
    print('[mode] READ ONLY - no RL or distillation updates')
    print('[matp] ',guidance_root if matp is not None else 'explicitly disabled')
    for scen_idx,name in enumerate(scenarios):
        collected=attempts=0
        while collected<args.samples_per_scenario:
            if attempts>=args.max_seed_attempts:raise RuntimeError(f'Failed to collect {name} in {attempts} attempts')
            env_seed=_state_seed(args.seed,scen_idx,attempts); attempts+=1
            env=SCENARIOS[name](config={'show_trajectories':False,'show_future_trajectories':False},render_mode=None)
            try:
                env.reset(seed=env_seed)
                _,_,terminated,truncated,_=env.step(int(args.group_action))
                if terminated or truncated:
                    attempt_log.append({'scenario':name,'seed':env_seed,'reason':'terminated or truncated'});continue
                features=_scenario_features(env,model_adapter)
                frozen,selected=_frozen_reward_context(model,features,env)
                context=replace(frozen,config=cfg)
                road=RoadBoundaryField.from_road(
                    env.road,sampling_m=args.road_sampling_m,
                    inside_scale_m=args.road_inside_scale_m,
                    inside_bonus=args.road_inside_bonus,
                    outside_slope=args.road_outside_slope)
                # CRITICAL: W4 scores planning-time frozen poses, not the
                # live post env.step() vehicle poses. Match it exactly.
                snapshot=_planning_state_snapshot(env)
                if snapshot is not None:
                    role_pose=np.asarray(snapshot[0][args.role],dtype=np.float64)
                    pose_source='planning_snapshot'
                else:
                    role_pose=_vehicle_pose(env.controlled_vehicles[args.role])
                    pose_source='live_fallback'
                live_pose=_vehicle_pose(env.controlled_vehicles[args.role])
                pose_translation_error_m=float(np.linalg.norm(role_pose[:2]-live_pose[:2]))
                pose_heading_error_deg=float(np.degrees(np.arctan2(
                    np.sin(role_pose[2]-live_pose[2]),np.cos(role_pose[2]-live_pose[2]))))
                proxy.bind_road(road,role_pose,balanced_road_weight=args.road_balanced_weight)
                noise_seed=_sample_noise_seed(args.noise_seed,scen_idx,collected)
                gen=torch.Generator(device=device.type).manual_seed(noise_seed)
                with torch.no_grad():trace=sampler.sample(features,generator=gen)
                full_candidates=trace.candidates.detach().clone()
                original=full_candidates[:, :args.guide_count].clone()
                if original.ndim!=5 or original.shape[:3]!=(3,args.guide_count,10):
                    raise RuntimeError('Unexpected sampler contract: '+str(original.shape))
                mode=int(selected[args.role].item())
                if not bool(features['mode_valid_mask'][args.role,mode].item()):
                    raise RuntimeError('Frozen selected mode is invalid')
                xy=original[args.role,:,mode,:,:].clone()
                base_reward,base_info,base_result=paired_score(reward_adapter,features,context,
                    original,role=args.role,mode=mode,task=args.reward_type)
                parity=proxy.validate_against_production(xy,base_result,role=args.role,mode=mode,tol=args.reward_parity_tol)
                # Independent noise-only G48 reference from exactly the same state and seed.
                if args.guide_count==args.group_size:
                    noise48_best=float(np.max(base_reward))
                else:
                    noise48_rewards,_,_=paired_score(reward_adapter,features,context,full_candidates,
                        role=args.role,mode=mode,task=args.reward_type)
                    noise48_best=float(np.max(noise48_rewards))
                with torch.no_grad():
                    raw_geom=proxy.components(xy)
                    raw_road=road.components(xy,role_pose,cfg)
                with torch.enable_grad():
                    grads=proxy.gradients(xy)
                    cosine=gradient_cosine(grads).cpu().numpy()
                    guided,hist=guide_all(xy,proxy,iters=args.guidance_iters,
                        step_m=args.guidance_step_m,trust_rms_m=args.guidance_trust_rms_m,
                        max_point_move_m=args.guidance_max_point_move_m)
                for di,a in enumerate(DIRECTIONS):
                    for b in DIRECTIONS[di+1:]:
                        aa=guided[a].detach().cpu().numpy()
                        bb=guided[b].detach().cpu().numpy()
                        disp=np.linalg.norm(aa-bb,axis=-1)
                        for g in range(args.guide_count):
                            shape_rows.append({'scenario':name,'state_id':collected,'group_id':g,
                                'pair':f'{a}/{b}','pair_ade_m':float(disp[g].mean()),
                                'pair_fde_m':float(disp[g,-1]),
                                'pair_lateral_ade_m':float(np.abs(aa[g,:,1]-bb[g,:,1]).mean())})
                for a_i,a in enumerate(DIRECTIONS):
                    gnorm=torch.linalg.vector_norm(grads[a].flatten(start_dim=1),dim=-1).cpu().numpy()
                    with torch.no_grad():
                        guided_geom=proxy.components(guided[a])
                        guided_road=road.components(guided[a],role_pose,cfg)
                        candidates=original.clone()
                        candidates[args.role,:,mode,:,:]=guided[a]
                        r_guided,i_guided,_=paired_score(reward_adapter,features,context,candidates,
                            role=args.role,mode=mode,task=args.reward_type)
                        if matp is not None:
                            physical,diag=project_w1(guided[a].detach(),module=matp,reward_config=cfg)
                            matp_geom=proxy.components(physical)
                            matp_road=road.components(physical,role_pose,cfg)
                            # MATP V4.3 uses its own dense geometric-curvature metric.
                            # Read its native diagnostics rather than conflating it
                            # with W4's analytic spline curvature definition.
                            for key in ('max_abs_curvature_before','max_abs_curvature_after'):
                                if key not in diag:
                                    raise RuntimeError('Frozen MATP source lacks '+key)
                            native_before=np.asarray(diag['max_abs_curvature_before'].detach().cpu()).reshape(-1)
                            native_after=np.asarray(diag['max_abs_curvature_after'].detach().cpu()).reshape(-1)
                            if native_before.size!=args.guide_count or native_after.size!=args.guide_count:
                                raise RuntimeError('MATP curvature diagnostics must be one scalar per candidate')
                            candidates[args.role,:,mode,:,:]=physical
                            r_matp,i_matp,_=paired_score(reward_adapter,features,context,candidates,
                                role=args.role,mode=mode,task=args.reward_type)
                        else:
                            physical=None;r_matp=None;i_matp=None
                    for i in range(args.guide_count):
                        delta=metrics_xy(xy[i].cpu().numpy(),guided[a][i].cpu().numpy())
                        orig_curv=float(raw_geom['max_abs_curvature'][i])
                        guided_curv=float(native_before[i]) if matp is not None else float(guided_geom['max_abs_curvature'][i])
                        row={'scenario':name,'state_id':collected,'env_seed':env_seed,'noise_seed':noise_seed,
                             'vehicle_role':args.role,'mode_id':mode,'group_id':i,'direction':a,'road_pose_source':pose_source,
                             'snapshot_vs_live_pose_offset_m':pose_translation_error_m,
                             'snapshot_vs_live_heading_delta_deg':pose_heading_error_deg,
                             'noise48_best_reward':noise48_best,
                             'guided_exceeds_noise48_best':int(r_guided[i]>noise48_best+1e-6),
                             'reward_before':float(base_reward[i]),'reward_guided':float(r_guided[i]),
                             'before_w4_reward':float(base_info['legacy_w4_reward'][i]),
                             'guided_w4_reward':float(i_guided['legacy_w4_reward'][i]),
                             'before_analytic_max_curvature':orig_curv,
                             'guided_analytic_max_curvature':float(guided_geom['max_abs_curvature'][i]),
                             'guided_matp_native_max_curvature':float(native_before[i]) if matp is not None else float('nan'),
                             'guided_max_curvature':guided_curv,
                             'before_analytic_curvature_feasible':int(orig_curv<=0.02+1e-6),
                             'guided_curvature_feasible':int(guided_curv<=0.02+1e-6),
                             'guided_delta_reward':float(r_guided[i]-base_reward[i]),
                             'road_reward_before':float(raw_road['road'][i]),
                             'road_reward_guided':float(guided_road['road'][i]),
                             'road_reward_guided_gain':float(guided_road['road'][i]-raw_road['road'][i]),
                             'road_union_min_margin_before_m':float(raw_road['road_min_margin_m'][i]),
                             'road_union_min_margin_guided_m':float(guided_road['road_min_margin_m'][i]),
                             'road_union_offroad_guided':int(float(guided_road['road_min_margin_m'][i]) < 0.0),
                             'initial_gradient_norm':float(gnorm[i]),
                             **{'guided_'+k:v for k,v in delta.items()}}
                        for prefix,info in [('before',base_info),('guided',i_guided)]:
                            for field in ('unsafe','collision','out_of_drivable','comfort_penalty','minimum_road_margin_m'):
                                row[prefix+'_'+field]=float(info[field][i])
                        if matp is not None:
                            after_curv=float(native_after[i])
                            row['reward_matp']=float(r_matp[i]);row['matp_vs_guided_reward']=float(r_matp[i]-r_guided[i])
                            row['matp_w4_reward']=float(i_matp['legacy_w4_reward'][i])
                            row['matp_max_curvature']=after_curv
                            row['matp_analytic_max_curvature']=float(matp_geom['max_abs_curvature'][i])
                            row['road_reward_matp']=float(matp_road['road'][i])
                            row['road_union_min_margin_matp_m']=float(matp_road['road_min_margin_m'][i])
                            row['road_union_offroad_matp']=int(float(matp_road['road_min_margin_m'][i]) < 0.0)
                            for field in ('matp_iterations_used','matp_active_constraints_sum','matp_trust_saturation_fraction',
                                          'matp_archive_feasible_fraction','matp_stall_triggers','update_l2_m'):
                                if field in diag:
                                    dd=np.asarray(diag[field].detach().cpu()).reshape(-1)
                                    if len(dd)==args.guide_count:row[field]=float(dd[i])
                            row['matp_curvature_feasible']=int(after_curv<=0.02+1e-6)
                            row['matp_curvature_fixed']=int(guided_curv>0.02+1e-6 and after_curv<=0.02+1e-6)
                            row.update({'matp_'+field:float(i_matp[field][i]) for field in ('unsafe','collision','out_of_drivable','comfort_penalty','minimum_road_margin_m')})
                            row.update({'matp_'+k:v for k,v in metrics_xy(guided[a][i].cpu().numpy(),physical[i].cpu().numpy()).items()})
                        rows.append(row)
                        road_diag.append({'scenario':name,'state_id':collected,'group_id':i,
                            'direction':a,'pose_offset_m':pose_translation_error_m,
                            'heading_delta_deg':pose_heading_error_deg,'legacy_offroad_guided':int(float(i_guided['out_of_drivable'][i])>0.5),
                            'union_offroad_guided':row['road_union_offroad_guided'],
                            'legacy_min_margin_guided_m':float(i_guided['minimum_road_margin_m'][i]),
                            'union_min_margin_guided_m':row['road_union_min_margin_guided_m'],
                            'native_matp_curvature_guided':guided_curv,
                            'native_matp_curvature_after':float(native_after[i]) if matp is not None else float('nan'),
                            'analytic_curvature_guided':float(guided_geom['max_abs_curvature'][i]),
                            'analytic_curvature_after':float(matp_geom['max_abs_curvature'][i]) if matp is not None else float('nan')})
                    if a_i==0:
                        grad_rows.extend([{'scenario':name,'state_id':collected,'group_id':g,
                            **{f'{a}__{b}':float(cosine[g,ai,bi]) for ai,a in enumerate(DIRECTIONS) for bi,b in enumerate(DIRECTIONS)}}
                            for g in range(args.guide_count)])
                    if len(gallery)<100:
                        if a_i==0:gallery.append({'scenario':name,'sample_index':collected,
                            'base':xy[0].cpu().numpy().copy()})
                        if a_i==0:
                            gallery[-1]['role_pose']=role_pose.copy()
                            gallery[-1]['road_outline']=[np.asarray(poly.exterior.coords) for poly in __import__('highway_env.planner.diffusion.guide_distill.road_boundary',fromlist=['_polygon_components'])._polygon_components(road.geometry)]
                        gallery[-1][a]={'guided':guided[a][0].cpu().numpy().copy(),
                                        'matp':physical[0].cpu().numpy().copy() if physical is not None else None}
                print(f'[state] {name}:{collected+1:02d}/{args.samples_per_scenario} parity={parity} '
                      + ' '.join(f'{a}:{np.mean([r["guided_delta_reward"] for r in rows if r["scenario"]==name and r["state_id"]==collected and r["direction"]==a]):+.3f}' for a in DIRECTIONS),flush=True)
                collected+=1
            finally:
                if hasattr(env,'close'):env.close()
    save_csv(out/'paired_trajectories.csv',rows)
    save_csv(out/'gradient_cosines.csv',grad_rows)
    save_csv(out/'pairwise_guided_shape.csv',shape_rows)
    save_csv(out/'skipped_states.csv',attempt_log)
    save_csv(out/'road_curvature_diagnostics.csv',road_diag)
    summary=create_summary(rows,grad_rows,proxy.semantics,matp is not None)
    summary['pairwise_guided_shape']={pair:{
        'N':sum(r['pair']==pair for r in shape_rows),
        'ade_mean_m':safe_mean([r['pair_ade_m'] for r in shape_rows if r['pair']==pair]),
        'lateral_ade_mean_m':safe_mean([r['pair_lateral_ade_m'] for r in shape_rows if r['pair']==pair])}
        for pair in sorted(set(r['pair'] for r in shape_rows))}
    summary['road_geometry_diagnostics']={
        'n':len(road_diag),
        'snapshot_vs_live_offset_mean_m':safe_mean([x['pose_offset_m'] for x in road_diag]),
        'legacy_offroad_fraction':safe_mean([x['legacy_offroad_guided'] for x in road_diag]),
        'outer_union_offroad_fraction':safe_mean([x['union_offroad_guided'] for x in road_diag]),
        'legacy_positive_union_negative_fraction':safe_mean([int(x['legacy_offroad_guided']==1 and x['union_offroad_guided']==0) for x in road_diag]),
        'legacy_negative_union_positive_fraction':safe_mean([int(x['legacy_offroad_guided']==0 and x['union_offroad_guided']==1) for x in road_diag]),
        }
    for scen in sorted(set(x['scenario'] for x in road_diag)):
        a=[x for x in road_diag if x['scenario']==scen]
        summary['road_geometry_diagnostics'][scen]={
            'N':len(a),'old_offroad':safe_mean([x['legacy_offroad_guided'] for x in a]),
            'union_offroad':safe_mean([x['union_offroad_guided'] for x in a]),
            'old_flag_union_in':sum(x['legacy_offroad_guided']==1 and x['union_offroad_guided']==0 for x in a),
            'old_in_union_out':sum(x['legacy_offroad_guided']==0 and x['union_offroad_guided']==1 for x in a),
            'curvature_native_analytic_abs_diff_mean':safe_mean([abs(x['native_matp_curvature_guided']-x['analytic_curvature_guided']) for x in a if np.isfinite(x['native_matp_curvature_guided'])])}
    summary['road_gradient_fallback_counts']=dict(getattr(proxy,'gradient_fallback_counts',{}))
    print('[road-gradient-diagnostics]',summary['road_gradient_fallback_counts'],flush=True)
    summary['configuration']={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    summary['source_provenance']={
        'current_task_reward_sha256':hashlib.sha256((ROOT/'highway_env/planner/diffusion/grpo/task_reward.py').read_bytes()).hexdigest(),
        'current_scoring_sha256':hashlib.sha256((ROOT/'highway_env/planner/diffusion/trajectory_mode_reward/scoring.py').read_bytes()).hexdigest(),
        'matp_guidance_sha256':(hashlib.sha256((guidance_root/'highway_env/planner/diffusion/guidance.py').read_bytes()).hexdigest() if matp is not None else None),
        'checkpoint_sha256':hashlib.sha256(args.checkpoint.resolve().read_bytes()).hexdigest(),
        'matp_curvature_limit':0.02 if matp is not None else None,
        'road_geometry_contract':'shapely lane ribbon union; outermost boundary; tracking-aware 4 corners',
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    if not args.no_plots:create_plots(rows,grad_rows,gallery,out,matp is not None,shape_rows)
    print('[PASS] probe complete:',out)
    for name,s in summary['directions'].items():
        print(name,'guided gain',round(s['guided_delta_mean'],5),'unsafe',round(s['guided_unsafe_rate'],4),
              'matp gain vs guide',round(s['matp_vs_guided_delta_mean'],5) if matp is not None else 'off')
    return 0

if __name__=='__main__':raise SystemExit(main())
