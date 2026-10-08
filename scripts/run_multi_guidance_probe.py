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
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    guidance_root=args.guidance_root.expanduser().resolve()
    matp=None if args.no_matp else load_matp(guidance_root)
    cfg=TrajectoryModeRewardConfig()
    proxy=DifferentiableTaskReward(cfg,device)
    model_adapter,model=_build_model(CheckpointSpec(name='pretrained',path=args.checkpoint.resolve()),device)
    sampler=GroupDiffusionSampler(model,group_size=args.group_size,eta=args.eta)
    reward_adapter=CandidateRewardAdapter(resolve_reward_evaluator('auto'))
    rows=[];grad_rows=[];shape_rows=[];gallery=[];attempt_log=[]
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
                        candidates=original.clone()
                        candidates[args.role,:,mode,:,:]=guided[a]
                        r_guided,i_guided,_=paired_score(reward_adapter,features,context,candidates,
                            role=args.role,mode=mode,task=args.reward_type)
                        if matp is not None:
                            physical,diag=project_w1(guided[a].detach(),module=matp,reward_config=cfg)
                            matp_geom=proxy.components(physical)
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
                             'vehicle_role':args.role,'mode_id':mode,'group_id':i,'direction':a,
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
                            row['matp_curvature_feasible']=int(after_curv<=0.02+1e-6)
                            row['matp_curvature_fixed']=int(guided_curv>0.02+1e-6 and after_curv<=0.02+1e-6)
                            row.update({'matp_'+field:float(i_matp[field][i]) for field in ('unsafe','collision','out_of_drivable','comfort_penalty','minimum_road_margin_m')})
                            row.update({'matp_'+k:v for k,v in metrics_xy(guided[a][i].cpu().numpy(),physical[i].cpu().numpy()).items()})
                        rows.append(row)
                    if a_i==0:
                        grad_rows.extend([{'scenario':name,'state_id':collected,'group_id':g,
                            **{f'{a}__{b}':float(cosine[g,ai,bi]) for ai,a in enumerate(DIRECTIONS) for bi,b in enumerate(DIRECTIONS)}}
                            for g in range(args.guide_count)])
                    if len(gallery)<100:
                        if a_i==0:gallery.append({'scenario':name,'sample_index':collected,
                            'base':xy[0].cpu().numpy().copy()})
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
    summary=create_summary(rows,grad_rows,proxy.semantics,matp is not None)
    summary['pairwise_guided_shape']={pair:{
        'N':sum(r['pair']==pair for r in shape_rows),
        'ade_mean_m':safe_mean([r['pair_ade_m'] for r in shape_rows if r['pair']==pair]),
        'lateral_ade_mean_m':safe_mean([r['pair_lateral_ade_m'] for r in shape_rows if r['pair']==pair])}
        for pair in sorted(set(r['pair'] for r in shape_rows))}
    summary['configuration']={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    summary['source_provenance']={
        'current_task_reward_sha256':hashlib.sha256((ROOT/'highway_env/planner/diffusion/grpo/task_reward.py').read_bytes()).hexdigest(),
        'current_scoring_sha256':hashlib.sha256((ROOT/'highway_env/planner/diffusion/trajectory_mode_reward/scoring.py').read_bytes()).hexdigest(),
        'matp_guidance_sha256':(hashlib.sha256((guidance_root/'highway_env/planner/diffusion/guidance.py').read_bytes()).hexdigest() if matp is not None else None),
        'checkpoint_sha256':hashlib.sha256(args.checkpoint.resolve().read_bytes()).hexdigest(),
        'matp_curvature_limit':0.02 if matp is not None else None,
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    if not args.no_plots:create_plots(rows,grad_rows,gallery,out,matp is not None,shape_rows)
    print('[PASS] probe complete:',out)
    for name,s in summary['directions'].items():
        print(name,'guided gain',round(s['guided_delta_mean'],5),'unsafe',round(s['guided_unsafe_rate'],4),
              'matp gain vs guide',round(s['matp_vs_guided_delta_mean'],5) if matp is not None else 'off')
    return 0

if __name__=='__main__':raise SystemExit(main())
