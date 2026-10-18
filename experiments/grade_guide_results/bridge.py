#!/usr/bin/env python3
"""Shared replay and reward helpers for reward-gradient mean guidance experiment.

Installed in AllMerge root. Imports current guideDistill worktree dynamically.
Does not optimize, patch, or save model parameters. Uses common random numbers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('all_merge-guideDistill'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--scenario', choices=('all','straight','curved','merge_in','merge_out'), default='all')
    p.add_argument('--samples-per-scenario',type=int,default=2)
    p.add_argument('--group-size',type=int,default=8)
    p.add_argument('--role',type=int,default=0,choices=(0,1,2))
    p.add_argument('--mode-scope',choices=('selected','all'),default='selected')
    p.add_argument('--profiles',default='uniform_y,ramp_y,bump_y,late_y,progress_x')
    p.add_argument('--amplitudes-m',default='-0.30,-0.10,-0.03,0,0.03,0.10,0.30')
    p.add_argument('--eta',type=float,default=0.02)
    p.add_argument('--min-std',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=20261017)
    p.add_argument('--noise-seed',type=int,default=70261017)
    p.add_argument('--group-action',type=int,default=0)
    p.add_argument('--max-seed-attempts',type=int,default=100)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir',type=Path,default=Path('all_merge-guideDistill/outputs/mean_shift_probe_v1'))
    p.add_argument('--centerline-weight',type=float,default=0.10)
    p.add_argument('--no-score',action='store_true',help='Skip production reward/constraint evaluations')
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p


def parse_values(text):
    values = [float(x) for x in text.split(',') if x.strip()]
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('amplitudes-m requires finite comma-separated values')
    return values


def profile_points(name, horizon=8):
    """Unit amplitude at maximum. xy arrays are in physical metres."""
    t = np.arange(1,horizon+1, dtype=np.float64)/horizon
    p = np.zeros((horizon,2), dtype=np.float64)
    if name == 'uniform_y': p[:,1]=1.0
    elif name == 'ramp_y': p[:,1]=t
    elif name == 'bump_y': p[:,1]=np.sin(np.pi*t)
    elif name == 'late_y': p[:,1]=t**2
    elif name == 'progress_x': p[:,0]=t
    else: raise ValueError('Unknown profile '+name)
    return p


def score_selected(candidates, reward_adapter, features, context, role, modes):
    """Use the exact current V5.2 task scorer, no reward reimplementation."""
    from highway_env.planner.diffusion.grpo.task_reward import task_reward_from_w4_result
    result = reward_adapter.evaluate_result(candidates,features=features,context=context)
    task = task_reward_from_w4_result(result,context=context,device=candidates.device,
                                       dtype=candidates.dtype,reward_type='progress_comfort')
    def get(name):
        src=getattr(result,name,None)
        if src is None: src=result.components.get(name)
        if src is None: raise RuntimeError(f'Scorer missing {name}')
        return np.asarray(src)
    # [role,mode,group] in production scorer
    names=('unsafe','collision','out_of_drivable','minimum_road_margin_m',
           'curvature_max_abs','centerline_late_error_m')
    outputs={m:{} for m in modes}
    for m in modes:
        outputs[m]['reward']=task[role,:,m].detach().cpu().numpy().astype(float)
        for key in names:
            outputs[m][key]=np.asarray(get(key)[role,m,:]).astype(float)
    return outputs


def gaussian_kl_shift(delta_norm: torch.Tensor, std: torch.Tensor):
    """KL(N(mu+delta,sigma^2I) || N(mu,sigma^2I)) for a single mode."""
    return float((0.5 * (delta_norm / std).square()).sum().item())


def setup_worktree(root:Path):
    root=root.resolve()
    if not (root/'highway_env/planner/diffusion/grpo/sampling.py').is_file():
        raise FileNotFoundError(f'Not an AllMerge guideDistill checkout: {root}')
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    # Avoid local copies of code: imports always come from user-supplied worktree.
    return root


def terminal_decode(model, features, all_modes_xt0, group_size):
    from highway_env.planner.diffusion.grpo.sampling import _expand_group
    if all_modes_xt0.shape[0] != features['coarse_trajectories'].shape[0]:
        raise ValueError('Mismatch: state batch')
    expanded=_expand_group(features,group_size)
    scene=model.scene_encoder(expanded)
    anchors=model.adapter.normalize_trajectory(expanded['coarse_trajectories'])
    b=all_modes_xt0.shape[0]
    t=torch.zeros((b*group_size,),dtype=torch.long,device=all_modes_xt0.device)
    with torch.no_grad():
        pred,_=model.denoiser(all_modes_xt0.reshape(b*group_size,*all_modes_xt0.shape[2:]),
                             t,scene,navigation_point_norm=anchors[:,:,-1,:])
        phys=model.adapter.denormalize_trajectory(pred)
    return phys.reshape(b,group_size,*phys.shape[1:])


def per_mode_rows(base, change, role, mode, delta_phys, delta_norm, std, reward_before=None, reward_after=None):
    """One row per group candidate. Records both physical geometry and Gaussian policy distances."""
    out=[]
    p=base[role,:,mode].detach().cpu().numpy()
    q=change[role,:,mode].detach().cpu().numpy()
    diff=q-p
    d=delta_phys.detach().cpu().numpy()
    dn=delta_norm.detach().cpu().numpy()
    norm=float(np.linalg.norm(d))
    kl=gaussian_kl_shift(delta_norm,std)
    for g in range(p.shape[0]):
        displacement=np.linalg.norm(diff[g],axis=-1)
        denom=float(np.sum(d*d))
        gain=float((diff[g]*d).sum()/denom) if denom>1e-14 else 0.
        response=float(np.linalg.norm(diff[g])/(norm+1e-12)) if norm>0 else 0.
        nonrigid=diff[g]-diff[g].mean(axis=0,keepdims=True)
        lateral_neighbor=np.diff(diff[g,:,1])
        row=dict(group=g,role=role,mode=mode,mean_shift_max_m=float(np.abs(d).max()),
                 mean_shift_l2_m=norm,mean_shift_x_end_m=float(d[-1,0]),mean_shift_y_end_m=float(d[-1,1]),
                 mean_shift_norm_l2=float(np.linalg.norm(dn)),
                 kl_nats=kl, rms_waypoint_disp_m=float(np.sqrt(np.mean(displacement**2))),
                 ade_waypoint_disp_m=float(np.mean(displacement)),
                 max_waypoint_disp_m=float(displacement.max()),
                 end_dx_m=float(diff[g,-1,0]),end_dy_m=float(diff[g,-1,1]),
                 gain_along_shift=gain,response_l2=response,
                 path_l2_m=float(np.linalg.norm(diff[g])),
                 nonrigid_rms_m=float(np.sqrt(np.mean(np.sum(nonrigid**2,axis=1)))),
                 lateral_neighbor_change_max_m=float(np.max(np.abs(lateral_neighbor))),
                 mean_dx_m=float(diff[g,:,0].mean()), mean_dy_m=float(diff[g,:,1].mean()),
                 base_x_end_m=float(p[g,-1,0]),base_y_end_m=float(p[g,-1,1]),
                 modified_x_end_m=float(q[g,-1,0]),modified_y_end_m=float(q[g,-1,1]))
        if reward_before is not None and reward_after is not None:
            for field in reward_before[mode]:
                before=float(reward_before[mode][field][g]);after=float(reward_after[mode][field][g])
                row['base_'+field]=before;row['modified_'+field]=after
                if field=='reward':row['reward_gain']=after-before
        out.append(row)
    return out


def save_csv(path,rows):
    if not rows: return
    columns=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def plot_outputs(rows, gallery, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=[r for r in rows if float(r['amplitude_m'])!=0]
    fig,ax=plt.subplots(figsize=(8,5))
    for profile in sorted(set(r['profile'] for r in data)):
        rr=[r for r in data if r['profile']==profile]
        xs=sorted(set(float(r['amplitude_m']) for r in rr))
        ys=[float(np.mean([r['rms_waypoint_disp_m'] for r in rr if float(r['amplitude_m'])==x])) for x in xs]
        ax.plot(xs,ys,'o-',label=profile)
    ax.set_xlabel('Guided Gaussian mean shift amplitude (m)')
    ax.set_ylabel('Final 8-point RMS displacement (m)')
    ax.set_title('Response of final trajectory to Gaussian mean shift')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(output/'01_mean_to_final_response.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    grouped=sorted(set(r['profile'] for r in data))
    for profile in grouped:
        rr=[r for r in data if r['profile']==profile]
        xs=sorted(set(abs(float(r['amplitude_m'])) for r in rr))
        yy=[float(np.mean([r['kl_nats'] for r in rr if abs(float(r['amplitude_m']))==x])) for x in xs]
        ax.plot(xs,yy,'o-',label=profile)
    ax.set_xlabel('Absolute mean shift amplitude (m)');ax.set_ylabel('Gaussian KL (nats per candidate/mode)')
    ax.set_yscale('symlog',linthresh=1);ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(output/'02_conditional_gaussian_kl.png',dpi=180);plt.close(fig)
    if any('reward_gain' in r for r in data):
        fig,ax=plt.subplots(figsize=(8,5))
        for profile in grouped:
            rr=[r for r in data if r['profile']==profile]
            xs=sorted(set(float(r['amplitude_m']) for r in rr))
            ys=[float(np.mean([r['reward_gain'] for r in rr if float(r['amplitude_m'])==x])) for x in xs]
            ax.plot(xs,ys,'o-',label=profile)
        ax.axhline(0,color='gray',linestyle='--',lw=1)
        ax.set_xlabel('Guided mean shift amplitude (m)');ax.set_ylabel('V5.2 reward change')
        ax.set_title('Reward response (not a policy training result)')
        ax.grid(alpha=.25);ax.legend();fig.tight_layout()
        fig.savefig(output/'03_reward_vs_mean_shift.png',dpi=180);plt.close(fig)
    if gallery:
        import random
        sample=list(gallery)
        random.Random(7).shuffle(sample)
        fig,axes=plt.subplots(3,3,figsize=(12,10))
        for ax,item in zip(axes.flat,sample[:9]):
            base=np.asarray(item['base']);new=np.asarray(item['modified'])
            ax.plot(base[:,0],base[:,1],'-o',ms=2,label='Original')
            ax.plot(new[:,0],new[:,1],'-o',ms=2,label='Mean shifted')
            ax.quiver(base[:,0],base[:,1],new[:,0]-base[:,0],new[:,1]-base[:,1],
                      angles='xy',scale_units='xy',scale=1,alpha=.5,width=.004)
            ax.set_title(f"{item['scenario']} | {item['profile']} | {item['amplitude_m']:+.2f}m",fontsize=9)
            ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='datalim')
        for ax in axes.flat[len(sample[:9]):]:ax.axis('off')
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout();fig.savefig(output/'04_trajectory_gallery_3x3.png',dpi=180);plt.close(fig)


def run(args):
    if args.group_size<2 or args.samples_per_scenario<1: raise ValueError('group-size >=2, samples >=1')
    if args.eta<=0 or args.min_std<=0: raise ValueError('eta/min-std must be positive')
    profiles=[a.strip() for a in args.profiles.split(',') if a.strip()]
    if not profiles: raise ValueError('No profiles')
    for x in profiles:profile_points(x)
    amplitudes=parse_values(args.amplitudes_m)
    if 0. not in amplitudes: amplitudes=[0.]+amplitudes
    root=setup_worktree(args.root)
    checkpoint=args.checkpoint.resolve()
    if not checkpoint.is_file(): raise FileNotFoundError('checkpoint: '+str(checkpoint))
    out=args.output_dir.resolve()
    if args.dry_run:
        print(json.dumps({'worktree':str(root),'checkpoint':str(checkpoint),'output':str(out),
                          'profiles':profiles,'amplitudes_m':amplitudes,
                          'experiment':'READ ONLY; common RNG; no training; no MATP'},indent=2));return
    # Import only after root is inserted, so running in AllMerge root targets correct checkout.
    from scripts.run_grpo_noise_probe import (SCENARIOS,CheckpointSpec,_build_model,
                                                _frozen_reward_context,_scenario_features,
                                                _state_seed,_sample_noise_seed)
    from highway_env.planner.diffusion.grpo.sampling import GroupDiffusionSampler
    from highway_env.planner.diffusion.grpo.reward_adapter import CandidateRewardAdapter,resolve_reward_evaluator
    from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
    device=torch.device(args.device)
    adapter,model=_build_model(CheckpointSpec('mean_shift_probe',checkpoint),device)
    sampler=GroupDiffusionSampler(model,group_size=args.group_size,eta=args.eta,min_std=args.min_std)
    reward_adapter=CandidateRewardAdapter(resolve_reward_evaluator('auto')) if not args.no_score else None
    cfg=replace(TrajectoryModeRewardConfig(),task_centerline_weight=args.centerline_weight)
    out.mkdir(parents=True,exist_ok=True)
    rows=[];gallery=[];attempt_log=[];max_replay_error=0.
    scenarios=list(SCENARIOS.keys()) if args.scenario=='all' else [args.scenario]
    for sidx,sname in enumerate(scenarios):
        count=attempts=0
        while count<args.samples_per_scenario:
            if attempts>=args.max_seed_attempts:raise RuntimeError(f'Cannot collect {sname} valid states')
            env_seed=_state_seed(args.seed,sidx,attempts);attempts+=1
            env=SCENARIOS[sname](config={'show_trajectories':False,'show_future_trajectories':False},render_mode=None)
            try:
                env.reset(seed=env_seed)
                _,_,done,trunc,_=env.step(args.group_action)
                if done or trunc:
                    attempt_log.append(dict(scenario=sname,env_seed=env_seed,reason='terminated'));continue
                features=_scenario_features(env,adapter)
                frozen,selected=_frozen_reward_context(model,features,env)
                context=replace(frozen,config=cfg)
                noise_seed=_sample_noise_seed(args.noise_seed,sidx,count)
                gen=torch.Generator(device=device.type).manual_seed(noise_seed)
                with torch.no_grad():
                    trace=sampler.sample(features,generator=gen)
                if len(trace.states)!=1 or trace.timesteps!=(8,0):
                    raise RuntimeError('This probe requires exactly the current 8->0 randomized transition; got '+str(trace.timesteps))
                base_xt0=trace.next_states[0].detach()
                baseline=terminal_decode(model,features,base_xt0,args.group_size)
                error=float((baseline-trace.candidates).abs().max().item())
                max_replay_error=max(error,max_replay_error)
                if error>2e-3:
                    raise RuntimeError(f'Baseline replay mismatch {error:.6g}m; cannot interpret mean shifts')
                states=trace.states[0].reshape(3*args.group_size,*trace.states[0].shape[2:])
                # Get true baseline DDIM mean/std using existing scheduler, with no resampling.
                from highway_env.planner.diffusion.grpo.sampling import _expand_group
                expand=_expand_group(features,args.group_size)
                anchor=model.adapter.normalize_trajectory(expand['coarse_trajectories'])
                t8=torch.full((states.shape[0],),8,dtype=torch.long,device=device)
                with torch.no_grad():
                    scene=model.scene_encoder(expand)
                    predicted,_=model.denoiser(states,t8,scene,navigation_point_norm=anchor[:,:,-1,:])
                    baseline_transition=sampler.transition.step(states,predicted,8,0,
                                             prev_sample=base_xt0.reshape_as(states))
                lp_err=float((baseline_transition.log_prob.reshape(3,args.group_size,10)-trace.old_log_prob[:,:,0]).abs().max().item())
                if lp_err>1e-3:
                    raise RuntimeError(f'Old log-prob replay failed: {lp_err:.4g}')
                std=baseline_transition.std
                mode_indices=[int(selected[args.role].item())] if args.mode_scope=='selected' else [int(i) for i in torch.where(features['mode_valid_mask'][args.role])[0]]
                if not mode_indices:raise RuntimeError('No valid modes')
                base_scores=score_selected(baseline,reward_adapter,features,context,args.role,mode_indices) if reward_adapter else None
                for profile in profiles:
                    unit=torch.as_tensor(profile_points(profile,horizon=baseline.shape[-2]),dtype=baseline.dtype,device=device)
                    for amplitude in amplitudes:
                        delta_phys=unit*float(amplitude)
                        delta_norm=adapter.normalize_trajectory(delta_phys)
                        new_xt0=base_xt0.clone()
                        # Same random draw z; shift mean ONLY for selected role and modes.
                        new_xt0[args.role,:,mode_indices,:,:]+=delta_norm[None,None,:,:]
                        if float(amplitude)==0:modified=baseline
                        else:modified=terminal_decode(model,features,new_xt0,args.group_size)
                        modified_scores=score_selected(modified,reward_adapter,features,context,args.role,mode_indices) if reward_adapter else None
                        untouched_modes=[m for m in range(baseline.shape[2]) if m not in mode_indices]
                        mode_leak_max=float((modified[args.role,:,untouched_modes]-baseline[args.role,:,untouched_modes]).abs().max().item()) if untouched_modes else 0.
                        other_roles=[r for r in range(baseline.shape[0]) if r!=args.role]
                        other_role_leak_max=float((modified[other_roles]-baseline[other_roles]).abs().max().item()) if other_roles else 0.
                        for mode in mode_indices:
                            rr=per_mode_rows(baseline,modified,args.role,mode,delta_phys,delta_norm,std,
                                             base_scores,modified_scores)
                            mean_base=baseline_transition.mean.reshape(3,args.group_size,10,8,2)
                            eps_base=(base_xt0[args.role,:,mode]-mean_base[args.role,:,mode])/std
                            logq=trace.old_log_prob[args.role,:,0,mode]
                            llr=(eps_base*(delta_norm/std)).sum(dim=(-1,-2)) + gaussian_kl_shift(delta_norm,std)
                            for r in rr:
                                g=r['group']
                                r['non_target_mode_max_abs_m']=mode_leak_max
                                r['other_vehicle_max_abs_m']=other_role_leak_max
                                r['guided_action_logq']=float(logq[g].item())
                                r['guided_action_logp_base']=float((logq[g]-llr[g]).item())
                                r['guided_logratio_q_over_p']=float(llr[g].item())
                                r.update(scenario=sname,state_id=count,env_seed=env_seed,noise_seed=noise_seed,
                                         profile=profile,amplitude_m=amplitude,eta=args.eta,ddim_std_norm=float(std.item()),
                                         baseline_replay_max_error_m=error)
                            rows.extend(rr)
                            if amplitude!=0 and len(gallery)<10000:
                                # Explicit fixed group0, not best-reward cherry-picking.
                                gallery.append(dict(scenario=sname,profile=profile,amplitude_m=amplitude,
                                     base=baseline[args.role,0,mode].cpu().numpy(),
                                     modified=modified[args.role,0,mode].cpu().numpy()))
                print(f'[state] {sname} {count+1}/{args.samples_per_scenario} modes={mode_indices} '
                      f'baseline_replay_error={error:.6f}m std_norm={float(std.item()):.7g}')
                count+=1
            finally:
                env.close()
    save_csv(out/'mean_shift_samples.csv',rows)
    save_csv(out/'skipped_states.csv',attempt_log)
    summary={
        'status':'read_only_complete','method':'DDIM mean shift with identical epsilon; no weights updated',
        'root':str(root),'checkpoint':str(checkpoint),
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'config':vars(args)|{'root':str(root),'checkpoint':str(checkpoint),'output_dir':str(out)},
        'max_base_replay_error_m':max_replay_error,
        'num_rows':len(rows),'num_unique_states':args.samples_per_scenario*len(scenarios),
        'summary_by_profile':{},'summary_by_scenario':{},
        'interpretation':'Gaussian KL is conditional transition distribution KL, not terminal trajectory KL; reward is observed, not trained.'}
    for profile in profiles:
        r=[x for x in rows if x['profile']==profile and x['amplitude_m']!=0]
        summary['summary_by_profile'][profile]={
            'mean_final_rms_disp_m':float(np.mean([x['rms_waypoint_disp_m'] for x in r])) if r else None,
            'mean_gain_along_shift':float(np.mean([x['gain_along_shift'] for x in r])) if r else None,
            'mean_reward_gain':float(np.mean([x['reward_gain'] for x in r])) if r and 'reward_gain' in r[0] else None}
    for name in scenarios:
        r=[x for x in rows if x['scenario']==name and x['amplitude_m']!=0]
        summary['summary_by_scenario'][name]={
            'mean_final_rms_disp_m':float(np.mean([x['rms_waypoint_disp_m'] for x in r])) if r else None,
            'positive_reward_gain_fraction':float(np.mean([x['reward_gain']>0 for x in r])) if r and 'reward_gain' in r[0] else None}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    if not args.no_plots:plot_outputs(rows,gallery,out)
    print('[OK] results:',out)
    print('[OK] exact baseline replay error (max m):',max_replay_error)
    print('[OK] output rows:',len(rows))


if __name__=='__main__':run(get_parser().parse_args())
