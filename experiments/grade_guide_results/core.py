#!/usr/bin/env python3
"""Frozen reward-gradient guidance ON the DDIM 8->0 Gaussian mean.

No model, optimizer, PPO, MATP or on-policy training is changed. All comparisons
hold the same sampled Gaussian residual and recompute the terminal denoiser.
Primary actual reward: existing current-branch 10Hz cubic-dense V5.2 production
`progress_comfort`, including the original collision/road/curvature diagnostics.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .math_utils import metric_gradient,normalized_direction,cap_total_shift,geometry_metrics
from . import bridge


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('all_merge-guideDistill'))
    p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--scenario',choices=('all','straight','curved','merge_in','merge_out'),default='all')
    p.add_argument('--samples-per-scenario',type=int,default=2)
    p.add_argument('--group-size',type=int,default=8)
    p.add_argument('--role',type=int,default=0,choices=(0,1,2))
    p.add_argument('--mode-scope',choices=('selected','all'),default='selected')
    p.add_argument('--step-sizes-m',default='0,0.03,0.10,0.30,0.60,1.00',
        help='SINGLE gradient direction at baseline, physical RMS move per candidate')
    p.add_argument('--iter-steps',type=int,default=4,
        help='Iterated reward-gradient steps (0 to disable). Always report raw trial and accepted step.')
    p.add_argument('--iter-step-m',type=float,default=0.20)
    p.add_argument('--line-search-factors',default='1,0.5,0.25,0.125')
    p.add_argument('--max-x-shift-m',type=float,default=10.)
    p.add_argument('--max-y-shift-m',type=float,default=3.)
    p.add_argument('--max-rms-shift-m',type=float,default=3.)
    p.add_argument('--eta',type=float,default=.02)
    p.add_argument('--min-std',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=20261017)
    p.add_argument('--noise-seed',type=int,default=70261017)
    p.add_argument('--group-action',type=int,default=0)
    p.add_argument('--max-seed-attempts',type=int,default=100)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--centerline-weight',type=float,default=.10)
    p.add_argument('--parity-tol',type=float,default=.002)
    p.add_argument('--road-sampling-m',type=float,default=.4)
    p.add_argument('--output-dir',type=Path,default=Path('all_merge-guideDistill/outputs/reward_guided_mean_v3'))
    p.add_argument('--gallery-states-per-scenario',type=int,default=2)
    p.add_argument('--gallery-group',type=int,default=0)
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--plot-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p


def write_csv(path,rows):
    if not rows:return
    cols=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)


def scalars(x):
    return x.detach().cpu().numpy().astype(float)


def make_eval(model, features, g):
    """Build the exact frozen terminal denoiser without torch.no_grad on its input.

    Scene encoding and parameters are frozen; gradient only flows to DDIM x0 input.
    Entire mode set retained, so any cross-mode influence is represented.
    """
    from highway_env.planner.diffusion.grpo.sampling import _expand_group
    exp=_expand_group(features,g)
    with torch.no_grad():
        scene=model.scene_encoder(exp)
        anchors=model.adapter.normalize_trajectory(exp['coarse_trajectories'])
    timesteps=torch.zeros(3*g,device=anchors.device,dtype=torch.long)
    def evaluate(xt0):
        pred,_=model.denoiser(xt0.reshape(3*g,*xt0.shape[2:]),timesteps,
                              scene,navigation_point_norm=anchors[:,:,-1,:])
        phys=model.adapter.denormalize_trajectory(pred)
        return phys.reshape(3,g,*phys.shape[1:])
    return evaluate


def gradient_direction(base_input, current_delta, role, mode, decode, proxy, physical_scale):
    """Chain rule: grad of dense reward wrt final 8pt -> Jacobian^T -> Gaussian mean.

    Crucially, do NOT set dmu proportional to dR/d(final trajectory).
    """
    # GRU backward in eval() under CUDA/cuDNN raises an exception. Native RNN
    # kernels preserve eval semantics and allow input-gradient calculation.
    with torch.backends.cudnn.flags(enabled=False), torch.enable_grad():
        var=base_input.detach().clone().requires_grad_(True)
        pred=decode(var)
        xy=pred[role,:,mode]
        grad_xy=proxy.gradient_of(xy.detach()) # production-aligned dense 10Hz V5.2
        grad_norm=torch.autograd.grad(xy,var,grad_outputs=grad_xy)[0][role,:,mode].detach()
    grad_phys=metric_gradient(grad_norm,physical_scale)
    return normalized_direction(grad_phys)


def evaluate_candidate(base_xt0,delta_phys,role,mode,model,decode,scale):
    v=base_xt0.detach().clone()
    v[role,:,mode] += (delta_phys/scale)
    with torch.no_grad(): return decode(v).detach()


def proxy_reward(production_proxy, candidate,role,mode):
    with torch.no_grad():
        val=production_proxy.components(candidate[role,:,mode])['balanced']
    if not bool(torch.isfinite(val).all()):raise FloatingPointError('Nonfinite dense proxy reward')
    return val.detach()


def score_prod(reward_adapter,candidate,features,context,role,mode):
    scores=bridge.score_selected(candidate,reward_adapter,features,context,role,[mode])
    return {k:np.asarray(v,dtype=float) for k,v in scores[mode].items()}


def build_rows(scenario,state,env_seed,noise_seed,role,mode,method,step,alpha,
               baseline,modified,base_reward,modified_reward,base_comp,mod_comp,
               shift_phys,grad_rms,std,proxy_base,proxy_mod,parity_error):
    b=baseline[role,:,mode]
    m=modified[role,:,mode]
    geo=geometry_metrics(b,m)
    from highway_env.planner.diffusion.trajectory_mode_reward.spline_dense import evaluate_dense_spline
    from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
    # Real dense output displacement, using same production spline contract.
    # Same dense interpolation time grid as the production task evaluator.
    dense_b,_,_=evaluate_dense_spline(b,TrajectoryModeRewardConfig())
    dense_m,_,_=evaluate_dense_spline(m,TrajectoryModeRewardConfig())
    dd=dense_m-dense_b
    dense_rms=torch.sqrt(dd[:,1:].square().sum(-1).mean(-1))
    dense_nr=dd[:,1:]-dd[:,1:].mean(-2,keepdim=True)
    dense_shape=torch.sqrt(dense_nr.square().sum(-1).mean(-1))
    rawshift=torch.sqrt(shift_phys.square().sum(-1).mean(-1))
    gaussian_kl=.5*(shift_phys/std).square().sum((-1,-2))
    safe_keys=('unsafe','collision','out_of_drivable','minimum_road_margin_m',
               'curvature_max_abs','centerline_late_error_m')
    result=[]
    for g in range(b.shape[0]):
        row=dict(scenario=scenario,state_id=state,env_seed=env_seed,noise_seed=noise_seed,
          group=g,role=role,mode=mode,method=method,step=step,step_size_m=alpha,
          baseline_reward=float(base_reward[g]),guided_reward=float(modified_reward[g]),
          reward_gain=float(modified_reward[g]-base_reward[g]),
          proxy_reward_before=float(proxy_base[g]),proxy_reward_after=float(proxy_mod[g]),
          proxy_reward_gain=float(proxy_mod[g]-proxy_base[g]),
          proxy_production_abs_err_after=abs(float(proxy_mod[g])-float(modified_reward[g])),
          gradient_rms_initial=float(grad_rms[g]),
          mean_shift_rms_m=float(rawshift[g]),
          mean_shift_max_x_m=float(shift_phys[g,:,0].abs().max()),
          mean_shift_max_y_m=float(shift_phys[g,:,1].abs().max()),
          conditional_kl_nats=float(gaussian_kl[g]),
          dense_rms_displacement_m=float(dense_rms[g]),
          dense_nonrigid_rms_m=float(dense_shape[g]),
          parity_max_err_before=parity_error,
          **{k:float(v[g]) for k,v in geo.items()})
        for k in safe_keys:
            if k in base_comp and k in mod_comp:
                row['base_'+k]=float(base_comp[k][g]);row['guided_'+k]=float(mod_comp[k][g]);
        row['safe_improved']=int(row.get('base_unsafe',1)>.5 and row.get('guided_unsafe',1)<.5)
        row['safe_regressed']=int(row.get('base_unsafe',1)<.5 and row.get('guided_unsafe',1)>.5)
        result.append(row)
    return result


def run(args):
    if args.plot_only:
        from .stats import plot
        plot(args.output_dir.resolve());return
    if args.group_size<2 or args.samples_per_scenario<1 or args.iter_steps<0:raise ValueError('Invalid group, states or iterations')
    if min(args.eta,args.min_std,args.max_x_shift_m,args.max_y_shift_m,args.max_rms_shift_m)<=0:raise ValueError('Positive variances / caps required')
    alphas=bridge.parse_values(args.step_sizes_m)
    if any(x<0 for x in alphas):raise ValueError('Step sizes must be nonnegative')
    factors=bridge.parse_values(args.line_search_factors)
    if not factors or any(x<=0 or x>1 for x in factors):raise ValueError('line search must contain factors in (0,1]')
    root=bridge.setup_worktree(args.root)
    checkpoint=args.checkpoint.resolve() if args.checkpoint is not None else None
    out=args.output_dir.resolve()
    config_json={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    if args.dry_run:
        print(json.dumps({'worktree':str(root),'checkpoint':str(checkpoint),'config':config_json,
             'reward':'current V5.2 task reward, 41 point 10Hz spline',
             'no_model_weights_changed':True},indent=2));return
    if checkpoint is None or not checkpoint.is_file():raise FileNotFoundError(f'Checkpoint required: {checkpoint}')
    from scripts.run_grpo_noise_probe import (SCENARIOS,CheckpointSpec,_build_model,
          _frozen_reward_context,_scenario_features,_state_seed,_sample_noise_seed)
    from highway_env.planner.diffusion.grpo.sampling import GroupDiffusionSampler
    from highway_env.planner.diffusion.grpo.reward_adapter import CandidateRewardAdapter,resolve_reward_evaluator
    from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
    from highway_env.planner.diffusion.guide_distill.reward_gradient import DifferentiableTaskReward
    from highway_env.planner.diffusion.guide_distill.road_boundary import RoadBoundaryField
    from highway_env.planner.diffusion.trajectory_mode_reward.state_adapter import _planning_state_snapshot,_vehicle_pose
    device=torch.device(args.device)
    adapter,model=_build_model(CheckpointSpec('reward_guided_mean_v3',checkpoint),device)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    sampler=GroupDiffusionSampler(model,group_size=args.group_size,eta=args.eta,min_std=args.min_std)
    reward_adapter=CandidateRewardAdapter(resolve_reward_evaluator('auto'))
    cfg=replace(TrajectoryModeRewardConfig(),task_centerline_weight=args.centerline_weight)
    out.mkdir(parents=True,exist_ok=True)
    rows=[];gallery=[];attempts=[];max_replay=0.;parity_max=0.
    names=list(SCENARIOS.keys()) if args.scenario=='all' else [args.scenario]
    for si,sname in enumerate(names):
        accepted=tries=0
        while accepted<args.samples_per_scenario:
            if tries>=args.max_seed_attempts:raise RuntimeError('No valid state for '+sname)
            env_seed=_state_seed(args.seed,si,tries);tries+=1
            env=SCENARIOS[sname](config={'show_trajectories':False,'show_future_trajectories':False},render_mode=None)
            try:
                env.reset(seed=env_seed)
                _,_,term,trunc,_=env.step(args.group_action)
                if term or trunc:
                    attempts.append({'scenario':sname,'env_seed':env_seed,'reason':'terminated'});continue
                features=_scenario_features(env,adapter)
                frozen,selected=_frozen_reward_context(model,features,env)
                context=replace(frozen,config=cfg)
                trace=None
                gen=torch.Generator(device=device.type).manual_seed(_sample_noise_seed(args.noise_seed,si,accepted))
                with torch.no_grad():trace=sampler.sample(features,generator=gen)
                if len(trace.states)!=1 or trace.timesteps!=(8,0):raise RuntimeError('Requires current DDIM 8->0')
                base_xt0=trace.next_states[0].detach()
                decode=make_eval(model,features,args.group_size)
                with torch.no_grad(): baseline=decode(base_xt0).detach()
                replay_error=float((baseline-trace.candidates).abs().max().item())
                max_replay=max(max_replay,replay_error)
                if replay_error>2e-3:raise RuntimeError(f'Pair mismatch {replay_error:.7g}')
                mode_indices=([int(selected[args.role].item())] if args.mode_scope=='selected' else
                      [int(m) for m in torch.where(features['mode_valid_mask'][args.role])[0]])
                snapshot=_planning_state_snapshot(env)
                pose=(np.asarray(snapshot[0][args.role],dtype=np.float64) if snapshot is not None else
                       _vehicle_pose(env.controlled_vehicles[args.role]))
                road=RoadBoundaryField.from_road(env.road,sampling_m=args.road_sampling_m)
                proxy=DifferentiableTaskReward(cfg,device)
                proxy.bind_road(road,pose)
                proxy.bind_centerline(features['target_lane_polyline'][args.role,:,:2].detach().cpu().numpy())
                scale=adapter.trajectory_scale.to(device=device,dtype=base_xt0.dtype)
                transition_replay= sampler.transition.step(
                    trace.states[0].reshape(3*args.group_size,10,8,2),
                    _predicted_t8(model,features,trace.states[0],args.group_size),8,0,
                    prev_sample=base_xt0.reshape(3*args.group_size,10,8,2))
                replay_logprob_error=float((transition_replay.log_prob.reshape(3,args.group_size,10)
                      -trace.old_log_prob[:,:,0]).abs().max().item())
                if replay_logprob_error>1e-3:
                    raise RuntimeError(f'Behavior log-prob replay mismatch {replay_logprob_error}')
                std_norm=transition_replay.std
                for mode in mode_indices:
                    initial_scores=score_prod(reward_adapter,baseline,features,context,args.role,mode)
                    base_reward=initial_scores['reward']
                    # Require agreement with production dense reward BEFORE reporting improvements.
                    prod_result=reward_adapter.evaluate_result(baseline,features=features,context=context)
                    parity=proxy.validate_against_production(baseline[args.role,:,mode],prod_result,
                                                           role=args.role,mode=mode,tol=args.parity_tol)
                    parity_err=max(parity.values());parity_max=max(parity_max,parity_err)
                    with torch.no_grad():proxy_base=proxy_reward(proxy,baseline,args.role,mode)
                    unit,g_rms,good=gradient_direction(base_xt0,torch.zeros_like(base_xt0[args.role,:,mode]),
                                                args.role,mode,decode,proxy,scale)
                    # A. Frozen first-order direction, full signed response curve, no cherry-picking.
                    for alpha in alphas:
                        delta=cap_total_shift(unit*alpha,max_x=args.max_x_shift_m,max_y=args.max_y_shift_m,
                                              max_rms=args.max_rms_shift_m)
                        modified=baseline if alpha==0 else evaluate_candidate(base_xt0,delta,args.role,mode,model,decode,scale)
                        ps=proxy_reward(proxy,modified,args.role,mode)
                        scores=score_prod(reward_adapter,modified,features,context,args.role,mode)
                        rr=build_rows(sname,accepted,env_seed,_sample_noise_seed(args.noise_seed,si,accepted),
                                   args.role,mode,'single_step',0,alpha,baseline,modified,
                                   base_reward,scores['reward'],initial_scores,scores,delta,g_rms,
                                   float(std_norm.item())*scale,proxy_base,ps,parity_err)
                        rows.extend(rr)
                        if len(gallery)<10000 and accepted<args.gallery_states_per_scenario and alpha>0:
                            from highway_env.planner.diffusion.trajectory_mode_reward.spline_dense import evaluate_dense_spline
                            db,_,_=evaluate_dense_spline(baseline[args.role,args.gallery_group,mode],cfg)
                            dg,_,_=evaluate_dense_spline(modified[args.role,args.gallery_group,mode],cfg)
                            gallery.append({'scenario':sname,'state_id':accepted,'mode':mode,'group':args.gallery_group,
                                'method':'single_step','step_size_m':alpha,
                                'baseline':scalars(baseline[args.role,args.gallery_group,mode]),
                                'guided':scalars(modified[args.role,args.gallery_group,mode]),
                                'baseline_dense':scalars(db),'guided_dense':scalars(dg)})
                    # B. Multiple re-evaluated gradient steps, per-sample backtracking acceptance.
                    if args.iter_steps:
                        cur_delta=torch.zeros_like(base_xt0[args.role,:,mode])
                        current=baseline
                        previous=proxy_base.clone()
                        for k in range(1,args.iter_steps+1):
                            current_xt0=base_xt0.detach().clone()
                            current_xt0[args.role,:,mode]+=cur_delta/scale
                            direction,g_rms_step,valid=gradient_direction(current_xt0,cur_delta,
                                                          args.role,mode,decode,proxy,scale)
                            raw=cap_total_shift(cur_delta+direction*args.iter_step_m,
                                max_x=args.max_x_shift_m,max_y=args.max_y_shift_m,max_rms=args.max_rms_shift_m)
                            raw_sample=evaluate_candidate(base_xt0,raw,args.role,mode,model,decode,scale)
                            raw_proxy=proxy_reward(proxy,raw_sample,args.role,mode)
                            raw_score=score_prod(reward_adapter,raw_sample,features,context,args.role,mode)
                            rows.extend(build_rows(sname,accepted,env_seed,_sample_noise_seed(args.noise_seed,si,accepted),
                                args.role,mode,'iter_raw',k,args.iter_step_m,baseline,raw_sample,
                                base_reward,raw_score['reward'],initial_scores,raw_score,raw,g_rms_step,
                                float(std_norm.item())*scale,proxy_base,raw_proxy,parity_err))
                            # Reject if the *dense guidance reward* does not rise, but preserve raw log.
                            accepted_delta=cur_delta.clone()
                            accepted_proxy=previous.clone()
                            for f in factors:
                                candidate_delta=cap_total_shift(cur_delta+direction*(args.iter_step_m*f),
                                    max_x=args.max_x_shift_m,max_y=args.max_y_shift_m,max_rms=args.max_rms_shift_m)
                                candidate=evaluate_candidate(base_xt0,candidate_delta,args.role,mode,model,decode,scale)
                                pr=proxy_reward(proxy,candidate,args.role,mode)
                                good_step=(pr>accepted_proxy+1e-7)&valid
                                accepted_delta=torch.where(good_step[:,None,None],candidate_delta,accepted_delta)
                                accepted_proxy=torch.where(good_step,pr,accepted_proxy)
                            cur_delta=accepted_delta
                            current=evaluate_candidate(base_xt0,cur_delta,args.role,mode,model,decode,scale)
                            previous=accepted_proxy
                            final_score=score_prod(reward_adapter,current,features,context,args.role,mode)
                            rows.extend(build_rows(sname,accepted,env_seed,_sample_noise_seed(args.noise_seed,si,accepted),
                                args.role,mode,'iter_accepted',k,args.iter_step_m,baseline,current,
                                base_reward,final_score['reward'],initial_scores,final_score,cur_delta,g_rms_step,
                                float(std_norm.item())*scale,proxy_base,previous,parity_err))
                            if accepted<args.gallery_states_per_scenario:
                                from highway_env.planner.diffusion.trajectory_mode_reward.spline_dense import evaluate_dense_spline
                                db,_,_=evaluate_dense_spline(baseline[args.role,args.gallery_group,mode],cfg)
                                dg,_,_=evaluate_dense_spline(current[args.role,args.gallery_group,mode],cfg)
                                gallery.append({'scenario':sname,'state_id':accepted,'mode':mode,'group':args.gallery_group,
                                    'method':'iter_accepted','step_size_m':k,
                                    'baseline':scalars(baseline[args.role,args.gallery_group,mode]),
                                    'guided':scalars(current[args.role,args.gallery_group,mode]),
                                    'baseline_dense':scalars(db),'guided_dense':scalars(dg)})
                print(f'[state] {sname} {accepted+1}/{args.samples_per_scenario} '
                      f'modes={mode_indices} baseline replay={replay_error:.7g} '
                      f'parity={parity_err:.7g}',flush=True)
                accepted+=1
            finally:env.close()
    write_csv(out/'samples.csv',rows);write_csv(out/'skipped_states.csv',attempts)
    (out/'gallery.json').write_text(json.dumps(gallery,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')
    def stats(rr):
        if not rr:return {}
        r=np.asarray([v['reward_gain'] for v in rr],dtype=float)
        safe=np.asarray([v.get('guided_unsafe',float('nan')) for v in rr],dtype=float)
        return dict(n=len(rr),mean_gain=float(r.mean()),median_gain=float(np.median(r)),
                    positive_fraction=float((r>1e-6).mean()),safe_fraction=float((safe<.5).mean()),
                    mean_dense_rms_displacement_m=float(np.mean([v['dense_rms_displacement_m'] for v in rr])))
    summary=dict(method='reward-Jacobian guided DDIM Gaussian mean; no weight updates',
        target='production progress_comfort V5.2 and differentiable task guidance on 10Hz spline (41 incl t0)',
        checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        worktree=str(root),config=config_json,baseline_max_replay_error_m=max_replay,
        proxy_production_max_abs_error=parity_max,n_rows=len(rows),
        max_proxy_vs_production_err_guided=max(r['proxy_production_abs_err_after'] for r in rows),
        by_method={m:stats([r for r in rows if r['method']==m and (m!='single_step' or r['step_size_m']>0)]) for m in ('single_step','iter_raw','iter_accepted')},
        by_scenario={s:stats([r for r in rows if r['scenario']==s and r['method']=='iter_accepted' and r['step']==args.iter_steps]) for s in names},
        note='Single_step is unselected raw result; iter_accepted selects by dense differentiable reward (not production safety). Reward includes sparse terminal progress plus dense 10Hz comfort/road/curvature/centerline; collision is hard diagnosis, not differentiable task gradient.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    if not args.no_plots:
        from .stats import plot
        plot(out)
    print('[COMPLETE]',out,'rows=',len(rows),'replay_error=',max_replay,'reward_parity=',parity_max)


def _predicted_t8(model,features,states,g):
    from highway_env.planner.diffusion.grpo.sampling import _expand_group
    exp=_expand_group(features,g)
    with torch.no_grad():
        scene=model.scene_encoder(exp)
        anchors=model.adapter.normalize_trajectory(exp['coarse_trajectories'])
        t=torch.full((3*g,),8,dtype=torch.long,device=anchors.device)
        pred,_=model.denoiser(states.reshape(3*g,10,8,2),t,scene,navigation_point_norm=anchors[:,:,-1,:])
    return pred

if __name__=='__main__':run(cli().parse_args())
