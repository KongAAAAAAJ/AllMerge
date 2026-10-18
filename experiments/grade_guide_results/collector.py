#!/usr/bin/env python3
"""Frozen guided-Gaussian mean reward comparison, paired random controls, larger state set.

All controls use identical scene, mode, G-group source, stochastic DDIM epsilon,
original model weights, terminal Denoiser, 10Hz Clamped Cubic Spline and production reward.
Gradient and random proposals are UNSELECTED in the primary comparisons.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys,traceback
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from .math_utils import normalized_direction,cap_total_shift
from .random_controls import random_like,input_match_error
from . import bridge
from . import core
from .parity import validate_reward_parity


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--scenario',choices=('all','straight','curved','merge_in','merge_out'),default='all')
    p.add_argument('--samples-per-scenario',type=int,default=20)
    p.add_argument('--group-size',type=int,default=8)
    p.add_argument('--role',type=int,choices=(0,1,2),default=0)
    p.add_argument('--mode-scope',choices=('selected','all'),default='selected')
    p.add_argument('--step-sizes-m',default='0,0.10,0.30,0.60,1.0,2.0,4.0,8.0,12.0,16.0',
                   help='Physical RMS mean shifts. Caps apply afterwards, logged as actual shifts.')
    p.add_argument('--max-x-shift-m',type=float,default=20.)
    p.add_argument('--max-y-shift-m',type=float,default=6.)
    p.add_argument('--max-rms-shift-m',type=float,default=16.)
    p.add_argument('--gradient-axis',choices=('xy','x','y'),default='xy',
                   help='xy full Jacobian reward gradient; x/y project reward gradient to one axis before normalization')
    p.add_argument('--random-controls',default='random_sign,random_gaussian')
    p.add_argument('--random-repeats',type=int,default=2)
    p.add_argument('--run-seed',type=int,default=0,help='Seed identifier written to all results')
    p.add_argument('--seed',type=int,default=20261017)
    p.add_argument('--noise-seed',type=int,default=70261017)
    p.add_argument('--random-seed',type=int,default=90261017)
    p.add_argument('--gallery-seed',type=int,default=20261018)
    p.add_argument('--gallery-states-per-scenario',type=int,default=3)
    p.add_argument('--gallery-amplitudes-m',default='0.10,0.30,0.60,1.0,2.0')
    p.add_argument('--gallery-group',type=int,default=-1,help='-1 draws fixed random group per state')
    p.add_argument('--eta',type=float,default=.02)
    p.add_argument('--min-std',type=float,default=1e-5)
    p.add_argument('--group-action',type=int,default=0)
    p.add_argument('--max-seed-attempts',type=int,default=200)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--centerline-weight',type=float,default=.10)
    p.add_argument('--parity-tol',type=float,default=.002)
    p.add_argument('--road-sampling-m',type=float,default=.4)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/reward_guided_mean_v5/seed_0'))
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--plot-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p


def checked_config(args):
    alphas=bridge.parse_values(args.step_sizes_m)
    if any(a<0 for a in alphas) or 0 not in alphas or len(set(alphas))!=len(alphas):
        raise ValueError('Step amplitudes must be unique, nonnegative and include 0')
    controls=[x.strip() for x in args.random_controls.split(',') if x.strip()]
    if not controls or len(set(controls))!=len(controls) or not set(controls).issubset({'random_sign','random_gaussian'}):
        raise ValueError('Random controls must be random_sign and/or random_gaussian')
    if (args.samples_per_scenario<1 or args.group_size<2 or args.random_repeats<1 or
        min(args.max_x_shift_m,args.max_y_shift_m,args.max_rms_shift_m,args.eta,args.min_std)<=0 or
        args.gallery_states_per_scenario<0):raise ValueError('Invalid experiment sizes/caps')
    return alphas,controls


def dense_xy(x,cfg):
    from highway_env.planner.diffusion.trajectory_mode_reward.spline_dense import evaluate_dense_spline
    xy,_,_=evaluate_dense_spline(x,cfg)
    return xy.detach().cpu().numpy().astype(float)


def score_production(reward_adapter,candidate,features,context,role,mode):
    return core.score_prod(reward_adapter,candidate,features,context,role,mode)


def summarize_row(scenario,state,env_seed,noise_seed,role,mode,method,repeat,alpha,
                  baseline,modified,base_score,mod_score,delta,grad_rms,std,parity,origin_delta):
    rows=core.build_rows(scenario,state,env_seed,noise_seed,role,mode,method,0,alpha,
           baseline,modified,base_score['reward'],mod_score['reward'],base_score,mod_score,
           delta,grad_rms,std,np.full((delta.shape[0],),np.nan),np.full((delta.shape[0],),np.nan),parity)
    rel=input_match_error(origin_delta,delta)
    shift=torch.sqrt(delta.square().sum(-1).mean(-1))
    xshift=torch.sqrt(delta[...,0].square().mean(-1))
    yshift=torch.sqrt(delta[...,1].square().mean(-1))
    for g,r in enumerate(rows):
        r.update(random_repeat=repeat,actual_rms_shift_m=float(shift[g]),
             x_rms_shift_m=float(xshift[g]),y_rms_shift_m=float(yshift[g]),
             target_rms_m=float(torch.sqrt(origin_delta[g].square().sum(-1).mean()).item()),
             input_match_rel_error=float(rel[g]),reward_valid=int(math.isfinite(r['reward_gain'])),
             group_state_key=f'{scenario}/{state}/{mode}/{g}',failed=0)
    return rows


def draw_seed(base,scenario_idx,state,mode,method,repeat):
    # Stable deterministic mapping, independent of future state count and call order.
    meth={'random_sign':11,'random_gaussian':23}[method]
    return int(base+scenario_idx*100_000_000+state*1_000_000+mode*10000+repeat*50+meth)


def run(args):
    if args.plot_only:
        raise ValueError('Use standalone plot.py with merged V5 samples.csv and trajectories.csv')
    alphas,controls=checked_config(args)
    root=bridge.setup_worktree(args.root)
    checkpoint=args.checkpoint.resolve() if args.checkpoint else None
    out=args.output_dir.resolve()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    if args.dry_run:
        print(json.dumps({'config':config,'root':str(root),'checkpoint':str(checkpoint),
           'design':'matched same state/mode/noise; unselected gradient vs random; 41pt 10Hz spline'},indent=2));return
    if checkpoint is None or not checkpoint.is_file():raise FileNotFoundError('Checkpoint not found: '+str(checkpoint))
    from scripts.run_grpo_noise_probe import SCENARIOS,CheckpointSpec,_build_model,_frozen_reward_context,_scenario_features,_state_seed,_sample_noise_seed
    from highway_env.planner.diffusion.grpo.sampling import GroupDiffusionSampler
    from highway_env.planner.diffusion.grpo.reward_adapter import CandidateRewardAdapter,resolve_reward_evaluator
    from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
    from highway_env.planner.diffusion.guide_distill.reward_gradient import DifferentiableTaskReward
    from highway_env.planner.diffusion.guide_distill.road_boundary import RoadBoundaryField
    from highway_env.planner.diffusion.trajectory_mode_reward.state_adapter import _planning_state_snapshot,_vehicle_pose
    device=torch.device(args.device)
    adapter,model=_build_model(CheckpointSpec('reward_guided_mean_v5',checkpoint),device)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    sampler=GroupDiffusionSampler(model,group_size=args.group_size,eta=args.eta,min_std=args.min_std)
    scorer=CandidateRewardAdapter(resolve_reward_evaluator('auto'))
    cfg=replace(TrajectoryModeRewardConfig(),task_centerline_weight=args.centerline_weight)
    out.mkdir(parents=True,exist_ok=True)
    output_rows=[];errors=[];pool=[];parity_max=0.;replay_max=0.
    names=list(SCENARIOS.keys()) if args.scenario=='all' else [args.scenario]
    gallery_amp={float(x) for x in bridge.parse_values(args.gallery_amplitudes_m)}
    rng_gallery=np.random.default_rng(args.gallery_seed)
    for si,sname in enumerate(names):
        accepted=tries=0
        while accepted<args.samples_per_scenario:
            if tries>=args.max_seed_attempts:raise RuntimeError('Unable to collect required states for '+sname)
            env_seed=_state_seed(args.seed,si,tries);tries+=1
            env=SCENARIOS[sname](config={'show_trajectories':False,'show_future_trajectories':False},render_mode=None)
            try:
                env.reset(seed=env_seed)
                _,_,term,trunc,_=env.step(args.group_action)
                if term or trunc:
                    errors.append({'scenario':sname,'env_seed':env_seed,'stage':'state','message':'terminated'})
                    continue
                features=_scenario_features(env,adapter)
                frozen,selected=_frozen_reward_context(model,features,env)
                context=replace(frozen,config=cfg)
                noise_seed=_sample_noise_seed(args.noise_seed,si,accepted)
                gen=torch.Generator(device=device.type).manual_seed(noise_seed)
                with torch.no_grad():trace=sampler.sample(features,generator=gen)
                if len(trace.states)!=1 or trace.timesteps!=(8,0):raise RuntimeError('V4 requires random 8->0 DDIM')
                input0=trace.next_states[0].detach()
                decode=core.make_eval(model,features,args.group_size)
                with torch.no_grad():base=decode(input0).detach()
                replay_error=float((base-trace.candidates).abs().max().item())
                replay_max=max(replay_max,replay_error)
                if replay_error>2e-3:raise RuntimeError('DDIM baseline replay mismatch '+str(replay_error))
                tr=sampler.transition.step(trace.states[0].reshape(3*args.group_size,10,8,2),
                      core._predicted_t8(model,features,trace.states[0],args.group_size),8,0,
                      prev_sample=input0.reshape(3*args.group_size,10,8,2))
                lp_err=float((tr.log_prob.reshape(3,args.group_size,10)-trace.old_log_prob[:,:,0]).abs().max().item())
                if lp_err>1e-3:raise RuntimeError('DDIM log-prob replay mismatch '+str(lp_err))
                std=tr.std*adapter.trajectory_scale.to(input0.device,dtype=input0.dtype)
                modes=([int(selected[args.role].item())] if args.mode_scope=='selected' else
                        [int(m) for m in torch.where(features['mode_valid_mask'][args.role])[0]])
                snapshot=_planning_state_snapshot(env)
                pose=(np.asarray(snapshot[0][args.role],dtype=np.float64) if snapshot is not None else
                       _vehicle_pose(env.controlled_vehicles[args.role]))
                road=RoadBoundaryField.from_road(env.road,sampling_m=args.road_sampling_m)
                proxy=DifferentiableTaskReward(cfg,device)
                proxy.bind_road(road,pose)
                proxy.bind_centerline(features['target_lane_polyline'][args.role,:,:2].detach().cpu().numpy())
                scale=adapter.trajectory_scale.to(device=device,dtype=input0.dtype)
                for mode in modes:
                    base_score=score_production(scorer,base,features,context,args.role,mode)
                    prod_result=scorer.evaluate_result(base,features=features,context=context)
                    parity=validate_reward_parity(proxy,base[args.role,:,mode],prod_result,
                                  role=args.role,mode=mode,tol=args.parity_tol)
                    parity_err=max(parity.values());parity_max=max(parity_max,parity_err)
                    direction,grad_rms,valid=core.gradient_direction(input0,
                        torch.zeros_like(input0[args.role,:,mode]),args.role,mode,decode,proxy,scale)
                    if args.gradient_axis != 'xy':
                        # Coordinate-restricted steepest reward ascent; keeps chain-rule direction.
                        mask=torch.tensor([1.,0.] if args.gradient_axis=='x' else [0.,1.],
                               dtype=direction.dtype,device=direction.device)
                        projected=direction*mask
                        direction,axis_grad_rms,valid=normalized_direction(projected)
                        grad_rms=grad_rms*axis_grad_rms
                    # state/group for gallery are sampled independently of reward.
                    gal_g=(args.gallery_group if args.gallery_group>=0 else int(rng_gallery.integers(args.group_size)))
                    state_gallery={'scenario':sname,'state_id':accepted,'env_seed':env_seed,'mode':mode,
                          'centerline_xy':features['target_lane_polyline'][args.role,:,:2].detach().cpu().numpy().astype(float),
                          'baseline_unsafe':float(base_score['unsafe'][gal_g]),
                          'group':gal_g,'baseline_dense':dense_xy(base[args.role,gal_g,mode],cfg),
                          'baseline_sparse':base[args.role,gal_g,mode].detach().cpu().numpy().astype(float),
                          'baseline_reward':float(base_score['reward'][gal_g]),'traces':[]}
                    for ai,alpha in enumerate(alphas):
                        if alpha==0:
                            # Save one baseline row per real candidate (not one for each random repeat).
                            zero=torch.zeros_like(direction)
                            rr=summarize_row(sname,accepted,env_seed,noise_seed,args.role,mode,'baseline',0,alpha,
                                base,base,base_score,base_score,zero,grad_rms,std,parity_err,zero)
                            for g,r in enumerate(rr):r['gradient_valid']=int(valid[g])
                            output_rows.extend(rr)
                            continue
                        delta=cap_total_shift(direction*alpha,max_x=args.max_x_shift_m,
                                     max_y=args.max_y_shift_m,max_rms=args.max_rms_shift_m)
                        proposals=[('reward_gradient',0,delta)]
                        for method in controls:
                            for rep in range(args.random_repeats):
                                r=random_like(delta,method=method,
                                    seed=draw_seed(args.random_seed,si,accepted,mode,method,rep))
                                r=cap_total_shift(r,max_x=args.max_x_shift_m,
                                    max_y=args.max_y_shift_m,max_rms=args.max_rms_shift_m)
                                proposals.append((method,rep,r))
                        for method,rep,perturb in proposals:
                            try:
                                modified=core.evaluate_candidate(input0,perturb,args.role,mode,model,decode,scale)
                                scores=score_production(scorer,modified,features,context,args.role,mode)
                                rr=summarize_row(sname,accepted,env_seed,noise_seed,args.role,mode,method,rep,alpha,
                                        base,modified,base_score,scores,perturb,grad_rms,std,parity_err,delta)
                                for g,r in enumerate(rr):
                                    r['gradient_valid']=int(valid[g])
                                    if not r['reward_valid']:
                                        errors.append({'scenario':sname,'state_id':accepted,'mode':mode,
                                                       'amplitude_m':alpha,'method':method,'repeat':rep,
                                                       'group':g,'stage':'nonfinite_reward','message':'nonfinite production reward'})
                                output_rows.extend(rr)
                                if alpha in gallery_amp:
                                    state_gallery['traces'].append({'method':method,'repeat':rep,
                                        'unsafe':float(scores['unsafe'][gal_g]),
                                        'amplitude_m':alpha,'dense':dense_xy(modified[args.role,gal_g,mode],cfg),
                                        'sparse':modified[args.role,gal_g,mode].detach().cpu().numpy().astype(float),
                                        'reward':float(scores['reward'][gal_g]),'actual_rms_m':float(rr[gal_g]['actual_rms_shift_m'])})
                            except Exception as exc:
                                errors.append({'scenario':sname,'state_id':accepted,'mode':mode,
                                               'amplitude_m':alpha,'method':method,'repeat':rep,
                                               'stage':'candidate','message':repr(exc)})
                                # A scored proposal failure is retained as missing, never silently replaced by baseline.
                                print(f'[WARN] candidate scoring failed {sname} state={accepted} amp={alpha}'
                                      f' method={method} repeat={rep}: {type(exc).__name__}: {exc}',flush=True)
                    pool.append(state_gallery)
                accepted+=1
                print(f'[STATE] {sname} {accepted}/{args.samples_per_scenario} '
                      f'replay={replay_error:.3g} parity={parity_err:.3g}',flush=True)
            finally:env.close()
    # All states retain one reproducibly selected G-group's dense trajectory
    # for offline *reward-based* state selection in the independent plot.py.
    # Every sample row remains unselected and has the original V4 indicators.
    for r in output_rows:
        r['seed']=args.run_seed
    gallery_rows=[]
    for p in pool:
        identity=dict(seed=args.run_seed,scenario=p['scenario'],state_id=p['state_id'],
                      env_seed=p['env_seed'],mode=p['mode'],group=p['group'])
        baseline=dict(**identity,method='baseline',random_repeat=0,step_size_m=0.,
            guided_reward=p['baseline_reward'],guided_unsafe=p['baseline_unsafe'],
            baseline_reward=p['baseline_reward'],
            dense_xy_json=json.dumps(np.asarray(p['baseline_dense']).tolist(),separators=(',',':')),
            sparse_xy_json=json.dumps(np.asarray(p['baseline_sparse']).tolist(),separators=(',',':')),
            centerline_xy_json=json.dumps(np.asarray(p['centerline_xy']).tolist(),separators=(',',':')))
        gallery_rows.append(baseline)
        for tr in p['traces']:
            gallery_rows.append(dict(**identity,method=tr['method'],random_repeat=tr['repeat'],
                step_size_m=tr['amplitude_m'],guided_reward=tr['reward'],
                guided_unsafe=tr['unsafe'],baseline_reward=p['baseline_reward'],
                dense_xy_json=json.dumps(np.asarray(tr['dense']).tolist(),separators=(',',':')),
                sparse_xy_json=json.dumps(np.asarray(tr['sparse']).tolist(),separators=(',',':')),
                centerline_xy_json=''))
    core.write_csv(out/'trajectories.csv',gallery_rows)
    
    core.write_csv(out/'samples.csv',output_rows)
    if errors:
        core.write_csv(out/'failed_proposals.csv',errors)
    else:
        (out/'failed_proposals.csv').write_text('scenario,state_id,amplitude_m,method,repeat,stage,message\n',encoding='utf-8-sig')
    from .stats import statistics
    summary=statistics(output_rows,errors,config,names)
    summary.update(run_seed=args.run_seed,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        baseline_replay_max_error_m=replay_max,proxy_production_baseline_max_error=parity_max,
        note='UNSELECTED reward gradient vs raw random proposals. Production 41pt dense spline V5.2 reward. '
            'No RL training; no policy update, no MATP. Paired rows have same state, group, mode, DDIM noise. '
            'Random_sign exactly matches gradient coordinate magnitudes; random_gaussian approximately matches axis energies.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    # V5 plots are generated offline from the merged CSV by plot.py only.
    print('[COMPLETE]',out,'rows=',len(output_rows),'errors=',len(errors),'replay=',replay_max,'parity=',parity_max)

def main(argv=None):
    return run(cli().parse_args(argv))

if __name__=='__main__':
    main()
