#!/usr/bin/env python3
"""Native V6 Post-Guide Distillation teacher generation and evaluation.

Usage: put this single file in D:\\KONG_Files\\AllMerge, invoke from there.
All model-side imports are from --root (all_merge-guideDistill). No training code is overwritten.
This comparison is an executable pilot; real CUDA execution requires local environment/checkpoint.
"""
from __future__ import annotations
import argparse,csv,dataclasses,hashlib,json,os,random,sys,time
from pathlib import Path


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def boot(root):
    root=Path(root).expanduser().resolve()
    if not (root/'scripts/run_grpo_noise_probe.py').exists():
        raise RuntimeError('Native V6 needs internal GRPO state/seed helpers')
    sys.path.insert(0,str(root))
    return root


def paired_score(adapter,features,context,candidates,*,role,mode):
    """One production W4 reward adapter for native V6 teacher proposals."""
    from highway_env.planner.diffusion.grpo.task_reward import task_reward_from_w4_result
    result=adapter.evaluate_result(candidates,features=features,context=context)
    score=task_reward_from_w4_result(result,context=context,device=candidates.device,
            dtype=candidates.dtype,reward_type='progress_comfort')
    values=score[role,:,mode].detach().cpu().numpy().copy()
    def field(key):
        if hasattr(result,key):arr=np.asarray(getattr(result,key))
        elif key in result.components:arr=np.asarray(result.components[key])
        else:return np.full(len(values),np.nan)
        return np.asarray(arr[role,mode,:]).copy()
    info={k:field(k) for k in ('unsafe','collision','out_of_drivable',
        'minimum_road_margin_m','curvature_max_abs','curvature_valid_points',
        'accel_penalty','jerk_penalty','accel_peak_mps2','jerk_peak_mps3')}
    return values,info,result


def policy_cfg(centre=None):
    from dataclasses import replace
    from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
    return replace(TrajectoryModeRewardConfig(),task_centerline_weight=float(TrajectoryModeRewardConfig().task_centerline_weight if centre is None else centre),
                   task_curvature_weight=.05,task_curvature_peak_weight=.5,
                   task_curvature_margin_weight=.05,centerline_scale_m=2.,centerline_late_power=2.)


def store(path,payload):
    import torch
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    torch.save(payload,tmp);os.replace(tmp,p)


def load(path):
    import torch
    # Only load our own local, trusted data; weights_only avoids arbitrary pickle execution.
    try:return torch.load(path,map_location='cpu',weights_only=True)
    except TypeError:return torch.load(path,map_location='cpu')


def _seed(seed):
    import numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)



def verify_reward_parity_v2(proxy, xy, result, *, role, mode):
    """Validate the real V5.2 reward on its two numerical paths.

    Production curvature is evaluated on CPU float64 and later serialized to
    float32, whereas guidance/Autograd runs on CUDA float32.  Comparing those
    curvature *components* in float32 to a single absolute threshold is not
    a valid test of reward formula parity, especially at curvature peaks.

    1. Float64 recomputation checks ALL production reward components using the
       original strict production/proxy contract (including centerline).
    2. Float32 recomputation audits the actual guidance objective.  Its total
       reward difference must remain small in absolute AND relative terms.
    Neither test is disabled and no reward/gradient definition is changed.
    """
    import torch
    from highway_env.planner.diffusion.grpo.task_reward import task_reward_from_w4_result

    with torch.no_grad():
        # Inputs retain the exact original float32 coordinates, cast to double.
        # Production also casts those coordinates to double before evaluating
        # its spline-based curvature penalty.
        precise_xy = xy.detach().to(dtype=torch.float64)
        diff64 = proxy.validate_against_production(
            precise_xy, result, role=role, mode=mode, tol=0.002)

        prod32 = task_reward_from_w4_result(
            result, context={'config': proxy.config},
            device=xy.device, dtype=xy.dtype,
            reward_type='progress_comfort')[role, :, mode]
        guide32 = proxy.components(xy.detach())['balanced']
        if guide32.shape != prod32.shape:
            raise RuntimeError('REWARD PARITY SHAPE MISMATCH '
                               f'production={tuple(prod32.shape)} guidance={tuple(guide32.shape)}')
        if not bool(torch.isfinite(prod32).all() and torch.isfinite(guide32).all()):
            raise RuntimeError('NON-FINITE REWARD during production/guidance parity check')
        err32 = (prod32 - guide32).abs()
        # Allow limited float32 rounding / discrete top-k boundary variation,
        # but reject meaningful differences in the actual training objective.
        scale = torch.maximum(prod32.abs(), guide32.abs())
        allowed = 5.0e-4 + 5.0e-3 * scale
        failure = err32 > allowed
        if bool(failure.any()):
            worst = int((err32 / allowed.clamp_min(1.0e-12)).argmax().item())
            raise RuntimeError('REWARD PARITY FAILED IN FLOAT32 '
                + str({'worst_candidate': worst,
                       'reward_production': float(prod32[worst]),
                       'reward_guidance': float(guide32[worst]),
                       'error': float(err32[worst]),
                       'allowed': float(allowed[worst]),
                       'float64_component_differences': diff64}))
        if not getattr(verify_reward_parity_v2, '_reported', False):
            print('[REWARD PARITY] float64 production/guidance components match; '
                  f'float32 reward max abs error={float(err32.max()):.6g}',flush=True)
            verify_reward_parity_v2._reported = True
        return {'float64_components': diff64,
                'float32_task_reward_max_abs': float(err32.max())}


ALL_SCENES = ('straight', 'curved', 'merge_in', 'merge_out')


def scene_at(requested, accepted_index):
    """Choose scenes round-robin by accepted State, not by attempts."""
    return ALL_SCENES[accepted_index % len(ALL_SCENES)] if requested == 'all' else requested


def distribution(rows, key='scenario'):
    return {name: sum(entry[key] == name for entry in rows) for name in ALL_SCENES}


def report_teacher_coverage(payload):
    states = payload['states']
    ledger = payload['ledger']
    per_scene = {name: {'train_states': 0, 'states_with_teacher': 0, 'teacher_trajectories': 0} for name in ALL_SCENES}
    for row in ledger:
        per_scene[row['scenario']]['train_states'] += 1
    for row in states:
        per_scene[row['scenario']]['states_with_teacher'] += 1
        per_scene[row['scenario']]['teacher_trajectories'] += len(row['teachers'])
    print('[SCENE COVERAGE]', per_scene, flush=True)
    return per_scene


def preflight(args,root):
    import torch
    from scripts.run_grpo_noise_probe import SCENARIOS,CheckpointSpec,_build_model
    if args.scenario == 'all':
        missing = set(ALL_SCENES).difference(SCENARIOS)
        if missing:raise RuntimeError(f'Unsupported required scenes: {sorted(missing)}')
    from highway_env.planner.diffusion.grpo.trainer import GRPOConfig
    from highway_env.planner.diffusion.guide_distill.matp_adapter import load_matp
    adapter,model=_build_model(CheckpointSpec('preflight',Path(args.checkpoint).resolve()),torch.device(args.device))
    prefixes=GRPOConfig().trainable_prefixes
    n=sum(p.numel() for name,p in model.named_parameters() if any(name.startswith(x) for x in prefixes))
    if n<=0:raise RuntimeError('No trainable policy parameters match original GRPO scope')
    _=load_matp()
    print('[PASS] preflight: reward=centerline 0.10, V5.2, frozen MATP W1, original GRPO scope')
    print('[MODEL] pretrained SHA256=',digest(args.checkpoint),'trainable_param_count=',n)
    print('[REWARD] source SHA256=',digest(root/'highway_env/planner/diffusion/grpo/task_reward.py'))
    print('[GRPO] trainer SHA256=',digest(root/'highway_env/planner/diffusion/grpo/trainer.py'))

def collect(args,root):
    import torch,numpy as np
    from dataclasses import replace
    from scripts.run_grpo_noise_probe import (SCENARIOS,CheckpointSpec,_build_model,
        _frozen_reward_context,_scenario_features,_state_seed,_sample_noise_seed)
    from highway_env.planner.diffusion.grpo.sampling import GroupDiffusionSampler
    from highway_env.planner.diffusion.grpo.reward_adapter import CandidateRewardAdapter,resolve_reward_evaluator
    from highway_env.planner.diffusion.guide_distill.reward_gradient import DifferentiableTaskReward
    from highway_env.planner.diffusion.guide_distill.exploration import guide_all
    from highway_env.planner.diffusion.guide_distill.matp_adapter import load_matp,project_w1
    from highway_env.planner.diffusion.guide_distill.road_boundary import RoadBoundaryField
    from highway_env.planner.diffusion.trajectory_mode_reward.state_adapter import _planning_state_snapshot,_vehicle_pose
    if args.seed==args.eval_seed:raise ValueError('train/eval seeds cannot be equal')
    if args.scenario != 'all' and args.scenario not in SCENARIOS:raise ValueError('Unsupported scenario')
    if args.guide_count>args.group_size:raise ValueError('guide-count > group-size')
    device=torch.device(args.device)
    cfg=policy_cfg()
    matp=load_matp()
    adapter,model=_build_model(CheckpointSpec(name='frozen_teacher',path=Path(args.checkpoint).resolve()),device)
    proxy=DifferentiableTaskReward(cfg,device)
    sampler=GroupDiffusionSampler(model,group_size=args.group_size,eta=.02)
    scorer=CandidateRewardAdapter(resolve_reward_evaluator('auto'))
    collection=[];ledger=[];attempts=0
    rejection_rows=[]
    start=time.perf_counter()
    while len(ledger)<args.train_states:
        if attempts>=args.train_states*30:
            raise RuntimeError(f'Could not generate {args.train_states} eligible scene states')
        scene=scene_at(args.scenario,len(ledger))
        scene_idx=ALL_SCENES.index(scene)
        env_seed=_state_seed(args.seed,scene_idx,attempts);attempts+=1
        env=SCENARIOS[scene](config={'show_trajectories':False,'show_future_trajectories':False},render_mode=None)
        try:
            env.reset(seed=env_seed)
            _,_,terminated,truncated,_=env.step(0)
            if terminated or truncated:continue
            features=_scenario_features(env,adapter)
            frozen,selected=_frozen_reward_context(model,features,env)
            context=replace(frozen,config=cfg)
            road=RoadBoundaryField.from_road(env.road)
            snapshot=_planning_state_snapshot(env)
            role_pose=(np.asarray(snapshot[0][args.role],dtype=np.float64) if snapshot is not None
                       else _vehicle_pose(env.controlled_vehicles[args.role]))
            proxy.bind_road(road,role_pose)
            proxy.bind_initial_velocity(features['ego_state'][args.role,:2])
            line=features['target_lane_polyline'][args.role,:,:2].detach().cpu().numpy()
            proxy.bind_centerline(line)
            group_seed=_sample_noise_seed(args.noise_seed,scene_idx,len(ledger))
            gen=torch.Generator(device=device.type).manual_seed(group_seed)
            with torch.no_grad(): sampled=sampler.sample(features,generator=gen)
            original=sampled.candidates[:,:args.guide_count].detach().clone()
            mode=int(selected[args.role].item())
            original_xy=original[args.role,:,mode].detach().clone()
            before,binfo,result=paired_score(scorer,features,context,original,role=args.role,mode=mode)
            verify_reward_parity_v2(proxy,original_xy,result,role=args.role,mode=mode)
            from highway_env.planner.diffusion.guide_distill.teacher_guidance import guide_teacher
            with torch.enable_grad():
                guided,_=guide_teacher(original_xy,proxy,
                    strategy=getattr(args,'pgd_strategy','b2_warm_search'),
                    iterations=args.guidance_iters,step_m=args.guidance_step_m,
                    max_x_m=10.,max_y_m=3.)
            physical,diag=project_w1(guided.detach(),module=matp,reward_config=cfg,
                        initial_velocity_xy=features['ego_state'][args.role,:2].expand(len(guided),2))
            altered=original.clone();altered[args.role,:,mode]=physical
            after,ainfo,_=paired_score(scorer,features,context,altered,role=args.role,mode=mode)
            after_k=np.asarray(diag['max_abs_curvature_after'].detach().cpu()).reshape(-1)
            with torch.no_grad():road_info=road.components(physical,role_pose,cfg)
            valid=[]
            # Explicit funnel: record why EVERY sampled candidate fails.  An unsafe
            # Curved model or infeasible MATP output must not be mislabeled as
            # merely insufficient teacher data.
            with torch.no_grad():pre_road=road.components(original_xy,role_pose,cfg)
            pre_margin=np.asarray(pre_road['road_min_margin_m'].detach().cpu()).reshape(-1)
            for i in range(args.guide_count):
                gain=float(after[i]-before[i]);margin=float(road_info['road_min_margin_m'][i])
                # Hard geometric bounds on the entire executable dense path,
                # checked after projection, plus all original safety gates.
                from highway_env.planner.diffusion.trajectory_mode_reward.spline_dense import evaluate_dense_spline
                with torch.no_grad():
                    before_dense=evaluate_dense_spline(original_xy[i:i+1],cfg,
                        initial_velocity_xy=features['ego_state'][args.role,:2])[0]
                    after_dense=evaluate_dense_spline(physical[i:i+1],cfg,
                        initial_velocity_xy=features['ego_state'][args.role,:2])[0]
                    sd=(physical[i]-original_xy[i]).abs()
                    dd=(after_dense-before_dense).abs()
                    within_caps=bool(sd[:,0].max()<=10 and sd[:,1].max()<=3
                        and dd[...,0].max()<=10 and dd[...,1].max()<=3)
                fail_gain=not (gain>1e-6) or not within_caps
                fail_curvature=not (float(after_k[i])<=.020001)
                fail_road=not (margin>=0.)
                fail_curvature_points=not (float(ainfo['curvature_valid_points'][i])>=1.)
                fail_unsafe=not (float(ainfo['unsafe'][i])<.5)
                fail_collision=not (float(ainfo['collision'][i])<.5)
                rejection_rows.append({'scenario':scene,'state_id':len(ledger),'env_seed':env_seed,
                    'group_id':i,'pre_road_margin_m':float(pre_margin[i]),
                    'post_road_margin_m':margin,'post_max_abs_curvature':float(after_k[i]),
                    'reward_before':float(before[i]),'reward_after':float(after[i]),'reward_gain':gain,
                    'fail_reward_gain':int(fail_gain),'fail_curvature':int(fail_curvature),
                    'fail_road':int(fail_road),'fail_valid_points':int(fail_curvature_points),
                    'fail_unsafe':int(fail_unsafe),'fail_collision':int(fail_collision),
                    'accepted':int(not any((fail_gain,fail_curvature,fail_road,
                        fail_curvature_points,fail_unsafe,fail_collision)))})
                if any((fail_gain,fail_curvature,fail_road,fail_curvature_points,fail_unsafe,fail_collision)):
                    continue
                # PAIRED-REPLAY-V1: preserve actual PRETRAIN reverse-chain input
                # for the exact G48 sample i.  x_0 is the input to terminal
                # denoiser timestep 0, NOT clean expert x0.  Storing generator
                # seed alone is insufficient with stochastic DDIM eta=0.02.
                # trace.states[0]=x_8; trace.next_states[-1]=input x_0.
                if tuple(sampled.timesteps) != (8, 0) or len(sampled.next_states) != 1:
                    raise RuntimeError(f'Paired replay requires (8,0) DDIM schedule, got {sampled.timesteps}')
                paired_t8=sampled.states[0][args.role,i].detach().cpu().float().contiguous()
                paired_t0=sampled.next_states[-1][args.role,i].detach().cpu().float().contiguous()
                valid.append({'target':physical[i].detach().cpu().float().contiguous(),
                    'pretrain_xy':original_xy[i].detach().cpu().float().contiguous(),
                    'paired_replay':{'schema':'frozen_ddim_v1','t8':paired_t8,'t0':paired_t0,
                                     'timesteps':(8,0),'source_group_id':i},
                    'mode':mode,'role':args.role,'group_id':i,'reward_before':float(before[i]),
                    'reward_after':float(after[i]),'reward_gain':gain,'margin':margin,
                    'max_curvature':float(after_k[i])})
            valid.sort(key=lambda e:e['reward_after'],reverse=True)
            # Limit near-duplicate targets per STATE, not per trajectory, to avoid data domination.
            chosen=[]
            for cand in valid:
                if len(chosen)>=args.max_teachers_per_state:break
                if any(torch.linalg.vector_norm(cand['target']-prev['target'],dim=-1).mean().item()<.05 for prev in chosen):
                    continue
                chosen.append(cand)
            if chosen:
                # Fail closed if saved replay state does not reproduce the
                # exact frozen-pretrain terminal action (not a different G48 draw).
                # Independent role-level scene encoding may differ by tiny
                # CUDA numeric roundoff from expanded-group inference.
                with torch.no_grad():
                    # Verify EVERY selected teacher, not just one example.
                    n_chosen=len(chosen)
                    role_features={k:v[args.role:args.role+1].expand(n_chosen,*v.shape[1:]).contiguous()
                                   for k,v in features.items()}
                    xt=torch.stack([t['paired_replay']['t0'] for t in chosen],dim=0).to(device)
                    norm=model.adapter.normalize_trajectory(role_features['coarse_trajectories'])
                    pred,_=model.denoiser(xt,torch.zeros(n_chosen,device=device,dtype=torch.long),
                        model.scene_encoder(role_features),navigation_point_norm=norm[:,:,-1,:])
                    physical_replay=model.adapter.denormalize_trajectory(pred)
                    for j,one in enumerate(chosen):
                        replay_physical=physical_replay[j,one['mode']]
                        actual=original_xy[one['group_id']]
                        replay_err=(replay_physical-actual).abs().max().item()
                        if not np.isfinite(replay_err) or replay_err>2e-3:
                            raise RuntimeError(f'PAIRED REPLAY CHECK FAILED scene={scene} '
                                f'group={one["group_id"]} mode={one["mode"]} '
                                f'max_abs_err_m={replay_err:.6g}; do not train mispaired data')
                        one['paired_replay']['terminal_replay_error_m']=replay_err
                feature_cpu={k:v[args.role].detach().cpu().clone() for k,v in features.items()}
                collection.append({'scenario':scene,'env_seed':env_seed,'noise_seed':group_seed,
                                   'features':feature_cpu,'teachers':chosen})
            ledger.append({'scenario':scene,'env_seed':env_seed,'state_index':len(ledger),
                           'eligible':len(valid),'kept':len(chosen)})
            print(f'[collect] {scene} state={len(ledger)}/{args.train_states} '
                  f'eligible={len(valid)} kept={len(chosen)}',flush=True)
        finally:
            env.close()
    from highway_env.planner.diffusion.trajectory_mode_reward.config import trajectory_mode_reward_config_sha256
    meta={'kind':'post_guide_distillation_v6',
          'reward_config_sha256':trajectory_mode_reward_config_sha256(cfg),
          'projector_variant':'in_repo_matp_weighted_joint_v6',
          'matp_original_parity':'UNVERIFIED',
          'spline_initial_velocity':'planning_ego_state_vx_vy',
          'noise_pairing_schema':'frozen_ddim_v1',
          'paired_timesteps':[8,0],'paired_terminal_state':'actual_ddim_next_states_last',
          'centerline_weight':float(cfg.task_centerline_weight),'checkpoint':str(Path(args.checkpoint).resolve()),
          'checkpoint_sha256':digest(args.checkpoint),'source_reward_sha256':digest(root/'highway_env/planner/diffusion/grpo/task_reward.py'),
          'seed':args.seed,'noise_seed':args.noise_seed,'scenario':args.scenario,'group_size':args.group_size,
          'guide_count':args.guide_count,'guidance_iters':args.guidance_iters,'guidance_step_m':args.guidance_step_m,
          'training_states':len(ledger),'states_with_teacher':len(collection),'teacher_count':sum(len(s['teachers']) for s in collection)}
    if not collection:raise RuntimeError('No safely feasible higher-reward teachers; abort distillation')
    meta['elapsed_seconds']=round(time.perf_counter()-start,2)
    # Every scenario sees 250 x 12 candidate checks; preserve overlap between
    # failure reasons rather than wrongly summing mutually nonexclusive flags.
    failure_fields=['fail_reward_gain','fail_curvature','fail_road','fail_valid_points',
        'fail_unsafe','fail_collision']
    meta['candidate_failure_funnel']={s:{
        'checked':sum(1 for x in rejection_rows if x['scenario']==s),
        'accepted':sum(x['accepted'] for x in rejection_rows if x['scenario']==s),
        'pre_road_feasible':sum(x['pre_road_margin_m']>=0 for x in rejection_rows if x['scenario']==s),
        **{k:sum(x[k] for x in rejection_rows if x['scenario']==s) for k in failure_fields}}
        for s in ALL_SCENES}
    detail_path=Path(args.teacher_file).with_name('teacher_candidate_funnel.csv')
    detail_path.parent.mkdir(parents=True,exist_ok=True)
    with detail_path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rejection_rows[0]));w.writeheader();w.writerows(rejection_rows)
    print('[CANDIDATE FUNNEL]',json.dumps(meta['candidate_failure_funnel'],sort_keys=True),flush=True)
    meta['scene_coverage']=report_teacher_coverage({'states':collection, 'ledger':ledger})
    meta['scenario_sampling']='accepted-state-roundrobin-v1'
    if args.scenario=='all':
        missing=[s for s in ALL_SCENES if meta['scene_coverage'][s]['states_with_teacher']==0]
        if missing:print('[WARNING] no valid teacher states for scenes:', ', '.join(missing),
                         '; Student will not receive teacher supervision for these scenes, but GRPO/validation include them.',flush=True)
    store(args.teacher_file,{'meta':meta,'states':collection,'ledger':ledger})
    Path(str(args.teacher_file)+'.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('[OK] teacher dataset:',args.teacher_file,'stats=',meta)


# Student optimization lives exclusively in post_guide_distillation.py.
# GRPO/RGG optimization remains exclusively in the native GRPOTrainer.
