#!/usr/bin/env python3
"""Resume-safe, exact-step student and unchanged GRPOTrainer optimizer updates.

Run from D:\\KONG_Files\\AllMerge.  Only creates output artifacts; never edits worktrees.
Uses compare_unified_guidance_vs_grpo_20261012.py for state/features/reward.
"""
from __future__ import annotations
import argparse, csv, dataclasses, hashlib, json, os, random, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
from . import teacher as base


def sha(p):
    return base.digest(Path(p))


def rng_capture(local_rng=None):
    import torch, numpy as np
    result={'python':random.getstate(),'numpy_global':(lambda v:(v[0],v[1].tolist(),v[2],v[3],v[4]))(np.random.get_state()),
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    if local_rng is not None:result['numpy_local']=local_rng.bit_generator.state
    return result


def rng_restore(state,local_rng=None):
    import torch,numpy as np
    random.setstate(state['python']);
    ng=state['numpy_global'];np.random.set_state((ng[0],np.asarray(ng[1],dtype=np.uint32),ng[2],ng[3],ng[4]))
    torch.set_rng_state(state['torch_cpu'])
    if torch.cuda.is_available() and state.get('torch_cuda'):
        if len(state['torch_cuda'])!=torch.cuda.device_count():
            raise RuntimeError('CUDA topology changed on resume; cannot guarantee exact RNG continuity')
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    if local_rng is not None:local_rng.bit_generator.state=state['numpy_local']


def make_fingerprint(args,root,teachers):
    return {
        'pretrain_sha256':sha(args.checkpoint),'teacher_sha256':sha(args.teacher_file),
        'reward_sha256':sha(root/'highway_env/planner/diffusion/grpo/task_reward.py'),
        'grpo_trainer_sha256':sha(root/'highway_env/planner/diffusion/grpo/trainer.py'),
        'scenario':args.scenario,'role':args.role,'seed':args.seed,
        'noise_seed':args.noise_seed,'learning_rate':args.lr,'group_size':args.group_size,
        'max_grad_norm':5.0,'reward_kind':'unified_v6','mode_alignment':'teacher_mode_locked_v2','method':args.method,
        'model_impl_sha256':sha(root/'highway_env/planner/diffusion/structured_model.py'),
        'state_count':len(teachers['ledger']),
        'noise_pairing_schema':'frozen_ddim_v1','early_stage_weight':args.early_stage_weight}


def confirm_fingerprint(got,expected):
    for key,value in expected.items():
        if got.get(key)!=value:raise RuntimeError(f'Cannot resume: fingerprint {key} expected={value!r} found={got.get(key)!r}')


def build_model(args,device):
    import torch
    from highway_env.planner.diffusion.config import build_structured_diffusion_config
    from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
    from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter
    from pretraining.checkpoint_io import load_checkpoint_file
    state,config=load_checkpoint_file(args.checkpoint,map_location='cpu')
    config=build_structured_diffusion_config(**config)
    model=StructuredDiffusionPlanner(config,PlannerTensorAdapter(config,device)).to(device)
    model.load_state_dict(state,strict=True)
    return model,config


def student_init(args,device):
    import torch
    from highway_env.planner.diffusion.grpo.trainer import GRPOConfig
    model,config=build_model(args,device)
    model.requires_grad_(False)
    # Same parameter scope as the original GRPO, but Student must explicitly
    # re-enable autograd AFTER freezing the pretrained planner.
    prefixes = GRPOConfig().trainable_prefixes
    trainable = []
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad_(True)
            trainable.append(param)
    if not trainable:
        raise RuntimeError('No matched student params')
    if not all(param.requires_grad for param in trainable):
        raise RuntimeError('Student optimizer contains frozen parameters')
    opt=torch.optim.AdamW(trainable,lr=args.lr)
    return model,opt,trainable,config


def grpo_init(args,device):
    from highway_env.planner.diffusion.grpo.trainer import GRPOTrainer,GRPOConfig
    from highway_env.planner.diffusion.grpo.reward_adapter import CandidateRewardAdapter,resolve_reward_evaluator
    model,_=build_model(args,device)
    cfg=GRPOConfig(group_size=args.group_size,learning_rate=args.lr,eta=.02,clip_eps=.2,kl_coef=.01,
                   max_grad_norm=5.,update_epochs=2,task_reward_type='progress_comfort',constraint_strategy='none')
    trainer=GRPOTrainer(model,CandidateRewardAdapter(resolve_reward_evaluator('auto')),config=cfg)
    return trainer


def log_start(path,columns,start_step):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with path.open('r',newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
        if rows and any(int(row['step'])>start_step for row in rows):
            rows=[r for r in rows if int(r['step'])<=start_step]
            tmp=path.with_suffix('.truncate_tmp')
            with tmp.open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
            os.replace(tmp,path)
        last=int(rows[-1]['step']) if rows else 0
        if last>start_step:raise RuntimeError('Training log ahead of checkpoint')
        if rows and last<start_step:print(f'[WARN] log ends at {last}, checkpoint at {start_step}; logging resumes here',flush=True)
    else:
        with path.open('w',newline='',encoding='utf-8') as f:
            csv.DictWriter(f,fieldnames=columns).writeheader()


def paired_replay_loss(model, features, target, teacher_mode, replay, *, early_stage_weight=0.0):
    """Supervise EXACT G48/DDIM teacher sample's recorded denoiser inputs.

    The terminal x_t is the actual x at t=0 after stochastic 8->0 transition,
    NOT a fresh Gaussian perturbation of the teacher target or a seed replay.
    Identical SmoothL1 physical-space x0 regression + CE mode objective to
    original forward_train(dense_loss_enabled=False); optional t=8 regression
    ablation is disabled by default to isolate terminal noise pairing.
    """
    import torch
    import torch.nn.functional as F
    if replay.get('schema')!='frozen_ddim_v1' or tuple(replay.get('timesteps',()))!=(8,0):
        raise RuntimeError('Teacher missing correct frozen DDIM paired replay; recollect teachers')
    if int(replay.get('source_group_id',-1))<0:
        raise RuntimeError('Invalid paired replay source_group_id')
    for field in ('t0','t8'):
        if field not in replay or tuple(replay[field].shape)!=tuple(features['coarse_trajectories'].shape[1:]):
            raise RuntimeError(f'Paired replay {field} missing or wrong [M,T,2] shape')
    if not(0.0<=early_stage_weight<=1.0):
        raise ValueError('early_stage_weight must be in [0,1]')
    mode=torch.tensor([int(teacher_mode)],device=target.device,dtype=torch.long)
    if not(0<=teacher_mode<int(features['coarse_trajectories'].shape[1])):
        raise ValueError('Teacher mode out of range')
    scene=model.scene_encoder(features)
    anchor=model.adapter.normalize_trajectory(features['coarse_trajectories'])
    nav=anchor[:,:,-1,:]
    def decode(t, field):
        xt=replay[field].to(device=target.device,dtype=anchor.dtype).unsqueeze(0)
        if xt.shape!=anchor.shape:
            raise RuntimeError(f'Replay {field} shape does not match anchor {xt.shape} vs {anchor.shape}')
        if not torch.isfinite(xt).all():
            raise FloatingPointError(f'Nonfinite replay state {field}')
        pred,logits=model.denoiser(xt,torch.full((1,),t,device=xt.device,dtype=torch.long),
                                   scene,navigation_point_norm=nav)
        physical=model.adapter.denormalize_trajectory(pred)
        return physical[:,int(teacher_mode)],logits
    prediction,logits=decode(0,'t0')
    terminal=F.smooth_l1_loss(prediction,target)
    classification=F.cross_entropy(logits,mode)
    early=terminal.new_zeros(())
    if early_stage_weight>0:
        early_prediction,_=decode(8,'t8')
        early=F.smooth_l1_loss(early_prediction,target)
    loss=terminal+classification+float(early_stage_weight)*early
    return {'loss':loss,'trajectory_regression_loss':terminal+float(early_stage_weight)*early,
            'trajectory_classification_loss':classification,
            'terminal_regression_loss':terminal,'early_regression_loss':early,'target_mode':mode}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=('student','grpo'),required=True)
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[4]))
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--teacher-file',required=True)
    p.add_argument('--scenario',default='merge_in')
    p.add_argument('--role',type=int,default=0)
    p.add_argument('--seed',type=int,default=20261020)
    p.add_argument('--noise-seed',type=int,default=70261020)
    p.add_argument('--group-size',type=int,default=48)
    p.add_argument('--lr',type=float,default=1e-6)
    p.add_argument('--max-grad-norm',type=float,default=5.0)
    p.add_argument('--early-stage-weight',type=float,default=0.0,help='0: terminal x_t0 replay only; optional t8 loss weight')
    p.add_argument('--start-step',type=int,required=True)
    p.add_argument('--stop-step',type=int,required=True)
    p.add_argument('--resume',help='Last completed step checkpoint, or omit for zero start')
    p.add_argument('--output',required=True)
    p.add_argument('--log',required=True)
    p.add_argument('--recovery',required=True)
    p.add_argument('--recovery-interval',type=int,default=250)
    a=p.parse_args()
    if not(0<=a.start_step<a.stop_step):raise ValueError('Need 0 <= start < stop')
    if a.start_step and not a.resume:raise ValueError('Nonzero start requires --resume')
    if not a.start_step and a.resume:raise ValueError('Zero start forbids --resume')
    if a.max_grad_norm!=5.0:raise ValueError('Original paired GRPO uses max_grad_norm=5.0')
    a.checkpoint=Path(a.checkpoint).resolve();a.teacher_file=Path(a.teacher_file).resolve()
    a.output=Path(a.output).resolve();a.log=Path(a.log).resolve();a.recovery=Path(a.recovery).resolve()
    root=base.boot(a.root)
    import torch,numpy as np
    from dataclasses import replace
    from scripts.run_grpo_noise_probe import (SCENARIOS,CheckpointSpec,_build_model,_scenario_features,_frozen_reward_context)
    from highway_env.planner.diffusion.grpo.trainer import GRPOConfig
    from train_grpo import _frozen_reward_context as grpo_context
    if a.scenario!='all' and a.scenario not in SCENARIOS:raise ValueError('Unsupported scenario '+a.scenario)
    teachers=base.load(a.teacher_file)
    meta=teachers['meta']
    from highway_env.planner.diffusion.trajectory_mode_reward.config import trajectory_mode_reward_config_sha256
    if meta.get('reward_config_sha256')!=trajectory_mode_reward_config_sha256(base.policy_cfg()):
        raise RuntimeError('Teacher V6 Reward config SHA mismatch: regenerate teachers')
    if meta.get('checkpoint_sha256')!=sha(a.checkpoint) or meta.get('centerline_weight')!=float(base.policy_cfg().task_centerline_weight):
        raise RuntimeError('V6 teacher/pretrain mismatch')
    if meta.get('scenario')!=a.scenario:raise RuntimeError('Teacher scenario differs')
    if a.scenario=='all' and meta.get('scenario_sampling')!='accepted-state-roundrobin-v1':
        raise RuntimeError('Multi-scene teacher dataset schema mismatch')
    if meta.get('noise_pairing_schema')!='frozen_ddim_v1' or meta.get('paired_timesteps')!=[8,0]:
        raise RuntimeError('Old teachers.pt cannot be used: paired x_t8/x_t0 states are missing; recollect into a new output dir')
    if len(teachers['ledger'])==0:raise RuntimeError('Empty teacher state ledger')
    fp=make_fingerprint(a,root,teachers)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':raise RuntimeError('CUDA expected for 5000/100000-step real experiment')
    base._seed(a.seed)
    local_rng=None
    if a.method=='student':
        model,opt,params,config=student_init(a,device)
        local_rng=np.random.default_rng(a.seed)
        if not teachers['states']:raise RuntimeError('No filtered teachers')
        columns=['step','scenario','teacher_mode','teacher_group_id','loss',
                 'reg_loss','terminal_reg_loss','early_reg_loss','cls_loss',
                 'grad_norm','elapsed_total_seconds']
    else:
        trainer=grpo_init(a,device)
        model=trainer.model;opt=trainer.optimizer
        columns=['step','scenario','env_seed','reward_mean','reference_kl','elapsed_total_seconds','metrics_json']
    elapsed_prefix=0.0
    if a.resume:
        state=base.load(a.resume)
        if state.get('continuous_long_version')!=1 or state.get('method')!=a.method:
            raise RuntimeError('Refusing to resume from old checkpoint without RNG+optimizer guarantee')
        if int(state['step'])!=a.start_step:
            raise RuntimeError(f'Resume step {state["step"]} != requested {a.start_step}')
        confirm_fingerprint(state['fingerprint'],fp)
        model.load_state_dict(state['planner_state_dict'],strict=True)
        opt.load_state_dict(state['optimizer_state_dict'])
        rng_restore(state['rng_state'],local_rng)
        elapsed_prefix=float(state.get('elapsed_total_seconds',0))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    log_start(a.log,columns,a.start_step)
    started=time.perf_counter()
    print(f'[CONTINUOUS] method={a.method} start={a.start_step} stop={a.stop_step} '
          f'checkpoint={a.resume} teacher_sha={fp["teacher_sha256"][:12]}',flush=True)
    for step in range(a.start_step+1,a.stop_step+1):
        if a.method=='student':
            entry=teachers['states'][int(local_rng.integers(0,len(teachers['states'])))]
            teacher=entry['teachers'][int(local_rng.integers(0,len(entry['teachers'])))]
            target=teacher['target'].to(device)[None]
            features={k:v[None].to(device) for k,v in entry['features'].items()}
            # The exact reference-selected mode that Guidance optimized is the
            # regression AND classification target.  Do not silently reassign
            # this teacher to its nearest coarse anchor.
            teacher_mode=int(teacher['mode'])
            mode_override=torch.tensor([teacher_mode],device=device,dtype=torch.long)
            model.train();opt.zero_grad(set_to_none=True)
            replay=teacher.get('paired_replay')
            if not isinstance(replay,dict) or replay.get('source_group_id')!=teacher['group_id']:
                raise RuntimeError('Teacher/replay group ID mismatch; refusing unpaired student update')
            loss_data=paired_replay_loss(model,features,target,teacher_mode,replay,
                                         early_stage_weight=a.early_stage_weight)
            actual=int(loss_data['target_mode'].reshape(-1)[0].detach().item())
            if actual!=teacher_mode:
                raise RuntimeError(f'Teacher-mode lock failed: expected {teacher_mode}, actual {actual}')
            loss=loss_data['loss']
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite student loss at step {step}')
            if not loss.requires_grad:
                raise RuntimeError(f'Student loss has no autograd graph at step {step}; check trainable scope / no_grad / detach')
            loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(params,a.max_grad_norm))
            if not np.isfinite(grad):raise FloatingPointError(f'Nonfinite student grad at step {step}')
            opt.step()
            row={'step':step,'scenario':entry['scenario'],'teacher_mode':teacher_mode,
                 'teacher_group_id':int(teacher['group_id']),'loss':float(loss.detach()),
                 'reg_loss':float(loss_data['trajectory_regression_loss'].detach()),
                 'terminal_reg_loss':float(loss_data['terminal_regression_loss'].detach()),
                 'early_reg_loss':float(loss_data['early_regression_loss'].detach()),
                 'cls_loss':float(loss_data['trajectory_classification_loss'].detach()),
                 'grad_norm':grad}
        else:
            state_info=teachers['ledger'][(step-1)%len(teachers['ledger'])]
            env=SCENARIOS[state_info['scenario']](config={'show_trajectories':False,'show_future_trajectories':False},render_mode=None)
            try:
                env.reset(seed=int(state_info['env_seed']))
                _,_,terminated,truncated,_=env.step(0)
                if terminated or truncated:raise RuntimeError('Saved training state now invalid')
                features=_scenario_features(env,trainer.model.adapter)
                frozen=grpo_context(trainer,features,env)
                context=replace(frozen,config=base.policy_cfg())
                generator=torch.Generator(device=device.type).manual_seed(a.noise_seed+step)
                metrics=trainer.train_step(features,context=context,generator=generator,paired_validation=False)
            finally:env.close()
            metrics={k:float(v) for k,v in metrics.items() if isinstance(v,(int,float))}
            if not np.isfinite(metrics.get('reward_mean',float('nan'))):
                raise FloatingPointError(f'Nonfinite GRPO reward at step {step}: {metrics}')
            row={'step':step,'scenario':state_info['scenario'],'env_seed':int(state_info['env_seed']),
                 'reward_mean':metrics.get('reward_mean',float('nan')),
                 'reference_kl':metrics.get('reference_kl',metrics.get('kl',float('nan'))),
                 'metrics_json':json.dumps(metrics,sort_keys=True)}
        elapsed_total=elapsed_prefix+time.perf_counter()-started
        row['elapsed_total_seconds']=round(elapsed_total,2)
        with a.log.open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writerow(row)
        if step%a.recovery_interval==0 or step==a.stop_step:
            payload={'continuous_long_version':1,'method':a.method,
                     'planner_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),
                     'allmerge_model_config':dataclasses.asdict(model.config),
                     'step':step,'fingerprint':fp,'rng_state':rng_capture(local_rng),
                     'elapsed_total_seconds':elapsed_total}
            base.store(a.recovery,payload)
            if step==a.stop_step:base.store(a.output,payload)
            print(f'[{a.method}] step={step} elapsed={elapsed_total:.1f}s '
                  f'reward={row.get("reward_mean",float("nan")):.5f} '
                  f'loss={row.get("loss",float("nan")):.6f}',flush=True)
    print(f'[PASS] {a.method} continuous checkpoint={a.output}',flush=True)

if __name__=='__main__':main()
