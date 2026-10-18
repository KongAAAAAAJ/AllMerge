"""Periodic current-Student teacher refresh with V6 Reward / MATP and paired-DDIM distillation.

Only the scheduler is new. Teacher collection, projection and the exact Student loss
are shared with offline Post-Guide Distillation. No independent optimizer or reward.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import time
from pathlib import Path

from . import teacher
from . import post_guide_distillation as distill


def refresh_points(total_steps: int, interval: int, start_step: int = 0):
    if total_steps <= 0 or interval <= 0 or not (0 <= start_step < total_steps):
        raise ValueError('Invalid Online PGD steps/refresh interval/start step')
    return list(range((start_step // interval) * interval, total_steps, interval))


def parameter_digest(model):
    return _model_sha(model)


def refresh_is_due(step, next_refresh_step, total_steps):
    return step < total_steps and step == next_refresh_step


def _teacher_path(policy_path, round_id, source_step):
    return Path(policy_path).parent/'teacher_bank'/f'round_{round_id:04d}_step_{source_step:06d}.pt'


def validate_teacher_bank(payload, *, source_step, source_sha256,
                          reward_config_sha256, scenario):
    meta=payload.get('meta',{})
    for key, expected in [('source_step',source_step),
                          ('source_model_sha256',source_sha256),
                          ('reward_config_sha256',reward_config_sha256),
                          ('scenario',scenario)]:
        if meta.get(key)!=expected:
            raise RuntimeError(f'Teacher bank {key} mismatch: expected={expected!r}, actual={meta.get(key)!r}')
    if (meta.get('noise_pairing_schema')!='frozen_ddim_v1' or
        meta.get('paired_timesteps')!=[8,0]):
        raise RuntimeError('Teacher bank paired replay schema mismatch')
    if not payload.get('states'):
        raise RuntimeError('Teacher bank has no valid teacher states')
    return True


def _model_sha(model):
    """Stable weights-only hash, independent of optimizer state and checkpoint format."""
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        v=tensor.detach().cpu().contiguous()
        h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def _fingerprint(args, root):
    return {
        'version':1,'method':'online_post_guide_distillation',
        'checkpoint_sha256':teacher.digest(args.checkpoint),
        'reward_sha256':teacher.digest(root/'highway_env/planner/diffusion/trajectory_mode_reward/task_objective.py'),
        'scenario':args.scenario,'seed':args.seed,'group_size':args.group_size,
        'lr':args.lr,'early_stage_weight':args.pgd_early_stage_weight,
        'refresh_interval':args.pgd_online_refresh_interval,
        'refresh_states':args.pgd_online_refresh_states,
        'guide_count':args.pgd_guide_count,'guide_iters':args.pgd_guide_iters,
        'guide_step_m':args.guide_step_m,'strategy':args.pgd_strategy,
    }


def run(args, root):
    import numpy as np
    import torch
    from argparse import Namespace
    root=Path(root).resolve()
    teacher.boot(root)
    if args.steps < 1 or args.pgd_online_refresh_interval < 1 or args.pgd_online_refresh_states < 1:
        raise ValueError('Online PGD requires positive steps, refresh interval and refresh states')
    if args.pgd_guide_count > args.group_size:
        raise ValueError('pgd-guide-count must be <= group-size')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for Online PGD training')
    device=torch.device('cuda')
    teacher._seed(args.seed)
    model,optimizer,params,model_config=distill.student_init(args,device)
    local_rng=np.random.default_rng(args.seed)
    output=Path(args.output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    recovery=output.with_name(output.stem+'_recovery.pt')
    history=output.with_suffix('.csv')
    bank_dir=output.parent/'teacher_bank'
    bank_dir.mkdir(parents=True,exist_ok=True)
    fingerprint=_fingerprint(args,root)
    start_step=0
    elapsed_prefix=0.0
    active_bank=None
    active_bank_sha=None
    next_refresh=0
    if args.pgd_resume:
        previous=teacher.load(Path(args.pgd_resume).resolve())
        if previous.get('online_pgd_version')!=1 or previous.get('fingerprint')!=fingerprint:
            raise RuntimeError('Online PGD resume fingerprint mismatch')
        start_step=int(previous['step'])
        if not (0 <= start_step < args.steps):
            raise ValueError('Resume step must be smaller than requested total steps')
        model.load_state_dict(previous['planner_state_dict'],strict=True)
        optimizer.load_state_dict(previous['optimizer_state_dict'])
        distill.rng_restore(previous['rng_state'],local_rng)
        elapsed_prefix=float(previous.get('elapsed_total_seconds',0))
        active_bank=Path(previous['active_teacher_bank']).resolve()
        active_bank_sha=previous['active_teacher_sha256']
        next_refresh=int(previous['next_refresh_step'])
        if not active_bank.is_file() or teacher.digest(active_bank)!=active_bank_sha:
            raise RuntimeError('Online PGD resume bank missing or SHA mismatch')
        if start_step > 0 and not history.exists():
            raise RuntimeError('Missing existing Online PGD CSV on resume')
    elif history.exists() or output.exists() or recovery.exists():
        raise FileExistsError('Existing Online PGD run; select new output or explicit --pgd-resume')

    columns=['step','round','teacher_bank_sha256','scenario','teacher_mode','teacher_group_id',
             'loss','reg_loss','terminal_reg_loss','early_reg_loss','cls_loss','grad_norm','elapsed_total_seconds']
    if not args.pgd_resume:
        with history.open('w',newline='',encoding='utf-8') as file:
            csv.DictWriter(file,fieldnames=columns).writeheader()
    started=time.perf_counter()
    def save(step, path):
        total_elapsed=elapsed_prefix+time.perf_counter()-started
        payload={'online_pgd_version':1,'method':'online_post_guide_distillation',
                 'planner_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                 'allmerge_model_config':dataclasses.asdict(model.config),
                 'step':step,'fingerprint':fingerprint,'rng_state':distill.rng_capture(local_rng),
                 'active_teacher_bank':str(active_bank),'active_teacher_sha256':active_bank_sha,
                 'next_refresh_step':next_refresh,'elapsed_total_seconds':total_elapsed}
        teacher.store(path,payload)

    bank_data=None
    for step in range(start_step,args.steps):
        if step >= next_refresh:
            round_id=step//args.pgd_online_refresh_interval
            bank_path=_teacher_path(output,round_id,step)
            if bank_path.exists():
                # Reusing an existing round without an explicit matching resume is unsafe.
                raise FileExistsError('Teacher bank already exists for new round: '+str(bank_path))
            source_sha=_model_sha(model)
            was_training=model.training
            model.eval()
            generator_args=Namespace(checkpoint=args.checkpoint,scenario=args.scenario,role=0,
                device='cuda',seed=args.seed+round_id*100003,
                noise_seed=args.seed+50000+round_id*100003,
                eval_seed=args.seed+100000+round_id*100003,
                group_size=args.group_size,guide_count=args.pgd_guide_count,
                guidance_iters=args.pgd_guide_iters,guidance_step_m=args.guide_step_m,
                max_teachers_per_state=4,train_states=args.pgd_online_refresh_states,
                teacher_file=bank_path,pgd_strategy=args.pgd_strategy)
            before_rng=distill.rng_capture(local_rng)
            try:
                with torch.no_grad():
                    teacher.preflight(generator_args,root) if step==0 else None
                teacher.collect(generator_args,root,source_model=model,
                                source_step=step,source_model_sha256=source_sha)
            finally:
                distill.rng_restore(before_rng,local_rng)
                model.train(was_training)
            bank_data=teacher.load(bank_path)
            if not bank_data['states']:
                raise RuntimeError('New teacher round has no eligible teacher states')
            from highway_env.planner.diffusion.trajectory_mode_reward.config import trajectory_mode_reward_config_sha256
            validate_teacher_bank(bank_data,source_step=step,source_sha256=source_sha,
                                  reward_config_sha256=trajectory_mode_reward_config_sha256(teacher.policy_cfg()),
                                  scenario=args.scenario)
            active_bank=bank_path
            active_bank_sha=teacher.digest(bank_path)
            next_refresh=step+args.pgd_online_refresh_interval
            # Recovery immediately after refresh makes interruption deterministic.
            save(step,recovery)
            print(f'[ONLINE-PGD] refresh step={step} bank={bank_path} '
                  f'teachers={len(bank_data["states"])}',flush=True)
        elif bank_data is None:
            bank_data=teacher.load(active_bank)
            if teacher.digest(active_bank)!=active_bank_sha:
                raise RuntimeError('Active teacher bank SHA changed')
        row=distill.student_train_step(model,optimizer,params,bank_data,local_rng,device,
             early_stage_weight=args.pgd_early_stage_weight,max_grad_norm=5.0)
        row.update(step=step+1,round=(step//args.pgd_online_refresh_interval),
                   teacher_bank_sha256=active_bank_sha,
                   elapsed_total_seconds=elapsed_prefix+time.perf_counter()-started)
        with history.open('a',newline='',encoding='utf-8') as file:
            csv.DictWriter(file,fieldnames=columns).writerow(row)
        if (step+1)%args.pgd_recovery_interval==0 or step+1==args.steps:
            save(step+1,recovery)
            if step+1==args.steps:
                save(step+1,output)
            print(f'[ONLINE-PGD] step={step+1}/{args.steps} '
                  f'loss={row["loss"]:.6f} bank={active_bank.name}',flush=True)
    print('[OK] Online PGD complete:',output,flush=True)
    return 0
