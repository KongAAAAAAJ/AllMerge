"""Trusted saved-teacher reading, metadata checks, and safe output helpers."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def load_teachers(path):
    import torch
    try:
        data=torch.load(path,map_location='cpu',weights_only=True)
    except TypeError as error:
        raise RuntimeError('A torch version with weights_only=True is required to safely read teachers.pt') from error
    if not isinstance(data,dict) or not isinstance(data.get('states'),list) or not isinstance(data.get('meta'),dict):
        raise RuntimeError('Unsupported teacher file: expected meta and states list')
    meta=data['meta']
    if meta.get('noise_pairing_schema')!='frozen_ddim_v1' or tuple(meta.get('paired_timesteps',()))!=(8,0):
        raise RuntimeError('Teacher bank must use frozen_ddim_v1 paired (8,0) replay')
    return data


def ensure_teacher_source(meta,pretrain_path):
    expected=meta.get('checkpoint_sha256')
    if not expected:
        raise RuntimeError('Missing checkpoint_sha256 in teacher metadata')
    actual=digest(pretrain_path)
    if actual!=expected:
        raise RuntimeError('Teacher bank checkpoint SHA does not match --pretrain-ckpt; cannot claim a paired comparison. '
                           f'expected={expected[:12]} actual={actual[:12]}')
    source=meta.get('teacher_source','frozen_pretrain')
    if source=='current_student' and int(meta.get('source_step',-1))!=0:
        raise RuntimeError('This teacher bank was produced by an intermediate Online PGD Student (teacher_source=current_student, '
                           f'source_step={meta.get("source_step")}). Its replay origin is not the initial Pretrain. '
                           'Use the initial frozen-Pretrain teacher bank for pretrain-versus-posttrain conclusions.')
    # Online round-0 teacher is initialized from pretrain but must still pass exact paired-t0 replay.
    return actual


def validate_reward_config(meta):
    from highway_env.planner.diffusion.guide_distill.teacher import policy_cfg
    from highway_env.planner.diffusion.trajectory_mode_reward.config import trajectory_mode_reward_config_sha256
    cfg=policy_cfg()
    actual=trajectory_mode_reward_config_sha256(cfg)
    if meta.get('reward_config_sha256')!=actual:
        raise RuntimeError('Teacher bank unified_v6 reward configuration hash mismatch; cannot compare reward reliably')
    return cfg


def rows_csv(path, rows, columns=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if columns is None:
        columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def json_out(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
