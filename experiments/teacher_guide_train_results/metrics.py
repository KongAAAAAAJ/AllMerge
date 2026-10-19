"""Reward/shape/safety metrics; no claims of full closed-loop safety."""
from __future__ import annotations
import numpy as np


def _rms(d):
    return float(np.sqrt(np.mean(np.sum(d*d,axis=-1))))


def metric_row(pre,post,teacher,score,info):
    pre=np.asarray(pre,dtype=np.float64);post=np.asarray(post,dtype=np.float64)
    teacher=np.asarray(teacher,dtype=np.float64)
    if any(p.shape!=(41,2) or not np.isfinite(p).all() for p in (pre,post,teacher)):
        raise RuntimeError('Expected three finite dense trajectories [41,2]')
    d=post-pre
    lens=np.linalg.norm(d,axis=-1)
    ix=int(np.argmax(lens))
    reward_pre,reward_post,reward_teacher=(float(score[k]) for k in ('pre','post','teacher'))
    a=_rms(pre-teacher);b=_rms(post-teacher)
    row={**info,
         'pretrain_reward':reward_pre,'posttrain_reward':reward_post,'teacher_reward':reward_teacher,
         'train_reward_gain':reward_post-reward_pre,
         'teacher_reward_gap_pre':reward_teacher-reward_pre,
         'teacher_reward_gap_post':reward_teacher-reward_post,
         'pre_teacher_dense_rms_m':a,'post_teacher_dense_rms_m':b,
         'teacher_distance_improve_m':a-b,
         'pre_post_dense_rms_m':_rms(d),
         'max_dense_shift_m':float(lens[ix]),'max_dense_shift_index':ix,
         'max_lateral_shift_m':float(np.abs(d[:,1]).max()),
         'dense_improvement_fraction':float(np.mean(np.linalg.norm(post-teacher,axis=-1)<np.linalg.norm(pre-teacher,axis=-1))),
        }
    return row


def attach_safety(row,safety):
    for which in ('pre','post','teacher'):
        item=safety.get(which,{})
        for key in ('unsafe','collision','out_of_drivable','minimum_road_margin_m','minimum_ttc_s',
                    'minimum_teammate_gap_m','minimum_background_gap_m','curvature_max_abs'):
            v=item.get(key)
            if v is not None and np.isfinite(float(v)):
                row[f'{which}_{key}']=float(v)
        if all(k in item and np.isfinite(float(item[k])) for k in ('unsafe','collision','out_of_drivable')):
            row[f'{which}_basic_safety_pass']=int(all(float(item[k])<.5 for k in ('unsafe','collision','out_of_drivable')))
    return row
