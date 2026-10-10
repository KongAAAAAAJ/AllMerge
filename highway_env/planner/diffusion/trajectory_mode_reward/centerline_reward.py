"""Differentiable target-lane-centerline penalty on the exact 41-point 10 Hz spline.

Reference MUST be the frozen Planner Feature `target_lane_polyline`, not the
nearest road lane. This preserves lane-change/merge intent.
"""
from __future__ import annotations
import torch


def centerline_reward_components(dense_xy: torch.Tensor, target_line, cfg):
    if dense_xy.ndim != 3 or dense_xy.shape[1:] != (41, 2):
        raise ValueError('Expected [N,41,2] production dense trajectory')
    n = dense_xy.shape[0]
    zero = dense_xy.new_zeros(n)
    if target_line is None:
        return {'centerline_penalty': zero,
                'centerline_mean_error_m':zero,
                'centerline_late_error_m':zero,
                'centerline_valid':zero}
    line=torch.as_tensor(target_line,device=dense_xy.device,dtype=dense_xy.dtype)
    if line.ndim != 2 or line.shape[1] != 2 or line.shape[0] < 2 or not bool(torch.isfinite(line).all()):
        raise ValueError('Invalid frozen target-lane polyline; expected finite [M,2]')
    start, end = line[:-1],line[1:]
    vec=end-start
    lengths2=vec.square().sum(-1)
    seg_valid=lengths2>1e-8
    if not bool(seg_valid.any()):
        raise ValueError('Target lane polyline has no nondegenerate segments')
    # Every dense future position has a distance to each *target lane*
    # segment, not to any nearby lane. Gradients flow to dense positions.
    pts=dense_xy[:,1:,:] # 0.1..4.0 s
    rel=pts[:,:,None,:]-start[None,None,:,:]
    frac=(rel*vec[None,None,:,:]).sum(-1)/lengths2.clamp_min(1e-8)
    foot=start[None,None,:,:]+frac.clamp(0.,1.)[...,None]*vec[None,None,:,:]
    sq=(pts[:,:,None,:]-foot).square().sum(-1)
    sq=torch.where(seg_valid[None,None,:],sq,torch.full_like(sq,float('inf')))
    min_sq=sq.amin(-1)
    # Stable at exactly zero error, without a derivative singularity.
    eps=1e-4
    distance=torch.sqrt(min_sq+eps*eps)-eps
    # No longitudinal pull toward a polyline endpoint after the frozen map
    # reference runs out. Exclude out-of-coverage points from the penalty.
    first_v=vec[seg_valid][0];last_v=vec[seg_valid][-1]
    first_pt=start[seg_valid][0];last_pt=end[seg_valid][-1]
    before=((pts-first_pt)*first_v).sum(-1)<-first_v.square().sum()
    after=((pts-last_pt)*last_v).sum(-1)>last_v.square().sum()
    in_range=~(before|after)
    # Explicit tail priority: t=4s has greater influence than t=0.1s.
    t=torch.linspace(.025,1.,40,device=dense_xy.device,dtype=dense_xy.dtype)
    w=(.05+t.pow(float(cfg.centerline_late_power)))[None,:]*in_range.to(dense_xy.dtype)
    err=distance/float(cfg.centerline_scale_m)
    huber=torch.where(err<1.,.5*err.square(),err-.5)
    penalty=(w*huber).sum(-1)/w.sum(-1).clamp_min(1e-9)
    late_mask=in_range[:,20:]
    late_error=(distance[:,20:]*late_mask).sum(-1)/late_mask.sum(-1).clamp_min(1)
    mean_error=(distance*in_range).sum(-1)/in_range.sum(-1).clamp_min(1)
    valid=(in_range[:,20:].to(dense_xy.dtype).mean(dim=-1)>=.5).to(dense_xy.dtype)
    return {'centerline_penalty': penalty,
            'centerline_mean_error_m':mean_error,
            'centerline_late_error_m':late_error,
            'centerline_valid':valid}
