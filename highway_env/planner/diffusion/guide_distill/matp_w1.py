"""Standalone MATP-inspired multi-active violation-weighted projection.

This source-only port uses the frozen W1 parameters and multi-active weighted
local projection described by the MATP V4.3 patch.  The exact V4.3 full
`guidance.py` was NOT in the provided archive. Do not claim byte-for-byte or
numerical parity with that unavailable implementation. A parity audit against
original W1 is required before treating this as a reproduction benchmark.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from highway_env.planner.diffusion.trajectory_mode_reward.spline_dense import evaluate_dense_spline

@dataclass(frozen=True)
class MatpW1Config:
    curvature_limit: float=.02
    active_threshold: float=.015
    tightening: float=.0005
    iters: int=6
    max_active: int=8
    regularization: float=1.e-4
    step_max_m: float=.15
    max_trust_m: float=1.
    initial_trust_m: float=.30
    beta: float=2.
    tolerance: float=1.e-5
    preserve_endpoint: bool=True


def curvature_series(xy:torch.Tensor, config, initial_velocity_xy=None):
    dense,_,_=evaluate_dense_spline(xy,config,initial_velocity_xy=initial_velocity_xy)
    step=dense[1:]-dense[:-1]
    heading=torch.atan2(step[:,1],step[:,0])
    dh=torch.atan2(torch.sin(heading[1:]-heading[:-1]),torch.cos(heading[1:]-heading[:-1]))
    scale=.5*(torch.linalg.vector_norm(step[:-1],dim=-1)+torch.linalg.vector_norm(step[1:],dim=-1)).clamp_min(.25)
    return dh/scale


def project_w1_local(xy:torch.Tensor, reward_config, *, initial_velocity_xy=None,
                      projector_config:MatpW1Config=MatpW1Config()):
    if xy.ndim!=3 or xy.shape[-2:]!=(8,2):
        raise ValueError('MATP needs [N,8,2]')
    if not torch.isfinite(xy).all():raise ValueError('Nonfinite input')
    cfg=projector_config
    anchor=xy.detach();out=anchor.clone(); peaks=[];accepted=[]
    trust=cfg.initial_trust_m
    for i in range(len(out)):
        x=out[i].detach().clone(); v0=(None if initial_velocity_xy is None else initial_velocity_xy[i])
        current=float(curvature_series(x,reward_config,v0).abs().max())
        best=x.clone();best_peak=current
        for iteration in range(cfg.iters):
            with torch.enable_grad():
                z=x.detach().clone().requires_grad_(True)
                curv=curvature_series(z,reward_config,v0)
                violation=curv.abs()-(cfg.curvature_limit-cfg.tightening)
                ids=torch.topk(violation,k=min(cfg.max_active,len(violation))).indices
                active=ids[violation[ids]> -max(0.,cfg.curvature_limit-cfg.active_threshold)]
                if len(active)==0:break
                vals=curv[active]
                signs=vals.detach().sign().clamp(min=-1.,max=1.)
                rhs=(vals.abs()-(cfg.curvature_limit-cfg.tightening)).clamp_min(0.).detach()
                if not bool((rhs>cfg.tolerance).any()):break
                jac=torch.stack([torch.autograd.grad(curv[j],z,retain_graph=True)[0].reshape(-1) for j in active])
            vmax=rhs.max().clamp_min(1.e-8)
            weights=(1.+cfg.beta*rhs/vmax).sqrt()
            J=signs[:,None]*jac*weights[:,None]
            b=rhs*weights
            # Preserve the end point, applying Jacobian only to the first 7 sparse points.
            if cfg.preserve_endpoint:J=J[:,:-2]
            reg=torch.eye(len(rhs),device=x.device,dtype=x.dtype)*cfg.regularization
            correction=-(J.T@torch.linalg.solve(J@J.T+reg,b))
            d=torch.zeros(16,device=x.device,dtype=x.dtype)
            d[:len(correction)]=correction
            d=d.reshape(8,2)
            maxmag=torch.linalg.vector_norm(d.reshape(-1)).clamp_min(1e-9)
            d=d*min(1.,cfg.step_max_m/float(maxmag))
            candidate=x+d
            distance=(candidate-anchor[i]).flatten().norm().clamp_min(1.e-9)
            candidate=anchor[i]+(candidate-anchor[i])*min(1.,trust/float(distance))
            if cfg.preserve_endpoint:candidate[-1]=anchor[i,-1]
            new_peak=float(curvature_series(candidate,reward_config,v0).abs().max())
            if new_peak+1e-8<best_peak:
                x=candidate.detach();best=x.clone();best_peak=new_peak
            else:break
        out[i]=best;peaks.append(best_peak);accepted.append(best_peak<=cfg.curvature_limit+cfg.tolerance)
    diag={'max_abs_curvature_after':torch.tensor(peaks,device=xy.device,dtype=xy.dtype),
          'feasible':torch.tensor(accepted,device=xy.device,dtype=torch.bool),
          'projector_variant':'in_repo_matp_weighted_joint_v6',
          'parity_with_frozen_v43_w1':'UNVERIFIED_original_source_unavailable'}
    return out,diag
