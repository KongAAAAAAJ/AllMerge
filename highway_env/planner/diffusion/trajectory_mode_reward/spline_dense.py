"""Single differentiable clamped-cubic decoding contract for Reward, Guidance and MATP.

8 sparse control points (t=.5..4.0 s), 41 dense points at 10 Hz.
The initial derivative matches the frozen MATP W1 adapter for the probe.
"""
from __future__ import annotations
from functools import lru_cache
import torch
from highway_env.planner.diffusion.trajectory_spline import ClampedCubicTrajectorySpline

@lru_cache(maxsize=12)
def _decoder(sparse_dt:float, dense_dt:float, device:str):
    return ClampedCubicTrajectorySpline(horizon_s=8*sparse_dt,
             sparse_dt=sparse_dt,dense_dt=dense_dt).to(torch.device(device))

def evaluate_dense_spline(xy:torch.Tensor, config):
    if xy.shape[-2:] != (8,2):
        raise ValueError(f'Expected control points [...,8,2], received {xy.shape}')
    dt=float(config.trajectory_dt_s)
    original_shape=xy.shape[:-2]
    flat=xy.reshape(-1,8,2)
    origin=torch.zeros_like(flat[:,0])
    start=torch.stack((flat[:,0,0].clamp_min(0)/dt,
                       torch.zeros_like(flat[:,0,0])),dim=-1)
    end=(flat[:,-1]-flat[:,-2])/dt
    spline=_decoder(dt,float(config.interpolation_dt_s),str(flat.device))
    dense,v,a=spline.evaluate(flat,origin,start,end_velocity_xy=end)
    return tuple(t.reshape(*original_shape,t.shape[-2],2) for t in (dense,v,a))

def headings_from_dense(dense:torch.Tensor):
    """Matches numpy W4 forward-filled secant headings, safe at degenerate segments."""
    delta=torch.diff(dense,dim=-2)
    valid=torch.linalg.vector_norm(delta,dim=-1)>1e-6
    dx=torch.where(valid,delta[...,0],torch.ones_like(delta[...,0]))
    dy=torch.where(valid,delta[...,1],torch.zeros_like(delta[...,1]))
    headings=torch.atan2(dy,dx)
    T=headings.shape[-1]
    idx=torch.arange(T,device=dense.device).expand_as(headings)
    last=torch.cummax(torch.where(valid,idx,torch.zeros_like(idx)),dim=-1).values
    h=torch.gather(headings,-1,last)
    h=torch.where(torch.cummax(valid.long(),dim=-1).values.bool(),h,torch.zeros_like(h))
    # Initial heading follows the first dense segment, at t=0.
    return torch.cat((h[...,:1],h),dim=-1)

def task_comfort_from_dense(dense:torch.Tensor,config):
    """Keep original 0.5*mean(|a|)/8 + 0.5*mean(|yaw_rate|) scoring."""
    dt=float(config.interpolation_dt_s)
    v=torch.linalg.vector_norm(torch.diff(dense,dim=-2),dim=-1)/dt
    a=torch.diff(v,dim=-1,prepend=v[...,:1])/dt
    head=headings_from_dense(dense)[...,1:]
    delta=head[...,1:]-head[...,:-1]
    dh=torch.atan2(torch.sin(delta),torch.cos(delta))
    yaw=torch.cat((torch.zeros_like(head[...,:1]),dh),dim=-1)/dt
    return (.5*torch.mean(torch.abs(a),dim=-1)/8.
           +.5*torch.mean(torch.abs(yaw),dim=-1)).clamp(0.,1.)
