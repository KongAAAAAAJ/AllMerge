"""Pure math: normalize a *physical mean-coordinate* reward gradient; radius constraints.

The projected step is tested in physical metres, then divided by the actual
trajectory normalization scale before being added to DDIM mean / its sampled x0 input.
"""
from __future__ import annotations
import torch


def metric_gradient(grad_norm: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """dR/d(mean_physical) given dR/d(mean_normalized)."""
    return grad_norm / scale


def normalized_direction(grad_phys: torch.Tensor, *, eps: float = 1e-12):
    """Each candidate: unit RMS over 8 two-dimensional displacement vectors."""
    rms = torch.sqrt(grad_phys.square().sum(-1).mean(-1))
    good = torch.isfinite(rms) & (rms > eps)
    unit = torch.where(good[:,None,None], grad_phys / rms.clamp_min(eps)[:,None,None], torch.zeros_like(grad_phys))
    return unit, rms, good


def cap_total_shift(delta: torch.Tensor, *, max_x: float, max_y: float, max_rms: float):
    """Clip the *cumulative* mean shift using global per-axis and RMS bounds.

    Same multiplicative scalar applied across all sparse waypoints of each group:
    direction remains aligned with the gradient unless a group reaches the trust limit.
    """
    if min(max_x,max_y,max_rms) <= 0: raise ValueError('Caps must be >0')
    x = delta[...,0].abs().amax(dim=-1).clamp_min(1e-12)
    y = delta[...,1].abs().amax(dim=-1).clamp_min(1e-12)
    rms = torch.sqrt(delta.square().sum(-1).mean(-1)).clamp_min(1e-12)
    ratio=torch.minimum(torch.ones_like(x),torch.minimum(
        torch.minimum(torch.as_tensor(max_x,device=delta.device,dtype=delta.dtype)/x,
                      torch.as_tensor(max_y,device=delta.device,dtype=delta.dtype)/y),
        torch.as_tensor(max_rms,device=delta.device,dtype=delta.dtype)/rms))
    return delta * ratio[:,None,None]


def geometry_metrics(base, guided):
    diff=guided-base
    disp=torch.linalg.vector_norm(diff,dim=-1)
    relative=diff-diff.mean(dim=-2,keepdim=True)
    return {
        'sparse_rms_m': torch.sqrt(disp.square().mean(-1)),
        'sparse_max_m': disp.amax(-1),
        'nonrigid_rms_m': torch.sqrt(relative.square().sum(-1).mean(-1)),
        'end_dx_m': diff[:,-1,0], 'end_dy_m':diff[:,-1,1],
    }
