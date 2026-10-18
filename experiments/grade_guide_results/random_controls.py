"""Paired random mean-displacement controls; physical XY units only."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def random_like(delta:torch.Tensor, *, method:str, seed:int) -> torch.Tensor:
    """delta [G,8,2]. sign: exact coordinate absolute values; gaussian: axis-energy matched.

    RNG is local; it never advances the diffusion sampler's random state.
    """
    gen=torch.Generator(device=delta.device.type)
    gen.manual_seed(int(seed))
    if method == 'random_sign':
        signs=torch.randint(0,2,delta.shape,generator=gen,device=delta.device,dtype=torch.long)
        return delta*(signs.to(delta.dtype)*2-1)
    if method != 'random_gaussian':
        raise ValueError('Unknown control '+method)
    noise=torch.randn(delta.shape,generator=gen,device=delta.device,dtype=delta.dtype)
    # Smooth in waypoint time to avoid artificially oscillatory 10 Hz trajectories.
    g,t,_=noise.shape
    n=noise.permute(0,2,1).reshape(g*2,1,t)
    kernel=torch.tensor([.25,.5,.25],device=delta.device,dtype=delta.dtype).reshape(1,1,3)
    n=F.conv1d(F.pad(n,(1,1),mode='replicate'),kernel).reshape(g,2,t).permute(0,2,1)
    target_axis=torch.linalg.vector_norm(delta,dim=1,keepdim=True)
    actual_axis=torch.linalg.vector_norm(n,dim=1,keepdim=True).clamp_min(1e-12)
    return n*(target_axis/actual_axis)


def input_match_error(target:torch.Tensor, control:torch.Tensor):
    """RMS relative mismatch, one per group. Sign matched => precisely zero."""
    def rms(x):return torch.sqrt(x.square().sum(-1).mean(-1))
    return (rms(control)-rms(target)).abs()/rms(target).clamp_min(1e-10)
