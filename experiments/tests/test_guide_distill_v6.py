"""Contract smoke tests: python -m experiments.tests.test_guide_distill_v6."""
from __future__ import annotations
import torch
from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
from highway_env.planner.diffusion.trajectory_mode_reward.spline_dense import evaluate_dense_spline
from highway_env.planner.diffusion.trajectory_mode_reward.task_objective import (
    smooth_tier,longitudinal_terms,combine_reward)
from highway_env.planner.diffusion.guide_distill.matp_adapter import load_matp,project_w1
from highway_env.planner.diffusion.guide_distill.teacher_guidance import guide_teacher


def test():
    cfg=TrajectoryModeRewardConfig()
    times=torch.arange(1,9,dtype=torch.float64)*.5
    xy=torch.stack((times*25,torch.zeros_like(times)),dim=-1)[None].requires_grad_(True)
    velocity=torch.tensor([[25.,0.]],dtype=torch.float64)
    dense,v,a=evaluate_dense_spline(xy,cfg,initial_velocity_xy=velocity)
    assert dense.shape==(1,41,2)
    assert torch.allclose(v[:,0],velocity,atol=1e-8)
    terms=longitudinal_terms(dense,cfg,initial_velocity_xy=velocity)
    assert abs(float(terms['progress_score'].detach())-1.)<1e-8
    # Progress beyond historical 100 m must still produce a nonzero gradient.
    terminal=times*30
    z=torch.stack((terminal,torch.zeros_like(terminal)),dim=-1)[None].requires_grad_(True)
    dz=evaluate_dense_spline(z,cfg,initial_velocity_xy=velocity)[0]
    progress=longitudinal_terms(dz,cfg,initial_velocity_xy=velocity)['progress_score']
    grad=torch.autograd.grad(progress.sum(),z)[0]
    assert float(grad.abs().sum())>0.
    assert float(smooth_tier(torch.tensor([9.]),4.,6.))>float(smooth_tier(torch.tensor([5.]),4.,6.))
    p=load_matp()
    corrected,diag=project_w1(xy.detach(),module=p,reward_config=cfg,initial_velocity_xy=velocity)
    assert corrected.shape==xy.shape and torch.isfinite(corrected).all()
    assert diag['projector_variant']=='in_repo_matp_weighted_joint_v6'
    # Reward has ONE explicit shared combination for all three methods.
    r=combine_reward({'progress_score':terms['progress_score'],
        'accel_penalty':terms['accel_penalty'],'jerk_penalty':terms['jerk_penalty'],
        'road_boundary_reward':torch.zeros(1,dtype=torch.float64),
        'curvature_penalty':torch.zeros(1,dtype=torch.float64),
        'centerline_penalty':torch.zeros(1,dtype=torch.float64),
        'lateral_comfort_penalty':torch.zeros(1,dtype=torch.float64)},cfg)
    assert r.shape==(1,) and torch.isfinite(r).all()
    class Dummy:
        def gradient_of(self,x):return torch.zeros_like(x)
        def components(self,x):return {'balanced':x[...,0].mean(-1)}
    guided,_=guide_teacher(xy.detach(),Dummy(),strategy='a2_v2',iterations=2)
    assert guided.shape==xy.shape and torch.isfinite(guided).all()
    print('[PASS] V6 spline ego velocity, continuous progress, kinematic tiers, MATP in-tree, PGD guidance')
if __name__=='__main__':test()
