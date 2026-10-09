"""V2.1 nonfinite-gradient regression, in local offpolicyEnv."""
from __future__ import annotations
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from highway_env.planner.diffusion.guide_distill.road_boundary import RoadBoundaryField
from highway_env.planner.diffusion.guide_distill.reward_gradient import DifferentiableTaskReward, DIRECTIONS
from highway_env.planner.diffusion.guide_distill.exploration import guide_all
from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig

class MockLane:
    length=90.
    def __init__(self,y):self.y=y
    def width_at(self,s):return 4.
    def position(self,s,l):return np.asarray([s,self.y+l])
class MockRoad:
    def __init__(self):self.network=SimpleNamespace(graph={'a':{'b':[MockLane(0),MockLane(4)]}})

class NanBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx,xy):return xy[:,:,1].sum(dim=1)
    @staticmethod
    def backward(ctx,grad):
        return torch.full((len(grad),8,2),float('nan'),dtype=grad.dtype,device=grad.device)
class DeliberatelyBadRoad:
    def components(self,xy,pose,cfg):return {'road':NanBackward.apply(xy)}

class TestRoadGradientStabilityV21(unittest.TestCase):
    def setUp(self):
        self.cfg=TrajectoryModeRewardConfig()
        x=torch.arange(1,9,dtype=torch.float32)*4.
        self.xy=torch.stack([x,torch.ones_like(x)*.8],dim=-1)[None].repeat(4,1,1)
        self.xy[1,:,1]*=-1
        self.xy[2,:,1]=0
        self.xy[3,:,1]=1.5
    def test_real_road_analytic_grads_finite(self):
        road=RoadBoundaryField.from_road(MockRoad())
        p=DifferentiableTaskReward(self.cfg,torch.device('cpu'))
        p.bind_road(road,[0.,0.,0.])
        grads=p.gradients(self.xy)
        for direction in DIRECTIONS:
            self.assertTrue(bool(grads[direction].isfinite().all()),direction)
        guided,_=guide_all(self.xy,p,iters=2)
        for direction in DIRECTIONS:
            self.assertTrue(bool(guided[direction].isfinite().all()),direction)
    def test_finite_difference_recovers_nonfinite_road_only(self):
        p=DifferentiableTaskReward(self.cfg,torch.device('cpu'))
        p.bind_road(DeliberatelyBadRoad(),[0.,0.,0.],balanced_road_weight=.25)
        grads=p.gradients(self.xy)
        self.assertEqual(p.gradient_fallback_counts['road'],4)
        self.assertEqual(p.gradient_fallback_counts['balanced'],4)
        self.assertTrue(bool(torch.allclose(grads['road'][:,:,1],torch.ones_like(grads['road'][:,:,1]),atol=.003)))
        self.assertTrue(bool(torch.allclose(grads['road'][:,:,0],torch.zeros_like(grads['road'][:,:,0]),atol=.003)))
        self.assertTrue(bool(grads['balanced'].isfinite().all()))
        guided,hist=guide_all(self.xy,p,iters=2)
        self.assertTrue(bool(guided['balanced'].isfinite().all()))
        self.assertGreaterEqual(hist['road'][-1],hist['road'][0]-1e-5)
if __name__=='__main__':unittest.main()
