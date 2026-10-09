"""Unittest: run with python -m unittest discover -s tests -p test_guide_distill_probe.py"""
from __future__ import annotations
import unittest
import numpy as np
import torch
from highway_env.planner.diffusion.guide_distill.reward_gradient import DifferentiableTaskReward, DIRECTIONS
from highway_env.planner.diffusion.guide_distill.exploration import guide_all, gradient_cosine
from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
from highway_env.planner.diffusion.guide_distill.road_boundary import RoadBoundaryField
from types import SimpleNamespace
class MockLane:
    length=100.
    def __init__(self,y):self.y=y
    def width_at(self,s):return 4.
    def position(self,s,lateral):return np.asarray([s,self.y+lateral])
class MockRoad:
    def __init__(self):self.network=SimpleNamespace(graph={'a':{'b':[MockLane(0),MockLane(4)]}})


class TestCurrentRewardGuidance(unittest.TestCase):
    def setUp(self):
        self.cfg=TrajectoryModeRewardConfig()
        self.proxy=DifferentiableTaskReward(self.cfg,torch.device('cpu'))
        self.proxy.bind_road(RoadBoundaryField.from_road(MockRoad()),np.array([0.,0.,0.]))
        x=torch.arange(1,9,dtype=torch.float32)*5.0
        y=torch.sin(torch.linspace(0,2,8))*0.5
        self.xy=torch.stack((x[None,:].expand(3,-1).clone(),y[None,:].expand(3,-1).clone()),dim=-1)
        self.xy[1,:,1]*=2
        self.xy[2,:,1]*=-2

    def test_current_components(self):
        v=self.proxy.components(self.xy)
        self.assertTrue(torch.allclose(v['native_balanced'],float(self.cfg.progress_weight)*v['raw_progress']-
                                         float(self.cfg.comfort_weight)*v['comfort_penalty'],atol=1e-8))
        self.assertNotIn('progress',DIRECTIONS)
        self.assertIn('road',DIRECTIONS)

    def test_gradients_and_bounded_guidance(self):
        grads=self.proxy.gradients(self.xy)
        self.assertEqual(set(grads),set(DIRECTIONS))
        cs=gradient_cosine(grads)
        self.assertEqual(tuple(cs.shape),(3,4,4))
        guided,_=guide_all(self.xy,self.proxy,iters=2,step_m=.1,trust_rms_m=.5,max_point_move_m=.3)
        for key,arr in guided.items():
            self.assertTrue(torch.isfinite(arr).all())
            self.assertLessEqual(float(torch.linalg.vector_norm(arr-self.xy,dim=-1).max()),.30002)
            self.assertGreaterEqual(float((self.proxy.components(arr)[key]-self.proxy.components(self.xy)[key]).min()),-1e-5)

    def test_zero_gradient_is_nan_cosine(self):
        z={key:torch.zeros_like(self.xy) for key in DIRECTIONS}
        self.assertTrue(torch.isnan(gradient_cosine(z)).all())

if __name__=='__main__':unittest.main()
