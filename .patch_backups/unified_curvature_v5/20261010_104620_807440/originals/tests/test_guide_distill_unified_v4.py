"""V4 regression: single reward guidance, 100m progress, target weight parity."""
from __future__ import annotations
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
from highway_env.planner.diffusion.grpo.task_reward import task_reward_from_w4_result
from highway_env.planner.diffusion.guide_distill.reward_gradient import DifferentiableTaskReward,DIRECTIONS
from highway_env.planner.diffusion.guide_distill.exploration import guide_all
from highway_env.planner.diffusion.guide_distill.road_boundary import RoadBoundaryField

class Lane:
    length=140.
    def width_at(self,s):return 8.
    def position(self,s,lateral):return np.array([s,lateral],dtype=float)
class Road:
    def __init__(self):self.network=SimpleNamespace(graph={'a':{'b':[Lane()]}})

class TestUnifiedGuidanceV4(unittest.TestCase):
    def setUp(self):
        self.cfg=TrajectoryModeRewardConfig()
        self.proxy=DifferentiableTaskReward(self.cfg,torch.device('cpu'))
        self.proxy.bind_road(RoadBoundaryField.from_road(Road()),np.zeros(3))
        self.xy=torch.stack([torch.linspace(5,75,8,dtype=torch.float64),
                             torch.linspace(0,.8,8,dtype=torch.float64)],dim=-1)[None]

    def test_config_and_only_one_direction(self):
        self.assertEqual(self.cfg.progress_norm_m,100.)
        self.assertAlmostEqual(self.cfg.task_progress_weight,.20)
        self.assertAlmostEqual(self.cfg.task_comfort_weight,.10)
        self.assertAlmostEqual(self.cfg.task_road_weight,.40)
        self.assertEqual(DIRECTIONS,('balanced',))
        for invalid in ('comfort','road','curvature_aux','progress'):
            with self.assertRaises(KeyError):self.proxy.gradient_of(self.xy,invalid)

    def test_non_saturated_progress_100m(self):
        point=self.xy.clone();point[:,-1,0]=75.
        self.assertAlmostEqual(float(self.proxy.components(point)['raw_progress'][0]),.75)
        point[:,-1,0]=101.
        self.assertAlmostEqual(float(self.proxy.components(point)['raw_progress'][0]),1.)

    def test_task_formula_and_production_match(self):
        components=self.proxy.components(self.xy)
        expected=(self.cfg.task_progress_weight*components['raw_progress']
                  -self.cfg.task_comfort_weight*components['comfort_penalty']
                  +self.cfg.task_road_weight*components['road'])
        self.assertTrue(torch.allclose(components['balanced'],expected,atol=1e-10))
        fields={'progress_score':components['raw_progress'].detach().numpy(),
                'comfort_penalty':components['comfort_penalty'].detach().numpy(),
                'road_boundary_reward':components['road'].detach().numpy()}
        result=SimpleNamespace(components={k:np.broadcast_to(v[None,None,:],(3,10,1)).copy() for k,v in fields.items()},
                               rewards=np.zeros((3,10,1),dtype=np.float32))
        actual=task_reward_from_w4_result(result,context={'config':self.cfg},
                     device=torch.device('cpu'),dtype=torch.float64,reward_type='progress_comfort')
        np.testing.assert_allclose(actual[0,0,0].detach().numpy(),expected[0].detach().numpy(),atol=1e-6)

    def test_gradients_and_monotone_unified(self):
        grad=self.proxy.gradient_of(self.xy)
        self.assertTrue(bool(torch.isfinite(grad).all()))
        guided,hist=guide_all(self.xy,self.proxy,iters=2,step_m=.1)
        self.assertEqual(set(guided),{'balanced'})
        self.assertGreaterEqual(hist['balanced'][-1],hist['balanced'][0]-1e-6)
        self.assertGreaterEqual(float(self.proxy.components(guided['balanced'])['balanced'][0]),
                                float(self.proxy.components(self.xy)['balanced'][0])-1e-6)

if __name__=='__main__':unittest.main()
