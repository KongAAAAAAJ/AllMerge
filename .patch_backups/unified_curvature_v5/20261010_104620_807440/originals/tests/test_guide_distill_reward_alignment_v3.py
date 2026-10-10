"""Reward-aligned Cubic Spline V3, run in the actual offpolicyEnv worktree."""
from __future__ import annotations
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
from highway_env.planner.diffusion.trajectory_mode_reward.spline_dense import evaluate_dense_spline
from highway_env.planner.diffusion.trajectory_mode_reward.geometry import _dense_local_trajectories_batch
from highway_env.planner.diffusion.trajectory_mode_reward.scoring import CounterfactualScoringMixin
from highway_env.planner.diffusion.guide_distill.road_boundary import RoadBoundaryField
from highway_env.planner.diffusion.guide_distill.reward_gradient import DifferentiableTaskReward

class Lane:
    length=130.
    def width_at(self, s): return 8.
    def position(self,s,lateral):return np.asarray([s,lateral],dtype=float)

class Road:
    def __init__(self):self.network=SimpleNamespace(graph={'a': {'b': [Lane()]}})

class Scorer(CounterfactualScoringMixin):
    def __init__(self,c):self.config=c

class TestSplineRewardAlignmentV3(unittest.TestCase):
    def setUp(self):
        self.config=TrajectoryModeRewardConfig()
        self.road=Road()
        self.xy=torch.stack((torch.linspace(6.,38.,8),
                             torch.linspace(0.,.4,8)),dim=-1)[None,:,:].double()

    def test_task_reward_same_existing_name(self):
        from highway_env.planner.diffusion.grpo.task_reward import (
            SUPPORTED_TASK_REWARDS, task_reward_from_w4_result,
        )
        self.assertIn('progress_comfort',SUPPORTED_TASK_REWARDS)
        self.assertNotIn('progress_comfort_road_spline_v1',SUPPORTED_TASK_REWARDS)
        shape=(3,10,2)
        values={'progress_score':np.full(shape,.5,dtype=np.float32),
                'comfort_penalty':np.full(shape,.2,dtype=np.float32),
                'road_boundary_reward':np.full(shape,.6,dtype=np.float32)}
        result=SimpleNamespace(components=values,rewards=np.zeros(shape,dtype=np.float32))
        actual=task_reward_from_w4_result(result,context={'config':self.config},
                     device=torch.device('cpu'),dtype=torch.float64,reward_type='progress_comfort')
        expected=self.config.task_progress_weight*.5-self.config.task_comfort_weight*.2+self.config.task_road_weight*.6
        np.testing.assert_allclose(actual.numpy(),expected,atol=1e-7)

    def test_shared_dense_41_samples(self):
        joint=np.repeat(self.xy.detach().numpy()[:,None],3,axis=1)
        prod,_=_dense_local_trajectories_batch(joint,self.config)
        dense,_,_=evaluate_dense_spline(self.xy,self.config)
        self.assertEqual(prod.shape,(1,3,41,3))
        np.testing.assert_allclose(prod[:,0,:,:2],dense.detach().numpy(),rtol=0,atol=1e-10)

    def test_reward_and_guidance_parity(self):
        joint=np.repeat(self.xy.detach().numpy()[0][None],3,axis=0)
        out=Scorer(self.config)._score_target_group(target_role=0,
            target_trajectories=self.xy.detach().numpy(),
            frozen_argmax_joint_trajectories=joint,
            poses=[np.zeros(3) for _ in range(3)],road=self.road,background_by_actor={})
        proxy=DifferentiableTaskReward(self.config,torch.device('cpu'))
        proxy.bind_road(RoadBoundaryField.from_road(self.road),np.zeros(3),
                        balanced_road_weight=self.config.task_road_weight)
        vals=proxy.components(self.xy)
        for component,key in [('progress_score','raw_progress'),('comfort_penalty','comfort_penalty'),
                              ('road_boundary_reward','road')]:
            np.testing.assert_allclose(out['components'][component],vals[key].detach().numpy(),
                                       rtol=0,atol=5e-6)
        production=(self.config.task_progress_weight*out['components']['progress_score']
                    -self.config.task_comfort_weight*out['components']['comfort_penalty']
                    +self.config.task_road_weight*out['components']['road_boundary_reward'])
        np.testing.assert_allclose(production,vals['balanced'].detach().numpy(),rtol=0,atol=5e-6)
        grad=proxy.gradient_of(self.xy,'balanced')
        self.assertTrue(bool(torch.isfinite(grad).all()))

if __name__=='__main__':unittest.main()
