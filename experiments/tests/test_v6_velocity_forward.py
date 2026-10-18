"""Regression for V6 planning-time per-role ego velocity forwarding.

Run from all_merge-guideDistill root:
    python -m experiments.tests.test_v6_velocity_forward
"""
from __future__ import annotations

import numpy as np
from unittest.mock import patch
from highway_env.planner.diffusion.trajectory_mode_reward.counterfactual import TrajectoryModeCounterfactualReward
from highway_env.planner.diffusion.trajectory_mode_reward.results import RewardGeometryContext


def run_test() -> None:
    scorer=TrajectoryModeCounterfactualReward()
    # Different, nonzero, ego-local velocities catch both missing and wrong-role propagation.
    velocities=(np.array([24.0,0.0]),np.array([18.0,1.5]),np.array([31.0,-0.4]))
    context=RewardGeometryContext(
        poses=tuple(np.array([float(r)*8., float(r)*4., 0.]) for r in range(3)),
        backgrounds=({}, {}, {}),road=object(),
        target_centerlines=tuple(np.column_stack((np.arange(41,dtype=float),np.zeros(41))) for _ in range(3)),
        initial_velocity_xy=velocities,
    )
    valid=np.zeros((3,10),dtype=bool)
    valid[:,0]=True
    frozen_all=np.zeros((3,10,8,2),dtype=np.float32)
    frozen_argmax=np.zeros((3,8,2),dtype=np.float32)
    candidates=np.zeros((3,10,2,8,2),dtype=np.float32)
    calls=[]

    def fake_target_group(*, target_role, initial_velocity_by_role=None, target_trajectories, **kwargs):
        assert initial_velocity_by_role is context.initial_velocity_xy, (
            'Lost RewardGeometryContext.initial_velocity_xy in scoring chain'
        )
        assert len(initial_velocity_by_role)==3
        np.testing.assert_array_equal(initial_velocity_by_role[target_role],velocities[target_role])
        count=target_trajectories.shape[0]
        calls.append((target_role,count))
        scores=np.full(count,float(target_role)+0.25,dtype=np.float32)
        flags=np.zeros(count,dtype=bool)
        return {
            'task_rewards':scores,
            'unsafe':flags.copy(),
            'collision':flags.copy(),
            'out_of_drivable':flags.copy(),
            'clearance_violation':flags.copy(),
            'components':{k:np.zeros(count,dtype=np.float32) for k in scorer.COMPONENT_NAMES},
        }

    with patch.object(scorer,'_score_target_group',side_effect=fake_target_group):
        pretrain=scorer.score_pretrain(
            object(),frozen_all,frozen_argmax,valid,geometry_context=context
        )
        result=scorer.score_candidates(
            object(),candidates,frozen_argmax,valid,pretrain,geometry_context=context
        )
    assert calls==[(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)],calls
    assert pretrain.task_rewards.shape==(3,10)
    assert result.task_rewards.shape==(3,10,2)
    for role in range(3):
        assert np.all(result.task_rewards[role,0]==role+0.25)
    print('[PASS] V6 frozen/current counterfactual scoring share correct per-role planning-time initial velocities')

if __name__=='__main__':run_test()
