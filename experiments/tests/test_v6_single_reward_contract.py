"""V6 task-reward contract smoke test.
Run from all_merge-guideDistill/: python -m experiments.tests.test_v6_single_reward_contract
"""
from __future__ import annotations

import numpy as np
import torch

from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
from highway_env.planner.diffusion.trajectory_mode_reward.results import (
    TrajectoryModePretrainRewardResult, TrajectoryModeRewardResult, _COMPONENT_NAMES,
)
from highway_env.planner.diffusion.trajectory_mode_reward.task_objective import combine_reward
from highway_env.planner.diffusion.grpo.task_reward import task_reward_from_result


def main():
    cfg = TrajectoryModeRewardConfig()
    # A representative, nontrivial mix of V6 components.
    common = dict(progress_score=0.9, accel_penalty=0.05, jerk_penalty=0.16,
                  road_boundary_reward=0.33, curvature_penalty=0.08,
                  centerline_penalty=0.15, lateral_comfort_penalty=0.04)
    expected = combine_reward({k: torch.tensor([v], dtype=torch.float32)
                               for k, v in common.items()}, cfg).item()
    assert np.isfinite(expected)
    comps = {k: np.zeros((3,10,2), dtype=np.float32) for k in _COMPONENT_NAMES}
    for key,value in common.items():
        comps[key][:] = value
    comps['centerline_valid'][:]=1.0
    precomps = {k: np.zeros((3,10), dtype=np.float32) for k in _COMPONENT_NAMES}
    precomps['centerline_valid'][:] = 1.0
    task = np.full((3,10,2), expected, dtype=np.float32)
    pretask = np.full((3,10), expected, dtype=np.float32)
    valid = np.ones((3,10), dtype=bool)
    flags = np.zeros((3,10,2), dtype=bool)
    pflags = np.zeros((3,10), dtype=bool)
    pre=TrajectoryModePretrainRewardResult(
        task_rewards=pretask,valid_mode_mask=valid, unsafe=pflags,
        collision=pflags,out_of_drivable=pflags,clearance_violation=pflags,
        components=precomps,
    )
    result=TrajectoryModeRewardResult(
        task_rewards=task,pretrain_task_rewards=pre.task_rewards,
        valid_mode_mask=valid, unsafe=flags, collision=flags,
        out_of_drivable=flags,clearance_violation=flags,
        pretrain_unsafe=pre.unsafe,pretrain_collision=pre.collision,
        pretrain_out_of_drivable=pre.out_of_drivable,
        pretrain_clearance_violation=pre.clearance_violation,
        components=comps,pretrain_components=pre.components,
    )
    assert not hasattr(result, 'rewards') and not hasattr(pre, 'rewards')
    grpo=task_reward_from_result(result,context={'config':cfg},
                                device=torch.device('cpu'),dtype=torch.float32,
                                reward_type='unified_v6')
    assert grpo.shape == (3,2,10)
    torch.testing.assert_close(grpo, torch.full_like(grpo,expected), atol=1e-6,rtol=1e-6)
    for obsolete in ('legacy_w4',):
        try:
            task_reward_from_result(result,context={'config':cfg},device=torch.device('cpu'),
                                    dtype=torch.float32,reward_type=obsolete)
        except ValueError:
            pass
        else:
            raise AssertionError('Obsolete W4 objective unexpectedly accepted')
    print('[PASS] V6-only task_rewards [3,10,N], GRPO [3,N,10], legacy field absent')

if __name__ == '__main__':
    main()
