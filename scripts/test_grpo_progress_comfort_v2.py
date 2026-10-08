"""Smoke checks for progress_comfort V2 (no simulator required)."""
from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace, ModuleType
from pathlib import Path
import sys

# Run the reward-unit test without importing highway_env's simulator package.
# Only the actual config/motion_quality/task_reward source files are loaded.
repo = Path(__file__).resolve().parents[1]
for fullname, rel in (
    ('highway_env', 'highway_env'),
    ('highway_env.planner', 'highway_env/planner'),
    ('highway_env.planner.diffusion', 'highway_env/planner/diffusion'),
    ('highway_env.planner.diffusion.trajectory_mode_reward',
     'highway_env/planner/diffusion/trajectory_mode_reward'),
    ('highway_env.planner.diffusion.grpo', 'highway_env/planner/diffusion/grpo'),
):
    if fullname not in sys.modules:
        package = ModuleType(fullname)
        package.__path__ = [str(repo / rel)]
        sys.modules[fullname] = package
adapter = ModuleType('highway_env.planner.diffusion.grpo.reward_adapter')
# legacy_w4 is deliberately not invoked in this test.
adapter.w4_to_grpo_rewards = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError('legacy_w4 unexpectedly invoked'))
sys.modules[adapter.__name__] = adapter

from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
from highway_env.planner.diffusion.trajectory_mode_reward.motion_quality import spline_motion_quality
from highway_env.planner.diffusion.grpo.task_reward import task_reward_from_w4_result


def main() -> None:
    cfg = TrajectoryModeRewardConfig()
    assert cfg.task_progress_weight == 1.0
    assert cfg.task_comfort_weight == 10.0
    assert cfg.task_kinematic_weight == 1.0

    t = np.arange(1, 9, dtype=np.float64) * 0.5
    straight = np.column_stack([10.0*t, np.zeros_like(t)])
    smooth = np.column_stack([10.0*t, 1.5*(1.0-np.cos(np.pi*t/4.0))])
    corner = np.column_stack([10.0*t, np.where(t < 2.0, 0.0, 2.5)])
    result = spline_motion_quality(np.stack([straight, smooth, corner]), cfg)
    s = result['smoothness_penalty']
    k = result['kinematic_score']
    assert np.isfinite(s).all() and np.isfinite(k).all()
    assert 0.0 <= s[0] < s[1] < s[2] <= 1.0, (s,k)
    assert 0.0 < k[2] < k[1] <= k[0] <= 1.0, (s,k)
    assert s.shape == (3,)

    # Same progress, different geometry -> the smoother curve wins.
    progress = np.ones((3, 10, 3), dtype=np.float32)
    smooth_pen = np.broadcast_to(s.reshape(1, 1, -1), progress.shape).copy()
    kin_score = np.broadcast_to(k.reshape(1, 1, -1), progress.shape).copy()
    fake = SimpleNamespace(components={
        'progress_score': progress,
        'smoothness_penalty': smooth_pen,
        'kinematic_score': kin_score,
    })
    scores = task_reward_from_w4_result(fake, context={'config': cfg},
                                         device=torch.device('cpu'), dtype=torch.float32,
                                         reward_type='progress_comfort')
    assert tuple(scores.shape) == (3,3,10)
    means = scores.mean(dim=(0,2)).numpy()
    assert means[0] > means[1] > means[2], means
    print('[PASS] Reward V2: 1*progress - 10*smoothness + 1*kinematic')
    print('[PASS] geometry ordering straight > smooth turn > kink')
    print('[PASS] legacy_w4 weights preserved; GRPO reward shape [3,3,10]')
    for name, val in zip(('straight', 'smooth', 'corner'), means):
        print(f'{name}: reward={val:.5f}')


if __name__ == '__main__':
    main()
