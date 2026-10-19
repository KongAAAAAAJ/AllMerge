"""CPU-only regression for precision of V6 PGD teacher scoring.

Run from the all_merge-guideDistill worktree:
    python -m experiments.tests.test_pgd_high_precision_reward
"""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import torch

from highway_env.planner.diffusion.guide_distill.reward_gradient import DifferentiableTaskReward
from highway_env.planner.diffusion.guide_distill.teacher import verify_reward_parity_v2
from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig


class _ZeroRoad:
    def components(self, xy, pose, config, *, initial_velocity_xy=None):
        return {'road': xy.new_zeros(xy.shape[0])}


def _make_proxy(high_precision):
    proxy = DifferentiableTaskReward(TrajectoryModeRewardConfig(), torch.device('cpu'),
                                    high_precision=high_precision)
    proxy.bind_initial_velocity(torch.tensor([25.0, 0.0]))
    proxy.bind_road(_ZeroRoad(), np.array([0.0, 0.0, 0.0]))
    proxy.bind_centerline(np.column_stack((np.linspace(-30, 140, 70), np.zeros(70))))
    return proxy


def run():
    times = torch.arange(1, 9, dtype=torch.float32) * .5
    xy = torch.stack((times * 25, .5 * torch.sin(times * .75)), -1)[None].repeat(3, 1, 1)
    xy[1, 1, 0] += .051
    xy[1, 3, 0] -= .029
    xy[2, 2, 1] += .11
    xy.requires_grad_(True)

    native = _make_proxy(False)
    pgd = _make_proxy(True)
    assert native.components(xy)['balanced'].dtype == torch.float32
    high = pgd.components(xy)
    assert high['balanced'].dtype == torch.float64
    assert torch.isfinite(high['balanced']).all()
    grad = pgd.gradient_of(xy)
    assert grad.dtype == xy.dtype and grad.shape == xy.shape and torch.isfinite(grad).all()
    exact = pgd.components(xy.detach().double())
    assert torch.allclose(high['balanced'].detach(), exact['balanced'], rtol=0, atol=1e-10)

    # Emulate the production ScoreResult layout [role, mode, candidate],
    # using CPU FP64 components quantized to the actual FP32 result dtype.
    parts = {
        'progress_score': 'raw_progress',
        'road_boundary_reward': 'road',
        'curvature_penalty': 'curvature_penalty',
        'centerline_penalty': 'centerline_penalty',
        'accel_penalty': 'accel_penalty',
        'jerk_penalty': 'jerk_penalty',
    }
    n = xy.shape[0]
    components = {}
    for production_name, proxy_name in parts.items():
        a = np.zeros((3, 10, n), dtype=np.float32)
        a[0, 0] = exact[proxy_name].detach().cpu().numpy().astype(np.float32)
        components[production_name] = a
    components['centerline_valid'] = np.ones((3, 10, n), dtype=np.float32)
    score = np.zeros((3, 10, n), dtype=np.float32)
    score[0, 0] = exact['balanced'].detach().cpu().numpy().astype(np.float32)
    result = SimpleNamespace(components=components, task_rewards=score,
                             valid_mode_mask=np.ones((3, 10), dtype=bool))
    assert verify_reward_parity_v2(pgd, xy.detach(), result, role=0, mode=0)
    print('[PASS] PGD FP64 score, FP32 waypoint gradients and V6 score parity')


if __name__ == '__main__':
    run()
