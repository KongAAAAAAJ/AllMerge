# STAGE_A_SPLINE_RUNTIME_V1
from __future__ import annotations

import math

import pytest
import torch

from highway_env.planner.diffusion.trajectory_spline import (
    ClampedCubicTrajectorySpline,
)


def _linear_sparse(batch=1, speed=12.0, device="cpu"):
    t = torch.arange(1, 9, dtype=torch.float32, device=device) * 0.5
    xy = torch.stack([speed * t, torch.zeros_like(t)], dim=-1)
    return xy.unsqueeze(0).repeat(batch, 1, 1)


def test_waypoint_exact_interpolation():
    spline = ClampedCubicTrajectorySpline()
    sparse = torch.tensor(
        [[[3.0, 0.2], [7.0, 0.7], [12.0, 1.5], [18.0, 2.0],
          [25.0, 2.6], [33.0, 2.0], [42.0, 1.0], [52.0, 0.0]]],
        dtype=torch.float32,
    )
    dense = spline.interpolate(
        sparse,
        start_xy=torch.zeros(1, 2),
        start_velocity_xy=torch.tensor([[6.0, 0.0]]),
    )
    recovered = dense[:, spline.sparse_dense_indices, :]
    assert torch.max(torch.abs(recovered - sparse)).item() < 1e-5


def test_constant_velocity_straight_line():
    speed = 12.0
    spline = ClampedCubicTrajectorySpline()
    sparse = _linear_sparse(speed=speed)
    pos, vel, acc = spline.evaluate(
        sparse,
        start_xy=torch.zeros(1, 2),
        start_velocity_xy=torch.tensor([[speed, 0.0]]),
    )
    expected_x = spline.dense_time_s * speed
    assert torch.allclose(pos[0, :, 0], expected_x, atol=1e-5, rtol=0.0)
    assert torch.allclose(pos[0, :, 1], torch.zeros_like(expected_x), atol=1e-6)
    assert torch.allclose(vel[0, :, 0], torch.full_like(expected_x, speed), atol=1e-5)
    assert torch.max(torch.abs(acc)).item() < 1e-4


def test_c1_c2_continuity():
    torch.manual_seed(3)
    spline = ClampedCubicTrajectorySpline()
    sparse = torch.cumsum(torch.randn(1, 8, 2) * 0.4 + torch.tensor([4.0, 0.0]), dim=1)
    eps = 1e-5
    for knot in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        q = torch.tensor([knot - eps, knot + eps], dtype=torch.float32)
        _, vel, acc = spline.evaluate(
            sparse,
            start_xy=torch.zeros(1, 2),
            start_velocity_xy=torch.tensor([[8.0, 0.0]]),
            query_time_s=q,
        )
        assert torch.max(torch.abs(vel[:, 0] - vel[:, 1])).item() < 2e-3
        assert torch.max(torch.abs(acc[:, 0] - acc[:, 1])).item() < 2e-3


def test_batch_shape_and_autograd():
    torch.manual_seed(4)
    spline = ClampedCubicTrajectorySpline()
    sparse = torch.randn(5, 8, 2, requires_grad=True)
    dense = spline.interpolate(
        sparse,
        start_xy=torch.zeros(5, 2),
        start_velocity_xy=torch.tensor([10.0, 0.0]),
    )
    assert dense.shape == (5, 41, 2)
    dense.square().mean().backward()
    assert sparse.grad is not None
    assert torch.isfinite(sparse.grad).all()
    assert sparse.grad.abs().sum().item() > 0.0


def test_nan_inf_rejected():
    spline = ClampedCubicTrajectorySpline()
    sparse = _linear_sparse()
    sparse[0, 2, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        spline.interpolate(
            sparse,
            start_xy=torch.zeros(1, 2),
            start_velocity_xy=torch.tensor([[10.0, 0.0]]),
        )


def test_extreme_but_reasonable_curve_is_finite():
    spline = ClampedCubicTrajectorySpline()
    t = torch.arange(1, 9, dtype=torch.float32) * 0.5
    x = 8.0 * t
    y = 5.0 * torch.sin(t * math.pi / 2.0)
    sparse = torch.stack([x, y], dim=-1).unsqueeze(0)
    pos, vel, acc = spline.evaluate(
        sparse,
        start_xy=torch.zeros(1, 2),
        start_velocity_xy=torch.tensor([[8.0, 0.0]]),
    )
    assert torch.isfinite(pos).all()
    assert torch.isfinite(vel).all()
    assert torch.isfinite(acc).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_gpu_consistency():
    torch.manual_seed(5)
    sparse_cpu = torch.randn(3, 8, 2)
    start_cpu = torch.zeros(3, 2)
    velocity_cpu = torch.tensor([[9.0, 0.2]]).repeat(3, 1)

    cpu_spline = ClampedCubicTrajectorySpline()
    gpu_spline = ClampedCubicTrajectorySpline().cuda()

    cpu = cpu_spline.interpolate(sparse_cpu, start_cpu, velocity_cpu)
    gpu = gpu_spline.interpolate(
        sparse_cpu.cuda(),
        start_cpu.cuda(),
        velocity_cpu.cuda(),
    ).cpu()
    assert torch.allclose(cpu, gpu, atol=2e-4, rtol=2e-4)
