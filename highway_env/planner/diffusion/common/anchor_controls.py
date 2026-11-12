"""Convert the existing 0.5-s XY motion anchors to 0.1-s control anchors.

This module only adapts the representation used for *diffusion initialization*.
The source remains the exact same coarse_trajectories as trajectory-space
sampling. It does not modify anchor generation, validity, planner objectives,
vehicle execution, or checkpoint weights.

With a future anchor generator that retains the sampled a/delta commands,
those controls can be used directly; the current builder stores XY only, so
its controls are reconstructed deterministically from smooth XY derivatives.
"""
from __future__ import annotations

import torch


def xy_anchors_to_controls(
    anchors_xy: torch.Tensor,
    initial_vx: torch.Tensor,
    *,
    sparse_dt: float = 0.5,
    dense_dt: float = 0.1,
    wheelbase_m: float = 5.5,
    max_accel_mps2: float = 2.5,
    max_steer_rad: float = 0.08,
) -> torch.Tensor:
    """Return bounded physical [a,front-wheel-delta] for each XY anchor.

    Args:
        anchors_xy: [B,M,K,2], positions at (1..K)*sparse_dt in the
            planning-time ego frame (the existing anchor contract).
        initial_vx: [B], initial ego-frame forward speed (m/s).

    Each segment uses a cubic Hermite interpolant with the measured initial
    forward speed and centered knot slopes. First and second derivatives at
    10-Hz midpoints determine a and kappa, then
        delta = atan(wheelbase * kappa).
    Controls are clipped to the same physical limits used by the planner.
    The conversion is approximate: it cannot guarantee an exact inverse for
    arbitrary geometrically constructed XY anchors (especially tight curves,
    low-speed reversals, and points violating the control limits).
    """
    if anchors_xy.ndim != 4 or anchors_xy.shape[-1] != 2:
        raise ValueError('anchors_xy must be [B,M,K,2]')
    batch, modes, knots, _ = anchors_xy.shape
    if knots < 2:
        raise ValueError('At least 2 sparse anchor points required')
    if initial_vx.reshape(-1).shape != (batch,):
        raise ValueError('initial_vx must have shape [B]')
    if sparse_dt <= 0 or dense_dt <= 0 or wheelbase_m <= 0:
        raise ValueError('Times and wheelbase must be positive')
    if max_accel_mps2 <= 0 or max_steer_rad <= 0:
        raise ValueError('Control bounds must be positive')
    ratio = sparse_dt / dense_dt
    stride = round(ratio)
    if stride < 1 or abs(stride - ratio) > 1e-6:
        raise ValueError('sparse_dt must be an integer multiple of dense_dt')
    if not torch.isfinite(anchors_xy).all() or not torch.isfinite(initial_vx).all():
        raise ValueError('Anchor positions and ego speed must be finite')

    # Knot 0 corresponds to the ego origin at t=0.
    origin = torch.zeros_like(anchors_xy[:, :, :1, :])
    positions = torch.cat((origin, anchors_xy), dim=-2)
    slopes = torch.zeros_like(positions)
    slopes[:, :, 0, 0] = initial_vx.reshape(batch, 1).clamp_min(0).expand(-1, modes)
    slopes[:, :, 1:-1] = (positions[:, :, 2:] - positions[:, :, :-2]) / (2 * sparse_dt)
    slopes[:, :, -1] = (positions[:, :, -1] - positions[:, :, -2]) / sparse_dt

    total = knots * stride
    idx = torch.arange(total, device=anchors_xy.device)
    segment = torch.div(idx, stride, rounding_mode='floor')
    u = (idx.remainder(stride).to(dtype=anchors_xy.dtype) + 0.5) / stride
    u = u.view(1, 1, total, 1)
    p0, p1 = positions[:, :, segment], positions[:, :, segment + 1]
    m0, m1 = slopes[:, :, segment], slopes[:, :, segment + 1]
    h = sparse_dt

    # Analytic time derivatives of the cubic Hermite polynomial.
    vel = ((6*u*u-6*u)/h*p0 + (3*u*u-4*u+1)*m0
           + (-6*u*u+6*u)/h*p1 + (3*u*u-2*u)*m1)
    acc = ((12*u-6)/(h*h)*p0 + (6*u-4)/h*m0
           + (-12*u+6)/(h*h)*p1 + (6*u-2)/h*m1)
    speed_sq = vel.square().sum(dim=-1)
    speed = speed_sq.clamp_min(0.0).sqrt()
    # Freeze steering and longitudinal acceleration at very small speed,
    # where geometry does not define a reliable driving direction.
    moving = speed > 0.3
    v_safe = speed.clamp_min(0.3)
    tangent_acc = (vel * acc).sum(dim=-1) / v_safe
    curvature = (vel[..., 0]*acc[..., 1] - vel[..., 1]*acc[..., 0]) / (v_safe**3)
    steer = torch.atan(wheelbase_m * curvature)
    tangent_acc = torch.where(moving, tangent_acc, torch.zeros_like(tangent_acc))
    steer = torch.where(moving, steer, torch.zeros_like(steer))
    return torch.stack((
        tangent_acc.clamp(-max_accel_mps2, max_accel_mps2),
        steer.clamp(-max_steer_rad, max_steer_rad),
    ), dim=-1)
