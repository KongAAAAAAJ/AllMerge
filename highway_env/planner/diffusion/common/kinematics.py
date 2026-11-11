"""Differentiable planar bicycle rollout for TRACE control-space Stage D1.

Controls: longitudinal acceleration [m/s^2], front-wheel steering [rad], at 10 Hz.
The output poses are ego-centric [x (m), y (m), yaw (rad)] including t=0.
This is a kinematic initialization baseline, NOT the TruckSim/Koopman surrogate.
"""
from __future__ import annotations

import torch


def rollout_bicycle(controls: torch.Tensor, initial_vx: torch.Tensor,
                    dt: float = 0.1, wheelbase_m: float = 5.5):
    """Return poses [B,H+1,3] and longitudinal speeds [B,H+1]."""
    if controls.ndim != 3 or controls.shape[-1] != 2:
        raise ValueError('controls must have shape [B,H,2]')
    if initial_vx.reshape(-1).numel() != controls.shape[0]:
        raise ValueError('initial_vx must have batch size B')
    if dt <= 0 or wheelbase_m <= 0:
        raise ValueError('dt and wheelbase_m must be positive')
    vx = initial_vx.reshape(-1).to(controls).clamp_min(0.0)
    x, y, psi = (torch.zeros_like(vx) for _ in range(3))
    poses = [torch.stack((x, y, psi), dim=-1)]
    speeds = [vx]
    for j in range(controls.shape[1]):
        a = controls[:, j, 0]
        delta = controls[:, j, 1]
        v_new = (vx + a * dt).clamp_min(0.0)
        v_mid = 0.5 * (vx + v_new)
        yaw_step = v_mid * torch.tan(delta) * (dt / wheelbase_m)
        psi_mid = psi + 0.5 * yaw_step
        x = x + v_mid * torch.cos(psi_mid) * dt
        y = y + v_mid * torch.sin(psi_mid) * dt
        psi = psi + yaw_step
        poses.append(torch.stack((x, y, psi), dim=-1))
        speeds.append(v_new)
        vx = v_new
    return torch.stack(poses, dim=1), torch.stack(speeds, dim=1)
