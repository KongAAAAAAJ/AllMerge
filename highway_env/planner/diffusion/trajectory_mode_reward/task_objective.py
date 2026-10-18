"""Single V6 differentiable reward algebra shared by production, GRPO and PGD.

All terms dimensionless.  No per-group normalization.  All longitudinal
kinematics are derived from the same 41-point decoded trajectory.
"""
from __future__ import annotations
import torch


def _as_tensor(v, like):
    return torch.as_tensor(v, device=like.device, dtype=like.dtype)


def smooth_tier(value: torch.Tensor, comfortable: float, feasible: float, *, beta: float = 8.):
    """Quadratic comfortable excess; stronger quadratic beyond feasibility.

    Normalized excess remains meaningful over the feasible range; excess in
    the extreme tail uses smooth Huber-like growth to avoid reward explosions.
    """
    if not 0 <= comfortable < feasible:
        raise ValueError('Require 0 <= comfortable < feasible')
    interval = feasible - comfortable
    c = ((value.abs() - comfortable) / interval).clamp_min(0.)
    f = ((value.abs() - feasible) / interval).clamp_min(0.)
    # Smooth C1 Huber tail, quadratic up to 1 and linear after 1.
    def huber(x): return torch.where(x <= 1., x.square(), 2. * x - 1.)
    return huber(c) + beta * huber(f)


def longitudinal_terms(dense: torch.Tensor, config, *, initial_velocity_xy=None):
    """dense [...,41,2], continuous differentiable where possible."""
    dt = float(config.interpolation_dt_s)
    dv = (dense[..., 1:, :] - dense[..., :-1, :]) / dt
    heading = dv / torch.linalg.vector_norm(dv, dim=-1, keepdim=True).clamp_min(1e-5)
    speed = torch.linalg.vector_norm(dv, dim=-1)
    if initial_velocity_xy is not None:
        v0 = _as_tensor(initial_velocity_xy, dense)
        v0 = torch.broadcast_to(v0, (*dense.shape[:-2], 2))
        speed0 = torch.linalg.vector_norm(v0, dim=-1, keepdim=True)
        # Include initial boundary velocity in acceleration, do not hide first-bin discontinuity.
        longitudinal_v = torch.cat([speed0, speed], dim=-1)
    else:
        longitudinal_v = torch.cat([speed[..., :1], speed], dim=-1)
    accel = torch.diff(longitudinal_v, dim=-1) / dt
    jerk = torch.diff(accel, dim=-1, prepend=accel[..., :1]) / dt
    ca = float(config.v6_accel_comfort_mps2)
    fa = float(config.v6_accel_feasible_mps2)
    cj = float(config.v6_jerk_comfort_mps3)
    fj = float(config.v6_jerk_feasible_mps3)
    a_cost = smooth_tier(accel, ca, fa, beta=float(config.v6_strong_penalty_beta))
    j_cost = smooth_tier(jerk, cj, fj, beta=float(config.v6_strong_penalty_beta))
    alpha = float(config.v6_peak_weight)
    a_summary = (1.-alpha)*a_cost.mean(dim=-1) + alpha*a_cost.amax(dim=-1)
    j_summary = (1.-alpha)*j_cost.mean(dim=-1) + alpha*j_cost.amax(dim=-1)
    # Fixed reference length, detached: cannot game progress normalization by modifying the first waypoint.
    initial_speed = (longitudinal_v[..., :1].detach().squeeze(-1))
    duration = (dense.shape[-2]-1) * dt
    reference = (initial_speed * duration).clamp_min(float(config.v6_min_progress_norm_m))
    progress = (dense[..., -1, 0]-dense[..., 0, 0]) / reference
    return {'progress_score':progress, 'accel_penalty':a_summary,
            'jerk_penalty':j_summary,'accel_peak_mps2':accel.abs().amax(dim=-1),
            'jerk_peak_mps3':jerk.abs().amax(dim=-1),
            'speed_final_mps':speed[..., -1],
            'accel_series_mps2':accel,'jerk_series_mps3':jerk}


def combine_reward(components:dict, cfg):
    """Only authoritative V6 total objective. Same function in all callers."""
    return (float(cfg.task_progress_weight)*components['progress_score']
            -float(cfg.v6_accel_weight)*components['accel_penalty']
            -float(cfg.v6_jerk_weight)*components['jerk_penalty']
            +float(cfg.task_road_weight)*components['road_boundary_reward']
            -float(cfg.task_curvature_weight)*components['curvature_penalty']
            -float(cfg.task_centerline_weight)*components['centerline_penalty']
            -float(cfg.v6_lateral_comfort_weight)*components['lateral_comfort_penalty'])
