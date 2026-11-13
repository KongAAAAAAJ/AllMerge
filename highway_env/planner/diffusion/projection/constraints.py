"""Full-horizon chassis stability constraints used by JTP."""
from __future__ import annotations

import math

import torch


def stability_limits(vx: torch.Tensor, payload_kg: torch.Tensor, road_mu: torch.Tensor,
                     cfg: dict) -> torch.Tensor:
    """Return per-step [r_lim, beta_lim, phi_lim, LTR_lim] bounds."""
    section = cfg["stability"]
    sideslip = section["sideslip"]
    roll = section["roll"]
    g = float(section["gravity_mps2"])
    mu = road_mu.reshape(()).to(vx).clamp_min(float(section["minimum_mu_for_limits"]))
    payload = payload_kg.reshape(()).to(vx).clamp_min(0.0)

    v_safe = vx.abs().clamp_min(float(section["minimum_speed_for_yaw_limit_mps"]))
    yaw_limit = (float(section["yaw_rate_safety_factor"]) * mu * g / v_safe)

    beta_scale = (mu / float(sideslip["reference_mu"])).pow(float(sideslip["friction_exponent"]))
    beta_scale = beta_scale.clamp(float(sideslip["scale_min"]), float(sideslip["scale_max"]))
    beta_limit = math.radians(float(sideslip["base_limit_deg"])) * beta_scale

    roll_mu_scale = (mu / float(roll["reference_mu"])).pow(float(roll["friction_exponent"]))
    roll_mu_scale = roll_mu_scale.clamp(float(roll["scale_min"]), float(roll["scale_max"]))
    payload_ratio = payload / float(roll["payload_reference_kg"])
    payload_scale = 1.0 / (1.0 + float(roll["payload_sensitivity"]) * payload_ratio)
    roll_limit = math.radians(float(roll["base_limit_deg"])) * roll_mu_scale * payload_scale

    ltr_limit = vx.new_tensor(float(section["ltr_limit"]))
    return torch.stack((yaw_limit, beta_limit.expand_as(vx), roll_limit.expand_as(vx),
                        ltr_limit.expand_as(vx)), dim=-1)


def stability_residual(states: torch.Tensor, ltr: torch.Tensor,
                       payload_kg: torch.Tensor, road_mu: torch.Tensor,
                       cfg: dict) -> torch.Tensor:
    """Return stacked upper/lower residuals; every element must be <= 0."""
    if states.ndim != 2 or states.shape[-1] != 5:
        raise ValueError("K2 states must be [H+1,5] = [vx,vy,r,roll,roll_rate]")
    if ltr.ndim != 1 or ltr.shape[0] != states.shape[0]:
        raise ValueError("LTR must be [H+1]")
    future = states[1:]
    beta = torch.atan2(future[:, 1], future[:, 0])
    q = torch.stack((future[:, 2], beta, future[:, 3], ltr[1:]), dim=-1)
    bounds = stability_limits(future[:, 0], payload_kg, road_mu, cfg)
    upper = q - bounds
    lower = -q - bounds
    return torch.cat((upper.reshape(-1), lower.reshape(-1)))
