"""Offline A2 teacher scores: constant-velocity collision + road compliance.

The scores are NOT Koopman dynamics or stability constraints. World state is
never used: all geometry is expressed in the observed initial ego frame.
"""
from __future__ import annotations

import torch


def anchor_safety_cost(features, xy, valid=None, *, collision_weight=1.0,
                       road_weight=1.0, ego_length_m=5.0, ego_width_m=2.0,
                       longitudinal_margin_m=1.0, lateral_margin_m=0.3,
                       dt=0.5):
    """Returns total/collision/road costs [B,M] and hard safety [B,M].

    xy is [B,M,T,2] and describes ego-relative future positions at t=dt..Tdt.
    Agent velocity features are relative; ego velocity is added back for the
    constant-velocity prediction in the fixed initial ego coordinate frame.
    Road compliance uses segment-wise distance to the union of known drivable
    lane centerline corridors. Invalid roadway metadata raises an error.
    """
    if xy.ndim != 4 or xy.shape[-1] != 2:
        raise ValueError('xy must be [B,M,T,2]')
    B, M, T, _ = xy.shape
    dtype, device = xy.dtype, xy.device
    agent = features['agent_states'].to(xy)
    agent_mask = features['agent_valid_mask'].bool()
    ego = features['ego_state'].to(xy)
    # Agent velocity is relative to ego; ego longitudinal and lateral speed
    # are in the initial vehicle coordinate frame.
    agent_vel = agent[..., 2:4] + ego[:, None, :2]
    times = torch.arange(1, T + 1, device=device, dtype=dtype) * dt
    agent_xy = agent[:, None, None, :, :2] + agent_vel[:, None, None] * times[None, None, :, None, None]
    # Pose tangent from origin to first point and central differences later.
    origin = torch.zeros_like(xy[:, :, :1])
    chain = torch.cat((origin, xy), dim=2)
    direction = chain[:, :, 2:] - chain[:, :, :-2]
    direction = torch.cat((chain[:, :, 1:2] - origin, direction), dim=2)
    heading = torch.atan2(direction[..., 1], direction[..., 0])
    relative = agent_xy - xy[:, :, :, None, :]
    c, s = heading.cos()[..., None], heading.sin()[..., None]
    along = (relative[..., 0] * c + relative[..., 1] * s).abs()
    across = (-relative[..., 0] * s + relative[..., 1] * c).abs()
    # Oriented agent-rectangle projection into ego tangent axes.
    agent_heading = torch.atan2(agent[..., 6], agent[..., 7])
    theta = agent_heading[:, None, None] - heading[..., None]
    cr, sr = theta.cos().abs(), theta.sin().abs()
    half_length = agent[:, None, None, :, 8].clamp_min(0.1) / 2
    half_width = agent[:, None, None, :, 9].clamp_min(0.1) / 2
    long_lim = ego_length_m / 2 + cr * half_length + sr * half_width + longitudinal_margin_m
    lat_lim = ego_width_m / 2 + sr * half_length + cr * half_width + lateral_margin_m
    penetration = torch.minimum((long_lim - along) / long_lim,
                                (lat_lim - across) / lat_lim).clamp_min(0)
    penetration = penetration.masked_fill(~agent_mask[:, None, None, :], 0)
    col = penetration.amax(dim=(2, 3))  # maximum predicted collision severity
    col_unsafe = penetration.amax(dim=(2, 3)) > 0

    maps = features['map_polylines'].to(xy)
    map_mask = features['map_valid_mask'].bool()
    # Only recognized drivable lanes are included; target-lane flag may
    # appear in addition to current/left/right flags.
    roadway = (maps[..., 6:10].amax(dim=-2).amax(dim=-1) > 0.5) & map_mask
    if not roadway.any(dim=1).all():
        raise ValueError('A2 supervision requires at least one valid drivable map polyline')
    segments_start = maps[:, None, None, :, :-1, :2]
    segments_end = maps[:, None, None, :, 1:, :2]
    delta = segments_end - segments_start
    rel = xy[:, :, :, None, None, :] - segments_start
    fraction = (rel * delta).sum(-1) / delta.square().sum(-1).clamp_min(1e-6)
    nearest = segments_start + fraction.clamp(0, 1)[..., None] * delta
    distance = (xy[:, :, :, None, None, :] - nearest).square().sum(-1).sqrt()
    distance = distance.amin(dim=-1)
    width = maps[:, None, None, :, 0, 4].clamp_min(0.1) / 2
    excess = (distance - width).clamp_min(0)
    excess = excess.masked_fill(~roadway[:, None, None, :], float('inf'))
    road_violation = excess.amin(dim=-1)  # [B,M,T]
    road = road_violation.square().mean(dim=-1)  # m^2
    road_unsafe = road_violation.amax(dim=-1) > 1e-4
    total = collision_weight * col + road_weight * road
    hard_safe = ~(col_unsafe | road_unsafe)
    if valid is not None:
        valid = valid.bool()
        total = total.masked_fill(~valid, 0.0)
        col = col.masked_fill(~valid, 0.0)
        road = road.masked_fill(~valid, 0.0)
        hard_safe = hard_safe & valid
    return {'total': total, 'collision': col, 'road': road, 'hard_safe': hard_safe}
