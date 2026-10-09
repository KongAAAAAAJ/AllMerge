"""Reward-aligned road union score, shared with GuideDistill Road Guidance.

Always uses the same 10Hz cubic spline and frozen planning-time poses.
"""
from __future__ import annotations
import weakref
import numpy as np
import torch

_ROAD_FIELDS=weakref.WeakKeyDictionary()

def production_road_reward(trajectories, pose, road, config):
    from highway_env.planner.diffusion.guide_distill.road_boundary import RoadBoundaryField
    try:
        field=_ROAD_FIELDS.get(road)
    except TypeError:
        field=None
    if field is None:
        field=RoadBoundaryField.from_road(road)
        try:
            _ROAD_FIELDS[road]=field
        except TypeError:
            pass
    with torch.no_grad():
        xy=torch.as_tensor(np.asarray(trajectories),dtype=torch.float64)
        val=field.components(xy,pose,config)['road']
    result=val.detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite production road boundary reward')
    return result
