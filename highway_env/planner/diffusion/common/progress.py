"""Differentiable road-centerline arc-length progress for TRACE planner loss."""
from __future__ import annotations

import torch


def _closest_polyline_s(points: torch.Tensor, polyline_xy: torch.Tensor):
    """Project [B,2] points on the closest nondegenerate polyline segments.

    The discrete nearest-segment index is piecewise constant, while the
    distance along that segment stays differentiable wrt the query point.
    """
    a, b = polyline_xy[:, :-1, :], polyline_xy[:, 1:, :]
    d = b - a
    lengths2 = d.square().sum(-1)
    if not bool((lengths2.sum(-1) > 1e-5).all()):
        raise ValueError('Progress loss needs nondegenerate target_lane_polyline')
    segment_len = lengths2.clamp_min(1e-12).sqrt()
    t = ((points[:, None, :] - a) * d).sum(-1) / lengths2.clamp_min(1e-12)
    t = t.clamp(0., 1.)
    closest = a + t[..., None] * d
    dist2 = (points[:, None, :] - closest).square().sum(-1)
    # Degenerate segments cannot win nearest selection.
    dist2 = dist2.masked_fill(lengths2 <= 1e-10, float('inf'))
    idx = dist2.argmin(-1)
    prefix = torch.cat((torch.zeros_like(segment_len[:, :1]),
                        torch.cumsum(segment_len, dim=1)[:, :-1]), dim=1)
    return (prefix + t * segment_len).gather(1, idx[:, None]).squeeze(-1)


def centerline_progress(terminal_pose: torch.Tensor, target_lane_polyline: torch.Tensor):
    """Signed path progress in metres from ego origin to terminal local pose.

    Args:
        terminal_pose: [B,3] local x, y, yaw.
        target_lane_polyline: [B,P,10] with point xy in columns 0:2.
    """
    if terminal_pose.ndim != 2 or terminal_pose.shape[-1] != 3:
        raise ValueError('terminal_pose must be [B,3]')
    if target_lane_polyline.ndim != 3 or target_lane_polyline.shape[0] != len(terminal_pose) or target_lane_polyline.shape[-1] < 2:
        raise ValueError('target_lane_polyline must be [B,P,D]')
    points = target_lane_polyline[..., :2]
    origin = torch.zeros_like(terminal_pose[:, :2])
    return (_closest_polyline_s(terminal_pose[:, :2], points)
            - _closest_polyline_s(origin, points))
