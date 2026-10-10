"""V5 MATP-aligned curvature reward component, shared by W4 scorer and Guidance.

Three consecutive 10Hz clamped cubic spline positions and the 0.25 m
segment-validity rule match the frozen MATP W1 geometry contract. This module
DOES NOT modify MATP, nor does it add another Guidance direction.
"""
from __future__ import annotations
import torch


def curvature_from_dense(dense_xy: torch.Tensor, cfg):
    """Return signed 3-point curvature [...,39] and MATP-compatible valid mask."""
    prev = dense_xy[..., 1:-1, :] - dense_xy[..., :-2, :]
    following = dense_xy[..., 2:, :] - dense_xy[..., 1:-1, :]
    chord = dense_xy[..., 2:, :] - dense_xy[..., :-2, :]
    lp = torch.linalg.vector_norm(prev, dim=-1)
    ln = torch.linalg.vector_norm(following, dim=-1)
    lc = torch.linalg.vector_norm(chord, dim=-1)
    threshold = float(cfg.curvature_min_segment_m)
    valid = (lp >= threshold) & (ln >= threshold) & (lc >= threshold)
    cross = prev[..., 0] * following[..., 1] - prev[..., 1] * following[..., 0]
    raw = 2.0 * cross / (lp * ln * lc).clamp_min(1.0e-6)
    return torch.where(valid, raw, torch.zeros_like(raw)), valid


def _huber_positive(x: torch.Tensor, delta: float):
    # Classic Huber. At x=0 the derivative is zero and gradients are finite.
    d = float(delta)
    return torch.where(x <= d, 0.5 * x.square() / d, x - 0.5 * d)


def curvature_reward_components(dense_xy: torch.Tensor, cfg):
    """All returned tensors are batch-shaped, differentiable where appropriate.

    P = P_violation + alpha*P_peak + gamma*P_margin.
    Severity W1-inspired weighting remains differentiable so autograd matches
    the gradient of the SAME scalar reward used by production scoring.
    """
    curv, valid = curvature_from_dense(dense_xy, cfg)
    abs_c = curv.abs()
    limit = float(cfg.curvature_limit_m_inv)
    threshold = float(cfg.curvature_active_threshold_m_inv)
    violation = torch.relu(abs_c / limit - 1.0) * valid.to(abs_c.dtype)
    n = min(int(cfg.curvature_topk), abs_c.shape[-1])
    active = valid & (abs_c >= threshold)
    scores = torch.where(active, abs_c, torch.full_like(abs_c, -1.0e6))
    _, active_indices = torch.topk(scores, k=n, dim=-1)
    chosen = torch.gather(violation, dim=-1, index=active_indices)
    selected_mask = torch.gather(active, dim=-1, index=active_indices)
    positive = torch.where(selected_mask, chosen, torch.zeros_like(chosen))
    # MATP W1: 1+beta*violation/max_violation per trajectory.
    max_v = positive.amax(dim=-1, keepdim=True)
    normalized = torch.where(max_v > 1.0e-9,
                             positive / max_v.clamp_min(1.0e-9),
                             torch.zeros_like(positive))
    weights = (1.0 + float(cfg.curvature_matp_beta) * normalized) * selected_mask.to(abs_c.dtype)
    violation_loss = (_huber_positive(positive, cfg.curvature_huber_delta) * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    peak_loss = _huber_positive(violation.amax(dim=-1), cfg.curvature_huber_delta)
    # Weak, continuous anticipatory penalty over 0.015..0.020. No penalty below threshold.
    denom = limit - threshold
    margin = ((abs_c - threshold) / denom).clamp(0.,1.) * valid.to(abs_c.dtype)
    margin_loss = margin.square().sum(dim=-1) / valid.to(abs_c.dtype).sum(dim=-1).clamp_min(1.0)
    total = (violation_loss + float(cfg.task_curvature_peak_weight) * peak_loss
             + float(cfg.task_curvature_margin_weight) * margin_loss)
    max_abs = torch.where(valid, abs_c, torch.zeros_like(abs_c)).amax(dim=-1)
    return {
        'curvature_penalty': total,
        'curvature_violation_penalty': violation_loss,
        'curvature_peak_penalty': peak_loss,
        'curvature_margin_penalty': margin_loss,
        'curvature_max_abs': max_abs,
        'curvature_valid_points': valid.to(abs_c.dtype).sum(dim=-1),
    }
