from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GRPOObjectiveResult:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_mean: torch.Tensor


def group_relative_advantage(
    rewards: torch.Tensor,
    *,
    group_dim: int = 1,
    eps: float = 1e-6,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standard GRPO group normalization without a critic.

    Expected reward layout is ``[B, G, M]``: batch, group, trajectory mode.
    Invalid modes can be masked with ``valid_mask[B, M]``.
    """
    if rewards.ndim != 3:
        raise ValueError(f"rewards must be [B,G,M], got {tuple(rewards.shape)}")
    mean = rewards.mean(dim=group_dim, keepdim=True)
    std = rewards.std(dim=group_dim, unbiased=False, keepdim=True)
    adv = (rewards - mean) / (std + eps)
    if valid_mask is not None:
        if valid_mask.shape != (rewards.shape[0], rewards.shape[2]):
            raise ValueError("valid_mask must have shape [B,M]")
        adv = adv.masked_fill(~valid_mask[:, None, :], 0.0)
    return adv


# GRPO RANK ADVANTAGE D V2
def rank_group_advantage(
    rewards: torch.Tensor,
    *,
    group_dim: int = 1,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Tie-aware rank advantage mapped to [-1, 1] inside each GRPO group.

    Exact reward ties receive deterministic average ranks. The result is then
    explicitly centered along ``group_dim``. For the normal AllMerge layout
    ``[B, G, M]``, ``valid_mask`` is ``[B, M]`` and invalid modes are zeroed.
    """
    if rewards.ndim != 3:
        raise ValueError(f"rewards must be [B,G,M], got {tuple(rewards.shape)}")
    if group_dim < 0:
        group_dim += rewards.ndim
    if group_dim < 0 or group_dim >= rewards.ndim:
        raise ValueError(f"invalid group_dim={group_dim}")

    group_size = int(rewards.shape[group_dim])
    if group_size <= 1:
        adv = torch.zeros_like(rewards)
    else:
        moved = rewards.movedim(group_dim, -1)
        flat = moved.reshape(-1, group_size)

        # Average rank for ties:
        # rank_i = count(R_j < R_i) + 0.5 * (count(R_j == R_i) - 1)
        less = (flat[:, None, :] < flat[:, :, None]).sum(dim=-1).to(flat.dtype)
        equal = (flat[:, None, :] == flat[:, :, None]).sum(dim=-1).to(flat.dtype)
        ranks = less + 0.5 * (equal - 1.0)

        adv_flat = 2.0 * ranks / float(group_size - 1) - 1.0
        adv_flat = adv_flat - adv_flat.mean(dim=-1, keepdim=True)
        adv = adv_flat.reshape(moved.shape).movedim(-1, group_dim)

    if valid_mask is not None:
        if valid_mask.shape != (rewards.shape[0], rewards.shape[2]):
            raise ValueError("valid_mask must have shape [B,M]")
        adv = adv.masked_fill(~valid_mask[:, None, :], 0.0)
    return adv

def _masked_mean(value: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return value.mean()
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(2)
    expanded = expanded.expand_as(value)
    denom = expanded.sum().clamp_min(1)
    return (value * expanded.to(value.dtype)).sum() / denom


def grpo_clipped_objective(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    valid_mask: Optional[torch.Tensor] = None,
) -> GRPOObjectiveResult:
    """PPO/GRPO clipped surrogate over diffusion transitions.

    ``new_log_prob`` and ``old_log_prob`` are ``[B,G,S,M]``; advantages are
    ``[B,G,M]`` and are shared across the S denoising decisions.
    """
    if new_log_prob.shape != old_log_prob.shape or new_log_prob.ndim != 4:
        raise ValueError("new/old log_prob must share shape [B,G,S,M]")
    if advantages.shape != (new_log_prob.shape[0], new_log_prob.shape[1], new_log_prob.shape[3]):
        raise ValueError("advantages must have shape [B,G,M]")

    delta = new_log_prob - old_log_prob
    ratio = torch.exp(delta)
    adv = advantages.unsqueeze(2)
    unclipped = -adv * ratio
    clipped = -adv * ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    element_loss = torch.maximum(unclipped, clipped)

    step_mask = None if valid_mask is None else valid_mask[:, None, None, :]
    policy_loss = _masked_mean(element_loss, step_mask)
    approx_kl = _masked_mean(0.5 * delta.square(), step_mask)
    clip_fraction = _masked_mean((torch.abs(ratio - 1.0) > clip_eps).to(ratio.dtype), step_mask)
    ratio_mean = _masked_mean(ratio, step_mask)
    return GRPOObjectiveResult(
        loss=policy_loss,
        policy_loss=policy_loss,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        ratio_mean=ratio_mean,
    )
