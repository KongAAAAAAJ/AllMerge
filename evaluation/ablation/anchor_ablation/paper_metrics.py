from __future__ import annotations

from typing import Mapping

import torch


CONTROL_SCALE = (2.5, 0.08)


def geometry_valid_mask(strategy: str, features: Mapping, targets: Mapping | None,
                        xy: torch.Tensor) -> torch.Tensor:
    """Geometric availability mask; A1 traffic mode_valid_mask is intentionally ignored."""
    strategy = str(strategy).lower()
    finite = torch.isfinite(xy).all(dim=(-1, -2))
    if strategy in ("dynamic", "a1"):
        path_length = torch.linalg.vector_norm(xy[:, :, 1:] - xy[:, :, :-1], dim=-1).sum(dim=-1)
        return finite & (path_length > 0.5)
    mask = features.get("mode_valid_mask")
    if mask is None:
        return finite
    return finite & mask.to(device=xy.device, dtype=torch.bool)


def _pairwise_mean_distance(values: torch.Tensor, valid: torch.Tensor,
                            scale: torch.Tensor | None = None) -> torch.Tensor:
    if scale is not None:
        values = values / scale.to(device=values.device, dtype=values.dtype)
    pair = torch.linalg.vector_norm(values[:, :, None] - values[:, None, :], dim=-1).mean(dim=-1)
    count = values.shape[1]
    upper = torch.triu(torch.ones((count, count), dtype=torch.bool, device=values.device), diagonal=1)
    pair_mask = valid[:, :, None] & valid[:, None, :] & upper[None]
    denominator = pair_mask.sum(dim=(1, 2)).clamp_min(1).to(values.dtype)
    result = (pair * pair_mask).sum(dim=(1, 2)) / denominator
    return torch.where(pair_mask.sum(dim=(1, 2)) > 0, result,
                       torch.full_like(result, float("nan")))


def compute_anchor_metrics(*, xy: torch.Tensor, controls: torch.Tensor,
                           expert_xy: torch.Tensor, valid: torch.Tensor,
                           safety: Mapping[str, torch.Tensor],
                           expected_anchor_count: int = 10) -> dict[str, torch.Tensor]:
    """Compute minADE, Safe Ratio, trajectory diversity and control diversity."""
    if xy.ndim != 4 or xy.shape[-2:] != (8, 2):
        raise ValueError(f"Anchor trajectories must be [B,M,8,2], got {tuple(xy.shape)}")
    if controls.shape != (xy.shape[0], xy.shape[1], 40, 2):
        raise ValueError(f"Anchor controls must be [B,M,40,2], got {tuple(controls.shape)}")
    if xy.shape[1] != int(expected_anchor_count):
        raise ValueError(f"Expected {expected_anchor_count} anchors, got {xy.shape[1]}")
    if expert_xy.shape != (xy.shape[0], 8, 2) or valid.shape != xy.shape[:2]:
        raise ValueError("Expert trajectory or geometry-valid mask has an incompatible shape")
    valid = valid.bool()
    counts = valid.sum(dim=1)
    if not torch.all(counts == int(expected_anchor_count)):
        bad = torch.nonzero(counts != int(expected_anchor_count)).flatten().tolist()
        raise RuntimeError(
            f"Every evaluated scene must contain {expected_anchor_count} geometric anchors; "
            f"invalid batch indices={bad}, counts={counts.tolist()}. "
            "Do not mark zero-filled A1 slots as valid."
        )
    ade = torch.linalg.vector_norm(xy - expert_xy[:, None], dim=-1).mean(dim=-1)
    minade = ade.masked_fill(~valid, torch.inf).min(dim=1).values
    hard_safe = safety["hard_safe"].bool() & valid
    safe_count = hard_safe.sum(dim=1)
    safe_ratio = safe_count.to(xy.dtype) / counts.to(xy.dtype)
    trajectory_diversity = _pairwise_mean_distance(xy, valid)
    control_diversity = _pairwise_mean_distance(
        controls, valid, scale=controls.new_tensor(CONTROL_SCALE)
    )
    return {
        "anchor_minade_m": minade,
        "safe_ratio": safe_ratio,
        "safe_anchor_count": safe_count,
        "valid_anchor_count": counts,
        "trajectory_diversity_m": trajectory_diversity,
        "control_diversity_normalized": control_diversity,
        "per_anchor_ade_m": ade,
        "per_anchor_safe": hard_safe,
    }


def compute_planning_minade(candidate_xy: torch.Tensor, expert_xy: torch.Tensor,
                            valid_modes: torch.Tensor) -> torch.Tensor:
    """Best ADE among valid diffusion-generated execution trajectories."""
    if candidate_xy.ndim != 4 or candidate_xy.shape[-2:] != (8, 2):
        raise ValueError(f"Diffusion candidates must be [B,M,8,2], got {tuple(candidate_xy.shape)}")
    if expert_xy.shape != (candidate_xy.shape[0], 8, 2):
        raise ValueError("Expert trajectory must be [B,8,2]")
    if valid_modes.shape != candidate_xy.shape[:2] or not valid_modes.bool().any(dim=1).all():
        raise ValueError("Every scene needs at least one geometrically valid diffusion candidate")
    distance = torch.linalg.vector_norm(candidate_xy - expert_xy[:, None], dim=-1)
    ade = distance.mean(dim=-1)
    return ade.masked_fill(~valid_modes.bool(), torch.inf).min(dim=1).values
