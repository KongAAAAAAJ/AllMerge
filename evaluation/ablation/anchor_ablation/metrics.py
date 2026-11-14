from __future__ import annotations

from typing import Dict, Optional

import torch


def geometry_valid_mask(strategy: str, features: dict, targets: Optional[dict], xy: torch.Tensor) -> torch.Tensor:
    """Return geometry availability, never reusing A1's traffic-filtered mask.

    A1 stores target_mode_geometry_valid separately in expert-shard diagnostics.
    If that diagnostic is absent, zero-padded coarse trajectories are used as a
    conservative geometry-availability fallback. A0/A2 manager masks already
    represent candidate availability rather than collision safety.
    """
    strategy = str(strategy).lower()
    if strategy == "dynamic":
        diagnostics = (targets or {}).get("diagnostics", {})
        geometry = diagnostics.get("target_mode_geometry_valid") if isinstance(diagnostics, dict) else None
        if geometry is not None:
            return geometry.to(device=xy.device, dtype=torch.bool)
        # A1's mode_valid_mask mixes geometry and traffic validity. Do not use it.
        nonzero_xy = torch.linalg.vector_norm(xy, dim=-1).sum(dim=-1) > 1.0e-5
        nonzero_control = None
        controls = features.get("coarse_controls")
        if controls is not None:
            nonzero_control = torch.linalg.vector_norm(controls.to(xy), dim=-1).sum(dim=-1) > 1.0e-5
        return nonzero_xy if nonzero_control is None else (nonzero_xy | nonzero_control)
    mask = features.get("mode_valid_mask")
    if mask is None:
        return torch.ones(xy.shape[:2], dtype=torch.bool, device=xy.device)
    return mask.to(device=xy.device, dtype=torch.bool)


def _pairwise_values(x: torch.Tensor, valid: torch.Tensor, *, normalize: Optional[torch.Tensor] = None):
    if normalize is not None:
        x = x / normalize.to(device=x.device, dtype=x.dtype)
    # x: [B,M,T,D]. Pairwise mean pointwise L2 distance [B,M,M].
    delta = x[:, :, None] - x[:, None, :]
    return torch.linalg.vector_norm(delta, dim=-1).mean(dim=-1)


def compute_anchor_metrics(
    *,
    xy: torch.Tensor,
    controls: torch.Tensor,
    expert_xy: torch.Tensor,
    valid: torch.Tensor,
    safety: Dict[str, torch.Tensor],
    trajectory_weight: float = 1.0,
    control_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute paper-table Anchor metrics for a batch.

    minADE uses the best valid anchor per scene. Safe ratio uses the common
    A2 teacher hard-safety definition for every method. The paper table labels
    Diversity in metres, so its reported value is mean pairwise trajectory ADE
    in physical coordinates. The paper's joint normalized trajectory/control
    distance is also retained as an auxiliary diagnostic matching A2 selection.
    """
    if xy.ndim != 4 or xy.shape[-2:] != (8, 2):
        raise ValueError(f"Anchor trajectories must be [B,M,8,2], got {tuple(xy.shape)}")
    if controls.shape[:2] != xy.shape[:2] or controls.shape[-2:] != (40, 2):
        raise ValueError("Anchor controls must be [B,M,40,2] and match trajectory candidates")
    if expert_xy.shape != (xy.shape[0], 8, 2):
        raise ValueError(f"Expert reference must be [B,8,2], got {tuple(expert_xy.shape)}")
    if valid.shape != xy.shape[:2] or not valid.bool().any(dim=1).all():
        raise ValueError("Each scene must have at least one geometry-valid Anchor")

    valid = valid.bool()
    ade = torch.linalg.vector_norm(xy - expert_xy[:, None], dim=-1).mean(dim=-1)
    minade = ade.masked_fill(~valid, torch.inf).min(dim=1).values

    hard_safe = safety["hard_safe"].bool() & valid
    safe_count = hard_safe.sum(dim=1)
    valid_count = valid.sum(dim=1)
    safe_ratio = safe_count.to(xy.dtype) / valid_count.clamp_min(1).to(xy.dtype)

    pair_traj = _pairwise_values(xy, valid)
    pair_ctrl = _pairwise_values(
        controls,
        valid,
        normalize=controls.new_tensor([2.5, 0.08]),
    )
    pair_traj_normalized = _pairwise_values(
        xy,
        valid,
        normalize=xy.new_tensor([120.0, 24.0]),
    )
    m = xy.shape[1]
    upper = torch.triu(torch.ones((m, m), dtype=torch.bool, device=xy.device), diagonal=1)
    pair_valid = valid[:, :, None] & valid[:, None, :] & upper[None]
    pair_count = pair_valid.sum(dim=(1, 2))
    denom = pair_count.clamp_min(1).to(xy.dtype)

    diversity_m = (pair_traj * pair_valid).sum(dim=(1, 2)) / denom
    joint_pair = trajectory_weight * pair_traj_normalized + control_weight * pair_ctrl
    joint_diversity = (joint_pair * pair_valid).sum(dim=(1, 2)) / denom
    control_diversity = (pair_ctrl * pair_valid).sum(dim=(1, 2)) / denom
    diversity_m = torch.where(pair_count > 0, diversity_m, torch.full_like(diversity_m, float("nan")))
    joint_diversity = torch.where(pair_count > 0, joint_diversity, torch.full_like(joint_diversity, float("nan")))
    control_diversity = torch.where(pair_count > 0, control_diversity, torch.full_like(control_diversity, float("nan")))

    return {
        "minade_m": minade,
        "safe_ratio": safe_ratio,
        "safe_count": safe_count,
        "valid_anchor_count": valid_count,
        "diversity_m": diversity_m,
        "control_diversity_normalized": control_diversity,
        "joint_diversity_normalized": joint_diversity,
        "per_anchor_ade_m": ade,
        "per_anchor_safe": hard_safe,
    }
