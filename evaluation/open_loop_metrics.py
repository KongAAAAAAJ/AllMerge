from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class OpenLoopMetricBatch:
    selected_ade: torch.Tensor
    selected_fde: torch.Tensor
    min_ade_all: torch.Tensor
    min_fde_all: torch.Tensor
    min_ade_valid: torch.Tensor
    min_fde_valid: torch.Tensor
    raw_pred_mode: torch.Tensor
    selected_pred_mode: torch.Tensor
    target_mode: torch.Tensor
    raw_mode_correct: torch.Tensor
    selected_mode_correct: torch.Tensor
    target_mode_traffic_valid: torch.Tensor
    raw_mode_entropy: torch.Tensor
    selected_mode_entropy: torch.Tensor
    valid_mode_count: torch.Tensor
    target_assignment_ade: torch.Tensor


def _trajectory_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    point_error = torch.linalg.vector_norm(
        prediction - target,
        dim=-1,
    )
    return point_error.mean(dim=-1), point_error[..., -1]


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    return -(
        probabilities
        * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)


def compute_open_loop_metrics(
    *,
    selected_trajectory: torch.Tensor,
    trajectory_candidates: torch.Tensor,
    raw_logits: torch.Tensor,
    masked_logits: torch.Tensor,
    selected_mode_idx: torch.Tensor,
    target_trajectory: torch.Tensor,
    target_mode: torch.Tensor,
    target_assignment_ade: torch.Tensor,
    mode_valid_mask: torch.Tensor,
) -> OpenLoopMetricBatch:
    """
    W5 open-loop metrics for the 9cc8b71 AllMerge contract.

    Two best-of-K views are kept intentionally:
      - minADE/FDE@M_all: standard metric across all M generated branches;
      - minADE/FDE@valid: diagnostic metric restricted to inference-selectable
        branches after mode_valid_mask.

    Training uses raw logits, while infer_multimodal() selects from masked
    logits. Therefore both raw and selected mode diagnostics are reported.
    """
    if trajectory_candidates.ndim != 4:
        raise ValueError(
            "trajectory_candidates must be [B,M,T,2], got "
            f"{tuple(trajectory_candidates.shape)}"
        )
    if selected_trajectory.shape != target_trajectory.shape:
        raise ValueError(
            "selected_trajectory and target_trajectory shape mismatch: "
            f"{tuple(selected_trajectory.shape)} vs "
            f"{tuple(target_trajectory.shape)}"
        )

    selected_ade, selected_fde = _trajectory_errors(
        selected_trajectory,
        target_trajectory,
    )

    target_expanded = target_trajectory[:, None, :, :]
    candidate_ade, candidate_fde = _trajectory_errors(
        trajectory_candidates,
        target_expanded,
    )

    # Standard best-of-M candidate-set quality.
    min_ade_all = candidate_ade.min(dim=1).values
    min_fde_all = candidate_fde.min(dim=1).values

    # Actual inference-selectable candidate-set quality.
    valid = mode_valid_mask.bool()
    if valid.shape != candidate_ade.shape:
        raise ValueError(
            "mode_valid_mask shape mismatch: "
            f"{tuple(valid.shape)} vs expected "
            f"{tuple(candidate_ade.shape)}"
        )
    if not valid.any(dim=1).all():
        bad = torch.nonzero(
            ~valid.any(dim=1),
            as_tuple=False,
        ).flatten().tolist()
        raise RuntimeError(
            f"No valid inference mode for batch indices {bad}"
        )

    inf_ade = torch.full_like(candidate_ade, float("inf"))
    inf_fde = torch.full_like(candidate_fde, float("inf"))
    min_ade_valid = torch.where(
        valid,
        candidate_ade,
        inf_ade,
    ).min(dim=1).values
    min_fde_valid = torch.where(
        valid,
        candidate_fde,
        inf_fde,
    ).min(dim=1).values

    target_mode = target_mode.long()
    raw_pred_mode = raw_logits.argmax(dim=-1)
    selected_pred_mode = selected_mode_idx.long()

    target_mode_traffic_valid = torch.gather(
        valid,
        dim=1,
        index=target_mode[:, None],
    ).squeeze(1)

    return OpenLoopMetricBatch(
        selected_ade=selected_ade,
        selected_fde=selected_fde,
        min_ade_all=min_ade_all,
        min_fde_all=min_fde_all,
        min_ade_valid=min_ade_valid,
        min_fde_valid=min_fde_valid,
        raw_pred_mode=raw_pred_mode,
        selected_pred_mode=selected_pred_mode,
        target_mode=target_mode,
        raw_mode_correct=raw_pred_mode.eq(target_mode),
        selected_mode_correct=selected_pred_mode.eq(target_mode),
        target_mode_traffic_valid=target_mode_traffic_valid,
        raw_mode_entropy=_entropy_from_logits(raw_logits),
        selected_mode_entropy=_entropy_from_logits(masked_logits),
        valid_mode_count=valid.sum(dim=1),
        target_assignment_ade=target_assignment_ade,
    )


def finite_metric_check(metrics: OpenLoopMetricBatch) -> None:
    values: Dict[str, torch.Tensor] = {
        "selected_ADE": metrics.selected_ade,
        "selected_FDE": metrics.selected_fde,
        "minADE_all": metrics.min_ade_all,
        "minFDE_all": metrics.min_fde_all,
        "minADE_valid": metrics.min_ade_valid,
        "minFDE_valid": metrics.min_fde_valid,
        "raw_mode_entropy": metrics.raw_mode_entropy,
        "selected_mode_entropy": metrics.selected_mode_entropy,
        "target_assignment_ADE": metrics.target_assignment_ade,
    }
    bad = [
        name
        for name, value in values.items()
        if not torch.isfinite(value).all()
    ]
    if bad:
        raise RuntimeError(
            f"Open-loop metrics contain NaN/Inf: {bad}"
        )
