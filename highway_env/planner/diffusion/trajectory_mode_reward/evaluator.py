"""Shared standalone/GRPO entry points for trajectory-mode reward."""

from __future__ import annotations

from .config import TrajectoryModeRewardConfig
from .counterfactual import (
    TrajectoryModeCounterfactualReward,
)
from .results import TrajectoryModeRewardResult


def evaluate_candidates(
    env: object,
    candidates: object,
    frozen_all_mode_trajectories: object,
    frozen_argmax_joint_trajectories: object,
    valid_mode_mask: object,
    *,
    config: TrajectoryModeRewardConfig | None = None,
) -> TrajectoryModeRewardResult:
    """Evaluate [3,10,N,8,2] candidates with one shared reward pipeline."""
    scorer = TrajectoryModeCounterfactualReward(
        config=config
    )
    return scorer.score_counterfactuals(
        env=env,
        candidates=candidates,
        frozen_all_mode_trajectories=(
            frozen_all_mode_trajectories
        ),
        frozen_argmax_joint_trajectories=(
            frozen_argmax_joint_trajectories
        ),
        valid_mode_mask=valid_mode_mask,
    )
