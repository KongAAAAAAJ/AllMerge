from __future__ import annotations

# DECOUPLED_TASK_CONSTRAINT_V2_SEMANTIC

from typing import Any, Mapping

import numpy as np
import torch

from highway_env.planner.diffusion.trajectory_mode_reward.config import (
    TrajectoryModeRewardConfig,
)
from .reward_adapter import w4_to_grpo_rewards

# Name # All reward results are produced by the single canonical V6 objective.
SUPPORTED_TASK_REWARDS = ("unified_v6",)


def _context_reward_config(context: Any) -> TrajectoryModeRewardConfig:
    config = None
    if context is not None:
        if isinstance(context, Mapping):
            config = context.get("config")
        else:
            config = getattr(context, "config", None)
    if config is None:
        return TrajectoryModeRewardConfig()
    if not isinstance(config, TrajectoryModeRewardConfig):
        raise TypeError(
            "task reward requires TrajectoryModeRewardConfig when context supplies config"
        )
    return config


def task_reward_from_result(
    result: Any,
    *,
    context: Any,
    device: torch.device,
    dtype: torch.dtype,
    reward_type: str,
) -> torch.Tensor:
    """Read the sole official V6 score from production Result [3,10,N]."""
    if reward_type not in SUPPORTED_TASK_REWARDS:
        raise ValueError(f"Only V6 reward is supported, got {reward_type!r}")
    config = _context_reward_config(context)
    if float(config.task_centerline_weight) > 0.:
        valid = result.components.get('centerline_valid')
        if valid is None:
            raise RuntimeError('V6 result missing centerline_valid')
        valid = np.asarray(valid)
        mask = np.asarray(result.valid_mode_mask)[:, :, None]
        if np.any((valid < 0.5) & mask):
            raise RuntimeError('V6 target-lane centerline reference missing for a valid mode')
    return w4_to_grpo_rewards(result.task_rewards, device=device, dtype=dtype)
