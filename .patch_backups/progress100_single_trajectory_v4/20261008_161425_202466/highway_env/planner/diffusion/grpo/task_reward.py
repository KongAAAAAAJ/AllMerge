from __future__ import annotations

# DECOUPLED_TASK_CONSTRAINT_V2_SEMANTIC

from typing import Any, Mapping

import numpy as np
import torch

from highway_env.planner.diffusion.trajectory_mode_reward.config import (
    TrajectoryModeRewardConfig,
)
from .reward_adapter import w4_to_grpo_rewards

SUPPORTED_TASK_REWARDS = ("legacy_w4", "progress_comfort")


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


def _component_to_grpo(
    value: Any, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[:2] != (3, 10):
        raise ValueError(f"W4 component must be [3,10,N], got {array.shape}")
    return torch.as_tensor(
        np.ascontiguousarray(np.transpose(array, (0, 2, 1))),
        device=device,
        dtype=dtype,
    )


def task_reward_from_w4_result(
    result: Any,
    *,
    context: Any,
    device: torch.device,
    dtype: torch.dtype,
    reward_type: str,
) -> torch.Tensor:
    """Return the GRPO task objective, deliberately separated from constraints.

    legacy_w4 preserves the historical reward exactly.

    progress_comfort keeps only efficiency and smoothness:
      progress_weight * progress_score - comfort_weight * comfort_penalty

    collision / road / TTC / background-gap / teammate-gap are deliberately
    excluded and are handled only by the shared ConstraintEvaluator.
    """
    if reward_type not in SUPPORTED_TASK_REWARDS:
        raise ValueError(
            f"unsupported task reward {reward_type!r}; supported={SUPPORTED_TASK_REWARDS}"
        )
    if reward_type == "legacy_w4":
        return w4_to_grpo_rewards(result.rewards, device=device, dtype=dtype)

    config = _context_reward_config(context)
    progress = _component_to_grpo(
        result.components["progress_score"], device=device, dtype=dtype
    )
    smoothness = _component_to_grpo(
        result.components["smoothness_penalty"], device=device, dtype=dtype
    )
    kinematic = _component_to_grpo(
        result.components["kinematic_score"], device=device, dtype=dtype
    )
    return (
        float(config.task_progress_weight) * progress
        - float(config.task_comfort_weight) * smoothness
        + float(config.task_kinematic_weight) * kinematic
    )


def legacy_w4_reward_from_result(
    result: Any, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return w4_to_grpo_rewards(result.rewards, device=device, dtype=dtype)
