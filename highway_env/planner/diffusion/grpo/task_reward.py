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

    progress_comfort uses the unified 10Hz spline objective:
      task_progress_weight * progress_score - task_comfort_weight * comfort_penalty
      + task_road_weight * road_boundary_reward
      - task_curvature_weight * curvature_penalty
      - task_centerline_weight * centerline_penalty

    Road BOUNDARY continuous reward is part of progress_comfort; collision,
    TTC and vehicle gap remain hard constraint / safety signals.
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
    comfort = _component_to_grpo(
        result.components["comfort_penalty"], device=device, dtype=dtype
    )
    if "road_boundary_reward" not in result.components:
        raise RuntimeError("Current progress_comfort scorer missing road_boundary_reward")
    road = _component_to_grpo(
        result.components["road_boundary_reward"], device=device, dtype=dtype
    )
    if "curvature_penalty" not in result.components:
        raise RuntimeError("V5 progress_comfort scorer missing curvature_penalty")
    curvature = _component_to_grpo(
        result.components["curvature_penalty"], device=device, dtype=dtype
    )
    centerline = torch.zeros_like(progress)
    if float(config.task_centerline_weight) > 0.:
        if "centerline_penalty" not in result.components or "centerline_valid" not in result.components:
            raise RuntimeError('V5.2 scorer missing target-lane centerline components')
        centerline = _component_to_grpo(result.components['centerline_penalty'], device=device, dtype=dtype)
        valid = _component_to_grpo(result.components['centerline_valid'],device=device,dtype=dtype)
        # Ignore invalid mode slots but NEVER silently omit missing target geometry
        # on any mode whose reward is actually used.
        mode_mask = torch.as_tensor(result.valid_mode_mask,device=device,dtype=torch.bool)[:,None,:].expand_as(valid)
        if bool(((valid < .5) & mode_mask).any()):
            raise RuntimeError('V5.2 target-lane centerline reference is missing for a valid mode')
    return (
        float(config.task_progress_weight) * progress
        - float(config.task_comfort_weight) * comfort
        + float(config.task_road_weight) * road
        - float(config.task_curvature_weight) * curvature
        - float(config.task_centerline_weight) * centerline
    )


def legacy_w4_reward_from_result(
    result: Any, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return w4_to_grpo_rewards(result.rewards, device=device, dtype=dtype)
