from __future__ import annotations

# LAGRANGIAN_CONSTRAINED_GRPO_BASELINE_V2_SEMANTIC

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from highway_env.planner.diffusion.trajectory_mode_reward.config import (
    TrajectoryModeRewardConfig,
)

SUPPORTED_CONSTRAINTS = (
    "collision",
    "road",
    "ttc",
    "background_gap",
    "teammate_gap",
)

@dataclass(frozen=True)
class ConstraintBatch:
    names: tuple[str, ...]
    signed_residual: torch.Tensor
    violation: torch.Tensor
    feasible_mask: torch.Tensor
    reference_signed_residual: torch.Tensor
    reference_violation: torch.Tensor
    reference_feasible_mask: torch.Tensor

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
            "GRPO constraint evaluation requires TrajectoryModeRewardConfig "
            "when reward context supplies a config"
        )
    return config

def _current_to_grpo(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[:2] != (3, 10):
        raise ValueError(f"expected current W4 array [3,10,G], got {array.shape}")
    return torch.as_tensor(
        np.ascontiguousarray(np.transpose(array, (0, 2, 1))),
        device=device,
        dtype=dtype,
    )

def _reference_to_grpo(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    array = np.asarray(value)
    if array.shape != (3, 10):
        raise ValueError(f"expected reference W4 array [3,10], got {array.shape}")
    return torch.as_tensor(array, device=device, dtype=dtype)

def _clip_signed(value: torch.Tensor, cap: float) -> torch.Tensor:
    return value.clamp(min=-1.0, max=float(cap))

def _constraint_residuals(
    *,
    result: Any,
    config: TrajectoryModeRewardConfig,
    names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    residual_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_components = result.components
    reference_components = result.pretrain_components
    current_values: list[torch.Tensor] = []
    reference_values: list[torch.Tensor] = []

    for name in names:
        if name == "collision":
            current = _current_to_grpo(result.collision, device=device, dtype=dtype)
            reference = _reference_to_grpo(
                result.pretrain_collision, device=device, dtype=dtype
            )
        elif name == "road":
            current_margin = _current_to_grpo(
                current_components["minimum_road_margin_m"], device=device, dtype=dtype
            )
            reference_margin = _reference_to_grpo(
                reference_components["minimum_road_margin_m"], device=device, dtype=dtype
            )
            scale = float(config.road_outside_scale_m)
            current = -current_margin / scale
            reference = -reference_margin / scale
        elif name == "ttc":
            current_ttc = _current_to_grpo(
                current_components["minimum_ttc_s"], device=device, dtype=dtype
            )
            reference_ttc = _reference_to_grpo(
                reference_components["minimum_ttc_s"], device=device, dtype=dtype
            )
            threshold = float(config.ttc_warning_s)
            current = (threshold - current_ttc) / threshold
            reference = (threshold - reference_ttc) / threshold
        elif name == "background_gap":
            current_gap = _current_to_grpo(
                current_components["minimum_background_gap_m"], device=device, dtype=dtype
            )
            reference_gap = _reference_to_grpo(
                reference_components["minimum_background_gap_m"], device=device, dtype=dtype
            )
            threshold = float(config.background_safe_gap_m)
            current = (threshold - current_gap) / threshold
            reference = (threshold - reference_gap) / threshold
        elif name == "teammate_gap":
            current_gap = _current_to_grpo(
                current_components["minimum_teammate_gap_m"], device=device, dtype=dtype
            )
            reference_gap = _reference_to_grpo(
                reference_components["minimum_teammate_gap_m"], device=device, dtype=dtype
            )
            threshold = float(config.platoon_safe_gap_m)
            current = (threshold - current_gap) / threshold
            reference = (threshold - reference_gap) / threshold
        else:
            raise ValueError(
                f"unsupported GRPO constraint {name!r}; supported={SUPPORTED_CONSTRAINTS}"
            )
        current_values.append(_clip_signed(current, residual_cap))
        reference_values.append(_clip_signed(reference, residual_cap))

    return torch.stack(current_values, dim=-1), torch.stack(reference_values, dim=-1)

def evaluate_w4_constraints(
    result: Any,
    *,
    context: Any,
    device: torch.device,
    dtype: torch.dtype,
    names: Sequence[str] = SUPPORTED_CONSTRAINTS,
    residual_cap: float = 5.0,
) -> ConstraintBatch:
    names = tuple(str(name) for name in names)
    if not names:
        raise ValueError("at least one GRPO constraint is required")
    unknown = sorted(set(names) - set(SUPPORTED_CONSTRAINTS))
    if unknown:
        raise ValueError(
            f"unsupported GRPO constraints {unknown}; supported={SUPPORTED_CONSTRAINTS}"
        )
    if len(set(names)) != len(names):
        raise ValueError("GRPO constraint names must be unique")
    if not np.isfinite(float(residual_cap)) or float(residual_cap) <= 0.0:
        raise ValueError("constraint residual_cap must be positive and finite")

    config = _context_reward_config(context)
    current, reference = _constraint_residuals(
        result=result,
        config=config,
        names=names,
        device=device,
        dtype=dtype,
        residual_cap=float(residual_cap),
    )
    violation = torch.relu(current)
    reference_violation = torch.relu(reference)
    return ConstraintBatch(
        names=names,
        signed_residual=current,
        violation=violation,
        feasible_mask=(violation <= 0.0).all(dim=-1),
        reference_signed_residual=reference,
        reference_violation=reference_violation,
        reference_feasible_mask=(reference_violation <= 0.0).all(dim=-1),
    )
