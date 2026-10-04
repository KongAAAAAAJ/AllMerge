from __future__ import annotations

# LAGRANGIAN_CONSTRAINED_GRPO_BASELINE_V2_SEMANTIC

from dataclasses import dataclass
from typing import Dict

import torch

from .constraints import ConstraintBatch
from .objective import group_relative_advantage

def _expanded_valid_mask(valid_mask: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if valid_mask is None:
        base = torch.ones(
            target.shape[0], target.shape[2], dtype=torch.bool, device=target.device
        )
    else:
        base = valid_mask.to(device=target.device, dtype=torch.bool)
        if tuple(base.shape) != (target.shape[0], target.shape[2]):
            raise ValueError("valid_mode_mask must be [vehicle,mode]")
    expanded = base[:, None, :]
    while expanded.ndim < target.ndim:
        expanded = expanded.unsqueeze(-1)
    return expanded.expand_as(target)

def _masked_mean_per_constraint(value: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError("constraint tensor must be [vehicle,group,mode,constraint]")
    mask = _expanded_valid_mask(valid_mask, value)
    denom = mask.to(value.dtype).sum(dim=(0, 1, 2)).clamp_min(1.0)
    return (value * mask.to(value.dtype)).sum(dim=(0, 1, 2)) / denom

def _masked_scalar_mean(value: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError("value must be [vehicle,group,mode]")
    mask = _expanded_valid_mask(valid_mask, value)
    denom = mask.sum().clamp_min(1)
    return (value * mask.to(value.dtype)).sum() / denom

@dataclass
class LagrangianConstraintConfig:
    dual_lr: float = 0.05
    lambda_init: float = 0.0
    lambda_max: float = 20.0

    def validate(self) -> None:
        if self.dual_lr < 0.0:
            raise ValueError("lagrangian dual_lr must be >= 0")
        if self.lambda_init < 0.0:
            raise ValueError("lagrangian lambda_init must be >= 0")
        if self.lambda_max <= 0.0:
            raise ValueError("lagrangian lambda_max must be > 0")
        if self.lambda_init > self.lambda_max:
            raise ValueError("lagrangian lambda_init cannot exceed lambda_max")

class LagrangianConstraintStrategy:
    def __init__(
        self,
        names: tuple[str, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
        config: LagrangianConstraintConfig | None = None,
    ) -> None:
        self.names = tuple(names)
        self.config = config or LagrangianConstraintConfig()
        self.config.validate()
        self.lambdas = torch.full(
            (len(self.names),),
            float(self.config.lambda_init),
            device=device,
            dtype=dtype,
        )

    def compute(
        self,
        task_rewards: torch.Tensor,
        constraints: ConstraintBatch,
        *,
        valid_mask: torch.Tensor | None,
        advantage_eps: float,
        update_dual: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        if constraints.names != self.names:
            raise ValueError(
                f"constraint names changed: strategy={self.names} batch={constraints.names}"
            )
        if constraints.violation.shape[:3] != task_rewards.shape:
            raise ValueError("constraint/task reward shapes do not match")

        lambdas_before = self.lambdas.detach().clone()
        penalty = torch.sum(
            constraints.violation * lambdas_before.view(1, 1, 1, -1),
            dim=-1,
        )
        effective_rewards = task_rewards - penalty
        advantages = group_relative_advantage(
            effective_rewards,
            eps=advantage_eps,
            valid_mask=valid_mask,
        )

        signed_mean = _masked_mean_per_constraint(constraints.signed_residual, valid_mask)
        violation_mean = _masked_mean_per_constraint(constraints.violation, valid_mask)
        violation_fraction = _masked_mean_per_constraint(
            (constraints.violation > 0.0).to(task_rewards.dtype), valid_mask
        )

        if update_dual:
            with torch.no_grad():
                self.lambdas.add_(float(self.config.dual_lr) * signed_mean)
                self.lambdas.clamp_(0.0, float(self.config.lambda_max))

        metrics: Dict[str, float] = {
            "constraint/feasible_fraction": float(
                _masked_scalar_mean(
                    constraints.feasible_mask.to(task_rewards.dtype), valid_mask
                ).detach()
            ),
            "constraint/max_violation_mean": float(
                _masked_scalar_mean(constraints.violation.amax(dim=-1), valid_mask).detach()
            ),
            "lagrangian/penalty_mean": float(
                _masked_scalar_mean(penalty, valid_mask).detach()
            ),
            "lagrangian/effective_reward_mean": float(
                _masked_scalar_mean(effective_rewards, valid_mask).detach()
            ),
        }
        for index, name in enumerate(self.names):
            metrics[f"constraint/{name}_signed_mean"] = float(signed_mean[index].detach())
            metrics[f"constraint/{name}_violation_mean"] = float(violation_mean[index].detach())
            metrics[f"constraint/{name}_violation_fraction"] = float(
                violation_fraction[index].detach()
            )
            metrics[f"lagrangian/lambda_{name}_before"] = float(
                lambdas_before[index].detach()
            )
            metrics[f"lagrangian/lambda_{name}"] = float(self.lambdas[index].detach())
        return effective_rewards, advantages, metrics

    def state_dict(self) -> dict:
        return {
            "names": self.names,
            "lambdas": self.lambdas.detach().cpu(),
            "config": {
                "dual_lr": self.config.dual_lr,
                "lambda_init": self.config.lambda_init,
                "lambda_max": self.config.lambda_max,
            },
        }
