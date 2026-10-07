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

# ACTIVE_CONSTRAINT_GRPO_V1
@dataclass
class ActiveConstraintConfig:
    feasible_fraction: float = 0.50
    cvar_alpha: float = 0.25
    temperature: float = 0.20
    support_gate: bool = False
    support_margin: float = 1e-3

    def validate(self) -> None:
        if not 0.0 <= self.feasible_fraction <= 1.0:
            raise ValueError("active feasible_fraction must be in [0,1]")
        if not 0.0 < self.cvar_alpha <= 1.0:
            raise ValueError("active cvar_alpha must be in (0,1]")
        if self.temperature <= 0.0:
            raise ValueError("active temperature must be > 0")
        if self.support_margin < 0.0:
            raise ValueError("active support_margin must be >= 0")


def _valid_vehicle_mode_mask(
    valid_mask: torch.Tensor | None,
    task_rewards: torch.Tensor,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(
            task_rewards.shape[0], task_rewards.shape[2],
            dtype=torch.bool, device=task_rewards.device,
        )
    result = valid_mask.to(device=task_rewards.device, dtype=torch.bool)
    if tuple(result.shape) != (task_rewards.shape[0], task_rewards.shape[2]):
        raise ValueError("valid_mode_mask must be [vehicle,mode]")
    return result


def _cvar_group_severity(violation: torch.Tensor, alpha: float) -> torch.Tensor:
    """Upper-tail CVaR over GRPO group dimension.

    Input ``[vehicle, group, mode, constraint]`` -> ``[vehicle, mode, constraint]``.
    """
    if violation.ndim != 4:
        raise ValueError("violation must be [vehicle,group,mode,constraint]")
    group_size = int(violation.shape[1])
    topk = max(1, min(group_size, int(torch.ceil(torch.tensor(alpha * group_size)).item())))
    values = torch.topk(violation, k=topk, dim=1, largest=True, sorted=False).values
    return values.mean(dim=1)


def _vm_masked_mean(value: torch.Tensor, vm_mask: torch.Tensor) -> torch.Tensor:
    expanded = vm_mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value)
    denom = expanded.sum().clamp_min(1)
    return (value * expanded.to(value.dtype)).sum() / denom


class ActiveConstraintStrategy:
    """Feasibility-restoration GRPO with hard or soft active constraints.

    Each vehicle/mode owns one GRPO group.  A group enters task-reward mode if
    either the frozen pretrained reference is feasible or the current group has
    enough feasible candidates.  Otherwise task reward is suspended and GRPO
    ranks candidates by reduction of the current active constraint bottleneck.
    """

    def __init__(
        self,
        names: tuple[str, ...],
        *,
        mode: str,
        config: ActiveConstraintConfig | None = None,
    ) -> None:
        if mode not in {"hard_worst", "soft_active"}:
            raise ValueError("active constraint mode must be hard_worst or soft_active")
        self.names = tuple(names)
        self.mode = mode
        self.config = config or ActiveConstraintConfig()
        self.config.validate()

    def _weights(self, severity: torch.Tensor) -> torch.Tensor:
        if self.mode == "hard_worst":
            index = severity.argmax(dim=-1)
            return torch.nn.functional.one_hot(
                index, num_classes=severity.shape[-1]
            ).to(dtype=severity.dtype)
        return torch.softmax(severity / float(self.config.temperature), dim=-1)

    def compute(
        self,
        task_rewards: torch.Tensor,
        constraints: ConstraintBatch,
        *,
        valid_mask: torch.Tensor | None,
        advantage_eps: float,
        update_dual: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        del update_dual  # common strategy API; active strategies have no dual state.
        if constraints.names != self.names:
            raise ValueError(
                f"constraint names changed: strategy={self.names} batch={constraints.names}"
            )
        if constraints.violation.shape[:3] != task_rewards.shape:
            raise ValueError("constraint/task reward shapes do not match")

        vm_mask = _valid_vehicle_mode_mask(valid_mask, task_rewards)
        severity = _cvar_group_severity(
            constraints.violation, float(self.config.cvar_alpha)
        )
        weights = self._weights(severity)

        weighted_violation = torch.sum(
            constraints.violation * weights[:, None, :, :], dim=-1
        )
        restoration_rewards = -weighted_violation

        feasible_fraction_vm = constraints.feasible_mask.to(task_rewards.dtype).mean(dim=1)
        reference_feasible = constraints.reference_feasible_mask
        reward_mode = reference_feasible | (
            feasible_fraction_vm >= float(self.config.feasible_fraction)
        )
        reward_mode = reward_mode & vm_mask
        restore_mode = (~reward_mode) & vm_mask

        effective_rewards = torch.where(
            reward_mode[:, None, :], task_rewards, restoration_rewards
        )
        advantages = group_relative_advantage(
            effective_rewards,
            eps=advantage_eps,
            valid_mask=valid_mask,
        )

        # Once a vehicle/mode is in reward mode, an infeasible candidate is never
        # allowed to become a positive GRPO example, even if its task reward is high.
        reward_mode_expanded = reward_mode[:, None, :].expand_as(advantages)
        infeasible_positive = (
            reward_mode_expanded
            & (~constraints.feasible_mask)
            & (advantages > 0.0)
        )
        advantages = torch.where(
            infeasible_positive,
            torch.minimum(advantages, torch.zeros_like(advantages)),
            advantages,
        )

        # Optional pretrained-relative support gate.  In restoration mode, update
        # only when at least one sampled candidate improves the active violation
        # relative to the frozen pretrained reference.
        reference_weighted_violation = torch.sum(
            constraints.reference_violation * weights, dim=-1
        )
        best_current_violation = weighted_violation.amin(dim=1)
        supported = (
            best_current_violation
            <= reference_weighted_violation - float(self.config.support_margin)
        )
        unsupported = restore_mode & (~supported)
        if self.config.support_gate:
            advantages = advantages.masked_fill(unsupported[:, None, :], 0.0)

        dominant = weights.argmax(dim=-1)
        eps = torch.finfo(weights.dtype).eps
        weight_entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=-1)
        if weights.shape[-1] > 1:
            weight_entropy = weight_entropy / torch.log(
                torch.tensor(float(weights.shape[-1]), device=weights.device, dtype=weights.dtype)
            )
        else:
            weight_entropy = torch.zeros_like(weight_entropy)

        metrics: Dict[str, float] = {
            "constraint/feasible_fraction": float(
                _masked_scalar_mean(
                    constraints.feasible_mask.to(task_rewards.dtype), valid_mask
                ).detach()
            ),
            "constraint/max_violation_mean": float(
                _masked_scalar_mean(
                    constraints.violation.amax(dim=-1), valid_mask
                ).detach()
            ),
            "active/reward_mode_fraction": float(
                _vm_masked_mean(reward_mode.to(task_rewards.dtype), vm_mask).detach()
            ),
            "active/restoration_mode_fraction": float(
                _vm_masked_mean(restore_mode.to(task_rewards.dtype), vm_mask).detach()
            ),
            "active/current_feasible_group_fraction_mean": float(
                _vm_masked_mean(feasible_fraction_vm, vm_mask).detach()
            ),
            "active/reference_feasible_fraction": float(
                _vm_masked_mean(reference_feasible.to(task_rewards.dtype), vm_mask).detach()
            ),
            "active/max_severity_mean": float(
                _vm_masked_mean(severity.amax(dim=-1), vm_mask).detach()
            ),
            "active/weight_entropy_mean": float(
                _vm_masked_mean(weight_entropy, vm_mask).detach()
            ),
            "active/restoration_reward_mean": float(
                _masked_scalar_mean(restoration_rewards, valid_mask).detach()
            ),
            "active/infeasible_positive_clamped_fraction": float(
                _masked_scalar_mean(
                    infeasible_positive.to(task_rewards.dtype), valid_mask
                ).detach()
            ),
            "active/unsupported_fraction": float(
                _vm_masked_mean(unsupported.to(task_rewards.dtype), vm_mask).detach()
            ),
            "active/support_gate_enabled": float(bool(self.config.support_gate)),
        }

        for index, name in enumerate(self.names):
            metrics[f"active/severity_{name}"] = float(
                _vm_masked_mean(severity[..., index], vm_mask).detach()
            )
            metrics[f"active/weight_{name}"] = float(
                _vm_masked_mean(weights[..., index], vm_mask).detach()
            )
            metrics[f"active/dominant_fraction_{name}"] = float(
                _vm_masked_mean(
                    (dominant == index).to(task_rewards.dtype), vm_mask
                ).detach()
            )

        return effective_rewards, advantages, metrics

    def state_dict(self) -> dict:
        return {
            "names": self.names,
            "mode": self.mode,
            "config": {
                "feasible_fraction": self.config.feasible_fraction,
                "cvar_alpha": self.config.cvar_alpha,
                "temperature": self.config.temperature,
                "support_gate": self.config.support_gate,
                "support_margin": self.config.support_margin,
            },
        }

