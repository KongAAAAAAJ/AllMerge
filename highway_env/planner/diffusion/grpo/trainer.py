from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import torch

from .objective import group_relative_advantage, grpo_clipped_objective
from .reward_adapter import w4_to_grpo_rewards
from .constraints import SUPPORTED_CONSTRAINTS, evaluate_w4_constraints
from .constraint_strategy import (
    ActiveConstraintConfig,
    ActiveConstraintStrategy,
    LagrangianConstraintConfig,
    LagrangianConstraintStrategy,
)
from .reward_adapter import CandidateRewardAdapter
from .sampling import DiffusionTrace, GroupDiffusionSampler
from .task_reward import (
    SUPPORTED_TASK_REWARDS,
    legacy_w4_reward_from_result,
    task_reward_from_w4_result,
)


@dataclass
class GRPOConfig:
    group_size: int = 4
    learning_rate: float = 5e-5
    eta: float = 0.02
    clip_eps: float = 0.2
    kl_coef: float = 0.01
    max_grad_norm: float = 10.0
    advantage_eps: float = 1e-6
    update_epochs: int = 2
    task_reward_type: str = "legacy_w4"
    constraint_strategy: str = "none"
    constraint_names: Tuple[str, ...] = SUPPORTED_CONSTRAINTS
    constraint_residual_cap: float = 5.0
    lagrangian_dual_lr: float = 0.05
    lagrangian_lambda_init: float = 0.0
    lagrangian_lambda_max: float = 20.0
    active_feasible_fraction: float = 0.50
    active_cvar_alpha: float = 0.25
    active_temperature: float = 0.20
    active_support_gate: bool = False
    active_support_margin: float = 1e-3
    trainable_prefixes: Tuple[str, ...] = (
        "denoiser.layers",
        "denoiser.reg_head",
        "denoiser.gru_refine_token_norm",
        "denoiser.gru_refine_h0",
        "denoiser.gru_refine_input",
        "denoiser.gru_refine",
        "denoiser.gru_refine_out",
    )


class GRPOTrainer:
    """Small GRPO core around the existing StructuredDiffusionPlanner.

    The trainer deliberately owns no reward definition and no planner network.
    It only coordinates sampling/replay, W4 reward delegation, group-relative
    advantages, clipped policy optimization, KL regularization and checkpointable
    optimizer state.
    """

    def __init__(
        self,
        model,
        reward_adapter: CandidateRewardAdapter,
        *,
        config: GRPOConfig | None = None,
    ) -> None:
        self.model = model
        self.reward_adapter = reward_adapter
        self.config = config or GRPOConfig()
        if self.config.update_epochs < 1:
            raise ValueError("update_epochs must be >= 1")
        if self.config.task_reward_type not in SUPPORTED_TASK_REWARDS:
            raise ValueError(
                f"unsupported task_reward_type={self.config.task_reward_type!r}; "
                f"supported={SUPPORTED_TASK_REWARDS}"
            )
        if self.config.constraint_strategy not in {"none", "lagrangian", "hard_worst", "soft_active"}:
            raise ValueError(
                "constraint_strategy must be one of: none, lagrangian, hard_worst, soft_active"
            )
        if not self.config.constraint_names:
            raise ValueError("constraint_names cannot be empty")
        unknown_constraints = sorted(
            set(self.config.constraint_names) - set(SUPPORTED_CONSTRAINTS)
        )
        if unknown_constraints:
            raise ValueError(
                f"unsupported constraint_names={unknown_constraints}; "
                f"supported={SUPPORTED_CONSTRAINTS}"
            )
        if self.config.constraint_residual_cap <= 0.0:
            raise ValueError("constraint_residual_cap must be > 0")
        if not 0.0 <= self.config.active_feasible_fraction <= 1.0:
            raise ValueError("active_feasible_fraction must be in [0,1]")
        if not 0.0 < self.config.active_cvar_alpha <= 1.0:
            raise ValueError("active_cvar_alpha must be in (0,1]")
        if self.config.active_temperature <= 0.0:
            raise ValueError("active_temperature must be > 0")
        if self.config.active_support_margin < 0.0:
            raise ValueError("active_support_margin must be >= 0")
        self._configure_trainable_parameters()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError("GRPO has no trainable parameters after freezing")
        self.optimizer = torch.optim.AdamW(parameters, lr=self.config.learning_rate)

        self.reference_model = copy.deepcopy(self.model).eval()
        self.reference_model.requires_grad_(False)
        self.sampler = GroupDiffusionSampler(
            self.model,
            group_size=self.config.group_size,
            eta=self.config.eta,
        )
        self.reference_sampler = GroupDiffusionSampler(
            self.reference_model,
            group_size=self.config.group_size,
            eta=self.config.eta,
        )

        self.constraint_strategy = None
        if self.config.constraint_strategy == "lagrangian":
            self.constraint_strategy = LagrangianConstraintStrategy(
                tuple(self.config.constraint_names),
                device=parameters[0].device,
                dtype=parameters[0].dtype,
                config=LagrangianConstraintConfig(
                    dual_lr=self.config.lagrangian_dual_lr,
                    lambda_init=self.config.lagrangian_lambda_init,
                    lambda_max=self.config.lagrangian_lambda_max,
                ),
            )
        elif self.config.constraint_strategy in {"hard_worst", "soft_active"}:
            self.constraint_strategy = ActiveConstraintStrategy(
                tuple(self.config.constraint_names),
                mode=self.config.constraint_strategy,
                config=ActiveConstraintConfig(
                    feasible_fraction=self.config.active_feasible_fraction,
                    cvar_alpha=self.config.active_cvar_alpha,
                    temperature=self.config.active_temperature,
                    support_gate=self.config.active_support_gate,
                    support_margin=self.config.active_support_margin,
                ),
            )

    def _configure_trainable_parameters(self) -> None:
        self.model.requires_grad_(False)
        matched = []
        for name, parameter in self.model.named_parameters():
            if any(name.startswith(prefix) for prefix in self.config.trainable_prefixes):
                parameter.requires_grad_(True)
                matched.append(name)
        if not matched:
            raise RuntimeError(
                "No parameters matched trainable_prefixes="
                f"{self.config.trainable_prefixes}. Check current StructuredDiffusionPlanner names."
            )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def refresh_reference(self) -> None:
        """Explicit reference refresh; never called implicitly during an update."""
        self.reference_model.load_state_dict(self.model.state_dict())
        self.reference_model.eval()
        self.reference_model.requires_grad_(False)

    def score_candidates(
        self,
        candidates: torch.Tensor,
        *,
        features: Dict[str, torch.Tensor],
        context: Any,
    ) -> torch.Tensor:
        """Score candidates with the configured task objective."""
        if self.config.task_reward_type == "legacy_w4":
            return self.reward_adapter(
                candidates, features=features, context=context
            )
        result = self.reward_adapter.evaluate_result(
            candidates, features=features, context=context
        )
        return task_reward_from_w4_result(
            result,
            context=context,
            device=candidates.device,
            dtype=candidates.dtype,
            reward_type=self.config.task_reward_type,
        )

    def collect(
        self,
        features: Dict[str, torch.Tensor],
        *,
        context: Any = None,
        generator: torch.Generator | None = None,
    ) -> tuple[DiffusionTrace, torch.Tensor, torch.Tensor, Dict[str, float]]:
        # Keep dropout disabled: diffusion transition log-prob must describe all policy stochasticity.
        self.model.eval()
        with torch.no_grad():
            trace = self.sampler.sample(features, generator=generator)
            strategy_metrics: Dict[str, float] = {}
            needs_full_result = (
                self.config.constraint_strategy != "none"
                or self.config.task_reward_type != "legacy_w4"
            )
            if not needs_full_result:
                rewards = self.reward_adapter(
                    trace.candidates,
                    features=features,
                    context=context,
                )
                legacy_rewards = rewards
                effective_rewards = rewards
                advantages = group_relative_advantage(
                    rewards,
                    eps=self.config.advantage_eps,
                    valid_mask=features.get("mode_valid_mask"),
                )
            else:
                result = self.reward_adapter.evaluate_result(
                    trace.candidates,
                    features=features,
                    context=context,
                )
                rewards = task_reward_from_w4_result(
                    result,
                    context=context,
                    device=trace.candidates.device,
                    dtype=trace.candidates.dtype,
                    reward_type=self.config.task_reward_type,
                )
                legacy_rewards = legacy_w4_reward_from_result(
                    result,
                    device=trace.candidates.device,
                    dtype=trace.candidates.dtype,
                )
                if self.config.constraint_strategy == "none":
                    effective_rewards = rewards
                    advantages = group_relative_advantage(
                        rewards,
                        eps=self.config.advantage_eps,
                        valid_mask=features.get("mode_valid_mask"),
                    )
                else:
                    constraints = evaluate_w4_constraints(
                        result,
                        context=context,
                        device=trace.candidates.device,
                        dtype=trace.candidates.dtype,
                        names=self.config.constraint_names,
                        residual_cap=self.config.constraint_residual_cap,
                    )
                    assert self.constraint_strategy is not None
                    effective_rewards, advantages, strategy_metrics = (
                        self.constraint_strategy.compute(
                            rewards,
                            constraints,
                            valid_mask=features.get("mode_valid_mask"),
                            advantage_eps=self.config.advantage_eps,
                            update_dual=True,
                        )
                    )
            strategy_metrics["reward/task_reward_mean"] = float(
                rewards.mean().detach()
            )
            strategy_metrics["reward/legacy_w4_reward_mean"] = float(
                legacy_rewards.mean().detach()
            )
            strategy_metrics["reward/effective_reward_mean"] = float(
                effective_rewards.mean().detach()
            )
        return trace, rewards, advantages, strategy_metrics

    @staticmethod
    def _clone_generator(
        generator: torch.Generator | None,
    ) -> torch.Generator:
        if generator is None:
            raise ValueError(
                "paired validation requires an explicit torch.Generator"
            )
        clone = torch.Generator(device=generator.device)
        clone.set_state(generator.get_state())
        return clone

    @staticmethod
    def _paired_reward_metrics(
        current_rewards: torch.Tensor,
        frozen_rewards: torch.Tensor,
        valid_mask: torch.Tensor | None,
        *,
        candidate_delta_m: torch.Tensor,
    ) -> Dict[str, float]:
        if current_rewards.shape != frozen_rewards.shape:
            raise ValueError(
                "paired current/frozen rewards must have identical shapes, got "
                f"{tuple(current_rewards.shape)} and {tuple(frozen_rewards.shape)}"
            )
        if current_rewards.ndim != 3:
            raise ValueError(
                "paired rewards must be [vehicle,group,mode]"
            )
        if valid_mask is None:
            mask = torch.ones(
                current_rewards.shape[0],
                current_rewards.shape[2],
                dtype=torch.bool,
                device=current_rewards.device,
            )
        else:
            mask = valid_mask.to(
                device=current_rewards.device,
                dtype=torch.bool,
            )
        if tuple(mask.shape) != (
            current_rewards.shape[0],
            current_rewards.shape[2],
        ):
            raise ValueError(
                "valid_mode_mask shape does not match paired rewards"
            )

        role_current = []
        role_frozen = []
        metrics: Dict[str, float] = {}
        for role in range(current_rewards.shape[0]):
            role_mask = mask[role]
            if not bool(role_mask.any()):
                raise ValueError(
                    f"paired validation vehicle {role} has no valid modes"
                )
            current_value = current_rewards[role, :, role_mask].mean()
            frozen_value = frozen_rewards[role, :, role_mask].mean()
            role_current.append(current_value)
            role_frozen.append(frozen_value)
            metrics[f"validation/vehicle_{role}_reward_gain"] = float(
                (current_value - frozen_value).detach()
            )

        current_mean = torch.stack(role_current).mean()
        frozen_mean = torch.stack(role_frozen).mean()
        gain = current_mean - frozen_mean
        group_size = int(current_rewards.shape[1])
        metrics.update(
            {
                "validation/paired_group_current_reward_mean": float(
                    current_mean.detach()
                ),
                "validation/paired_group_frozen_reward_mean": float(
                    frozen_mean.detach()
                ),
                "validation/paired_group_reward_gain": float(gain.detach()),
                f"validation/paired_n{group_size}_reward_gain": float(
                    gain.detach()
                ),
                "validation/paired_candidate_delta_m": float(
                    candidate_delta_m.detach()
                ),
            }
        )
        return metrics

    @staticmethod
    def _selected_reward_metrics(
        current_rewards: torch.Tensor,
        frozen_rewards: torch.Tensor,
        current_logits: torch.Tensor,
        frozen_logits: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> Dict[str, float]:
        if current_rewards.shape != frozen_rewards.shape:
            raise ValueError("selected reward comparison requires matching shapes")
        if current_logits.shape != current_rewards.shape:
            raise ValueError("current logits/rewards shape mismatch")
        if frozen_logits.shape != frozen_rewards.shape:
            raise ValueError("frozen logits/rewards shape mismatch")

        cur_logits = current_logits
        ref_logits = frozen_logits
        if valid_mask is not None:
            mask = valid_mask.to(device=cur_logits.device, dtype=torch.bool)
            expected = (cur_logits.shape[0], cur_logits.shape[2])
            if tuple(mask.shape) != expected:
                raise ValueError(
                    "valid_mode_mask shape mismatch for selected reward"
                )
            expanded = mask[:, None, :].expand_as(cur_logits)
            neg_inf = torch.finfo(cur_logits.dtype).min
            cur_logits = cur_logits.masked_fill(~expanded, neg_inf)
            ref_logits = ref_logits.masked_fill(~expanded, neg_inf)

        cur_idx = cur_logits.argmax(dim=-1, keepdim=True)
        ref_idx = ref_logits.argmax(dim=-1, keepdim=True)
        cur_selected = current_rewards.gather(-1, cur_idx).squeeze(-1)
        ref_selected = frozen_rewards.gather(-1, ref_idx).squeeze(-1)

        metrics: Dict[str, float] = {
            "validation/selected_vehicle_reward_mean": float(
                cur_selected.mean().detach()
            ),
            "validation/selected_pretrain_vehicle_reward_mean": float(
                ref_selected.mean().detach()
            ),
            "validation/selected_vehicle_reward_gain": float(
                (cur_selected.mean() - ref_selected.mean()).detach()
            ),
        }
        for role in range(cur_selected.shape[0]):
            metrics[
                f"validation/selected_vehicle_{role}_reward_gain"
            ] = float(
                (
                    cur_selected[role].mean()
                    - ref_selected[role].mean()
                ).detach()
            )
        return metrics


    def paired_validation(
        self,
        trace: DiffusionTrace,
        current_rewards: torch.Tensor,
        features: Dict[str, torch.Tensor],
        *,
        context: Any,
        generator: torch.Generator,
    ) -> Dict[str, float]:
        """Same-noise current-vs-frozen task and constraint comparison."""
        self.reference_model.eval()
        with torch.no_grad():
            frozen_trace = self.reference_sampler.sample(
                features,
                generator=generator,
            )
            frozen_rewards = self.score_candidates(
                frozen_trace.candidates,
                features=features,
                context=context,
            )
            delta_m = torch.linalg.vector_norm(
                trace.candidates[..., :2]
                - frozen_trace.candidates[..., :2],
                dim=-1,
            ).mean()

            # Constraint diagnostics are intentionally independent of task reward.
            current_result = self.reward_adapter.evaluate_result(
                trace.candidates,
                features=features,
                context=context,
            )
            frozen_result = self.reward_adapter.evaluate_result(
                frozen_trace.candidates,
                features=features,
                context=context,
            )
            current_constraints = evaluate_w4_constraints(
                current_result,
                context=context,
                device=trace.candidates.device,
                dtype=trace.candidates.dtype,
                names=self.config.constraint_names,
                residual_cap=self.config.constraint_residual_cap,
            )
            frozen_constraints = evaluate_w4_constraints(
                frozen_result,
                context=context,
                device=frozen_trace.candidates.device,
                dtype=frozen_trace.candidates.dtype,
                names=self.config.constraint_names,
                residual_cap=self.config.constraint_residual_cap,
            )

        metrics = self._paired_reward_metrics(
            current_rewards,
            frozen_rewards,
            features.get("mode_valid_mask"),
            candidate_delta_m=delta_m,
        )
        metrics.update(
            self._selected_reward_metrics(
                current_rewards,
                frozen_rewards,
                trace.logits,
                frozen_trace.logits,
                features.get("mode_valid_mask"),
            )
        )

        valid_mask = features.get("mode_valid_mask")
        if valid_mask is None:
            valid = torch.ones(
                current_constraints.violation.shape[0],
                current_constraints.violation.shape[2],
                dtype=torch.bool,
                device=current_constraints.violation.device,
            )
        else:
            valid = valid_mask.to(
                device=current_constraints.violation.device,
                dtype=torch.bool,
            )
        expanded = valid[:, None, :, None].expand_as(current_constraints.violation)
        denom = expanded.to(current_constraints.violation.dtype).sum().clamp_min(1.0)

        def masked_mean(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 3:
                m = valid[:, None, :].expand_as(x)
            elif x.ndim == 4:
                m = expanded
            else:
                raise ValueError("constraint validation tensor must be rank 3 or 4")
            return (x * m.to(x.dtype)).sum() / m.to(x.dtype).sum().clamp_min(1.0)

        cur_feasible = masked_mean(current_constraints.feasible_mask.to(current_rewards.dtype))
        ref_feasible = masked_mean(frozen_constraints.feasible_mask.to(current_rewards.dtype))
        cur_max = masked_mean(current_constraints.violation.amax(dim=-1))
        ref_max = masked_mean(frozen_constraints.violation.amax(dim=-1))
        metrics.update(
            {
                "validation/current_constraint_feasible_fraction": float(cur_feasible.detach()),
                "validation/frozen_constraint_feasible_fraction": float(ref_feasible.detach()),
                "validation/constraint_feasible_fraction_gain": float((cur_feasible-ref_feasible).detach()),
                "validation/current_constraint_max_violation_mean": float(cur_max.detach()),
                "validation/frozen_constraint_max_violation_mean": float(ref_max.detach()),
                "validation/constraint_max_violation_change": float((cur_max-ref_max).detach()),
            }
        )
        for j, name in enumerate(current_constraints.names):
            cur_v = masked_mean(current_constraints.violation[..., j])
            ref_v = masked_mean(frozen_constraints.violation[..., j])
            metrics[f"validation/current_constraint_{name}_violation_mean"] = float(cur_v.detach())
            metrics[f"validation/frozen_constraint_{name}_violation_mean"] = float(ref_v.detach())
            metrics[f"validation/constraint_{name}_violation_change"] = float((cur_v-ref_v).detach())
        return metrics

    @staticmethod
    def _reference_kl(new_log_prob: torch.Tensor, ref_log_prob: torch.Tensor) -> torch.Tensor:
        # k3 estimator from log-ratio; non-negative and zero when policies match.
        log_ratio = ref_log_prob - new_log_prob
        return (torch.exp(log_ratio) - log_ratio - 1.0).mean()

    def update(
        self,
        trace: DiffusionTrace,
        advantages: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        # Keep the full planner in eval mode so Transformer/MLP dropout
        # stays disabled and diffusion replay remains deterministic.
        self.model.eval()
        # cuDNN RNN backward requires the GRU module itself to be in
        # training mode. The refinement GRU has num_layers=1 and no
        # internal dropout, so enabling train mode only for nn.GRU
        # preserves deterministic replay semantics.
        for module in self.model.modules():
            if isinstance(module, torch.nn.GRU):
                module.train()
                module.flatten_parameters()
        # GRPO_CUDNN_GRU_BACKWARD_FIX_V1
        self.optimizer.zero_grad(set_to_none=True)
        trainable_params = [
            p for p in self.model.parameters()
            if p.requires_grad
        ]
        before_update = [
            p.detach().clone() for p in trainable_params
        ]
        new_log_prob = self.sampler.replay(trace)
        objective = grpo_clipped_objective(
            new_log_prob,
            trace.old_log_prob,
            advantages,
            clip_eps=self.config.clip_eps,
            valid_mask=valid_mask,
        )

        with torch.no_grad():
            ref_log_prob = self.sampler.replay(trace, model=self.reference_model)
        ref_kl = self._reference_kl(new_log_prob, ref_log_prob)
        loss = objective.policy_loss + self.config.kl_coef * ref_kl
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite GRPO loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=self.config.max_grad_norm,
        )
        self.optimizer.step()
        update_sq = torch.zeros(
            (), device=trainable_params[0].device
        )
        for parameter, before in zip(
            trainable_params, before_update
        ):
            update_sq = update_sq + (
                parameter.detach() - before
            ).float().square().sum()
        parameter_update_norm = torch.sqrt(update_sq)
        return {
            "loss": float(loss.detach()),
            "policy_loss": float(objective.policy_loss.detach()),
            "reference_kl": float(ref_kl.detach()),
            "approx_kl": float(objective.approx_kl.detach()),
            "clip_fraction": float(objective.clip_fraction.detach()),
            "ratio_mean": float(objective.ratio_mean.detach()),
            "ratio_std": float(objective.ratio_std.detach()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach()),
            "parameter_update_norm": float(
                parameter_update_norm.detach()
            ),
        }

    def train_step(
        self,
        features: Dict[str, torch.Tensor],
        *,
        context: Any = None,
        generator: torch.Generator | None = None,
        paired_validation: bool = False,
    ) -> Dict[str, float]:
        paired_generator = (
            self._clone_generator(generator)
            if paired_validation
            else None
        )
        trace, rewards, advantages, strategy_metrics = self.collect(
            features, context=context, generator=generator
        )
        validation_metrics: Dict[str, float] = {}
        if paired_validation:
            assert paired_generator is not None
            validation_metrics = self.paired_validation(
                trace,
                rewards,
                features,
                context=context,
                generator=paired_generator,
            )
        update_metrics = []
        for _ in range(self.config.update_epochs):
            update_metrics.append(
                self.update(
                    trace,
                    advantages,
                    valid_mask=features.get("mode_valid_mask"),
                )
            )
        metrics = dict(update_metrics[-1])
        metrics.update(
            update_epochs=float(self.config.update_epochs),
            epoch1_ratio_mean=float(update_metrics[0]["ratio_mean"]),
            epoch1_approx_kl=float(update_metrics[0]["approx_kl"]),
            epoch1_clip_fraction=float(update_metrics[0]["clip_fraction"]),
            reward_mean=float(rewards.mean().detach()),
            reward_std=float(rewards.std(unbiased=False).detach()),
            advantage_mean=float(advantages.mean().detach()),
            advantage_std=float(advantages.std(unbiased=False).detach()),
        )
        metrics.update(strategy_metrics)
        metrics.update(validation_metrics)
        return metrics


    def constraint_state_dict(self) -> Dict[str, Any]:
        if self.constraint_strategy is None:
            return {"strategy": "none"}
        return {
            "strategy": self.config.constraint_strategy,
            "state": self.constraint_strategy.state_dict(),
        }

# GRPO_FORMAL_METRICS_V3

# LAGRANGIAN_CONSTRAINED_GRPO_BASELINE_V2_SEMANTIC

# ACTIVE_CONSTRAINT_GRPO_V1_SEMANTIC

# DECOUPLED_TASK_CONSTRAINT_V2_SEMANTIC
