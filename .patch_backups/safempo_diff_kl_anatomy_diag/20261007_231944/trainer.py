from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

from highway_env.planner.diffusion.grpo.constraints import evaluate_w4_constraints
from highway_env.planner.diffusion.grpo.task_reward import (
    legacy_w4_reward_from_result,
    task_reward_from_w4_result,
)
from highway_env.planner.diffusion.grpo.trainer import GRPOConfig, GRPOTrainer

from .target import SafeMPOTargetBuilder, SafeMPOTargetConfig


@dataclass
class SafeMPODiffConfig(GRPOConfig):
    """SafeMPO-Diff V1: G-particle E-step + KL distribution distillation."""

    safempo_kl_epsilon: float = 0.10
    safempo_kappa: float = 10.0
    safempo_constraint_beta: float = 1.0
    safempo_lambda_init: float = 1.0
    safempo_lambda_min: float = 1e-6
    safempo_lambda_max: float = 10000.0
    safempo_nu_init: float = 1.0
    safempo_nu_min: float = 1e-5
    safempo_nu_max: float = 1000.0
    safempo_active_range_eps: float = 1e-6
    safempo_dual_maxiter: int = 128
    safempo_dual_ftol: float = 1e-9


class SafeMPODiffTrainer(GRPOTrainer):
    """SafeMPO-Diff policy trainer.

    E-step:
      * sample G trajectories from pi_old;
      * evaluate task reward and multi-constraint violations;
      * solve the finite-particle SafeMPO dual for lambda_j and nu;
      * build q*(g | vehicle, mode).

    M-step:
      * replay the exact sampled diffusion transitions;
      * form the student categorical distribution from trajectory log-ratios
        p_theta(g) = softmax(sum_t(log pi_theta - log pi_old));
      * minimize KL(q* || p_theta).

    The reference-model KL, trainable scope, optimizer, sampler and fixed-state
    validation are inherited unchanged from the GRPO baseline infrastructure.
    """

    def __init__(self, *args, config: SafeMPODiffConfig | None = None, **kwargs) -> None:
        cfg = config or SafeMPODiffConfig()
        if cfg.constraint_strategy != "none":
            raise ValueError(
                "SafeMPO-Diff requires constraint_strategy='none'; constraints "
                "are handled in the SafeMPO E-step."
            )
        if cfg.task_reward_type == "legacy_w4":
            raise ValueError(
                "SafeMPO-Diff requires a task-only reward such as progress_comfort; "
                "legacy_w4 already mixes safety into reward."
            )
        super().__init__(*args, config=cfg, **kwargs)
        self.config: SafeMPODiffConfig
        self.target_builder = SafeMPOTargetBuilder(
            self.config.constraint_names,
            config=SafeMPOTargetConfig(
                kl_epsilon=self.config.safempo_kl_epsilon,
                kappa=self.config.safempo_kappa,
                constraint_beta=self.config.safempo_constraint_beta,
                lambda_init=self.config.safempo_lambda_init,
                lambda_min=self.config.safempo_lambda_min,
                lambda_max=self.config.safempo_lambda_max,
                nu_init=self.config.safempo_nu_init,
                nu_min=self.config.safempo_nu_min,
                nu_max=self.config.safempo_nu_max,
                active_range_eps=self.config.safempo_active_range_eps,
                maxiter=self.config.safempo_dual_maxiter,
                ftol=self.config.safempo_dual_ftol,
            ),
        )

    @staticmethod
    def _vm_mask(valid_mask: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
        # target is [B,G,M]
        if valid_mask is None:
            return torch.ones(
                (target.shape[0], target.shape[2]),
                dtype=torch.bool,
                device=target.device,
            )
        mask = valid_mask.to(device=target.device, dtype=torch.bool)
        if tuple(mask.shape) != (target.shape[0], target.shape[2]):
            raise ValueError("valid_mode_mask must be [B,M]")
        return mask

    @classmethod
    def _masked_vm_mean(
        cls,
        value: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # value [B,M]
        if value.ndim != 2:
            raise ValueError("SafeMPO vehicle/mode value must be [B,M]")
        if valid_mask is None:
            return value.mean()
        mask = valid_mask.to(device=value.device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(value.shape):
            raise ValueError("valid_mode_mask shape mismatch")
        denom = mask.sum().clamp_min(1)
        return (value * mask.to(value.dtype)).sum() / denom

    @staticmethod
    def _masked_step_mean(
        value: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # value [B,G,S,M]
        if valid_mask is None:
            return value.mean()
        mask = valid_mask[:, None, None, :].to(
            device=value.device, dtype=torch.bool
        )
        mask = mask.expand_as(value)
        denom = mask.sum().clamp_min(1)
        return (value * mask.to(value.dtype)).sum() / denom

    def collect_safempo(
        self,
        features: Dict[str, torch.Tensor],
        *,
        context: Any = None,
        generator: torch.Generator | None = None,
    ):
        self.model.eval()
        with torch.no_grad():
            trace = self.sampler.sample(features, generator=generator)
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
            constraints = evaluate_w4_constraints(
                result,
                context=context,
                device=trace.candidates.device,
                dtype=trace.candidates.dtype,
                names=self.config.constraint_names,
                residual_cap=self.config.constraint_residual_cap,
            )
            target = self.target_builder.solve(
                rewards,
                constraints.violation,
                valid_mask=features.get("mode_valid_mask"),
            )

            valid_mask = features.get("mode_valid_mask")
            if valid_mask is None:
                expanded = torch.ones_like(
                    constraints.feasible_mask, dtype=torch.bool
                )
            else:
                expanded = valid_mask[:, None, :].to(
                    device=rewards.device, dtype=torch.bool
                ).expand_as(constraints.feasible_mask)
            denom = expanded.sum().clamp_min(1)
            feasible_fraction = (
                constraints.feasible_mask.to(rewards.dtype)
                * expanded.to(rewards.dtype)
            ).sum() / denom
            max_violation = constraints.violation.amax(dim=-1)
            max_violation_mean = (
                max_violation * expanded.to(max_violation.dtype)
            ).sum() / denom

            metrics = dict(target.metrics)
            metrics.update(
                {
                    "constraint/feasible_fraction": float(feasible_fraction.detach()),
                    "constraint/max_violation_mean": float(max_violation_mean.detach()),
                    "reward/task_reward_mean": float(rewards.mean().detach()),
                    "reward/legacy_w4_reward_mean": float(legacy_rewards.mean().detach()),
                    # No penalty-shaped reward is used in SafeMPO-Diff.
                    "reward/effective_reward_mean": float(rewards.mean().detach()),
                }
            )
            for j, name in enumerate(constraints.names):
                v = constraints.violation[..., j]
                metrics[f"constraint/{name}_violation_mean"] = float(
                    (v * expanded.to(v.dtype)).sum().detach() / denom
                )
                metrics[f"constraint/{name}_violation_fraction"] = float(
                    (
                        (v > 0.0).to(v.dtype) * expanded.to(v.dtype)
                    ).sum().detach()
                    / denom
                )
        return trace, rewards, target.target_q, metrics

    def update_safempo(
        self,
        trace,
        target_q: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None,
        reference_log_prob: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        self.model.eval()
        for module in self._model_gru_modules:
            module.train()
            module.flatten_parameters()

        self.optimizer.zero_grad(set_to_none=True)
        trainable_params = self._trainable_parameters
        before_update = [p.detach().clone() for p in trainable_params]

        new_log_prob = self.sampler.replay(trace)
        delta = new_log_prob - trace.old_log_prob

        # True sampled reverse-chain likelihood ratio for every trajectory.
        # [B,G,S,M] -> [B,G,M].  At theta==theta_old every ratio is exactly 1,
        # therefore the particle student distribution starts uniformly at 1/G.
        trajectory_log_ratio = delta.sum(dim=2)
        student_log_q = torch.log_softmax(trajectory_log_ratio, dim=1)
        student_q = student_log_q.exp()
        teacher_q = target_q.detach().clamp_min(1e-12)
        teacher_log_q = teacher_q.log()
        kl_per_vm = torch.sum(
            teacher_q * (teacher_log_q - student_log_q), dim=1
        )
        distill_kl = self._masked_vm_mean(kl_per_vm, valid_mask)

        if reference_log_prob is None:
            with torch.no_grad():
                ref_log_prob = self.sampler.replay(
                    trace, model=self.reference_model
                )
        else:
            ref_log_prob = reference_log_prob
        ref_kl = self._reference_kl(new_log_prob, ref_log_prob)
        loss = distill_kl + float(self.config.kl_coef) * ref_kl
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite SafeMPO-Diff loss")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params, max_norm=self.config.max_grad_norm
        )
        self.optimizer.step()

        update_sq = torch.zeros((), device=trainable_params[0].device)
        for parameter, before in zip(trainable_params, before_update):
            update_sq = update_sq + (
                parameter.detach() - before
            ).float().square().sum()
        parameter_update_norm = torch.sqrt(update_sq)

        approx_kl = self._masked_step_mean(0.5 * delta.square(), valid_mask)
        ratio = torch.exp(delta)
        ratio_mean = self._masked_step_mean(ratio, valid_mask)
        ratio_std = torch.sqrt(
            self._masked_step_mean(
                (ratio - ratio_mean).square(), valid_mask
            ).clamp_min(0.0)
        )
        old_uniform = 1.0 / float(target_q.shape[1])
        student_old_kl_vm = torch.sum(
            student_q
            * (student_log_q - torch.log(
                torch.as_tensor(
                    old_uniform,
                    device=student_log_q.device,
                    dtype=student_log_q.dtype,
                )
            )),
            dim=1,
        )
        student_old_kl = self._masked_vm_mean(
            student_old_kl_vm, valid_mask
        )
        target_l1_vm = torch.sum(
            torch.abs(student_q - teacher_q), dim=1
        )
        target_l1 = self._masked_vm_mean(target_l1_vm, valid_mask)

        return {
            "loss": float(loss.detach()),
            "policy_loss": float(distill_kl.detach()),
            "reference_kl": float(ref_kl.detach()),
            "approx_kl": float(approx_kl.detach()),
            "clip_fraction": 0.0,
            "ratio_mean": float(ratio_mean.detach()),
            "ratio_std": float(ratio_std.detach()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach()),
            "parameter_update_norm": float(parameter_update_norm.detach()),
            "safempo/distill_kl": float(distill_kl.detach()),
            "safempo/student_old_particle_kl": float(student_old_kl.detach()),
            "safempo/student_target_l1": float(target_l1.detach()),
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
            self._clone_generator(generator) if paired_validation else None
        )
        trace, rewards, target_q, safempo_metrics = self.collect_safempo(
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

        reference_log_prob = None
        if self.config.update_epochs > 1:
            self.reference_model.eval()
            self._flatten_gru_parameters(self._reference_gru_modules)
            with torch.no_grad():
                reference_log_prob = self.sampler.replay(
                    trace, model=self.reference_model
                )

        update_metrics = []
        for _ in range(self.config.update_epochs):
            update_metrics.append(
                self.update_safempo(
                    trace,
                    target_q,
                    valid_mask=features.get("mode_valid_mask"),
                    reference_log_prob=reference_log_prob,
                )
            )

        metrics = dict(update_metrics[-1])
        metrics.update(
            update_epochs=float(self.config.update_epochs),
            epoch1_ratio_mean=float(update_metrics[0]["ratio_mean"]),
            epoch1_approx_kl=float(update_metrics[0]["approx_kl"]),
            epoch1_clip_fraction=0.0,
            reward_mean=float(rewards.mean().detach()),
            reward_std=float(rewards.std(unbiased=False).detach()),
            # Compatibility fields: SafeMPO-Diff has no signed GRPO advantage.
            advantage_mean=0.0,
            advantage_std=0.0,
        )
        metrics.update(safempo_metrics)
        metrics.update(validation_metrics)
        return metrics

    def constraint_state_dict(self) -> Dict[str, Any]:
        return {
            "strategy": "safempo_diff_v1",
            "last_lambda": self.target_builder._last_lambda.copy(),
            "last_nu": float(self.target_builder._last_nu),
        }
