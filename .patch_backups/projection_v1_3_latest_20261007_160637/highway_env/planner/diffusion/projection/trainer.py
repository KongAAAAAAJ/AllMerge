from __future__ import annotations

# GRPO_PROJECTION_BASELINE_V1_20261007
# GRPO_PROJECTION_BASELINE_V1_1_SPEEDV1_20261007
# GRPO_PROJECTION_BASELINE_V1_2_GRU_TRAINMODE_20261007

from dataclasses import dataclass
from typing import Any, Dict

import torch

from highway_env.planner.diffusion.grpo.constraints import evaluate_w4_constraints
from highway_env.planner.diffusion.grpo.objective import grpo_clipped_objective
from highway_env.planner.diffusion.grpo.task_reward import (
    legacy_w4_reward_from_result,
    task_reward_from_w4_result,
)
from highway_env.planner.diffusion.grpo.trainer import GRPOConfig, GRPOTrainer


@dataclass
class ProjectionConfig(GRPOConfig):
    """Explicit pre-optimizer gradient projection baseline.

    Preferred gradient = task GRPO gradient + reference-KL gradient.
    Active safety gradients define half-spaces:
        g_constraint^T g_projected >= 0
    so a gradient-descent step -g_projected is first-order non-increasing
    for every active constraint surrogate.

    No Lagrangian multiplier or reward penalty is used.
    """
    projection_advantage_beta: float = 0.99
    projection_q_low: float = 0.05
    projection_q_high: float = 0.95
    projection_active_threshold: float = 0.0
    projection_passes: int = 8
    projection_tolerance: float = 1e-7

    def validate_projection(self) -> None:
        if not 0.0 <= self.projection_advantage_beta < 1.0:
            raise ValueError("projection_advantage_beta must be in [0,1)")
        if not 0.0 <= self.projection_q_low < self.projection_q_high <= 1.0:
            raise ValueError("invalid projection Q quantiles")
        if self.projection_active_threshold < 0.0:
            raise ValueError("projection_active_threshold must be >= 0")
        if self.projection_passes < 1:
            raise ValueError("projection_passes must be >= 1")
        if self.projection_tolerance < 0.0:
            raise ValueError("projection_tolerance must be >= 0")


class _PerChannelQClipScaler:
    def __init__(self, *, beta: float, q_low: float, q_high: float, eps: float):
        self.beta = float(beta)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.eps = float(eps)
        self.var_ema: Dict[str, float] = {}

    @staticmethod
    def _mask(valid_mask: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
        if valid_mask is None:
            base = torch.ones(
                (value.shape[0], value.shape[2]),
                dtype=torch.bool,
                device=value.device,
            )
        else:
            base = valid_mask.to(device=value.device, dtype=torch.bool)
        return base[:, None, :].expand_as(value)

    def __call__(
        self,
        key: str,
        value: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if value.ndim != 3:
            raise ValueError(f"{key}: expected [vehicle,group,mode], got {value.shape}")
        mask = self._mask(valid_mask, value)
        centered = value - value.mean(dim=1, keepdim=True)
        centered = torch.where(mask, centered, torch.zeros_like(centered))
        cv = centered[mask].float()
        if cv.numel() == 0:
            return torch.zeros_like(value), {
                "scale": 1.0,
                "pre_std": 0.0,
                "post_std": 0.0,
                "clip_fraction": 0.0,
            }

        batch_var = float(cv.square().mean().detach())
        old = self.var_ema.get(key)
        var = batch_var if old is None else (
            self.beta * old + (1.0 - self.beta) * batch_var
        )
        self.var_ema[key] = float(var)
        scale = max(float(var) ** 0.5, self.eps)

        pre = centered / scale
        pv = pre[mask].float()
        qlo = torch.quantile(pv, self.q_low)
        qhi = torch.quantile(pv, self.q_high)
        adv = pre.clamp(qlo.to(pre.dtype), qhi.to(pre.dtype))
        adv = torch.where(mask, adv, torch.zeros_like(adv))
        av = adv[mask].float()
        changed = ((pv < qlo) | (pv > qhi)).float().mean()

        return adv, {
            "scale": scale,
            "pre_std": float(pv.std(unbiased=False).detach()),
            "post_std": float(av.std(unbiased=False).detach()),
            "clip_fraction": float(changed.detach()),
        }


class ProjectionTrainer(GRPOTrainer):
    """GRPO with explicit multi-constraint gradient projection."""

    def __init__(self, *args, config: ProjectionConfig | None = None, **kwargs):
        cfg = config or ProjectionConfig()
        cfg.validate_projection()
        if cfg.constraint_strategy != "none":
            raise ValueError(
                "ProjectionTrainer requires constraint_strategy='none'; "
                "constraints are handled explicitly in gradient space."
            )
        super().__init__(*args, config=cfg, **kwargs)
        self.config: ProjectionConfig
        self._projection_scaler = _PerChannelQClipScaler(
            beta=self.config.projection_advantage_beta,
            q_low=self.config.projection_q_low,
            q_high=self.config.projection_q_high,
            eps=self.config.advantage_eps,
        )
        self._projection_constraint_advantages: tuple[torch.Tensor, ...] = ()
        self._projection_constraint_active: tuple[bool, ...] = ()
        self._projection_constraint_violation_mean: tuple[float, ...] = ()
        self._projection_reference_log_prob: torch.Tensor | None = None
        self._projection_collect_metrics: Dict[str, float] = {}

    @staticmethod
    def _expanded_mask(
        valid_mask: torch.Tensor | None, value: torch.Tensor
    ) -> torch.Tensor:
        if valid_mask is None:
            base = torch.ones(
                (value.shape[0], value.shape[2]),
                dtype=torch.bool,
                device=value.device,
            )
        else:
            base = valid_mask.to(device=value.device, dtype=torch.bool)
        mask = base[:, None, :]
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        return mask.expand_as(value)

    @classmethod
    def _masked_constraint_mean(
        cls,
        value: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        mask = cls._expanded_mask(valid_mask, value)
        denom = mask.to(value.dtype).sum(dim=(0, 1, 2)).clamp_min(1.0)
        return (value * mask.to(value.dtype)).sum(dim=(0, 1, 2)) / denom

    @staticmethod
    def _flat_grads(grads, parameters) -> torch.Tensor:
        pieces = []
        for grad, parameter in zip(grads, parameters):
            if grad is None:
                pieces.append(
                    torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    .reshape(-1)
                    .float()
                )
            else:
                pieces.append(grad.detach().reshape(-1).float())
        if not pieces:
            raise RuntimeError("projection has no trainable gradient tensors")
        return torch.cat(pieces)

    @staticmethod
    def _assign_flat_grad(
        flat: torch.Tensor,
        parameters,
    ) -> None:
        offset = 0
        for parameter in parameters:
            n = parameter.numel()
            piece = flat[offset : offset + n].view_as(parameter)
            parameter.grad = piece.to(
                device=parameter.device,
                dtype=parameter.dtype,
            ).clone()
            offset += n
        if offset != flat.numel():
            raise RuntimeError("flat gradient size mismatch")

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        denom = a.norm() * b.norm()
        if float(denom) <= 1e-20:
            return 0.0
        return float(torch.dot(a, b) / denom)

    def _dykstra_halfspace_projection(
        self,
        preferred: torch.Tensor,
        constraint_grads: list[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Project onto intersection {g | c_j^T g >= 0} using Dykstra."""
        if not constraint_grads:
            return preferred, {
                "passes_used": 0.0,
                "correction_norm": 0.0,
                "correction_ratio": 0.0,
                "min_dot_after": 0.0,
                "min_cos_after": 0.0,
            }

        x = preferred.clone()
        corrections = [torch.zeros_like(x) for _ in constraint_grads]
        passes_used = 0

        for pass_index in range(int(self.config.projection_passes)):
            for j, c in enumerate(constraint_grads):
                norm2 = torch.dot(c, c)
                if float(norm2) <= 1e-20:
                    continue
                y = x + corrections[j]
                dot = torch.dot(c, y)
                if float(dot) < 0.0:
                    projected = y - (dot / norm2) * c
                else:
                    projected = y
                corrections[j] = y - projected
                x = projected

            passes_used = pass_index + 1
            min_dot = min(float(torch.dot(c, x)) for c in constraint_grads)
            scale = max(
                float(x.norm())
                * max(float(c.norm()) for c in constraint_grads),
                1e-20,
            )
            if min_dot >= -float(self.config.projection_tolerance) * scale:
                break

        correction = x - preferred
        min_dot_after = min(float(torch.dot(c, x)) for c in constraint_grads)
        cos_after = [self._cosine(c, x) for c in constraint_grads]
        return x, {
            "passes_used": float(passes_used),
            "correction_norm": float(correction.norm()),
            "correction_ratio": float(
                correction.norm() / preferred.norm().clamp_min(1e-20)
            ),
            "min_dot_after": min_dot_after,
            "min_cos_after": min(cos_after) if cos_after else 0.0,
        }

    def collect(
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
            valid_mask = features.get("mode_valid_mask")

            task_adv, task_diag = self._projection_scaler(
                "task", rewards, valid_mask=valid_mask
            )

            violation_mean = self._masked_constraint_mean(
                constraints.violation, valid_mask
            )
            signed_mean = self._masked_constraint_mean(
                constraints.signed_residual, valid_mask
            )

            constraint_advantages = []
            active = []
            metrics: Dict[str, float] = {
                "reward/task_reward_mean": float(rewards.mean().detach()),
                "reward/legacy_w4_reward_mean": float(
                    legacy_rewards.mean().detach()
                ),
                "reward/effective_reward_mean": float(rewards.mean().detach()),
                "projection/task_adv_std": float(task_diag["post_std"]),
                "constraint/feasible_fraction": float(
                    constraints.feasible_mask.float().mean().detach()
                ),
                "constraint/max_violation_mean": float(
                    constraints.violation.amax(dim=-1).mean().detach()
                ),
            }

            for index, name in enumerate(self.config.constraint_names):
                safety_reward = -constraints.violation[..., index]
                adv_j, diag_j = self._projection_scaler(
                    f"constraint/{name}",
                    safety_reward,
                    valid_mask=valid_mask,
                )
                constraint_advantages.append(adv_j)
                is_active = (
                    float(violation_mean[index].detach())
                    > float(self.config.projection_active_threshold)
                )
                active.append(bool(is_active))
                metrics[f"constraint/{name}_signed_mean"] = float(
                    signed_mean[index].detach()
                )
                metrics[f"constraint/{name}_violation_mean"] = float(
                    violation_mean[index].detach()
                )
                metrics[f"projection/{name}_active"] = float(is_active)
                metrics[f"projection/{name}_adv_std"] = float(
                    diag_j["post_std"]
                )
                metrics[f"projection/{name}_adv_scale"] = float(
                    diag_j["scale"]
                )
                metrics[f"projection/{name}_qclip_fraction"] = float(
                    diag_j["clip_fraction"]
                )

            # The parent Speed V1 train_step computes frozen-reference replay
            # once and passes it to update(reference_log_prob=...).  Keep this
            # field only as a compatibility fallback for direct/manual update().
            self._projection_reference_log_prob = None
            self._projection_constraint_advantages = tuple(
                constraint_advantages
            )
            self._projection_constraint_active = tuple(active)
            self._projection_constraint_violation_mean = tuple(
                float(v.detach()) for v in violation_mean
            )
            self._projection_collect_metrics = metrics

        return trace, rewards, task_adv, metrics

    def update(
        self,
        trace,
        advantages: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        reference_log_prob: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        if len(self._projection_constraint_advantages) != len(
            self.config.constraint_names
        ):
            raise RuntimeError("projection constraint advantages are unavailable")

        # GRU refinement is trainable. cuDNN RNN backward requires the
        # forward replay that builds this autograd graph to run in train mode.
        # collect() remains eval/no_grad; only differentiable replay/update is train.
        self.model.train()
        parameters = tuple(
            p for p in self.model.parameters() if p.requires_grad
        )
        if not parameters:
            raise RuntimeError("no trainable parameters for projection update")

        new_log_prob = self.sampler.replay(trace)
        task_objective = grpo_clipped_objective(
            new_log_prob,
            trace.old_log_prob,
            advantages,
            clip_eps=self.config.clip_eps,
            valid_mask=valid_mask,
        )
        ref_log_prob = reference_log_prob
        if ref_log_prob is None:
            ref_log_prob = self._projection_reference_log_prob
        if ref_log_prob is None:
            # Compatibility fallback only. Normal Speed V1 training passes the
            # cached tensor and therefore does not execute this reference replay.
            self.reference_model.eval()
            with torch.no_grad():
                ref_log_prob = self.sampler.replay(
                    trace, model=self.reference_model
                ).detach()
        else:
            ref_log_prob = ref_log_prob.detach()
        ref_kl = self._reference_kl(new_log_prob, ref_log_prob)
        preferred_loss = (
            task_objective.policy_loss + self.config.kl_coef * ref_kl
        )

        preferred_grads = torch.autograd.grad(
            preferred_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        preferred_flat = self._flat_grads(preferred_grads, parameters)

        active_names = []
        active_grads = []
        gradient_metrics: Dict[str, float] = {}

        for index, name in enumerate(self.config.constraint_names):
            if not self._projection_constraint_active[index]:
                continue
            c_adv = self._projection_constraint_advantages[index]
            c_objective = grpo_clipped_objective(
                new_log_prob,
                trace.old_log_prob,
                c_adv,
                clip_eps=self.config.clip_eps,
                valid_mask=valid_mask,
            )
            c_grads = torch.autograd.grad(
                c_objective.policy_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            c_flat = self._flat_grads(c_grads, parameters)
            if float(c_flat.norm()) <= 1e-20:
                continue
            active_names.append(name)
            active_grads.append(c_flat)
            gradient_metrics[
                f"projection/preferred_{name}_grad_cosine"
            ] = self._cosine(preferred_flat, c_flat)
            gradient_metrics[
                f"projection/{name}_grad_norm"
            ] = float(c_flat.norm())

        projected_flat, proj_diag = self._dykstra_halfspace_projection(
            preferred_flat,
            active_grads,
        )

        for name, c_flat in zip(active_names, active_grads):
            gradient_metrics[
                f"projection/final_{name}_grad_cosine"
            ] = self._cosine(projected_flat, c_flat)

        self.optimizer.zero_grad(set_to_none=True)
        before_update = [
            p.detach().clone() for p in parameters
        ]
        self._assign_flat_grad(projected_flat, parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        update_sq = torch.zeros(
            (), device=parameters[0].device, dtype=torch.float32
        )
        for parameter, before in zip(parameters, before_update):
            update_sq = update_sq + (
                parameter.detach() - before
            ).float().square().sum()
        parameter_update_norm = torch.sqrt(update_sq)

        metrics: Dict[str, float] = {
            "loss": float(preferred_loss.detach()),
            "policy_loss": float(task_objective.policy_loss.detach()),
            "reference_kl": float(ref_kl.detach()),
            "approx_kl": float(task_objective.approx_kl.detach()),
            "clip_fraction": float(task_objective.clip_fraction.detach()),
            "ratio_mean": float(task_objective.ratio_mean.detach()),
            "ratio_std": float(task_objective.ratio_std.detach()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach()),
            "parameter_update_norm": float(parameter_update_norm.detach()),
            "projection/active_constraint_count": float(len(active_grads)),
            "projection/preferred_grad_norm": float(preferred_flat.norm()),
            "projection/projected_grad_norm": float(projected_flat.norm()),
            "projection/passes_used": proj_diag["passes_used"],
            "projection/correction_norm": proj_diag["correction_norm"],
            "projection/correction_ratio": proj_diag["correction_ratio"],
            "projection/min_dot_after": proj_diag["min_dot_after"],
            "projection/min_cos_after": proj_diag["min_cos_after"],
        }
        metrics.update(gradient_metrics)
        return metrics

    def constraint_state_dict(self) -> dict:
        return {
            "strategy": "projection_v1",
            "constraint_names": tuple(self.config.constraint_names),
            "projection": {
                "advantage_beta": self.config.projection_advantage_beta,
                "q_low": self.config.projection_q_low,
                "q_high": self.config.projection_q_high,
                "active_threshold": self.config.projection_active_threshold,
                "passes": self.config.projection_passes,
                "tolerance": self.config.projection_tolerance,
            },
        }
