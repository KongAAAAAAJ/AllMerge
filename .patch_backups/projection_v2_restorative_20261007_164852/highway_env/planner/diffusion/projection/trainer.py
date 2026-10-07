from __future__ import annotations

# GRPO_PROJECTION_BASELINE_V1_3_LATEST_20261007

from dataclasses import dataclass
from typing import Dict

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
            raise ValueError("invalid projection quantiles")
        if self.projection_active_threshold < 0.0:
            raise ValueError("projection_active_threshold must be >= 0")
        if self.projection_passes < 1:
            raise ValueError("projection_passes must be >= 1")
        if self.projection_tolerance < 0.0:
            raise ValueError("projection_tolerance must be >= 0")


class _PerChannelQClipScaler:
    """Independent EMA scale + Q5/Q95 clipping for task/constraint channels."""

    def __init__(
        self,
        *,
        beta: float,
        q_low: float,
        q_high: float,
        eps: float,
    ) -> None:
        self.beta = float(beta)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.eps = float(eps)
        self.var_ema: Dict[str, float] = {}

    @staticmethod
    def _mask(
        valid_mask: torch.Tensor | None,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if valid_mask is None:
            base = torch.ones(
                (value.shape[0], value.shape[2]),
                dtype=torch.bool,
                device=value.device,
            )
        else:
            if tuple(valid_mask.shape) != (value.shape[0], value.shape[2]):
                raise ValueError("valid_mask must be [B,M]")
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
            raise ValueError(
                f"{key}: advantage source must be [B,G,M], got {tuple(value.shape)}"
            )
        mask = self._mask(valid_mask, value)
        centered = value - value.mean(dim=1, keepdim=True)
        centered = torch.where(mask, centered, torch.zeros_like(centered))
        valid = centered[mask].float()

        if valid.numel() == 0:
            return torch.zeros_like(value), {
                "scale": 1.0,
                "pre_std": 0.0,
                "post_std": 0.0,
                "clip_fraction": 0.0,
            }

        if not bool(torch.isfinite(valid).all()):
            raise FloatingPointError(
                f"non-finite values before projection advantage normalization: {key}"
            )

        batch_var = float(valid.square().mean().detach().cpu())
        old = self.var_ema.get(key)
        var_ema = (
            batch_var
            if old is None
            else self.beta * old + (1.0 - self.beta) * batch_var
        )
        if not torch.isfinite(torch.tensor(var_ema)):
            raise FloatingPointError(f"non-finite EMA variance for {key}")
        self.var_ema[key] = float(var_ema)

        scale = max(float(var_ema) ** 0.5, self.eps)
        pre = centered / scale
        pv = pre[mask].float()
        qlo = torch.quantile(pv, self.q_low)
        qhi = torch.quantile(pv, self.q_high)
        advantage = pre.clamp(qlo.to(pre.dtype), qhi.to(pre.dtype))
        advantage = torch.where(mask, advantage, torch.zeros_like(advantage))

        av = advantage[mask].float()
        if not bool(torch.isfinite(av).all()):
            raise FloatingPointError(f"non-finite Q-clipped advantage for {key}")

        changed = ((pv < qlo) | (pv > qhi)).float().mean()
        return advantage, {
            "scale": float(scale),
            "pre_std": float(pv.std(unbiased=False).detach().cpu()),
            "post_std": float(av.std(unbiased=False).detach().cpu()),
            "clip_fraction": float(changed.detach().cpu()),
        }


class ProjectionTrainer(GRPOTrainer):
    """Explicit multi-constraint gradient projection baseline.

    Preferred gradient:
        g0 = grad(L_task_GRPO + beta_KL * KL)

    Each currently violated constraint builds a safety GRPO loss from
    Q5/Q95-normalized reward -violation:
        gj = grad(L_safety_j)

    The optimizer gradient is the Euclidean projection of g0 onto
        {g | gj^T g >= 0 for all active j}.
    Therefore the descent step -g is first-order non-increasing for every
    active safety surrogate.

    There are deliberately no lambdas, penalty rewards, constraint priority,
    or positive restoration margins in this baseline.
    """

    def __init__(
        self,
        *args,
        config: ProjectionConfig | None = None,
        **kwargs,
    ) -> None:
        cfg = config or ProjectionConfig()
        cfg.validate_projection()
        if cfg.constraint_strategy != "none":
            raise ValueError(
                "ProjectionTrainer requires constraint_strategy='none'; "
                "constraints are handled in gradient space."
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

    @staticmethod
    def _expanded_mask(
        valid_mask: torch.Tensor | None,
        value: torch.Tensor,
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
        mask_f = mask.to(value.dtype)
        denom = mask_f.sum(dim=(0, 1, 2)).clamp_min(1.0)
        return (value * mask_f).sum(dim=(0, 1, 2)) / denom

    @staticmethod
    def _flat_grads(grads, parameters) -> torch.Tensor:
        pieces = []
        for grad, parameter in zip(grads, parameters):
            if grad is None:
                piece = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )
            else:
                piece = grad
            pieces.append(piece.detach().reshape(-1).float())
        if not pieces:
            raise RuntimeError("projection found no trainable gradient tensors")
        return torch.cat(pieces)

    @staticmethod
    def _assign_flat_grad(
        flat: torch.Tensor,
        parameters,
    ) -> None:
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            piece = flat[offset : offset + count].view_as(parameter)
            parameter.grad = piece.to(
                device=parameter.device,
                dtype=parameter.dtype,
            ).clone()
            offset += count
        if offset != flat.numel():
            raise RuntimeError("projected gradient size mismatch")

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        denom = a.norm() * b.norm()
        if float(denom) <= 1e-20:
            return 0.0
        return float((torch.dot(a, b) / denom).detach().cpu())

    @staticmethod
    def _require_finite(name: str, value: torch.Tensor) -> None:
        if not bool(torch.isfinite(value).all()):
            finite_fraction = float(
                torch.isfinite(value).float().mean().detach().cpu()
            )
            raise FloatingPointError(
                f"{name} contains NaN/Inf "
                f"(finite_fraction={finite_fraction:.6f})"
            )

    def _normalized_constraint_normals(
        self,
        constraint_grads: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        normals: list[torch.Tensor] = []
        for index, grad in enumerate(constraint_grads):
            self._require_finite(f"constraint_grad[{index}]", grad)
            norm = grad.norm()
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError(
                    f"constraint_grad[{index}] has non-finite norm"
                )
            if float(norm) <= 1e-20:
                continue
            # Positive scaling does not change the half-space gj^T g >= 0.
            # Unit normals substantially improve numerical conditioning.
            normals.append(grad / norm)
        return normals

    def _dykstra_halfspace_projection(
        self,
        preferred: torch.Tensor,
        constraint_grads: list[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        self._require_finite("preferred_grad", preferred)
        normals = self._normalized_constraint_normals(constraint_grads)
        if not normals:
            return preferred, {
                "passes_used": 0.0,
                "correction_norm": 0.0,
                "correction_ratio": 0.0,
                "min_dot_after": 0.0,
                "min_cos_after": 0.0,
            }

        x = preferred.clone()
        corrections = [torch.zeros_like(x) for _ in normals]
        passes_used = 0
        tolerance = float(self.config.projection_tolerance)

        for pass_index in range(int(self.config.projection_passes)):
            for index, normal in enumerate(normals):
                y = x + corrections[index]
                dot = torch.dot(normal, y)
                self._require_finite(
                    f"dykstra_dot_pass{pass_index}_constraint{index}",
                    dot.reshape(1),
                )
                projected = (
                    y - dot * normal
                    if float(dot) < 0.0
                    else y
                )
                corrections[index] = y - projected
                x = projected
                self._require_finite(
                    f"dykstra_x_pass{pass_index}_constraint{index}",
                    x,
                )

            passes_used = pass_index + 1
            dots = [float(torch.dot(normal, x).detach().cpu()) for normal in normals]
            if min(dots) >= -tolerance * max(float(x.norm()), 1.0):
                break

        self._require_finite("projected_grad", x)
        correction = x - preferred
        cos_after = [self._cosine(normal, x) for normal in normals]
        dots_after = [
            float(torch.dot(normal, x).detach().cpu())
            for normal in normals
        ]
        preferred_norm = preferred.norm().clamp_min(1e-20)
        return x, {
            "passes_used": float(passes_used),
            "correction_norm": float(correction.norm().detach().cpu()),
            "correction_ratio": float(
                (correction.norm() / preferred_norm).detach().cpu()
            ),
            "min_dot_after": min(dots_after),
            "min_cos_after": min(cos_after),
        }

    def collect(
        self,
        features: Dict[str, torch.Tensor],
        *,
        context=None,
        generator: torch.Generator | None = None,
    ):
        # EXACT latest-GRPO sampling semantics: dropout stays disabled.
        self.model.eval()
        with torch.no_grad():
            trace = self.sampler.sample(features, generator=generator)
            self._require_finite("sampled_candidates", trace.candidates)
            self._require_finite("sampled_old_log_prob", trace.old_log_prob)

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

            task_advantage, task_diag = self._projection_scaler(
                "task",
                rewards,
                valid_mask=valid_mask,
            )
            violation_mean = self._masked_constraint_mean(
                constraints.violation,
                valid_mask,
            )
            signed_mean = self._masked_constraint_mean(
                constraints.signed_residual,
                valid_mask,
            )

            constraint_advantages = []
            active = []
            metrics: Dict[str, float] = {
                "reward/task_reward_mean": float(rewards.mean().detach().cpu()),
                "reward/legacy_w4_reward_mean": float(
                    legacy_rewards.mean().detach().cpu()
                ),
                "reward/effective_reward_mean": float(
                    rewards.mean().detach().cpu()
                ),
                "projection/task_adv_std": float(task_diag["post_std"]),
                "constraint/feasible_fraction": float(
                    constraints.feasible_mask.float().mean().detach().cpu()
                ),
                "constraint/max_violation_mean": float(
                    constraints.violation.amax(dim=-1).mean().detach().cpu()
                ),
            }

            for index, name in enumerate(self.config.constraint_names):
                safety_reward = -constraints.violation[..., index]
                advantage_j, diag_j = self._projection_scaler(
                    f"constraint/{name}",
                    safety_reward,
                    valid_mask=valid_mask,
                )
                constraint_advantages.append(advantage_j)
                is_active = (
                    float(violation_mean[index].detach().cpu())
                    > float(self.config.projection_active_threshold)
                )
                active.append(bool(is_active))

                metrics[f"constraint/{name}_signed_mean"] = float(
                    signed_mean[index].detach().cpu()
                )
                metrics[f"constraint/{name}_violation_mean"] = float(
                    violation_mean[index].detach().cpu()
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

            self._projection_constraint_advantages = tuple(
                constraint_advantages
            )
            self._projection_constraint_active = tuple(active)

        return trace, rewards, task_advantage, metrics

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
            raise RuntimeError(
                "projection constraint advantages are unavailable"
            )

        # EXACT latest-GRPO update semantics:
        #   - full planner remains eval => Transformer/MLP dropout is disabled;
        #   - only nn.GRU modules are train => cuDNN stores reserve space for backward.
        self.model.eval()
        for module in self._model_gru_modules:
            module.train()
            module.flatten_parameters()

        self.optimizer.zero_grad(set_to_none=True)
        trainable_params = self._trainable_parameters
        before_update = [
            parameter.detach().clone()
            for parameter in trainable_params
        ]

        new_log_prob = self.sampler.replay(trace)
        self._require_finite("new_log_prob", new_log_prob)

        task_objective = grpo_clipped_objective(
            new_log_prob,
            trace.old_log_prob,
            advantages,
            clip_eps=self.config.clip_eps,
            valid_mask=valid_mask,
        )

        if reference_log_prob is None:
            self.reference_model.eval()
            self._flatten_gru_parameters(self._reference_gru_modules)
            with torch.no_grad():
                ref_log_prob = self.sampler.replay(
                    trace,
                    model=self.reference_model,
                )
        else:
            ref_log_prob = reference_log_prob
        ref_log_prob = ref_log_prob.detach()
        self._require_finite("reference_log_prob", ref_log_prob)

        ref_kl = self._reference_kl(new_log_prob, ref_log_prob)
        preferred_loss = (
            task_objective.policy_loss
            + float(self.config.kl_coef) * ref_kl
        )
        if not bool(torch.isfinite(preferred_loss)):
            raise FloatingPointError(
                "non-finite preferred task+KL loss before projection"
            )

        preferred_grads = torch.autograd.grad(
            preferred_loss,
            trainable_params,
            retain_graph=True,
            allow_unused=True,
        )
        preferred_flat = self._flat_grads(
            preferred_grads,
            trainable_params,
        )
        self._require_finite("preferred_grad", preferred_flat)

        active_names: list[str] = []
        active_grads: list[torch.Tensor] = []
        gradient_metrics: Dict[str, float] = {}

        for index, name in enumerate(self.config.constraint_names):
            if not self._projection_constraint_active[index]:
                continue

            constraint_objective = grpo_clipped_objective(
                new_log_prob,
                trace.old_log_prob,
                self._projection_constraint_advantages[index],
                clip_eps=self.config.clip_eps,
                valid_mask=valid_mask,
            )
            if not bool(torch.isfinite(constraint_objective.policy_loss)):
                raise FloatingPointError(
                    f"non-finite constraint policy loss: {name}"
                )

            grads_j = torch.autograd.grad(
                constraint_objective.policy_loss,
                trainable_params,
                retain_graph=True,
                allow_unused=True,
            )
            flat_j = self._flat_grads(grads_j, trainable_params)
            self._require_finite(f"constraint_grad/{name}", flat_j)

            norm_j = flat_j.norm()
            if float(norm_j) <= 1e-20:
                gradient_metrics[
                    f"projection/{name}_zero_grad"
                ] = 1.0
                continue

            active_names.append(name)
            active_grads.append(flat_j)
            gradient_metrics[
                f"projection/preferred_{name}_grad_cosine"
            ] = self._cosine(preferred_flat, flat_j)
            gradient_metrics[
                f"projection/{name}_grad_norm"
            ] = float(norm_j.detach().cpu())

        projected_flat, projection_diag = (
            self._dykstra_halfspace_projection(
                preferred_flat,
                active_grads,
            )
        )
        self._require_finite("projected_grad", projected_flat)

        for name, grad_j in zip(active_names, active_grads):
            gradient_metrics[
                f"projection/final_{name}_grad_cosine"
            ] = self._cosine(projected_flat, grad_j)

        self._assign_flat_grad(projected_flat, trainable_params)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=self.config.max_grad_norm,
            error_if_nonfinite=True,
        )

        # Verify again after clipping, before AdamW can consume the gradient.
        for index, parameter in enumerate(trainable_params):
            if parameter.grad is not None:
                self._require_finite(
                    f"clipped_grad[{index}]",
                    parameter.grad,
                )

        self.optimizer.step()

        # Do not allow a bad optimizer step to silently poison later validation.
        nonfinite_parameter = None
        for index, parameter in enumerate(trainable_params):
            if not bool(torch.isfinite(parameter).all()):
                nonfinite_parameter = index
                break
        if nonfinite_parameter is not None:
            with torch.no_grad():
                for parameter, before in zip(
                    trainable_params,
                    before_update,
                ):
                    parameter.copy_(before)
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "optimizer produced NaN/Inf in trainable parameter "
                f"index={nonfinite_parameter}; parameters were rolled back. "
                "Restart this smoke from the pretrained checkpoint."
            )

        update_sq = torch.zeros(
            (),
            device=trainable_params[0].device,
            dtype=torch.float32,
        )
        for parameter, before in zip(
            trainable_params,
            before_update,
        ):
            update_sq = update_sq + (
                parameter.detach() - before
            ).float().square().sum()
        parameter_update_norm = torch.sqrt(update_sq)

        metrics: Dict[str, float] = {
            "loss": float(preferred_loss.detach().cpu()),
            "policy_loss": float(
                task_objective.policy_loss.detach().cpu()
            ),
            "reference_kl": float(ref_kl.detach().cpu()),
            "approx_kl": float(
                task_objective.approx_kl.detach().cpu()
            ),
            "clip_fraction": float(
                task_objective.clip_fraction.detach().cpu()
            ),
            "ratio_mean": float(
                task_objective.ratio_mean.detach().cpu()
            ),
            "ratio_std": float(
                task_objective.ratio_std.detach().cpu()
            ),
            "grad_norm": float(
                torch.as_tensor(grad_norm).detach().cpu()
            ),
            "parameter_update_norm": float(
                parameter_update_norm.detach().cpu()
            ),
            "projection/active_constraint_count": float(
                len(active_grads)
            ),
            "projection/preferred_grad_norm": float(
                preferred_flat.norm().detach().cpu()
            ),
            "projection/projected_grad_norm": float(
                projected_flat.norm().detach().cpu()
            ),
            "projection/passes_used": projection_diag[
                "passes_used"
            ],
            "projection/correction_norm": projection_diag[
                "correction_norm"
            ],
            "projection/correction_ratio": projection_diag[
                "correction_ratio"
            ],
            "projection/min_dot_after": projection_diag[
                "min_dot_after"
            ],
            "projection/min_cos_after": projection_diag[
                "min_cos_after"
            ],
        }
        metrics.update(gradient_metrics)
        return metrics

    def constraint_state_dict(self) -> dict:
        return {
            "strategy": "projection_v1_3",
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
