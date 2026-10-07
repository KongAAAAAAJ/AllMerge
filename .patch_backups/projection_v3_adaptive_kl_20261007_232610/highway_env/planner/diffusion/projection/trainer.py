from __future__ import annotations

# GRPO_PROJECTION_V2_RESTORATIVE_20261007
# GRPO_PROJECTION_V2_1_DYKSTRA_RESCUE_20261007

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
    projection_passes: int = 16
    projection_tolerance: float = 1e-6
    projection_restoration_strength: float = 0.25
    projection_restoration_scale_beta: float = 0.95
    projection_restoration_scale_floor: float = 0.10
    projection_restoration_pressure_clip: float = 1.0
    projection_margin_backoff_factor: float = 0.5
    projection_margin_backoff_steps: int = 4

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
        if not 0.0 < self.projection_restoration_strength <= 1.0:
            raise ValueError("projection_restoration_strength must be in (0,1]")
        if not 0.0 <= self.projection_restoration_scale_beta < 1.0:
            raise ValueError("projection_restoration_scale_beta must be in [0,1)")
        if self.projection_restoration_scale_floor <= 0.0:
            raise ValueError("projection_restoration_scale_floor must be > 0")
        if self.projection_restoration_pressure_clip <= 0.0:
            raise ValueError("projection_restoration_pressure_clip must be > 0")
        if not 0.0 < self.projection_margin_backoff_factor < 1.0:
            raise ValueError("projection_margin_backoff_factor must be in (0,1)")
        if self.projection_margin_backoff_steps < 0:
            raise ValueError("projection_margin_backoff_steps must be >= 0")


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
    """Restorative multi-constraint pre-optimizer gradient projection.

    The preferred update is the *same clipped gradient budget* that Vanilla
    GRPO would send to the optimizer:

        g0_raw = grad(L_task_GRPO + beta_KL * KL)
        g0 = Proj_{||g||<=Gmax}(g0_raw)

    Each active safety channel builds a Q5/Q95-normalized safety surrogate
    from reward ``-violation`` and a unit gradient normal ``n_j``.  Current
    violation is converted to a dimensionless restoration pressure ``p_j``
    using an EMA scale, then to a positive margin

        b_j = restoration_strength * p_j * Gmax.

    We project onto the intersection

        n_j^T g >= b_j,     ||g|| <= Gmax.

    Therefore the safety requirement survives the usual global gradient-norm
    cap instead of being invalidated by clipping after projection.  Dykstra's
    algorithm handles the half-spaces and L2 ball jointly.  If positive
    margins conflict, all margins are backed off by one shared scalar; the
    zero-margin cone is a final guaranteed-feasible fallback and is logged.

    No Lagrange multipliers, reward penalties, constraint priorities, or
    learned safety weights are used.
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
        self._projection_restoration_pressure: tuple[float, ...] = ()
        self._projection_restoration_scale_ema = [
            None for _ in self.config.constraint_names
        ]
        self._projection_epoch_stats: list[dict[str, float]] = []

    @staticmethod
    def _ema(old: float | None, new: float, beta: float) -> float:
        return float(new) if old is None else float(beta * old + (1.0 - beta) * new)

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
    def _assign_flat_grad(flat: torch.Tensor, parameters) -> None:
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

    @staticmethod
    def _clip_to_l2_ball(value: torch.Tensor, radius: float) -> torch.Tensor:
        norm = value.norm()
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("non-finite gradient norm before budget projection")
        scale = (
            torch.as_tensor(float(radius), device=value.device, dtype=norm.dtype)
            / norm.clamp_min(1e-20)
        ).clamp(max=1.0)
        return value * scale.to(value.dtype)

    def _unit_constraint_normals(
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
            normals.append(grad / norm)
        return normals

    def _dykstra_shifted_halfspaces_ball_once(
        self,
        preferred: torch.Tensor,
        normals: list[torch.Tensor],
        margins: list[float],
        *,
        radius: float,
        max_passes: int | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        if len(normals) != len(margins):
            raise ValueError("normal/margin count mismatch")
        self._require_finite("preferred_budget_grad", preferred)
        if float(preferred.norm()) > float(radius) * (1.0 + 1e-5):
            raise ValueError("preferred gradient is outside projection budget ball")

        x = preferred.clone()
        # One Dykstra correction per half-space plus one for the L2 ball.
        corrections = [torch.zeros_like(x) for _ in range(len(normals) + 1)]
        tolerance = float(self.config.projection_tolerance)
        scale = max(float(radius), 1.0)
        passes_used = 0
        converged = False

        pass_limit = int(
            self.config.projection_passes if max_passes is None else max_passes
        )
        if pass_limit <= 0:
            raise ValueError("Dykstra max_passes must be positive")

        for pass_index in range(pass_limit):
            for index, (normal, margin) in enumerate(zip(normals, margins)):
                y = x + corrections[index]
                dot = torch.dot(normal, y)
                margin_tensor = torch.as_tensor(
                    float(margin), device=x.device, dtype=dot.dtype
                )
                deficit = (margin_tensor - dot).clamp_min(0.0)
                projected = y + deficit.to(y.dtype) * normal
                corrections[index] = y - projected
                x = projected

            ball_index = len(normals)
            y = x + corrections[ball_index]
            projected_ball = self._clip_to_l2_ball(y, radius)
            corrections[ball_index] = y - projected_ball
            x = projected_ball

            # One synchronization per pass for the convergence check, rather
            # than one synchronization per constraint projection.
            passes_used = pass_index + 1
            if normals:
                residual_tensor = torch.stack([
                    torch.dot(normal, x) - float(margin)
                    for normal, margin in zip(normals, margins)
                ])
                min_residual = float(residual_tensor.min().detach().cpu())
            else:
                min_residual = 0.0
            norm_excess = max(float(x.norm().detach().cpu()) - float(radius), 0.0)
            if (
                min_residual >= -tolerance * scale
                and norm_excess <= tolerance * scale
            ):
                converged = True
                break

        residuals = [
            float(torch.dot(normal, x).detach().cpu()) - float(margin)
            for normal, margin in zip(normals, margins)
        ]
        min_residual = min(residuals) if residuals else 0.0
        norm_value = float(x.norm().detach().cpu())
        return x, {
            "passes_used": float(passes_used),
            "converged": float(converged),
            "min_margin_residual_after": float(min_residual),
            "norm_after": norm_value,
            "ball_excess_after": max(norm_value - float(radius), 0.0),
        }

    def _restorative_projection(
        self,
        preferred: torch.Tensor,
        constraint_grads: list[torch.Tensor],
        pressures: list[float],
    ) -> tuple[torch.Tensor, Dict[str, float], list[torch.Tensor], list[float]]:
        self._require_finite("preferred_budget_grad", preferred)
        if len(constraint_grads) != len(pressures):
            raise ValueError("constraint gradient/pressure count mismatch")

        normals = self._unit_constraint_normals(constraint_grads)
        if len(normals) != len(constraint_grads):
            raise RuntimeError(
                "zero-norm constraint gradient reached restorative projection; "
                "zero gradients must be filtered before this call"
            )
        if not normals:
            return preferred, {
                "passes_used": 0.0,
                "correction_norm": 0.0,
                "correction_ratio": 0.0,
                "margin_scale_used": 1.0,
                "backoff_count": 0.0,
                "fallback_zero_margin": 0.0,
                "violated_before_count": 0.0,
                "violated_after_count": 0.0,
                "min_margin_residual_before": 0.0,
                "min_margin_residual_after": 0.0,
                "projected_grad_norm": float(preferred.norm().detach().cpu()),
                "hard_feasible": 1.0,
                "solver_rescue_passes": 0.0,
                "exact_zero_fallback": 0.0,
            }, normals, []

        radius = float(self.config.max_grad_norm)
        base_margins = [
            float(self.config.projection_restoration_strength)
            * max(0.0, min(float(p), float(self.config.projection_restoration_pressure_clip)))
            * radius
            for p in pressures
        ]
        factor = float(self.config.projection_margin_backoff_factor)
        attempts = int(self.config.projection_margin_backoff_steps)
        selected = None
        exact_zero_fallback = False

        for backoff_count in range(attempts + 1):
            margin_scale = factor ** backoff_count
            margins = [margin_scale * value for value in base_margins]
            candidate, diag = self._dykstra_shifted_halfspaces_ball_once(
                preferred,
                normals,
                margins,
                radius=radius,
            )
            if bool(diag["converged"]):
                selected = (
                    candidate,
                    diag,
                    margins,
                    margin_scale,
                    backoff_count,
                    False,
                )
                break

        if selected is None:
            # Numerical-rescue stage. A narrow feasible intersection can need
            # many more cyclic Dykstra passes than the normal fast path.
            # Retry the smallest positive margin using a larger pass budget
            # before dropping the restoration margin.
            rescue_passes = max(128, int(self.config.projection_passes) * 8)
            margin_scale = factor ** attempts
            margins = [margin_scale * value for value in base_margins]
            candidate, diag = self._dykstra_shifted_halfspaces_ball_once(
                preferred,
                normals,
                margins,
                radius=radius,
                max_passes=rescue_passes,
            )
            if bool(diag["converged"]):
                selected = (
                    candidate,
                    diag,
                    margins,
                    margin_scale,
                    attempts,
                    False,
                )

        if selected is None:
            # Guaranteed-feasible zero-margin fallback. Retry the same
            # zero-margin mathematical problem with the larger numerical
            # pass budget.
            margins = [0.0 for _ in base_margins]
            candidate, diag = self._dykstra_shifted_halfspaces_ball_once(
                preferred,
                normals,
                margins,
                radius=radius,
                max_passes=rescue_passes,
            )
            if not bool(diag["converged"]):
                # The zero-margin cone and the L2 ball both contain g=0.
                # If cyclic Dykstra still misses tolerance because of nearly
                # opposing normals, use that analytically exact feasible point.
                # It remains explicitly logged as a zero-margin fallback.
                candidate = torch.zeros_like(preferred)
                diag = {
                    "passes_used": float(rescue_passes),
                    "converged": 1.0,
                    "min_margin_residual_after": 0.0,
                    "norm_after": 0.0,
                    "ball_excess_after": 0.0,
                }
                exact_zero_fallback = True
            selected = (
                candidate,
                diag,
                margins,
                0.0,
                attempts + 1,
                True,
            )

        x, diag, margins, margin_scale, backoff_count, fallback = selected
        before_residuals = [
            float(torch.dot(normal, preferred).detach().cpu()) - margin
            for normal, margin in zip(normals, margins)
        ]
        after_residuals = [
            float(torch.dot(normal, x).detach().cpu()) - margin
            for normal, margin in zip(normals, margins)
        ]
        tolerance_abs = float(self.config.projection_tolerance) * max(radius, 1.0)
        violated_before = sum(value < -tolerance_abs for value in before_residuals)
        violated_after = sum(value < -tolerance_abs for value in after_residuals)
        correction = x - preferred
        preferred_norm = preferred.norm().clamp_min(1e-20)

        result = {
            "passes_used": float(diag["passes_used"]),
            "correction_norm": float(correction.norm().detach().cpu()),
            "correction_ratio": float(
                (correction.norm() / preferred_norm).detach().cpu()
            ),
            "margin_scale_used": float(margin_scale),
            "backoff_count": float(backoff_count),
            "fallback_zero_margin": float(fallback),
            "violated_before_count": float(violated_before),
            "violated_after_count": float(violated_after),
            "min_margin_residual_before": min(before_residuals),
            "min_margin_residual_after": min(after_residuals),
            "projected_grad_norm": float(x.norm().detach().cpu()),
            "hard_feasible": float(violated_after == 0),
            "solver_rescue_passes": float(
                max(
                    0.0,
                    float(diag["passes_used"])
                    - float(self.config.projection_passes),
                )
            ),
            "exact_zero_fallback": float(exact_zero_fallback),
        }
        return x, result, normals, margins

    def collect(
        self,
        features: Dict[str, torch.Tensor],
        *,
        context=None,
        generator: torch.Generator | None = None,
    ):
        # Exact latest-GRPO sampling semantics: all dropout stays disabled.
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
                "task", rewards, valid_mask=valid_mask
            )
            violation_mean = self._masked_constraint_mean(
                constraints.violation, valid_mask
            )
            signed_mean = self._masked_constraint_mean(
                constraints.signed_residual, valid_mask
            )
            abs_signed_mean = self._masked_constraint_mean(
                constraints.signed_residual.abs(), valid_mask
            )

            constraint_advantages = []
            active = []
            pressures = []
            metrics: Dict[str, float] = {
                "reward/task_reward_mean": float(rewards.mean().detach().cpu()),
                "reward/legacy_w4_reward_mean": float(
                    legacy_rewards.mean().detach().cpu()
                ),
                "reward/effective_reward_mean": float(rewards.mean().detach().cpu()),
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

                observed_scale = max(
                    float(abs_signed_mean[index].detach().cpu()),
                    float(self.config.projection_restoration_scale_floor),
                )
                old_scale = self._projection_restoration_scale_ema[index]
                scale_ema = self._ema(
                    old_scale,
                    observed_scale,
                    float(self.config.projection_restoration_scale_beta),
                )
                scale_ema = max(
                    scale_ema,
                    float(self.config.projection_restoration_scale_floor),
                )
                self._projection_restoration_scale_ema[index] = scale_ema
                pressure = float(violation_mean[index].detach().cpu()) / scale_ema
                pressure = max(
                    0.0,
                    min(
                        pressure,
                        float(self.config.projection_restoration_pressure_clip),
                    ),
                )
                is_active = (
                    float(violation_mean[index].detach().cpu())
                    > float(self.config.projection_active_threshold)
                    and pressure > 0.0
                )
                pressures.append(pressure)
                active.append(bool(is_active))

                metrics[f"constraint/{name}_signed_mean"] = float(
                    signed_mean[index].detach().cpu()
                )
                metrics[f"constraint/{name}_violation_mean"] = float(
                    violation_mean[index].detach().cpu()
                )
                metrics[f"projection/{name}_active"] = float(is_active)
                metrics[f"projection/{name}_restoration_scale"] = float(scale_ema)
                metrics[f"projection/{name}_restoration_pressure"] = float(pressure)
                metrics[f"projection/{name}_adv_std"] = float(diag_j["post_std"])
                metrics[f"projection/{name}_adv_scale"] = float(diag_j["scale"])
                metrics[f"projection/{name}_qclip_fraction"] = float(
                    diag_j["clip_fraction"]
                )

            self._projection_constraint_advantages = tuple(constraint_advantages)
            self._projection_constraint_active = tuple(active)
            self._projection_restoration_pressure = tuple(pressures)
            self._projection_epoch_stats = []

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
            raise RuntimeError("projection constraint advantages are unavailable")
        if len(self._projection_restoration_pressure) != len(
            self.config.constraint_names
        ):
            raise RuntimeError("projection restoration pressures are unavailable")

        # Exact latest-GRPO update semantics:
        # full planner remains eval; only nn.GRU modules enter train mode.
        self.model.eval()
        for module in self._model_gru_modules:
            module.train()
            module.flatten_parameters()

        self.optimizer.zero_grad(set_to_none=True)
        trainable_params = self._trainable_parameters
        before_update = [parameter.detach().clone() for parameter in trainable_params]

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
                ref_log_prob = self.sampler.replay(trace, model=self.reference_model)
        else:
            ref_log_prob = reference_log_prob
        ref_log_prob = ref_log_prob.detach()
        self._require_finite("reference_log_prob", ref_log_prob)

        ref_kl = self._reference_kl(new_log_prob, ref_log_prob)
        preferred_loss = (
            task_objective.policy_loss + float(self.config.kl_coef) * ref_kl
        )
        if not bool(torch.isfinite(preferred_loss)):
            raise FloatingPointError("non-finite preferred task+KL loss")

        preferred_grads = torch.autograd.grad(
            preferred_loss,
            trainable_params,
            retain_graph=True,
            allow_unused=True,
        )
        preferred_raw = self._flat_grads(preferred_grads, trainable_params)
        self._require_finite("preferred_raw_grad", preferred_raw)
        preferred = self._clip_to_l2_ball(
            preferred_raw,
            float(self.config.max_grad_norm),
        )

        active_names: list[str] = []
        active_grads: list[torch.Tensor] = []
        active_pressures: list[float] = []
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
                gradient_metrics[f"projection/{name}_zero_grad"] = 1.0
                continue

            active_names.append(name)
            active_grads.append(flat_j)
            active_pressures.append(self._projection_restoration_pressure[index])
            gradient_metrics[f"projection/{name}_grad_norm"] = float(
                norm_j.detach().cpu()
            )
            gradient_metrics[f"projection/preferred_{name}_grad_cosine"] = (
                self._cosine(preferred, flat_j)
            )

        projected, proj_diag, normals, used_margins = self._restorative_projection(
            preferred,
            active_grads,
            active_pressures,
        )
        self._require_finite("projected_grad", projected)

        radius = float(self.config.max_grad_norm)
        base_margin_factor = float(self.config.projection_restoration_strength) * radius
        for name, normal, pressure, used_margin in zip(
            active_names, normals, active_pressures, used_margins
        ):
            preferred_component = float(torch.dot(normal, preferred).detach().cpu())
            final_component = float(torch.dot(normal, projected).detach().cpu())
            base_margin = base_margin_factor * float(pressure)
            gradient_metrics[f"projection/{name}_base_margin"] = base_margin
            gradient_metrics[f"projection/{name}_used_margin"] = float(used_margin)
            gradient_metrics[f"projection/{name}_preferred_component"] = preferred_component
            gradient_metrics[f"projection/{name}_final_component"] = final_component
            gradient_metrics[f"projection/{name}_preferred_margin_residual"] = (
                preferred_component - float(used_margin)
            )
            gradient_metrics[f"projection/{name}_final_margin_residual"] = (
                final_component - float(used_margin)
            )
            gradient_metrics[f"projection/final_{name}_grad_cosine"] = (
                self._cosine(projected, normal)
            )

        self._assign_flat_grad(projected, trainable_params)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=self.config.max_grad_norm,
            error_if_nonfinite=True,
        )
        for index, parameter in enumerate(trainable_params):
            if parameter.grad is not None:
                self._require_finite(f"clipped_grad[{index}]", parameter.grad)

        self.optimizer.step()

        nonfinite_parameter = None
        for index, parameter in enumerate(trainable_params):
            if not bool(torch.isfinite(parameter).all()):
                nonfinite_parameter = index
                break
        if nonfinite_parameter is not None:
            with torch.no_grad():
                for parameter, before in zip(trainable_params, before_update):
                    parameter.copy_(before)
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "optimizer produced NaN/Inf in trainable parameter "
                f"index={nonfinite_parameter}; parameters were rolled back. "
                "Restart this run from the pretrained checkpoint."
            )

        update_sq = torch.zeros(
            (), device=trainable_params[0].device, dtype=torch.float32
        )
        for parameter, before in zip(trainable_params, before_update):
            update_sq = update_sq + (
                parameter.detach() - before
            ).float().square().sum()
        parameter_update_norm = torch.sqrt(update_sq)

        epoch_stat = {
            "correction_ratio": float(proj_diag["correction_ratio"]),
            "violated_before_count": float(proj_diag["violated_before_count"]),
            "violated_after_count": float(proj_diag["violated_after_count"]),
            "margin_scale_used": float(proj_diag["margin_scale_used"]),
            "min_margin_residual_after": float(proj_diag["min_margin_residual_after"]),
            "fallback_zero_margin": float(proj_diag["fallback_zero_margin"]),
            "solver_rescue_passes": float(proj_diag["solver_rescue_passes"]),
            "exact_zero_fallback": float(proj_diag["exact_zero_fallback"]),
        }
        self._projection_epoch_stats.append(epoch_stat)
        stats = self._projection_epoch_stats

        metrics: Dict[str, float] = {
            "loss": float(preferred_loss.detach().cpu()),
            "policy_loss": float(task_objective.policy_loss.detach().cpu()),
            "reference_kl": float(ref_kl.detach().cpu()),
            "approx_kl": float(task_objective.approx_kl.detach().cpu()),
            "clip_fraction": float(task_objective.clip_fraction.detach().cpu()),
            "ratio_mean": float(task_objective.ratio_mean.detach().cpu()),
            "ratio_std": float(task_objective.ratio_std.detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "parameter_update_norm": float(parameter_update_norm.detach().cpu()),
            "projection/active_constraint_count": float(len(active_grads)),
            "projection/preferred_raw_grad_norm": float(
                preferred_raw.norm().detach().cpu()
            ),
            "projection/preferred_budget_grad_norm": float(
                preferred.norm().detach().cpu()
            ),
            "projection/projected_grad_norm": float(
                projected.norm().detach().cpu()
            ),
            "projection/passes_used": float(proj_diag["passes_used"]),
            "projection/correction_norm": float(proj_diag["correction_norm"]),
            "projection/correction_ratio": float(proj_diag["correction_ratio"]),
            "projection/margin_scale_used": float(proj_diag["margin_scale_used"]),
            "projection/backoff_count": float(proj_diag["backoff_count"]),
            "projection/fallback_zero_margin": float(
                proj_diag["fallback_zero_margin"]
            ),
            "projection/violated_before_count": float(
                proj_diag["violated_before_count"]
            ),
            "projection/violated_after_count": float(
                proj_diag["violated_after_count"]
            ),
            "projection/min_margin_residual_before": float(
                proj_diag["min_margin_residual_before"]
            ),
            "projection/min_margin_residual_after": float(
                proj_diag["min_margin_residual_after"]
            ),
            "projection/hard_feasible": float(proj_diag["hard_feasible"]),
            "projection/solver_rescue_passes": float(
                proj_diag["solver_rescue_passes"]
            ),
            "projection/exact_zero_fallback": float(
                proj_diag["exact_zero_fallback"]
            ),
            "projection/collect_update_count": float(len(stats)),
            "projection/collect_correction_epoch_count": float(
                sum(s["correction_ratio"] > 1e-7 for s in stats)
            ),
            "projection/collect_any_correction": float(
                any(s["correction_ratio"] > 1e-7 for s in stats)
            ),
            "projection/collect_max_correction_ratio": max(
                s["correction_ratio"] for s in stats
            ),
            "projection/collect_max_violated_before_count": max(
                s["violated_before_count"] for s in stats
            ),
            "projection/collect_max_violated_after_count": max(
                s["violated_after_count"] for s in stats
            ),
            "projection/collect_min_margin_scale_used": min(
                s["margin_scale_used"] for s in stats
            ),
            "projection/collect_min_margin_residual_after": min(
                s["min_margin_residual_after"] for s in stats
            ),
            "projection/collect_any_zero_margin_fallback": float(
                any(s["fallback_zero_margin"] > 0.5 for s in stats)
            ),
            "projection/collect_any_solver_rescue": float(
                any(s["solver_rescue_passes"] > 0.5 for s in stats)
            ),
            "projection/collect_any_exact_zero_fallback": float(
                any(s["exact_zero_fallback"] > 0.5 for s in stats)
            ),
        }
        metrics.update(gradient_metrics)
        return metrics

    def constraint_state_dict(self) -> dict:
        return {
            "strategy": "projection_v2_restorative",
            "constraint_names": tuple(self.config.constraint_names),
            "restoration_scale_ema": tuple(self._projection_restoration_scale_ema),
            "projection": {
                "advantage_beta": self.config.projection_advantage_beta,
                "q_low": self.config.projection_q_low,
                "q_high": self.config.projection_q_high,
                "active_threshold": self.config.projection_active_threshold,
                "passes": self.config.projection_passes,
                "tolerance": self.config.projection_tolerance,
                "restoration_strength": self.config.projection_restoration_strength,
                "restoration_scale_beta": self.config.projection_restoration_scale_beta,
                "restoration_scale_floor": self.config.projection_restoration_scale_floor,
                "restoration_pressure_clip": self.config.projection_restoration_pressure_clip,
                "margin_backoff_factor": self.config.projection_margin_backoff_factor,
                "margin_backoff_steps": self.config.projection_margin_backoff_steps,
            },
        }
