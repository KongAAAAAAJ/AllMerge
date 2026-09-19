from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import torch

from .objective import group_relative_advantage, grpo_clipped_objective
from .reward_adapter import CandidateRewardAdapter
from .sampling import DiffusionTrace, GroupDiffusionSampler


@dataclass
class GRPOConfig:
    group_size: int = 4
    learning_rate: float = 5e-5
    eta: float = 0.02
    clip_eps: float = 0.2
    kl_coef: float = 0.01
    max_grad_norm: float = 10.0
    advantage_eps: float = 1e-6
    advantage_mode: str = "joint"
    update_epochs: int = 2
    trainable_prefixes: Tuple[str, ...] = (
        "denoiser.layers",
        "denoiser.reg_head",
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
        if self.config.advantage_mode not in ("joint", "rolewise"):
            raise ValueError(
                "advantage_mode must be one of {'joint', 'rolewise'}"
            )
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

    # GRPO ROLE-WISE CREDIT A ON DIAG V1
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.config.advantage_mode == "joint":
            return group_relative_advantage(
                rewards,
                eps=self.config.advantage_eps,
                valid_mask=valid_mask,
            )

        role_advantages = []
        for role in range(rewards.shape[0]):
            role_mask = (
                None
                if valid_mask is None
                else valid_mask[role : role + 1]
            )
            role_advantages.append(
                group_relative_advantage(
                    rewards[role : role + 1],
                    eps=self.config.advantage_eps,
                    valid_mask=role_mask,
                )
            )
        return torch.cat(role_advantages, dim=0)

    @staticmethod
    def _role_objectives(
        new_log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        *,
        clip_eps: float,
        valid_mask: torch.Tensor | None,
    ):
        role_objectives = {}
        for role in range(new_log_prob.shape[0]):
            role_mask = (
                None
                if valid_mask is None
                else valid_mask[role : role + 1]
            )
            if role_mask is not None and not bool(role_mask.any()):
                continue
            role_objectives[role] = grpo_clipped_objective(
                new_log_prob[role : role + 1],
                old_log_prob[role : role + 1],
                advantages[role : role + 1],
                clip_eps=clip_eps,
                valid_mask=role_mask,
            )
        return role_objectives

    def _policy_terms(
        self,
        new_log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ):
        global_objective = grpo_clipped_objective(
            new_log_prob,
            old_log_prob,
            advantages,
            clip_eps=self.config.clip_eps,
            valid_mask=valid_mask,
        )
        role_objectives = self._role_objectives(
            new_log_prob,
            old_log_prob,
            advantages,
            clip_eps=self.config.clip_eps,
            valid_mask=valid_mask,
        )

        if self.config.advantage_mode == "joint":
            policy_loss = global_objective.policy_loss
        else:
            if not role_objectives:
                raise ValueError("rolewise policy loss has no valid vehicle roles")
            policy_loss = torch.stack(
                [obj.policy_loss for obj in role_objectives.values()]
            ).mean()

        return global_objective, role_objectives, policy_loss

    @staticmethod
    def _role_advantage_diagnostics(
        advantages: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for role in range(advantages.shape[0]):
            if valid_mask is None:
                values = advantages[role].reshape(-1)
                valid_count = advantages.shape[-1]
            else:
                role_mask = valid_mask[role].to(
                    device=advantages.device,
                    dtype=torch.bool,
                )
                if not bool(role_mask.any()):
                    continue
                values = advantages[role, :, role_mask].reshape(-1)
                valid_count = int(role_mask.sum().item())

            metrics[f"diagnostics/vehicle_{role}_advantage_mean"] = float(
                values.mean().detach()
            )
            metrics[f"diagnostics/vehicle_{role}_advantage_std"] = float(
                values.std(unbiased=False).detach()
            )
            metrics[f"diagnostics/vehicle_{role}_valid_mode_count"] = float(
                valid_count
            )
        return metrics

    def collect(
        self,
        features: Dict[str, torch.Tensor],
        *,
        context: Any = None,
        generator: torch.Generator | None = None,
    ) -> tuple[DiffusionTrace, torch.Tensor, torch.Tensor]:
        # Keep dropout disabled: diffusion transition log-prob must describe all policy stochasticity.
        self.model.eval()
        with torch.no_grad():
            trace = self.sampler.sample(features, generator=generator)
            rewards = self.reward_adapter(
                trace.candidates,
                features=features,
                context=context,
            )
            advantages = self._compute_advantages(
                rewards,
                features.get("mode_valid_mask"),
            )
        return trace, rewards, advantages


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

    def paired_validation(
        self,
        trace: DiffusionTrace,
        current_rewards: torch.Tensor,
        features: Dict[str, torch.Tensor],
        *,
        context: Any,
        generator: torch.Generator,
    ) -> Dict[str, float]:
        """Same-noise current-vs-frozen reward comparison on one live state."""
        self.reference_model.eval()
        with torch.no_grad():
            frozen_trace = self.reference_sampler.sample(
                features,
                generator=generator,
            )
            frozen_rewards = self.reward_adapter(
                frozen_trace.candidates,
                features=features,
                context=context,
            )
            delta_m = torch.linalg.vector_norm(
                trace.candidates[..., :2]
                - frozen_trace.candidates[..., :2],
                dim=-1,
            ).mean()
        return self._paired_reward_metrics(
            current_rewards,
            frozen_rewards,
            features.get("mode_valid_mask"),
            candidate_delta_m=delta_m,
        )



    # GRPO DIAGNOSTICS BASE V1
    @staticmethod
    def _diagnostic_mask(
        rewards: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if rewards.ndim != 3:
            raise ValueError("diagnostic rewards must be [vehicle,group,mode]")
        if valid_mask is None:
            return torch.ones(
                rewards.shape[0],
                rewards.shape[2],
                dtype=torch.bool,
                device=rewards.device,
            )
        mask = valid_mask.to(device=rewards.device, dtype=torch.bool)
        if tuple(mask.shape) != (rewards.shape[0], rewards.shape[2]):
            raise ValueError("diagnostic valid_mode_mask must be [vehicle,mode]")
        return mask

    @staticmethod
    def _masked_scalar_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded = mask
        while expanded.ndim < values.ndim:
            expanded = expanded.unsqueeze(1)
        expanded = expanded.expand_as(values)
        denom = expanded.sum().clamp_min(1)
        return (values * expanded.to(values.dtype)).sum() / denom

    def rollout_diagnostics(
        self,
        trace: DiffusionTrace,
        current_rewards: torch.Tensor,
        features: Dict[str, torch.Tensor],
        *,
        context: Any,
        generator: torch.Generator,
    ) -> Dict[str, float]:
        """Read-only same-state/same-noise diagnostics against frozen Stage-1."""
        self.reference_model.eval()
        with torch.no_grad():
            frozen_trace = self.reference_sampler.sample(
                features,
                generator=generator,
            )
            frozen_rewards = self.reward_adapter(
                frozen_trace.candidates,
                features=features,
                context=context,
            )

            if current_rewards.shape != frozen_rewards.shape:
                raise ValueError("diagnostic current/frozen reward shapes differ")
            mask = self._diagnostic_mask(
                current_rewards,
                features.get("mode_valid_mask"),
            )
            gain = current_rewards - frozen_rewards
            gain_mask = mask[:, None, :].expand_as(gain)
            gain_values = gain[gain_mask]
            if gain_values.numel() == 0:
                raise ValueError("diagnostic reward mask has no valid entries")

            current_mean = current_rewards[gain_mask].mean()
            frozen_mean = frozen_rewards[gain_mask].mean()
            positive_fraction = (gain_values > 1e-6).to(gain.dtype).mean()

            waypoint_delta = torch.linalg.vector_norm(
                trace.candidates[..., :2]
                - frozen_trace.candidates[..., :2],
                dim=-1,
            ).mean(dim=-1)
            delta_mask = mask[:, None, :].expand_as(waypoint_delta)
            candidate_delta = waypoint_delta[delta_mask].mean()

            metrics: Dict[str, float] = {
                "diagnostics/online_paired_current_reward_mean": float(
                    current_mean.detach()
                ),
                "diagnostics/online_paired_frozen_reward_mean": float(
                    frozen_mean.detach()
                ),
                "diagnostics/online_paired_reward_gain": float(
                    (current_mean - frozen_mean).detach()
                ),
                "diagnostics/online_paired_positive_fraction": float(
                    positive_fraction.detach()
                ),
                "diagnostics/online_paired_candidate_delta_m": float(
                    candidate_delta.detach()
                ),
            }

            for role in range(current_rewards.shape[0]):
                role_mask = mask[role]
                if not bool(role_mask.any()):
                    continue
                role_current = current_rewards[role, :, role_mask].mean()
                role_frozen = frozen_rewards[role, :, role_mask].mean()
                metrics[f"diagnostics/vehicle_{role}_reward_gain"] = float(
                    (role_current - role_frozen).detach()
                )

        return metrics

    def group_diagnostics(
        self,
        trace: DiffusionTrace,
        rewards: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Measure reward ranking strength and geometric candidate diversity."""
        with torch.no_grad():
            mask = self._diagnostic_mask(
                rewards,
                features.get("mode_valid_mask"),
            )
            reward_std = rewards.std(dim=1, unbiased=False)
            reward_max = rewards.max(dim=1).values
            reward_min = rewards.min(dim=1).values
            reward_mean = rewards.mean(dim=1)
            reward_range = reward_max - reward_min
            best_minus_mean = reward_max - reward_mean

            valid_std = reward_std[mask]
            valid_range = reward_range[mask]
            valid_best = best_minus_mean[mask]

            candidates = trace.candidates[..., :2]
            group_size = int(candidates.shape[1])
            pairwise = candidates[:, :, None] - candidates[:, None, :]
            pairwise = torch.linalg.vector_norm(pairwise, dim=-1).mean(dim=-1)
            upper = torch.triu(
                torch.ones(
                    group_size,
                    group_size,
                    dtype=torch.bool,
                    device=candidates.device,
                ),
                diagonal=1,
            )
            pairwise = pairwise[:, upper, :]
            pair_mask = mask[:, None, :].expand_as(pairwise)
            pair_values = pairwise[pair_mask]

            metrics: Dict[str, float] = {
                "diagnostics/within_group_reward_std": float(valid_std.mean()),
                "diagnostics/within_group_reward_range": float(valid_range.mean()),
                "diagnostics/within_group_best_minus_mean": float(valid_best.mean()),
                "diagnostics/candidate_pairwise_distance_m": float(
                    pair_values.mean()
                ),
            }
            for role in range(rewards.shape[0]):
                role_mask = mask[role]
                if not bool(role_mask.any()):
                    continue
                metrics[
                    f"diagnostics/vehicle_{role}_within_group_reward_std"
                ] = float(reward_std[role, role_mask].mean())
                metrics[
                    f"diagnostics/vehicle_{role}_within_group_reward_range"
                ] = float(reward_range[role, role_mask].mean())
        return metrics

    @staticmethod
    def _grad_norm_from_tuple(grads) -> torch.Tensor:
        terms = [
            grad.detach().float().square().sum()
            for grad in grads
            if grad is not None
        ]
        if not terms:
            return torch.zeros(())
        return torch.stack(terms).sum().sqrt()

    @staticmethod
    def _grad_cosine_from_tuples(grads_a, grads_b) -> torch.Tensor:
        dot_terms = []
        a_terms = []
        b_terms = []
        for grad_a, grad_b in zip(grads_a, grads_b):
            if grad_a is None or grad_b is None:
                continue
            a = grad_a.detach().float()
            b = grad_b.detach().float()
            dot_terms.append((a * b).sum())
            a_terms.append(a.square().sum())
            b_terms.append(b.square().sum())
        if not dot_terms:
            return torch.zeros(())
        dot = torch.stack(dot_terms).sum()
        norm_a = torch.stack(a_terms).sum().sqrt()
        norm_b = torch.stack(b_terms).sum().sqrt()
        denom = norm_a * norm_b
        if float(denom) <= 1e-20:
            return torch.zeros((), device=dot.device)
        return dot / denom

    def gradient_diagnostics(
        self,
        trace: DiffusionTrace,
        advantages: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None,
    ) -> Dict[str, float]:
        """Read-only policy/KL/role gradient diagnostics; optimizer state is untouched."""
        self.model.eval()
        parameters = tuple(
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        if not parameters:
            raise RuntimeError("no trainable parameters for gradient diagnostics")

        new_log_prob = self.sampler.replay(trace)
        objective, role_objectives, policy_loss = self._policy_terms(
            new_log_prob,
            trace.old_log_prob,
            advantages,
            valid_mask,
        )
        with torch.no_grad():
            ref_log_prob = self.sampler.replay(
                trace,
                model=self.reference_model,
            )
        ref_kl = self._reference_kl(new_log_prob, ref_log_prob)

        policy_grads = torch.autograd.grad(
            policy_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        kl_grads = torch.autograd.grad(
            ref_kl,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        policy_norm = self._grad_norm_from_tuple(policy_grads)
        kl_norm = self._grad_norm_from_tuple(kl_grads)
        weighted_kl_norm = kl_norm * float(self.config.kl_coef)
        policy_kl_cosine = self._grad_cosine_from_tuples(
            policy_grads,
            kl_grads,
        )

        eps = 1e-20
        metrics: Dict[str, float] = {
            "diagnostics/policy_grad_norm": float(policy_norm),
            "diagnostics/reference_kl_grad_norm": float(kl_norm),
            "diagnostics/reference_kl_weighted_grad_norm": float(
                weighted_kl_norm
            ),
            "diagnostics/policy_kl_grad_cosine": float(policy_kl_cosine),
            "diagnostics/policy_to_weighted_kl_grad_norm_ratio": float(
                policy_norm / weighted_kl_norm.clamp_min(eps)
            ),
        }

        role_grads = []
        batch = int(new_log_prob.shape[0])
        for role in range(batch):
            role_mask = None if valid_mask is None else valid_mask[role : role + 1]
            role_objective = grpo_clipped_objective(
                new_log_prob[role : role + 1],
                trace.old_log_prob[role : role + 1],
                advantages[role : role + 1],
                clip_eps=self.config.clip_eps,
                valid_mask=role_mask,
            )
            grads = torch.autograd.grad(
                role_objective.policy_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            role_grads.append(grads)
            metrics[f"diagnostics/vehicle_{role}_grad_norm"] = float(
                self._grad_norm_from_tuple(grads)
            )

        for left in range(min(3, len(role_grads))):
            for right in range(left + 1, min(3, len(role_grads))):
                metrics[
                    f"diagnostics/vehicle_{left}{right}_grad_cosine"
                ] = float(
                    self._grad_cosine_from_tuples(
                        role_grads[left],
                        role_grads[right],
                    )
                )

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
        # eval() disables dropout but does not disable gradients.
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        new_log_prob = self.sampler.replay(trace)
        objective, role_objectives, policy_loss = self._policy_terms(
            new_log_prob,
            trace.old_log_prob,
            advantages,
            valid_mask,
        )

        with torch.no_grad():
            ref_log_prob = self.sampler.replay(trace, model=self.reference_model)
        ref_kl = self._reference_kl(new_log_prob, ref_log_prob)
        loss = policy_loss + self.config.kl_coef * ref_kl
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite GRPO loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=self.config.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "reference_kl": float(ref_kl.detach()),
            "approx_kl": float(objective.approx_kl.detach()),
            "clip_fraction": float(objective.clip_fraction.detach()),
            "ratio_mean": float(objective.ratio_mean.detach()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach()),
        }

    def train_step(
        self,
        features: Dict[str, torch.Tensor],
        *,
        context: Any = None,
        generator: torch.Generator | None = None,
        paired_validation: bool = False,
        diagnostics: bool = False,
        gradient_diagnostics: bool = False,
    ) -> Dict[str, float]:
        diagnostics = bool(diagnostics or gradient_diagnostics)
        paired_generator = (
            self._clone_generator(generator)
            if paired_validation
            else None
        )
        diagnostic_generator = (
            self._clone_generator(generator)
            if diagnostics
            else None
        )
        trace, rewards, advantages = self.collect(
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

        diagnostic_metrics: Dict[str, float] = {}
        if diagnostics:
            assert diagnostic_generator is not None
            diagnostic_metrics.update(
                self._role_advantage_diagnostics(
                    advantages,
                    features.get("mode_valid_mask"),
                )
            )
            diagnostic_metrics.update(
                self.group_diagnostics(trace, rewards, features)
            )
            diagnostic_metrics.update(
                self.rollout_diagnostics(
                    trace,
                    rewards,
                    features,
                    context=context,
                    generator=diagnostic_generator,
                )
            )
            if gradient_diagnostics:
                diagnostic_metrics.update(
                    self.gradient_diagnostics(
                        trace,
                        advantages,
                        valid_mask=features.get("mode_valid_mask"),
                    )
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
        metrics.update(validation_metrics)
        metrics.update(diagnostic_metrics)
        return metrics
