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

    # GRPO ROLE-WISE CREDIT A V3
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # joint is the exact original path.
        if self.config.advantage_mode == "joint":
            return group_relative_advantage(
                rewards,
                eps=self.config.advantage_eps,
                valid_mask=valid_mask,
            )

        # Explicit role isolation. Current baseline already normalizes group_dim=1
        # independently per vehicle, so this should be numerically identical.
        role_advantages = []
        for role in range(rewards.shape[0]):
            role_mask = None if valid_mask is None else valid_mask[role : role + 1]
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
        result = {}
        for role in range(new_log_prob.shape[0]):
            role_mask = None if valid_mask is None else valid_mask[role : role + 1]
            if role_mask is not None and not bool(role_mask.any()):
                continue
            result[role] = grpo_clipped_objective(
                new_log_prob[role : role + 1],
                old_log_prob[role : role + 1],
                advantages[role : role + 1],
                clip_eps=clip_eps,
                valid_mask=role_mask,
            )
        return result

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
            else:
                role_mask = valid_mask[role].to(
                    device=advantages.device,
                    dtype=torch.bool,
                )
                if not bool(role_mask.any()):
                    continue
                values = advantages[role, :, role_mask].reshape(-1)
            metrics[f"diagnostics/vehicle_{role}_advantage_mean"] = float(
                values.mean().detach()
            )
            metrics[f"diagnostics/vehicle_{role}_advantage_std"] = float(
                values.std(unbiased=False).detach()
            )
        return metrics

    def _role_gradient_diagnostics(self, role_objectives) -> Dict[str, float]:
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not parameters:
            return {}

        role_grads = {}
        role_norms = {}
        metrics: Dict[str, float] = {}

        for role, objective in role_objectives.items():
            grads = torch.autograd.grad(
                objective.policy_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            grads = tuple(None if g is None else g.detach() for g in grads)
            role_grads[role] = grads

            sq = torch.zeros(
                (),
                device=objective.policy_loss.device,
                dtype=objective.policy_loss.dtype,
            )
            for grad in grads:
                if grad is not None:
                    sq = sq + grad.square().sum()
            norm = torch.sqrt(sq)
            role_norms[role] = norm
            metrics[f"diagnostics/vehicle_{role}_grad_norm"] = float(norm.detach())

        for left, right in ((0, 1), (0, 2), (1, 2)):
            if left not in role_grads or right not in role_grads:
                continue
            dot = torch.zeros(
                (),
                device=role_norms[left].device,
                dtype=role_norms[left].dtype,
            )
            for grad_left, grad_right in zip(role_grads[left], role_grads[right]):
                if grad_left is not None and grad_right is not None:
                    dot = dot + (grad_left * grad_right).sum()
            denom = (role_norms[left] * role_norms[right]).clamp_min(1e-12)
            cosine = dot / denom
            metrics[
                f"diagnostics/vehicle_{left}{right}_grad_cosine"
            ] = float(cosine.detach())

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
        gradient_diagnostics: bool = False,
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
        diagnostic_metrics: Dict[str, float] = {}
        if gradient_diagnostics:
            diagnostic_metrics.update(
                self._role_gradient_diagnostics(role_objectives)
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

        metrics = {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "reference_kl": float(ref_kl.detach()),
            "approx_kl": float(objective.approx_kl.detach()),
            "clip_fraction": float(objective.clip_fraction.detach()),
            "ratio_mean": float(objective.ratio_mean.detach()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach()),
        }
        metrics.update(diagnostic_metrics)
        return metrics

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
        paired_generator = (
            self._clone_generator(generator)
            if paired_validation
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
        update_metrics = []
        for _ in range(self.config.update_epochs):
            update_metrics.append(
                self.update(
                    trace,
                    advantages,
                    valid_mask=features.get("mode_valid_mask"),
                    gradient_diagnostics=gradient_diagnostics,
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
        if diagnostics:
            metrics.update(
                self._role_advantage_diagnostics(
                    advantages,
                    features.get("mode_valid_mask"),
                )
            )
        metrics.update(validation_metrics)
        return metrics
