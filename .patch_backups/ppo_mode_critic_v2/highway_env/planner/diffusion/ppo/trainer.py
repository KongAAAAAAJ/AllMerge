from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from highway_env.planner.diffusion.grpo.trainer import GRPOConfig, GRPOTrainer


@dataclass
class PPOConfig(GRPOConfig):
    """Value-baseline PPO extension of the current AllMerge GRPO config.

    Actor objective remains the existing replayable diffusion ratio+clip objective.
    Only the advantage estimator is changed:
      GRPO: group-relative normalized reward
      PPO V1: one-step Monte-Carlo advantage R - V(s), optionally std-scaled
    """

    value_learning_rate: float = 1e-4
    value_hidden_dim: int = 256
    value_epochs: int = 2
    value_max_grad_norm: float = 5.0
    ppo_advantage_scale: str = "std"  # raw | std

    def __post_init__(self) -> None:
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        if self.value_learning_rate <= 0.0:
            raise ValueError("value_learning_rate must be > 0")
        if self.value_hidden_dim < 32:
            raise ValueError("value_hidden_dim must be >= 32")
        if self.value_epochs < 1:
            raise ValueError("value_epochs must be >= 1")
        if self.value_max_grad_norm <= 0.0:
            raise ValueError("value_max_grad_norm must be > 0")
        if self.ppo_advantage_scale not in {"raw", "std"}:
            raise ValueError("ppo_advantage_scale must be 'raw' or 'std'")


class _StateValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PPOTrainer(GRPOTrainer):
    """PPO ablation on top of the current diffusion-policy infrastructure.

    The current online loop scores a complete sampled trajectory with one terminal
    candidate reward, so temporal GAE collapses to A = R - V(s). V1 deliberately
    uses a state-only critic (one value per controlled vehicle), not V(s, action).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.config, PPOConfig):
            raise TypeError(
                "PPOTrainer requires PPOConfig; received "
                f"{type(self.config).__name__}"
            )
        if getattr(self.config, "constraint_strategy", "none") != "none":
            raise ValueError(
                "PPO V1 supports constraint_strategy='none' only. "
                "First isolate PPO vs GRPO before adding constrained PPO."
            )

        self.value_model: _StateValueNetwork | None = None
        self.value_optimizer: torch.optim.Optimizer | None = None
        self._pending_value_features: torch.Tensor | None = None
        self._pending_value_targets: torch.Tensor | None = None

    @staticmethod
    def _masked_mean(
        value: torch.Tensor,
        mask: torch.Tensor | None,
        dim: int,
    ) -> torch.Tensor:
        if mask is None:
            return value.mean(dim=dim)
        weight = mask.to(device=value.device, dtype=value.dtype)
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(-1)
        denom = weight.sum(dim=dim).clamp_min(1.0)
        return (value * weight).sum(dim=dim) / denom

    def _value_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        required = (
            "ego_state",
            "agent_states",
            "map_polylines",
            "target_point",
            "target_lane_polyline",
            "coarse_trajectories",
        )
        missing = [key for key in required if key not in features]
        if missing:
            raise KeyError(f"PPO critic missing planner features: {missing}")

        ego = features["ego_state"].float()
        target = features["target_point"].float()

        agents = features["agent_states"].float()
        agent_pool = self._masked_mean(
            agents, features.get("agent_valid_mask"), dim=1
        )

        map_poly = features["map_polylines"].float()
        map_per_poly = map_poly.mean(dim=2)
        map_pool = self._masked_mean(
            map_per_poly, features.get("map_valid_mask"), dim=1
        )

        target_lane = features["target_lane_polyline"].float().mean(dim=1)

        # All coarse modes are part of the observation/context. We expose the full
        # set to the critic, but never condition V on the sampled group/mode action.
        coarse = features["coarse_trajectories"].float()
        if coarse.ndim != 4:
            raise ValueError(
                "coarse_trajectories must be [vehicle,mode,horizon,2], got "
                f"{tuple(coarse.shape)}"
            )
        coarse_all = coarse.reshape(coarse.shape[0], -1)

        return torch.cat(
            [ego, target, agent_pool, map_pool, target_lane, coarse_all],
            dim=-1,
        ).detach()

    @staticmethod
    def _vm_mask(
        rewards: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if valid_mask is None:
            return torch.ones(
                rewards.shape[0],
                rewards.shape[2],
                dtype=torch.bool,
                device=rewards.device,
            )
        return valid_mask.to(device=rewards.device, dtype=torch.bool)

    def _state_return_target(
        self,
        rewards: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        vm_mask = self._vm_mask(rewards, valid_mask)
        expanded = vm_mask[:, None, :].expand_as(rewards)
        weight = expanded.to(rewards.dtype)
        denom = weight.sum(dim=(1, 2)).clamp_min(1.0)
        return (rewards * weight).sum(dim=(1, 2)) / denom

    def _ensure_value_model(
        self,
        value_features: torch.Tensor,
        rewards: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> None:
        if self.value_model is not None:
            return

        model = _StateValueNetwork(
            value_features.shape[-1], int(self.config.value_hidden_dim)
        ).to(device=value_features.device, dtype=value_features.dtype)

        init_value = self._state_return_target(rewards, valid_mask).mean()
        # Global initialization only. This intentionally avoids recreating the
        # current group's sample mean as the actor baseline on the first update.
        with torch.no_grad():
            final = model.net[-1]
            assert isinstance(final, nn.Linear)
            final.weight.zero_()
            final.bias.fill_(float(init_value.detach()))

        self.value_model = model
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(),
            lr=float(self.config.value_learning_rate),
            weight_decay=1e-4,
        )

    def collect(
        self,
        features: Dict[str, torch.Tensor],
        *,
        context: Any = None,
        generator: torch.Generator | None = None,
    ):
        # Keep the current sampler, reward and diffusion log-prob path unchanged.
        self.model.eval()
        with torch.no_grad():
            trace = self.sampler.sample(features, generator=generator)
            rewards = self.score_candidates(
                trace.candidates,
                features=features,
                context=context,
            )

        valid_mask = features.get("mode_valid_mask")
        value_features = self._value_features(features)
        self._ensure_value_model(value_features, rewards, valid_mask)
        assert self.value_model is not None

        with torch.no_grad():
            values = self.value_model(value_features)  # [vehicle]
            raw_advantages = rewards - values[:, None, None]

            vm_mask = self._vm_mask(rewards, valid_mask)
            expanded = vm_mask[:, None, :].expand_as(raw_advantages)
            raw_advantages = torch.where(
                expanded,
                raw_advantages,
                torch.zeros_like(raw_advantages),
            )
            valid_adv = raw_advantages[expanded]

            scale = torch.ones(
                (), device=rewards.device, dtype=rewards.dtype
            )
            if (
                self.config.ppo_advantage_scale == "std"
                and valid_adv.numel() > 1
            ):
                # Crucially: scale only. Do NOT subtract the current batch mean.
                scale = valid_adv.std(unbiased=False).clamp_min(
                    float(self.config.advantage_eps)
                )
            advantages = raw_advantages / scale

            value_targets = self._state_return_target(rewards, valid_mask)
            self._pending_value_features = value_features
            self._pending_value_targets = value_targets.detach()

            valid_scaled = advantages[expanded]
            positive_fraction = (
                (valid_scaled > 0.0).float().mean()
                if valid_scaled.numel()
                else torch.zeros((), device=rewards.device)
            )
            strategy_metrics = {
                "ppo/value_baseline_mean": float(values.mean().detach()),
                "ppo/value_target_mean": float(value_targets.mean().detach()),
                "ppo/advantage_raw_mean": float(
                    valid_adv.mean().detach() if valid_adv.numel() else 0.0
                ),
                "ppo/advantage_raw_std": float(
                    valid_adv.std(unbiased=False).detach()
                    if valid_adv.numel() > 1
                    else 0.0
                ),
                "ppo/advantage_scale": float(scale.detach()),
                "ppo/advantage_positive_fraction": float(
                    positive_fraction.detach()
                ),
            }

        return trace, rewards, advantages, strategy_metrics

    def _update_value_model(self) -> Dict[str, float]:
        if (
            self.value_model is None
            or self.value_optimizer is None
            or self._pending_value_features is None
            or self._pending_value_targets is None
        ):
            return {}

        x = self._pending_value_features
        target = self._pending_value_targets
        last_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        last_grad = torch.zeros((), device=x.device, dtype=x.dtype)

        self.value_model.train()
        for _ in range(int(self.config.value_epochs)):
            pred = self.value_model(x)
            loss = (pred - target).square().mean()
            self.value_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                self.value_model.parameters(),
                float(self.config.value_max_grad_norm),
            )
            self.value_optimizer.step()
            last_loss = loss.detach()
            last_grad = torch.as_tensor(grad).detach()

        with torch.no_grad():
            pred = self.value_model(x)
            residual_var = (target - pred).var(unbiased=False)
            target_var = target.var(unbiased=False)
            explained = torch.where(
                target_var > 1e-12,
                1.0 - residual_var / target_var.clamp_min(1e-12),
                torch.zeros_like(target_var),
            )
            metrics = {
                "ppo/value_loss": float(last_loss),
                "ppo/value_grad_norm": float(last_grad),
                "ppo/value_pred_mean_after": float(pred.mean()),
                "ppo/value_explained_variance": float(explained),
            }

        self._pending_value_features = None
        self._pending_value_targets = None
        return metrics

    def train_step(self, *args, **kwargs) -> Dict[str, float]:
        metrics = super().train_step(*args, **kwargs)
        metrics.update(self._update_value_model())
        return metrics

    def save_value_checkpoint(self, path: str | Path) -> None:
        if self.value_model is None or self.value_optimizer is None:
            return
        payload = {
            "format": "allmerge_ppo_value_v1",
            "value_model": self.value_model.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "config": {
                "value_learning_rate": self.config.value_learning_rate,
                "value_hidden_dim": self.config.value_hidden_dim,
                "value_epochs": self.config.value_epochs,
                "value_max_grad_norm": self.config.value_max_grad_norm,
                "ppo_advantage_scale": self.config.ppo_advantage_scale,
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)


# PPO_VALUE_BASELINE_V1_20261005
