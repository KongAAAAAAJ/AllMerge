from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from highway_env.planner.diffusion.grpo.trainer import GRPOConfig, GRPOTrainer


PPO_V2_MARKER = "PPO_MODE_CONDITIONED_VALUE_V2_20261005"
PPO_DIAG_FIELDS = (
    "ppo/value_baseline_mean",
    "ppo/value_target_mean",
    "ppo/value_bias_mean_before",
    "ppo/value_target_mode_spread",
    "ppo/value_pred_mode_spread_before",
    "ppo/advantage_raw_mean",
    "ppo/advantage_raw_std",
    "ppo/advantage_scale",
    "ppo/advantage_positive_fraction",
    "ppo/value_loss",
    "ppo/value_grad_norm",
    "ppo/value_pred_mean_after",
    "ppo/value_bias_mean_after",
    "ppo/value_pred_mode_spread_after",
    "ppo/value_mae_after",
    "ppo/value_explained_variance",
)


@dataclass
class PPOConfig(GRPOConfig):
    """Mode-conditioned value-baseline PPO for AllMerge.

    Actor path stays identical to the current replayable diffusion PPO/GRPO path.
    The only algorithmic baseline change is:

        GRPO: group-relative sample baseline per vehicle/mode
        PPO V2: learned V(s, m), where m is the coarse trajectory mode

    V2 conditions the critic on the same coarse endpoint used by the current GRU
    refinement head, avoiding the cross-mode mixing of PPO V1's state-only V(s).
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


class _ModeConditionedValueNetwork(nn.Module):
    """Shared MLP applied independently to every (vehicle, coarse-mode) pair."""

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
        # x: [vehicle, mode, feature]
        return self.net(x).squeeze(-1)  # [vehicle, mode]


class PPOTrainer(GRPOTrainer):
    """PPO V2 with learned coarse-mode-conditioned critic V(s, m)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.config, PPOConfig):
            raise TypeError(
                "PPOTrainer requires PPOConfig; received "
                f"{type(self.config).__name__}"
            )
        if getattr(self.config, "constraint_strategy", "none") != "none":
            raise ValueError(
                "PPO V2 supports constraint_strategy='none' only. "
                "First isolate mode-conditioned PPO vs vanilla GRPO."
            )

        self.value_model: _ModeConditionedValueNetwork | None = None
        self.value_optimizer: torch.optim.Optimizer | None = None
        self._pending_value_features: torch.Tensor | None = None
        self._pending_value_targets: torch.Tensor | None = None
        self._pending_value_mask: torch.Tensor | None = None
        self._pending_collect_metrics: Dict[str, float] = {}
        self.last_ppo_metrics: Dict[str, float] = {}
        self._ppo_diag_step = 0

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

    @staticmethod
    def _vm_mask(
        rewards: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if rewards.ndim != 3:
            raise ValueError(
                "PPO V2 expects rewards [vehicle,group,mode], got "
                f"{tuple(rewards.shape)}"
            )
        if valid_mask is None:
            return torch.ones(
                rewards.shape[0],
                rewards.shape[2],
                dtype=torch.bool,
                device=rewards.device,
            )
        mask = valid_mask.to(device=rewards.device, dtype=torch.bool)
        expected = (rewards.shape[0], rewards.shape[2])
        if tuple(mask.shape) != expected:
            raise ValueError(
                "mode_valid_mask must be [vehicle,mode], got "
                f"{tuple(mask.shape)} expected {expected}"
            )
        return mask

    def _shared_state_features(
        self,
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
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
            raise KeyError(f"PPO V2 critic missing planner features: {missing}")

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

        return torch.cat(
            [ego, target, agent_pool, map_pool, target_lane],
            dim=-1,
        )

    def _value_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build [vehicle, mode, feature] critic input.

        We intentionally condition each mode on its own coarse endpoint because the
        current GRU refinement head uses the coarse endpoint as its navigation
        condition. This makes critic conditioning match the actor's mode context.
        """
        shared = self._shared_state_features(features)

        coarse = features["coarse_trajectories"].float()
        if coarse.ndim != 4 or coarse.shape[-1] != 2:
            raise ValueError(
                "coarse_trajectories must be [vehicle,mode,horizon,2], got "
                f"{tuple(coarse.shape)}"
            )
        coarse_endpoint = coarse[:, :, -1, :]  # [vehicle, mode, 2]

        if coarse_endpoint.shape[0] != shared.shape[0]:
            raise ValueError(
                "vehicle dimension mismatch between state and coarse trajectories"
            )

        shared_vm = shared[:, None, :].expand(
            -1, coarse_endpoint.shape[1], -1
        )
        return torch.cat(
            [shared_vm, coarse_endpoint],
            dim=-1,
        ).detach()

    @staticmethod
    def _mode_return_target(rewards: torch.Tensor) -> torch.Tensor:
        # One-step Monte-Carlo estimate of E[R | s,m] from the G sampled refinements.
        return rewards.mean(dim=1)  # [vehicle, mode]

    @staticmethod
    def _valid_flat(
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return value[mask]

    @staticmethod
    def _mode_spread(
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean per-vehicle std across valid modes."""
        spreads = []
        for vehicle in range(value.shape[0]):
            valid = value[vehicle][mask[vehicle]]
            if valid.numel() > 1:
                spreads.append(valid.std(unbiased=False))
            elif valid.numel() == 1:
                spreads.append(torch.zeros((), device=value.device, dtype=value.dtype))
        if not spreads:
            return torch.zeros((), device=value.device, dtype=value.dtype)
        return torch.stack(spreads).mean()

    def _ensure_value_model(
        self,
        value_features: torch.Tensor,
        rewards: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> None:
        if self.value_model is not None:
            return

        model = _ModeConditionedValueNetwork(
            value_features.shape[-1], int(self.config.value_hidden_dim)
        ).to(device=value_features.device, dtype=value_features.dtype)

        vm_mask = self._vm_mask(rewards, valid_mask)
        targets = self._mode_return_target(rewards)
        valid_targets = targets[vm_mask]
        init_value = (
            valid_targets.mean()
            if valid_targets.numel()
            else targets.mean()
        )

        # Start from one global expected-return prior, not the current per-mode
        # sample mean. The critic must learn the mode differences.
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
        # Preserve current diffusion sampler, reward and replay log-prob path.
        self.model.eval()
        with torch.no_grad():
            trace = self.sampler.sample(features, generator=generator)
            rewards = self.score_candidates(
                trace.candidates,
                features=features,
                context=context,
            )

        valid_mask = features.get("mode_valid_mask")
        vm_mask = self._vm_mask(rewards, valid_mask)
        value_features = self._value_features(features)
        if tuple(value_features.shape[:2]) != tuple(vm_mask.shape):
            raise ValueError(
                "critic feature [vehicle,mode] shape does not match reward mask: "
                f"{tuple(value_features.shape[:2])} vs {tuple(vm_mask.shape)}"
            )

        self._ensure_value_model(value_features, rewards, valid_mask)
        assert self.value_model is not None

        with torch.no_grad():
            values = self.value_model(value_features)  # [vehicle, mode]
            value_targets = self._mode_return_target(rewards)  # [vehicle, mode]

            # Broadcast the learned per-mode baseline across the G samples.
            raw_advantages = rewards - values[:, None, :]

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
                # Scale only. Do NOT subtract current batch/group mean.
                scale = valid_adv.std(unbiased=False).clamp_min(
                    float(self.config.advantage_eps)
                )
            advantages = raw_advantages / scale

            self._pending_value_features = value_features
            self._pending_value_targets = value_targets.detach()
            self._pending_value_mask = vm_mask.detach()

            valid_values = values[vm_mask]
            valid_targets = value_targets[vm_mask]
            valid_scaled = advantages[expanded]
            positive_fraction = (
                (valid_scaled > 0.0).float().mean()
                if valid_scaled.numel()
                else torch.zeros((), device=rewards.device)
            )

            self._pending_collect_metrics = {
                "ppo/value_baseline_mean": float(
                    valid_values.mean().detach()
                    if valid_values.numel()
                    else values.mean().detach()
                ),
                "ppo/value_target_mean": float(
                    valid_targets.mean().detach()
                    if valid_targets.numel()
                    else value_targets.mean().detach()
                ),
                "ppo/value_bias_mean_before": float(
                    (valid_values - valid_targets).mean().detach()
                    if valid_values.numel()
                    else 0.0
                ),
                "ppo/value_target_mode_spread": float(
                    self._mode_spread(value_targets, vm_mask).detach()
                ),
                "ppo/value_pred_mode_spread_before": float(
                    self._mode_spread(values, vm_mask).detach()
                ),
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

        # Preserve the evolved GRPO trainer collect contract.
        return trace, rewards, advantages, dict(self._pending_collect_metrics)

    def _update_value_model(self) -> Dict[str, float]:
        if (
            self.value_model is None
            or self.value_optimizer is None
            or self._pending_value_features is None
            or self._pending_value_targets is None
            or self._pending_value_mask is None
        ):
            return {}

        x = self._pending_value_features
        target = self._pending_value_targets
        mask = self._pending_value_mask
        last_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        last_grad = torch.zeros((), device=x.device, dtype=x.dtype)

        self.value_model.train()
        for _ in range(int(self.config.value_epochs)):
            pred = self.value_model(x)
            pred_valid = pred[mask]
            target_valid = target[mask]
            if pred_valid.numel() == 0:
                raise RuntimeError("PPO V2 critic has no valid vehicle/mode targets")
            loss = (pred_valid - target_valid).square().mean()
            self.value_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                self.value_model.parameters(),
                float(self.config.value_max_grad_norm),
            )
            self.value_optimizer.step()
            last_loss = loss.detach()
            last_grad = torch.as_tensor(
                grad, device=x.device, dtype=x.dtype
            ).detach()

        with torch.no_grad():
            pred = self.value_model(x)
            pred_valid = pred[mask]
            target_valid = target[mask]
            residual = target_valid - pred_valid
            residual_var = residual.var(unbiased=False)
            target_var = target_valid.var(unbiased=False)
            explained = torch.where(
                target_var > 1e-12,
                1.0 - residual_var / target_var.clamp_min(1e-12),
                torch.zeros_like(target_var),
            )
            metrics = {
                "ppo/value_loss": float(last_loss),
                "ppo/value_grad_norm": float(last_grad),
                "ppo/value_pred_mean_after": float(pred_valid.mean()),
                "ppo/value_bias_mean_after": float(
                    (pred_valid - target_valid).mean()
                ),
                "ppo/value_pred_mode_spread_after": float(
                    self._mode_spread(pred, mask)
                ),
                "ppo/value_mae_after": float(residual.abs().mean()),
                "ppo/value_explained_variance": float(explained),
            }

        self._pending_value_features = None
        self._pending_value_targets = None
        self._pending_value_mask = None
        return metrics

    def _append_ppo_diag(self, metrics: Dict[str, float]) -> None:
        path_str = os.environ.get("ALLMERGE_PPO_DIAG_CSV", "").strip()
        if not path_str:
            return

        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.is_file() and path.stat().st_size > 0
        row = {"step": int(self._ppo_diag_step)}
        for key in PPO_DIAG_FIELDS:
            row[key] = metrics.get(key, float("nan"))

        with path.open("a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=("step", *PPO_DIAG_FIELDS),
            )
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def train_step(self, *args, **kwargs) -> Dict[str, float]:
        metrics = super().train_step(*args, **kwargs)
        value_metrics = self._update_value_model()

        ppo_metrics = dict(self._pending_collect_metrics)
        ppo_metrics.update(value_metrics)
        metrics.update(ppo_metrics)
        self.last_ppo_metrics = dict(ppo_metrics)

        self._ppo_diag_step += 1
        self._append_ppo_diag(ppo_metrics)
        return metrics

    def save_value_checkpoint(self, path: str | Path) -> None:
        if self.value_model is None or self.value_optimizer is None:
            return
        payload = {
            "format": "allmerge_ppo_mode_value_v2",
            "conditioning": "shared_state_plus_coarse_endpoint",
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


# PPO_MODE_CONDITIONED_VALUE_V2_20261005
