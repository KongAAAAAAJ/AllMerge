"""GRPO -> W4 trajectory-mode reward bridge.

There is exactly one production reward implementation:
``trajectory_mode_reward.evaluate_candidates``.

GRPO candidate layout:
    [3, N, 10, 8, 2]  = [vehicle, sample, mode, time, xy]

W4 reward layout:
    [3, 10, N, 8, 2]  = [vehicle, mode, sample, time, xy]

This module only performs contract conversion and returns the W4 scorer output.
It contains no reward mathematics.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping

import numpy as np
import torch


AUTO_REWARD_CANDIDATES = (
    "highway_env.planner.diffusion.trajectory_mode_reward:evaluate_candidates",
    "highway_env.planner.diffusion.trajectory_mode_reward.evaluator:evaluate_candidates",
)

_W4_PARAMETERS = frozenset(
    {
        "env",
        "candidates",
        "frozen_all_mode_trajectories",
        "frozen_argmax_joint_trajectories",
        "valid_mode_mask",
    }
)


def _import_symbol(path: str) -> Callable[..., Any]:
    module_name, sep, symbol = path.partition(":")
    if not sep:
        raise ValueError(
            "reward function must use 'module:function' notation"
        )
    module = importlib.import_module(module_name)
    fn = getattr(module, symbol)
    if not callable(fn):
        raise TypeError(f"{path} is not callable")
    return fn


def resolve_reward_evaluator(
    path: str = "auto",
) -> Callable[..., Any]:
    if path != "auto":
        return _import_symbol(path)

    errors = []
    for candidate in AUTO_REWARD_CANDIDATES:
        try:
            return _import_symbol(candidate)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{candidate}: {exc}")

    raise ImportError(
        "Could not locate W4 evaluate_candidates(). "
        "Pass --reward-fn module:function.\n"
        + "\n".join(errors)
    )


def fake_progress_reward(
    candidates: torch.Tensor,
    **_: Any,
) -> torch.Tensor:
    """Deterministic W3 smoke reward; never used as production reward."""
    return (
        candidates[..., -1, 0]
        - 0.05
        * candidates[..., :, 1]
        .abs()
        .mean(dim=-1)
    )


@dataclass(frozen=True)
class GRPORewardContext:
    """All state required by the shared W4 counterfactual scorer."""

    env: object
    frozen_all_mode_trajectories: object
    frozen_argmax_joint_trajectories: object
    valid_mode_mask: object
    config: object | None = None


def _is_w4_evaluator(
    evaluator: Callable[..., Any],
) -> bool:
    parameters = set(
        inspect.signature(evaluator).parameters
    )
    return _W4_PARAMETERS.issubset(
        parameters
    )


def _coerce_context(
    context: Any,
    *,
    features: Mapping[str, Any] | None = None,
) -> GRPORewardContext:
    if isinstance(
        context,
        GRPORewardContext,
    ):
        return context

    if isinstance(
        context,
        Mapping,
    ):
        required = (
            "env",
            "frozen_all_mode_trajectories",
            "frozen_argmax_joint_trajectories",
        )
        missing = [
            name
            for name in required
            if name not in context
        ]
        if missing:
            raise ValueError(
                "GRPO W4 reward context is missing: "
                + ", ".join(missing)
            )

        valid = context.get(
            "valid_mode_mask"
        )
        if (
            valid is None
            and features is not None
        ):
            valid = features.get(
                "mode_valid_mask"
            )
        if valid is None:
            raise ValueError(
                "GRPO W4 reward context requires "
                "valid_mode_mask"
            )

        return GRPORewardContext(
            env=context["env"],
            frozen_all_mode_trajectories=(
                context[
                    "frozen_all_mode_trajectories"
                ]
            ),
            frozen_argmax_joint_trajectories=(
                context[
                    "frozen_argmax_joint_trajectories"
                ]
            ),
            valid_mode_mask=valid,
            config=context.get("config"),
        )

    raise TypeError(
        "Production W4 reward requires "
        "GRPORewardContext or a mapping with env/frozen trajectories"
    )


def _to_numpy(
    value: Any,
) -> np.ndarray:
    if torch.is_tensor(value):
        return (
            value.detach()
            .cpu()
            .numpy()
        )
    return np.asarray(value)


def grpo_to_w4_candidates(
    candidates: Any,
) -> np.ndarray:
    """Convert [3,N,10,8,2] -> [3,10,N,8,2]."""
    values = _to_numpy(candidates)

    if (
        values.ndim != 5
        or values.shape[0] != 3
        or values.shape[2] != 10
        or values.shape[3:] != (8, 2)
        or values.shape[1] <= 0
    ):
        raise ValueError(
            "production GRPO candidates must be "
            "[3,N,10,8,2]"
        )
    if not np.isfinite(values).all():
        raise ValueError(
            "GRPO candidates contain NaN/Inf"
        )

    return np.ascontiguousarray(
        np.transpose(
            values,
            (0, 2, 1, 3, 4),
        ),
        dtype=np.float32,
    )


def w4_to_grpo_rewards(
    rewards: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert W4 [3,10,N] -> trainer [3,N,10]."""
    values = np.asarray(
        rewards,
        dtype=np.float32,
    )
    if (
        values.ndim != 3
        or values.shape[:2] != (3, 10)
        or values.shape[2] <= 0
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "W4 rewards must be finite [3,10,N]"
        )

    transposed = np.ascontiguousarray(
        np.transpose(
            values,
            (0, 2, 1),
        )
    )
    return torch.as_tensor(
        transposed,
        device=device,
        dtype=dtype,
    )



# GRPO REWARD DECOMPOSITION DIAGNOSTICS V1
def _reward_decomposition_metrics(result: Any) -> Dict[str, float]:
    # Summarize the exact W4 result already computed for the last rollout.
    if result is None:
        return {}

    required = (
        "rewards",
        "pretrain_rewards",
        "valid_mode_mask",
        "components",
        "pretrain_components",
    )
    if any(not hasattr(result, name) for name in required):
        return {}

    rewards = np.asarray(result.rewards, dtype=np.float64)
    pretrain_rewards = np.asarray(
        result.pretrain_rewards,
        dtype=np.float64,
    )
    valid = np.asarray(
        result.valid_mode_mask,
        dtype=np.bool_,
    )

    if (
        rewards.ndim != 3
        or rewards.shape[:2] != valid.shape
        or pretrain_rewards.shape != valid.shape
    ):
        raise ValueError(
            "reward decomposition expects rewards [V,M,N], "
            "pretrain_rewards/valid_mode_mask [V,M]"
        )

    metrics: Dict[str, float] = {}

    for role in range(rewards.shape[0]):
        role_valid = valid[role]
        if not bool(role_valid.any()):
            continue

        current_reward = rewards[role, role_valid, :]
        frozen_reward = pretrain_rewards[role, role_valid]
        reward_gain = current_reward - frozen_reward[:, None]

        prefix = f"diagnostics/reward_decomp/vehicle_{role}"

        metrics[f"{prefix}_reward_mean"] = float(current_reward.mean())
        metrics[f"{prefix}_reward_std"] = float(current_reward.std())
        metrics[f"{prefix}_pretrain_reward_mean"] = float(
            frozen_reward.mean()
        )
        metrics[f"{prefix}_reward_delta_vs_pretrain"] = float(
            reward_gain.mean()
        )
        metrics[f"{prefix}_candidate_positive_fraction"] = float(
            (reward_gain > 1.0e-6).mean()
        )
        metrics[f"{prefix}_candidate_best_gain"] = float(
            reward_gain.max()
        )
        metrics[f"{prefix}_candidate_gain_std"] = float(
            reward_gain.std()
        )

        current_components = result.components
        frozen_components = result.pretrain_components

        common_names = sorted(
            set(current_components).intersection(frozen_components)
        )
        for name in common_names:
            current = np.asarray(
                current_components[name],
                dtype=np.float64,
            )
            frozen = np.asarray(
                frozen_components[name],
                dtype=np.float64,
            )

            if current.shape != rewards.shape or frozen.shape != valid.shape:
                continue

            current_values = current[role, role_valid, :]
            frozen_values = frozen[role, role_valid]
            delta = current_values - frozen_values[:, None]

            metrics[f"{prefix}_{name}_mean"] = float(
                current_values.mean()
            )
            metrics[f"{prefix}_{name}_delta_vs_pretrain"] = float(
                delta.mean()
            )

            if name.startswith("minimum_"):
                metrics[f"{prefix}_{name}_p10"] = float(
                    np.percentile(current_values, 10.0)
                )

        for name in (
            "collision",
            "out_of_drivable",
            "clearance_violation",
            "unsafe",
        ):
            pre_name = f"pretrain_{name}"
            if (
                not hasattr(result, name)
                or not hasattr(result, pre_name)
            ):
                continue

            current = np.asarray(
                getattr(result, name),
                dtype=np.bool_,
            )
            frozen = np.asarray(
                getattr(result, pre_name),
                dtype=np.bool_,
            )
            if current.shape != rewards.shape or frozen.shape != valid.shape:
                continue

            current_rate = float(
                current[role, role_valid, :].mean()
            )
            frozen_rate = float(
                frozen[role, role_valid].mean()
            )
            metrics[f"{prefix}_{name}_rate"] = current_rate
            metrics[
                f"{prefix}_{name}_rate_delta_vs_pretrain"
            ] = current_rate - frozen_rate

    return metrics


@dataclass
class CandidateRewardAdapter:
    """Thin GRPO bridge; production W4 scoring is not reimplemented here."""

    evaluator: Callable[..., Any]
    last_result: Any = field(
        default=None,
        init=False,
        repr=False,
    )

    def reward_decomposition_diagnostics(
        self,
    ) -> Dict[str, float]:
        return _reward_decomposition_metrics(
            self.last_result
        )

    def evaluate_result(
        self,
        candidates: torch.Tensor,
        *,
        features: Mapping[str, Any],
        context: Any,
    ) -> Any:
        """Return the full W4 ``TrajectoryModeRewardResult``."""
        if not _is_w4_evaluator(
            self.evaluator
        ):
            raise TypeError(
                "evaluate_result() is only valid for the "
                "W4 evaluate_candidates contract"
            )

        reward_context = (
            _coerce_context(
                context,
                features=features,
            )
        )
        native_candidates = (
            grpo_to_w4_candidates(
                candidates
            )
        )

        kwargs = {
            "env": reward_context.env,
            "candidates": native_candidates,
            "frozen_all_mode_trajectories": (
                _to_numpy(
                    reward_context
                    .frozen_all_mode_trajectories
                )
            ),
            "frozen_argmax_joint_trajectories": (
                _to_numpy(
                    reward_context
                    .frozen_argmax_joint_trajectories
                )
            ),
            "valid_mode_mask": (
                _to_numpy(
                    reward_context
                    .valid_mode_mask
                ).astype(
                    np.bool_,
                    copy=False,
                )
            ),
        }
        if (
            reward_context.config
            is not None
        ):
            kwargs["config"] = (
                reward_context.config
            )

        return self.evaluator(
            **kwargs
        )

    def _legacy_call(
        self,
        candidates: torch.Tensor,
        *,
        features: Dict[str, torch.Tensor],
        context: Any,
    ) -> torch.Tensor:
        """Keep fake/custom W3 smoke rewards working."""
        signature = inspect.signature(
            self.evaluator
        )
        available = {
            "candidates": candidates,
            "trajectories": candidates,
            "trajectory_candidates": candidates,
            "features": features,
            "context": context,
            "env": context,
            "mode_valid_mask": features.get(
                "mode_valid_mask"
            ),
        }
        kwargs = {
            name: available[name]
            for name in signature.parameters
            if name in available
        }
        result = self.evaluator(
            **kwargs
        )
        if isinstance(
            result,
            dict,
        ):
            for key in (
                "rewards",
                "reward",
                "scores",
                "score",
                "total_reward",
            ):
                if key in result:
                    result = result[key]
                    break
            else:
                raise KeyError(
                    "custom evaluator dict output has no reward field"
                )

        if torch.is_tensor(result):
            return result.to(
                candidates.device,
                candidates.dtype,
            )
        return torch.as_tensor(
            result,
            device=candidates.device,
            dtype=candidates.dtype,
        )

    def __call__(
        self,
        candidates: torch.Tensor,
        *,
        features: Dict[str, torch.Tensor],
        context: Any = None,
    ) -> torch.Tensor:
        if _is_w4_evaluator(
            self.evaluator
        ):
            result = self.evaluate_result(
                candidates,
                features=features,
                context=context,
            )
            self.last_result = result
            return w4_to_grpo_rewards(
                result.rewards,
                device=candidates.device,
                dtype=candidates.dtype,
            )

        self.last_result = None
        result = self._legacy_call(
            candidates,
            features=features,
            context=context,
        )
        expected = candidates.shape[:3]
        if result.shape != expected:
            raise ValueError(
                "custom/fake GRPO reward must return "
                f"{tuple(expected)}, got {tuple(result.shape)}"
            )
        return result
