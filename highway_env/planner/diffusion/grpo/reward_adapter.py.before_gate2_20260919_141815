from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
import torch


AUTO_REWARD_CANDIDATES = (
    "highway_env.planner.diffusion.trajectory_mode_reward:evaluate_candidates",
    "highway_env.planner.diffusion.trajectory_mode_reward.evaluator:evaluate_candidates",
    "highway_env.planner.diffusion.trajectory_mode_reward.reward:evaluate_candidates",
)


def _import_symbol(path: str) -> Callable[..., Any]:
    module_name, sep, symbol = path.partition(":")
    if not sep:
        raise ValueError("reward function must use 'module:function' notation")
    module = importlib.import_module(module_name)
    fn = getattr(module, symbol)
    if not callable(fn):
        raise TypeError(f"{path} is not callable")
    return fn


def resolve_reward_evaluator(path: str = "auto") -> Callable[..., Any]:
    if path != "auto":
        return _import_symbol(path)
    errors = []
    for candidate in AUTO_REWARD_CANDIDATES:
        try:
            return _import_symbol(candidate)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise ImportError(
        "Could not locate W4 evaluate_candidates(). Pass --reward-fn module:function.\n"
        + "\n".join(errors)
    )


def fake_progress_reward(candidates: torch.Tensor, **_: Any) -> torch.Tensor:
    """Deterministic smoke-test reward; production code must use W4 evaluator."""
    return candidates[..., -1, 0] - 0.05 * candidates[..., :, 1].abs().mean(dim=-1)


@dataclass
class CandidateRewardAdapter:
    evaluator: Callable[..., Any]

    def __call__(
        self,
        candidates: torch.Tensor,
        *,
        features: Dict[str, torch.Tensor],
        context: Any = None,
    ) -> torch.Tensor:
        if candidates.ndim != 5:
            raise ValueError("candidates must be [B,G,M,T,2]")
        batch, group, modes, horizon, xy = candidates.shape
        flat = candidates.reshape(batch * group * modes, horizon, xy)
        signature = inspect.signature(self.evaluator)
        available = {
            "candidates": flat,
            "trajectories": flat,
            "trajectory_candidates": flat,
            "features": features,
            "context": context,
            "env": context,
            "mode_valid_mask": features.get("mode_valid_mask"),
        }
        kwargs = {
            name: available[name]
            for name in signature.parameters
            if name in available
        }
        result = self.evaluator(**kwargs)
        if isinstance(result, dict):
            for key in ("rewards", "reward", "scores", "score", "total_reward"):
                if key in result:
                    result = result[key]
                    break
            else:
                raise KeyError("W4 evaluator dict output has no reward/scores field")
        if isinstance(result, np.ndarray):
            result = torch.from_numpy(result).to(candidates.device, candidates.dtype)
        elif not torch.is_tensor(result):
            result = torch.as_tensor(result, device=candidates.device, dtype=candidates.dtype)
        else:
            result = result.to(candidates.device, candidates.dtype)

        if result.numel() == batch * group * modes:
            return result.reshape(batch, group, modes)
        if result.shape == (batch, group, modes):
            return result
        raise ValueError(
            f"W4 evaluator must return {batch * group * modes} candidate rewards; "
            f"got shape {tuple(result.shape)}"
        )
