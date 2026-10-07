from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch

MARKER = "GRPO_ADVANTAGE_NORM_V1_20261007"


def _parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--advantage-norm",
        choices=("group", "centered", "global"),
        required=True,
    )
    p.add_argument("--global-std-beta", type=float, default=0.99)
    p.add_argument("--advantage-diag-csv", type=Path, required=True)
    args, remaining = p.parse_known_args()
    if not 0.0 <= args.global_std_beta < 1.0:
        raise SystemExit("--global-std-beta must satisfy 0 <= beta < 1")
    return args, remaining


def _expand_valid_mask(
    rewards: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(rewards, dtype=torch.bool)
    mask = valid_mask.to(device=rewards.device, dtype=torch.bool)
    while mask.ndim < rewards.ndim:
        mask = mask.unsqueeze(1)
    try:
        return mask.expand_as(rewards)
    except RuntimeError as exc:
        raise RuntimeError(
            f"cannot broadcast valid_mask shape={tuple(valid_mask.shape)} "
            f"to rewards shape={tuple(rewards.shape)}"
        ) from exc


def _group_center(
    rewards: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.ndim < 2:
        raise RuntimeError(
            f"expected rewards with group dimension at dim=1, got {tuple(rewards.shape)}"
        )
    mean = rewards.mean(dim=1, keepdim=True)
    centered = rewards - mean
    mask = _expand_valid_mask(rewards, valid_mask)
    centered = torch.where(mask, centered, torch.zeros_like(centered))
    return centered, mask


def _finite_float(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu())


class AdvantageNormalizer:
    def __init__(
        self,
        *,
        mode: str,
        beta: float,
        diag_csv: Path,
        original_group_fn,
    ) -> None:
        self.mode = mode
        self.beta = float(beta)
        self.diag_csv = diag_csv
        self.original_group_fn = original_group_fn
        self.global_var: float | None = None
        self.call_index = 0

        self.diag_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.diag_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "call",
                    "mode",
                    "reward_mean",
                    "reward_std_all",
                    "group_std_mean",
                    "group_std_min",
                    "group_std_max",
                    "batch_global_std",
                    "denominator",
                    "advantage_mean",
                    "advantage_std",
                    "advantage_abs_mean",
                    "advantage_max_abs",
                ],
            )
            w.writeheader()

    def __call__(
        self,
        rewards: torch.Tensor,
        *,
        eps: float = 1e-6,
        valid_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        centered, mask = _group_center(rewards, valid_mask)
        valid_centered = centered[mask].float()
        valid_rewards = rewards[mask].float()

        if valid_centered.numel() == 0:
            return torch.zeros_like(rewards)

        group_var = centered.float().pow(2).mean(dim=1)
        group_std = torch.sqrt(group_var.clamp_min(0.0))
        if valid_mask is None:
            vm = torch.ones_like(group_std, dtype=torch.bool)
        else:
            vm = valid_mask.to(device=group_std.device, dtype=torch.bool)
            while vm.ndim < group_std.ndim:
                vm = vm.unsqueeze(0)
            vm = vm.expand_as(group_std)
        group_std_valid = group_std[vm]

        batch_var = valid_centered.pow(2).mean()
        batch_std = torch.sqrt(batch_var.clamp_min(0.0))

        if self.mode == "group":
            advantage = self.original_group_fn(
                rewards,
                eps=eps,
                valid_mask=valid_mask,
                **kwargs,
            )
            denominator = (
                group_std_valid.mean()
                if group_std_valid.numel()
                else batch_std
            )

        elif self.mode == "centered":
            advantage = centered
            denominator = torch.ones(
                (),
                device=rewards.device,
                dtype=rewards.dtype,
            )

        elif self.mode == "global":
            batch_var_f = _finite_float(batch_var)
            if self.global_var is None:
                self.global_var = batch_var_f
            else:
                self.global_var = (
                    self.beta * self.global_var
                    + (1.0 - self.beta) * batch_var_f
                )
            global_std = math.sqrt(max(self.global_var, 0.0))
            denom_value = max(global_std, float(eps))
            denominator = torch.tensor(
                denom_value,
                device=rewards.device,
                dtype=rewards.dtype,
            )
            advantage = centered / denominator

        else:
            raise RuntimeError(self.mode)

        advantage = torch.where(mask, advantage, torch.zeros_like(advantage))
        valid_adv = advantage[mask].float()

        row = {
            "call": self.call_index,
            "mode": self.mode,
            "reward_mean": _finite_float(valid_rewards.mean()),
            "reward_std_all": _finite_float(valid_rewards.std(unbiased=False)),
            "group_std_mean": (
                _finite_float(group_std_valid.mean())
                if group_std_valid.numel()
                else math.nan
            ),
            "group_std_min": (
                _finite_float(group_std_valid.min())
                if group_std_valid.numel()
                else math.nan
            ),
            "group_std_max": (
                _finite_float(group_std_valid.max())
                if group_std_valid.numel()
                else math.nan
            ),
            "batch_global_std": _finite_float(batch_std),
            "denominator": _finite_float(denominator.float().mean()),
            "advantage_mean": _finite_float(valid_adv.mean()),
            "advantage_std": _finite_float(valid_adv.std(unbiased=False)),
            "advantage_abs_mean": _finite_float(valid_adv.abs().mean()),
            "advantage_max_abs": _finite_float(valid_adv.abs().max()),
        }
        with self.diag_csv.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writerow(row)

        print(
            "[advnorm] "
            f"call={self.call_index:03d} mode={self.mode} "
            f"group_std={row['group_std_mean']:.6g} "
            f"global_std={row['batch_global_std']:.6g} "
            f"denom={row['denominator']:.6g} "
            f"adv_std={row['advantage_std']:.6g} "
            f"adv_max={row['advantage_max_abs']:.6g}"
        )
        self.call_index += 1
        return advantage


def main() -> int:
    args, remaining = _parse_wrapper_args()

    import highway_env.planner.diffusion.grpo.trainer as trainer_module

    original = trainer_module.group_relative_advantage
    normalizer = AdvantageNormalizer(
        mode=args.advantage_norm,
        beta=args.global_std_beta,
        diag_csv=args.advantage_diag_csv,
        original_group_fn=original,
    )
    trainer_module.group_relative_advantage = normalizer

    print("=" * 76)
    print("GRPO Advantage Normalization V1")
    print(f"mode={args.advantage_norm}")
    if args.advantage_norm == "global":
        print(f"global variance EMA beta={args.global_std_beta}")
    print(f"diag_csv={args.advantage_diag_csv}")
    print("core trainer/objective/reward/model files are NOT modified")
    print("=" * 76)

    from scripts import run_vanilla_learning_probe as probe

    sys.argv = [str(Path(probe.__file__).resolve()), *remaining]
    result = probe.main()
    return int(result) if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
