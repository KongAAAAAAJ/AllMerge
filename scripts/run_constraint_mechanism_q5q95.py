from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _expand_valid_mask(rewards, valid_mask):
    if rewards.ndim != 3:
        raise ValueError(
            f"Q5/Q95 baseline expects rewards [B,G,M], got {tuple(rewards.shape)}"
        )
    if valid_mask is None:
        return torch.ones_like(rewards, dtype=torch.bool)
    if tuple(valid_mask.shape) != (rewards.shape[0], rewards.shape[2]):
        raise ValueError(
            f"valid_mask must be [B,M], got {tuple(valid_mask.shape)}"
        )
    return valid_mask[:, None, :].expand_as(rewards).to(
        device=rewards.device, dtype=torch.bool
    )


class QuantileClippedGlobalAdvantage:
    """Frozen AllMerge GRPO baseline: global EMA normalization + Q5/Q95 clip."""

    def __init__(
        self,
        *,
        beta: float,
        q_low: float,
        q_high: float,
        diag_csv: Path,
    ):
        self.beta = float(beta)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.var_ema = None
        self.call = 0
        self.diag_csv = diag_csv
        self.diag_csv.parent.mkdir(parents=True, exist_ok=True)
        self.fields = [
            "call",
            "reward_mean",
            "reward_std_all",
            "centered_std",
            "global_std_ema",
            "preclip_adv_std",
            "postclip_adv_std",
            "advantage_abs_mean",
            "advantage_max_abs",
            "q_low_value",
            "q_high_value",
            "clip_changed_fraction",
        ]
        with self.diag_csv.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writeheader()

    @staticmethod
    def _f(x):
        if torch.is_tensor(x):
            return float(x.detach().float().cpu())
        return float(x)

    def __call__(
        self,
        rewards: torch.Tensor,
        *,
        group_dim: int = 1,
        eps: float = 1e-6,
        valid_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if group_dim != 1:
            raise ValueError(
                f"Q5/Q95 baseline expects group_dim=1, got {group_dim}"
            )

        mask = _expand_valid_mask(rewards, valid_mask)
        group_mean = rewards.mean(dim=1, keepdim=True)
        centered = rewards - group_mean
        centered = torch.where(mask, centered, torch.zeros_like(centered))

        rv = rewards[mask].float()
        cv = centered[mask].float()
        if cv.numel() == 0:
            return torch.zeros_like(rewards)

        batch_var = cv.square().mean()
        batch_var_f = self._f(batch_var)
        if self.var_ema is None:
            self.var_ema = batch_var_f
        else:
            self.var_ema = (
                self.beta * self.var_ema
                + (1.0 - self.beta) * batch_var_f
            )

        denom = max(math.sqrt(max(self.var_ema, 0.0)), float(eps))
        pre = centered / float(denom)
        pv = pre[mask].float()

        qlo = torch.quantile(pv, self.q_low)
        qhi = torch.quantile(pv, self.q_high)
        advantage = pre.clamp(qlo.to(pre.dtype), qhi.to(pre.dtype))
        advantage = torch.where(mask, advantage, torch.zeros_like(advantage))
        av = advantage[mask].float()

        changed = ((pv < qlo) | (pv > qhi)).float().mean()
        row = {
            "call": self.call,
            "reward_mean": self._f(rv.mean()),
            "reward_std_all": self._f(rv.std(unbiased=False)),
            "centered_std": self._f(cv.std(unbiased=False)),
            "global_std_ema": denom,
            "preclip_adv_std": self._f(pv.std(unbiased=False)),
            "postclip_adv_std": self._f(av.std(unbiased=False)),
            "advantage_abs_mean": self._f(av.abs().mean()),
            "advantage_max_abs": self._f(av.abs().max()),
            "q_low_value": self._f(qlo),
            "q_high_value": self._f(qhi),
            "clip_changed_fraction": self._f(changed),
        }
        with self.diag_csv.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)

        print(
            "[q5q95] "
            f"call={self.call:03d} "
            f"global_std={denom:.6g} "
            f"adv_std={row['postclip_adv_std']:.6g} "
            f"q=({row['q_low_value']:.4g},{row['q_high_value']:.4g}) "
            f"clipped={100.0*row['clip_changed_fraction']:.2f}%"
        )
        self.call += 1
        return advantage


def self_test():
    g = torch.Generator().manual_seed(7)
    rewards = torch.randn((3, 48, 5), generator=g) * 0.002
    rewards[0, 0, 0] = -0.05
    rewards[1, 1, 1] = 0.04
    mask = torch.ones((3, 5), dtype=torch.bool)

    tmp = Path("outputs") / "_q5q95_constraint_wrapper_selftest.csv"
    scaler = QuantileClippedGlobalAdvantage(
        beta=0.99, q_low=0.05, q_high=0.95, diag_csv=tmp
    )
    adv = scaler(rewards, valid_mask=mask)
    assert adv.shape == rewards.shape
    assert torch.isfinite(adv).all()
    rows = list(csv.DictReader(tmp.open("r", encoding="utf-8")))
    frac = float(rows[-1]["clip_changed_fraction"])
    if not 0.08 <= frac <= 0.12:
        raise RuntimeError(f"unexpected Q5/Q95 changed fraction: {frac}")
    try:
        tmp.unlink()
    except OSError:
        pass
    print("[PASS] Q5/Q95 synthetic self-test")
    return 0


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--q5q95-global-beta", type=float, default=0.99)
    p.add_argument("--q5q95-low", type=float, default=0.05)
    p.add_argument("--q5q95-high", type=float, default=0.95)
    p.add_argument("--q5q95-diag-csv", type=Path)
    p.add_argument("--run-log", type=Path)
    p.add_argument("--self-test", action="store_true")
    args, remaining = p.parse_known_args()

    if not 0.0 <= args.q5q95_global_beta < 1.0:
        raise SystemExit("--q5q95-global-beta must be in [0,1)")
    if not 0.0 <= args.q5q95_low < args.q5q95_high <= 1.0:
        raise SystemExit("invalid Q5/Q95 quantiles")
    if not args.self_test:
        if args.q5q95_diag_csv is None:
            raise SystemExit("--q5q95-diag-csv is required")
        if args.run_log is None:
            raise SystemExit("--run-log is required")
    return args, remaining


def main():
    args, remaining = parse_args()
    if args.self_test:
        return self_test()

    args.run_log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = args.run_log.open("w", encoding="utf-8", buffering=1)
    original_out, original_err = sys.stdout, sys.stderr
    sys.stdout = Tee(original_out, log_handle)
    sys.stderr = Tee(original_err, log_handle)

    try:
        import highway_env.planner.diffusion.grpo.objective as objective_module
        import highway_env.planner.diffusion.grpo.trainer as trainer_module
        import highway_env.planner.diffusion.grpo.constraint_strategy as strategy_module

        scaler = QuantileClippedGlobalAdvantage(
            beta=args.q5q95_global_beta,
            q_low=args.q5q95_low,
            q_high=args.q5q95_high,
            diag_csv=args.q5q95_diag_csv,
        )

        objective_module.group_relative_advantage = scaler
        trainer_module.group_relative_advantage = scaler
        strategy_module.group_relative_advantage = scaler

        print("=" * 78)
        print("AllMerge constrained GRPO mechanism ablation")
        print("baseline=Quantile-Clipped Global Advantage")
        print(
            f"global_ema_beta={args.q5q95_global_beta} "
            f"clip=Q{int(args.q5q95_low*100)}/Q{int(args.q5q95_high*100)}"
        )
        print("denoising_discount=disabled (gamma=1.0)")
        print("trainer + constraint_strategy share one Q5/Q95 scaler")
        print("core repository source files are unchanged; runtime patch only")
        print("=" * 78)

        import train_grpo
        sys.argv = [str(Path(train_grpo.__file__).resolve()), *remaining]
        rc = train_grpo.main()
        return int(rc) if rc is not None else 0
    finally:
        sys.stdout = original_out
        sys.stderr = original_err
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
