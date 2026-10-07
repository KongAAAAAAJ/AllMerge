from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch


def parse_wrapper_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--adv-scale-mode",
        choices=("global_temp", "percentile_floor", "mad_temp", "dppo"),
        required=True,
    )
    p.add_argument("--global-beta", type=float, default=0.99)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--percentile-low", type=float, default=0.05)
    p.add_argument("--percentile-high", type=float, default=0.95)
    p.add_argument("--percentile-floor", type=float, default=0.01)
    p.add_argument("--mad-eps", type=float, default=1e-6)
    p.add_argument("--dppo-gamma-denoising", type=float, default=0.90)
    p.add_argument("--dppo-clip-low", type=float, default=0.05)
    p.add_argument("--dppo-clip-high", type=float, default=0.95)
    p.add_argument("--adv-scale-diag-csv", type=Path, required=True)
    args, remaining = p.parse_known_args()

    if not 0.0 <= args.global_beta < 1.0:
        raise SystemExit("--global-beta must satisfy 0 <= beta < 1")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0")
    if not 0 <= args.percentile_low < args.percentile_high <= 1:
        raise SystemExit("invalid percentile range")
    if args.percentile_floor <= 0:
        raise SystemExit("--percentile-floor must be > 0")
    if not 0.0 < args.dppo_gamma_denoising <= 1.0:
        raise SystemExit("--dppo-gamma-denoising must be in (0,1]")
    if not 0 <= args.dppo_clip_low < args.dppo_clip_high <= 1:
        raise SystemExit("invalid DPPO advantage clip quantiles")
    return args, remaining


def expand_valid_mask(rewards, valid_mask):
    if valid_mask is None:
        return torch.ones_like(rewards, dtype=torch.bool)
    if tuple(valid_mask.shape) != (rewards.shape[0], rewards.shape[2]):
        raise ValueError(
            "valid_mask must be [B,M], got "
            f"{tuple(valid_mask.shape)} for rewards {tuple(rewards.shape)}"
        )
    return valid_mask[:, None, :].expand_as(rewards).to(
        device=rewards.device, dtype=torch.bool
    )


class AdvantageScaleV2:
    def __init__(self, args, original_group_fn):
        self.args = args
        self.original_group_fn = original_group_fn
        self.global_var_ema = None
        self.percentile_spread_ema = None
        self.mad_scale_ema = None
        self.call = 0
        self.csv_path = args.adv_scale_diag_csv
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = [
            "call", "mode",
            "reward_mean", "reward_std_all",
            "centered_std", "group_std_mean",
            "batch_global_std", "global_std_ema",
            "p_low", "p_high", "percentile_spread",
            "percentile_spread_ema", "percentile_floor",
            "mad_raw", "mad_sigma", "mad_scale_ema",
            "temperature", "denominator",
            "preclip_adv_std", "advantage_mean",
            "advantage_std", "advantage_abs_mean",
            "advantage_max_abs",
            "clip_q_low", "clip_q_high",
            "clip_changed_fraction",
            "dppo_gamma", "dppo_weight_mean",
            "dppo_weight_min", "dppo_weight_max",
        ]
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writeheader()

    @staticmethod
    def f(x):
        if torch.is_tensor(x):
            return float(x.detach().float().cpu())
        return float(x)

    @staticmethod
    def ema(old, new, beta):
        return new if old is None else beta * old + (1.0 - beta) * new

    def __call__(
        self,
        rewards: torch.Tensor,
        *,
        group_dim: int = 1,
        eps: float = 1e-6,
        valid_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if rewards.ndim != 3 or group_dim != 1:
            raise ValueError(
                "Advantage Scale V2 expects AllMerge rewards [B,G,M] "
                "with group_dim=1"
            )

        mask = expand_valid_mask(rewards, valid_mask)
        group_mean = rewards.mean(dim=1, keepdim=True)
        centered = rewards - group_mean
        centered = torch.where(mask, centered, torch.zeros_like(centered))
        rv = rewards[mask].float()
        cv = centered[mask].float()
        if rv.numel() == 0:
            return torch.zeros_like(rewards)

        group_std = centered.float().pow(2).mean(dim=1).sqrt()
        vm = (
            torch.ones_like(group_std, dtype=torch.bool)
            if valid_mask is None
            else valid_mask.to(device=group_std.device, dtype=torch.bool)
        )
        group_std_valid = group_std[vm]

        batch_var = cv.pow(2).mean()
        batch_std = torch.sqrt(batch_var.clamp_min(0.0))
        batch_var_f = self.f(batch_var)
        self.global_var_ema = self.ema(
            self.global_var_ema, batch_var_f, self.args.global_beta
        )
        global_std_ema = math.sqrt(max(self.global_var_ema, 0.0))

        p_low = torch.quantile(rv, self.args.percentile_low)
        p_high = torch.quantile(rv, self.args.percentile_high)
        spread = self.f(p_high - p_low)
        self.percentile_spread_ema = self.ema(
            self.percentile_spread_ema, spread, self.args.global_beta
        )

        median = torch.median(cv)
        mad_raw_t = torch.median((cv - median).abs())
        mad_raw = self.f(mad_raw_t)
        mad_sigma = 1.4826 * mad_raw
        self.mad_scale_ema = self.ema(
            self.mad_scale_ema, mad_sigma, self.args.global_beta
        )

        mode = self.args.adv_scale_mode
        clip_lo = math.nan
        clip_hi = math.nan
        clip_changed = 0.0
        preclip_std = math.nan

        if mode == "global_temp":
            denom = max(global_std_ema, eps)
            advantage = (
                self.args.temperature * centered / float(denom)
            )

        elif mode == "percentile_floor":
            denom = max(
                float(self.percentile_spread_ema),
                float(self.args.percentile_floor),
                float(eps),
            )
            advantage = centered / float(denom)

        elif mode == "mad_temp":
            denom = max(float(self.mad_scale_ema), self.args.mad_eps, eps)
            advantage = (
                self.args.temperature * centered / float(denom)
            )

        elif mode == "dppo":
            # Global normalization first, then DPPO-style robust clipping.
            denom = max(global_std_ema, eps)
            pre = centered / float(denom)
            pv = pre[mask].float()
            preclip_std = self.f(pv.std(unbiased=False))
            qlo = torch.quantile(pv, self.args.dppo_clip_low)
            qhi = torch.quantile(pv, self.args.dppo_clip_high)
            advantage = pre.clamp(qlo.to(pre.dtype), qhi.to(pre.dtype))
            clip_lo = self.f(qlo)
            clip_hi = self.f(qhi)
            clip_changed = self.f(
                ((pv < qlo) | (pv > qhi)).float().mean()
            )
        else:
            raise RuntimeError(mode)

        advantage = torch.where(mask, advantage, torch.zeros_like(advantage))
        av = advantage[mask].float()

        row = {
            "call": self.call,
            "mode": mode,
            "reward_mean": self.f(rv.mean()),
            "reward_std_all": self.f(rv.std(unbiased=False)),
            "centered_std": self.f(cv.std(unbiased=False)),
            "group_std_mean": (
                self.f(group_std_valid.mean())
                if group_std_valid.numel() else math.nan
            ),
            "batch_global_std": self.f(batch_std),
            "global_std_ema": global_std_ema,
            "p_low": self.f(p_low),
            "p_high": self.f(p_high),
            "percentile_spread": spread,
            "percentile_spread_ema": self.percentile_spread_ema,
            "percentile_floor": self.args.percentile_floor,
            "mad_raw": mad_raw,
            "mad_sigma": mad_sigma,
            "mad_scale_ema": self.mad_scale_ema,
            "temperature": (
                self.args.temperature
                if mode in ("global_temp", "mad_temp")
                else 1.0
            ),
            "denominator": denom,
            "preclip_adv_std": preclip_std,
            "advantage_mean": self.f(av.mean()),
            "advantage_std": self.f(av.std(unbiased=False)),
            "advantage_abs_mean": self.f(av.abs().mean()),
            "advantage_max_abs": self.f(av.abs().max()),
            "clip_q_low": clip_lo,
            "clip_q_high": clip_hi,
            "clip_changed_fraction": clip_changed,
            "dppo_gamma": (
                self.args.dppo_gamma_denoising if mode == "dppo" else 1.0
            ),
            "dppo_weight_mean": math.nan,
            "dppo_weight_min": math.nan,
            "dppo_weight_max": math.nan,
        }
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)

        print(
            "[adv-v2] "
            f"call={self.call:03d} mode={mode} "
            f"denom={denom:.6g} adv_std={row['advantage_std']:.6g} "
            f"adv_max={row['advantage_max_abs']:.6g}"
        )
        self.call += 1
        return advantage

    def write_dppo_weights(self, weights: torch.Tensor):
        if self.args.adv_scale_mode != "dppo":
            return
        # Update the latest row with objective-side denoising weights.
        rows = []
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        rows[-1]["dppo_weight_mean"] = str(self.f(weights.mean()))
        rows[-1]["dppo_weight_min"] = str(self.f(weights.min()))
        rows[-1]["dppo_weight_max"] = str(self.f(weights.max()))
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader()
            w.writerows(rows)


def build_dppo_objective(original_objective, objective_module, scaler):
    result_cls = objective_module.GRPOObjectiveResult

    def masked_mean(value, mask):
        if mask is None:
            return value.mean()
        expanded = mask
        while expanded.ndim < value.ndim:
            expanded = expanded.unsqueeze(2)
        expanded = expanded.expand_as(value)
        denom = expanded.sum().clamp_min(1)
        return (value * expanded.to(value.dtype)).sum() / denom

    def dppo_objective(
        new_log_prob,
        old_log_prob,
        advantages,
        *,
        clip_eps=0.2,
        valid_mask=None,
    ):
        if new_log_prob.shape != old_log_prob.shape or new_log_prob.ndim != 4:
            raise ValueError(
                "DPPO-style objective requires log_prob [B,G,S,M]; got "
                f"{tuple(new_log_prob.shape)}"
            )
        expected = (
            new_log_prob.shape[0],
            new_log_prob.shape[1],
            new_log_prob.shape[3],
        )
        if tuple(advantages.shape) != expected:
            raise ValueError(
                f"advantages must be {expected}, got {tuple(advantages.shape)}"
            )

        delta = new_log_prob - old_log_prob
        ratio = torch.exp(delta)
        adv = advantages.unsqueeze(2)

        s = new_log_prob.shape[2]
        idx = torch.arange(
            s, device=new_log_prob.device, dtype=new_log_prob.dtype
        )
        # Sampling index 0 is the high-noise/early reverse step; last is
        # closest to x0 and receives weight 1.
        power = (s - 1) - idx
        weights = torch.pow(
            torch.tensor(
                scaler.args.dppo_gamma_denoising,
                device=new_log_prob.device,
                dtype=new_log_prob.dtype,
            ),
            power,
        ).view(1, 1, s, 1)

        unclipped = -adv * ratio
        clipped = -adv * ratio.clamp(
            1.0 - clip_eps, 1.0 + clip_eps
        )
        element_loss = torch.maximum(unclipped, clipped) * weights

        step_mask = (
            None
            if valid_mask is None
            else valid_mask[:, None, None, :]
        )
        policy_loss = masked_mean(element_loss, step_mask)

        # Diagnostics deliberately remain unweighted so KL/clip numbers are
        # comparable to the other V2 variants.
        approx_kl = masked_mean(0.5 * delta.square(), step_mask)
        clip_fraction = masked_mean(
            (torch.abs(ratio - 1.0) > clip_eps).to(ratio.dtype),
            step_mask,
        )
        ratio_mean = masked_mean(ratio, step_mask)
        ratio_std = torch.sqrt(
            masked_mean(
                (ratio - ratio_mean).square(), step_mask
            ).clamp_min(0.0)
        )
        scaler.write_dppo_weights(weights)
        return result_cls(
            loss=policy_loss,
            policy_loss=policy_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
            ratio_mean=ratio_mean,
            ratio_std=ratio_std,
        )

    return dppo_objective


def main():
    args, remaining = parse_wrapper_args()

    import highway_env.planner.diffusion.grpo.objective as obj_module
    import highway_env.planner.diffusion.grpo.trainer as trainer_module

    scaler = AdvantageScaleV2(
        args,
        original_group_fn=trainer_module.group_relative_advantage,
    )
    trainer_module.group_relative_advantage = scaler

    if args.adv_scale_mode == "dppo":
        trainer_module.grpo_clipped_objective = build_dppo_objective(
            trainer_module.grpo_clipped_objective,
            obj_module,
            scaler,
        )

    print("=" * 78)
    print("GRPO Advantage Scale V2")
    print(f"mode={args.adv_scale_mode}")
    print(f"diag={args.adv_scale_diag_csv}")
    if args.adv_scale_mode == "global_temp":
        print(f"global EMA beta={args.global_beta} alpha={args.temperature}")
    elif args.adv_scale_mode == "percentile_floor":
        print(
            f"EMA P{int(args.percentile_low*100)}-P"
            f"{int(args.percentile_high*100)} spread, "
            f"floor={args.percentile_floor}"
        )
    elif args.adv_scale_mode == "mad_temp":
        print(
            f"MAD sigma=1.4826*MAD, EMA beta={args.global_beta}, "
            f"alpha={args.temperature}"
        )
    elif args.adv_scale_mode == "dppo":
        print(
            f"global EMA + Q{int(args.dppo_clip_low*100)}/"
            f"Q{int(args.dppo_clip_high*100)} advantage clip + "
            f"gamma_denoising={args.dppo_gamma_denoising}"
        )
    print("core repository source is unchanged; runtime patch only")
    print("=" * 78)

    from scripts import run_vanilla_learning_probe as probe

    sys.argv = [str(Path(probe.__file__).resolve()), *remaining]
    rc = probe.main()
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
