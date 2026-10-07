from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

MODES = (
    ("global_temp", "Global-Temp"),
    ("percentile_floor", "Percentile-Floor"),
    ("mad_temp", "MAD-Temp"),
    ("dppo", "DPPO-style"),
)


def rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def val(row, *keys):
    for key in keys:
        x = row.get(key)
        if x not in (None, ""):
            try:
                return float(x)
            except ValueError:
                pass
    return math.nan


def finite(xs):
    a = np.asarray(xs, dtype=np.float64)
    return a[np.isfinite(a)]


def mean(xs):
    a = finite(xs)
    return float(a.mean()) if a.size else math.nan


def maxv(xs):
    a = finite(xs)
    return float(a.max()) if a.size else math.nan


def last(xs):
    a = finite(xs)
    return float(a[-1]) if a.size else math.nan


def slope(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return math.nan
    return float(np.polyfit(x[m], y[m], 1)[0])


def summarize(name, label, root, max_grad_norm):
    p = rows(root / name / "learning_probe.csv")
    d = rows(root / name / "adv_scale_diag.csv")

    steps = [val(r, "step") for r in p]
    gains = [val(r, "probe/reward_gain", "reward_gain") for r in p]
    emas = [val(r, "probe/reward_gain_ema", "reward_gain_ema") for r in p]
    refkl = [val(r, "train/reference_kl", "reference_kl") for r in p]
    approxkl = [val(r, "train/approx_kl", "approx_kl") for r in p]
    ppo_clip = [val(r, "train/clip_fraction", "clip_fraction") for r in p]
    grad = [val(r, "train/grad_norm", "grad_norm") for r in p]
    upd = [
        val(r, "train/parameter_update_norm", "parameter_update_norm")
        for r in p
    ]
    delta = [
        val(r, "probe/candidate_delta_m", "candidate_delta_m") for r in p
    ]
    selected = [
        val(r, "probe/selected_reward_gain", "selected_reward_gain") for r in p
    ]

    xs = np.asarray(steps, dtype=np.float64)
    ys = np.asarray(gains, dtype=np.float64)
    if np.isfinite(xs).any():
        last_step = np.nanmax(xs)
        m = xs >= max(1.0, last_step - 9.0)
        slope10 = slope(xs[m], ys[m])
    else:
        slope10 = math.nan

    fg = finite(gains)
    if fg.size >= 2:
        incpos = float(np.mean(np.diff(fg) > 0))
    else:
        incpos = math.nan

    ga = finite(grad)
    if ga.size:
        grad_clip_fraction = float(np.mean(ga > max_grad_norm))
        clip_scale = np.minimum(
            1.0, max_grad_norm / np.maximum(ga, 1e-30)
        )
        mean_grad_clip_scale = float(clip_scale.mean())
    else:
        grad_clip_fraction = math.nan
        mean_grad_clip_scale = math.nan

    final_gain = last(gains)
    final_ema = last(emas)
    final_kl = last(refkl)
    max_kl = maxv(refkl)
    gain_per_kl = (
        final_gain / max(abs(final_kl), 1e-12)
        if math.isfinite(final_gain) and math.isfinite(final_kl)
        else math.nan
    )

    qualified = int(
        math.isfinite(final_gain)
        and math.isfinite(final_ema)
        and math.isfinite(slope10)
        and final_gain > 0
        and final_ema > 0
        and slope10 > 0
        and math.isfinite(max_kl)
        and max_kl <= 0.25
    )

    return {
        "name": name,
        "label": label,
        "qualified": qualified,
        "final_gain": final_gain,
        "final_gain_ema": final_ema,
        "slope_last10": slope10,
        "increment_positive_fraction": incpos,
        "selected_gain": last(selected),
        "final_reference_kl": final_kl,
        "max_reference_kl": max_kl,
        "mean_approx_kl": mean(approxkl),
        "mean_ppo_clip_fraction": mean(ppo_clip),
        "mean_raw_grad_norm": mean(grad),
        "max_raw_grad_norm": maxv(grad),
        "grad_clip_fraction": grad_clip_fraction,
        "mean_grad_clip_scale": mean_grad_clip_scale,
        "mean_parameter_update_norm": mean(upd),
        "final_candidate_delta_m": last(delta),
        "mean_advantage_std": mean(
            [val(r, "advantage_std") for r in d]
        ),
        "mean_denominator": mean(
            [val(r, "denominator") for r in d]
        ),
        "mean_clip_changed_fraction": mean(
            [val(r, "clip_changed_fraction") for r in d]
        ),
        "mean_dppo_weight": mean(
            [val(r, "dppo_weight_mean") for r in d]
        ),
        "gain_per_reference_kl": gain_per_kl,
    }


def fmt(x, scale=1.0, digits=3):
    try:
        x = float(x)
    except Exception:
        return "nan"
    if not math.isfinite(x):
        return "nan"
    return f"{x*scale:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--max-grad-norm", type=float, default=5.0)
    ap.add_argument("--baseline-root", type=Path, default=None)
    args = ap.parse_args()

    out = []
    for name, label in MODES:
        p = args.root / name / "learning_probe.csv"
        if p.exists():
            out.append(
                summarize(name, label, args.root, args.max_grad_norm)
            )

    if not out:
        raise SystemExit("No completed Advantage Scale V2 runs found.")

    ranked = sorted(
        out,
        key=lambda r: (
            r["qualified"],
            r["final_gain"]
            if math.isfinite(r["final_gain"]) else -1e30,
            r["gain_per_reference_kl"]
            if math.isfinite(r["gain_per_reference_kl"]) else -1e30,
        ),
        reverse=True,
    )

    print()
    print("=" * 168)
    print(
        "Advantage Scale V2 — reward-first comparison "
        "(KL constraint, clip/grad diagnostics)"
    )
    print("=" * 168)
    header = (
        f"{'rank':>4} {'method':<18} {'gain x1e6':>11} "
        f"{'EMA x1e6':>10} {'slope x1e6':>12} {'inc+':>6} "
        f"{'advStd':>8} {'gradMean':>10} {'gClip%':>8} "
        f"{'ppoClip':>8} {'refKL':>8} {'maxKL':>8} "
        f"{'delta(cm)':>10} {'gain/KL x1e6':>14} {'Q':>3}"
    )
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(ranked, 1):
        print(
            f"{rank:>4} {r['label']:<18} "
            f"{fmt(r['final_gain'],1e6):>11} "
            f"{fmt(r['final_gain_ema'],1e6):>10} "
            f"{fmt(r['slope_last10'],1e6):>12} "
            f"{fmt(r['increment_positive_fraction']):>6} "
            f"{fmt(r['mean_advantage_std'],1,4):>8} "
            f"{fmt(r['mean_raw_grad_norm'],1,2):>10} "
            f"{fmt(r['grad_clip_fraction'],100,1):>8} "
            f"{fmt(r['mean_ppo_clip_fraction']):>8} "
            f"{fmt(r['final_reference_kl']):>8} "
            f"{fmt(r['max_reference_kl']):>8} "
            f"{fmt(r['final_candidate_delta_m'],100):>10} "
            f"{fmt(r['gain_per_reference_kl'],1e6,2):>14} "
            f"{r['qualified']:>3}"
        )

    out_csv = args.root / "advantage_scale_v2_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print()
    print("Decision rule:")
    print("  1) Hard stability preference: max reference KL <= 0.25.")
    print("  2) Primary ranking: larger final/EMA reward gain with positive slope.")
    print("  3) Secondary: larger reward_gain / reference_KL.")
    print("  4) PPO clip / grad clip are diagnostics, not the optimization target.")
    print(f"[OK] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
