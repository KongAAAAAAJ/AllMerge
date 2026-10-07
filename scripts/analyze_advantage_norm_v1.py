from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

MODES = ("group", "centered", "global")


def _rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _f(row, *keys):
    for k in keys:
        if k in row and row[k] not in ("", None):
            try:
                return float(row[k])
            except ValueError:
                pass
    return math.nan


def _finite(values):
    a = np.asarray(values, dtype=np.float64)
    return a[np.isfinite(a)]


def _mean(values):
    a = _finite(values)
    return float(a.mean()) if a.size else math.nan


def _max(values):
    a = _finite(values)
    return float(a.max()) if a.size else math.nan


def _last(values):
    a = _finite(values)
    return float(a[-1]) if a.size else math.nan


def _slope(xs, ys):
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return math.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def summarize(mode: str, root: Path, max_grad_norm: float):
    probe_path = root / mode / "learning_probe.csv"
    diag_path = root / mode / "advantage_norm_diag.csv"
    if not probe_path.exists():
        raise FileNotFoundError(probe_path)
    if not diag_path.exists():
        raise FileNotFoundError(diag_path)

    p = _rows(probe_path)
    d = _rows(diag_path)

    steps = [_f(r, "step") for r in p]
    gains = [_f(r, "probe/reward_gain", "reward_gain") for r in p]
    emas = [_f(r, "probe/reward_gain_ema", "reward_gain_ema") for r in p]
    refkl = [_f(r, "train/reference_kl", "reference_kl") for r in p]
    approxkl = [_f(r, "train/approx_kl", "approx_kl") for r in p]
    clipfrac = [_f(r, "train/clip_fraction", "clip_fraction") for r in p]
    grad = [_f(r, "train/grad_norm", "grad_norm") for r in p]
    upd = [_f(r, "train/parameter_update_norm", "parameter_update_norm") for r in p]
    delta = [_f(r, "probe/candidate_delta_m", "candidate_delta_m") for r in p]

    finite_steps = np.asarray(steps, dtype=np.float64)
    finite_gains = np.asarray(gains, dtype=np.float64)
    if np.isfinite(finite_steps).any():
        last_step = np.nanmax(finite_steps)
        mask = finite_steps >= max(1.0, last_step - 9.0)
        slope10 = _slope(finite_steps[mask], finite_gains[mask])
    else:
        slope10 = math.nan

    grad_arr = _finite(grad)
    if grad_arr.size:
        clip_scale = np.minimum(
            1.0,
            max_grad_norm / np.maximum(grad_arr, 1e-30),
        )
        mean_clip_scale = float(clip_scale.mean())
        grad_clip_fraction = float(np.mean(grad_arr > max_grad_norm))
    else:
        mean_clip_scale = math.nan
        grad_clip_fraction = math.nan

    gain = _last(gains)
    final_refkl = _last(refkl)

    return {
        "mode": mode,
        "final_gain": gain,
        "final_gain_ema": _last(emas),
        "slope10": slope10,
        "final_reference_kl": final_refkl,
        "max_reference_kl": _max(refkl),
        "mean_approx_kl": _mean(approxkl),
        "mean_policy_clip_fraction": _mean(clipfrac),
        "mean_raw_grad_norm": _mean(grad),
        "max_raw_grad_norm": _max(grad),
        "mean_grad_clip_scale": mean_clip_scale,
        "grad_clip_fraction": grad_clip_fraction,
        "mean_parameter_update_norm": _mean(upd),
        "final_candidate_delta_m": _last(delta),
        "mean_group_std": _mean([_f(r, "group_std_mean") for r in d]),
        "mean_batch_global_std": _mean(
            [_f(r, "batch_global_std") for r in d]
        ),
        "mean_denominator": _mean([_f(r, "denominator") for r in d]),
        "mean_advantage_std": _mean(
            [_f(r, "advantage_std") for r in d]
        ),
        "mean_advantage_abs": _mean(
            [_f(r, "advantage_abs_mean") for r in d]
        ),
        "max_advantage_abs": _max(
            [_f(r, "advantage_max_abs") for r in d]
        ),
        "gain_per_refkl": (
            gain / max(abs(final_refkl), 1e-12)
            if math.isfinite(gain) and math.isfinite(final_refkl)
            else math.nan
        ),
        "probe_csv": str(probe_path),
        "diag_csv": str(diag_path),
    }


def fmt(x, scale=1.0, digits=4):
    try:
        x = float(x)
    except Exception:
        return "nan"
    if not math.isfinite(x):
        return "nan"
    return f"{x * scale:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--max-grad-norm", type=float, default=5.0)
    args = ap.parse_args()

    summaries = []
    for mode in MODES:
        try:
            summaries.append(
                summarize(mode, args.root, args.max_grad_norm)
            )
        except FileNotFoundError as exc:
            print(f"[WARN] missing {mode}: {exc}")

    if not summaries:
        raise SystemExit("No completed advantage-normalization runs found.")

    print()
    print("=" * 145)
    print("GRPO Advantage Normalization V1 — 15-step causal screen")
    print("=" * 145)
    header = (
        f"{'mode':<10} {'gain x1e6':>11} {'advStd':>9} {'denom':>10} "
        f"{'gradMean':>10} {'clipScale':>10} {'gradClip%':>10} "
        f"{'refKL':>8} {'maxKL':>8} {'ppoClip':>8} "
        f"{'delta(cm)':>10} {'gain/KL x1e6':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in summaries:
        print(
            f"{r['mode']:<10} "
            f"{fmt(r['final_gain'], 1e6, 3):>11} "
            f"{fmt(r['mean_advantage_std'], 1.0, 4):>9} "
            f"{fmt(r['mean_denominator'], 1.0, 6):>10} "
            f"{fmt(r['mean_raw_grad_norm'], 1.0, 2):>10} "
            f"{fmt(r['mean_grad_clip_scale'], 1.0, 4):>10} "
            f"{fmt(r['grad_clip_fraction'], 100.0, 1):>10} "
            f"{fmt(r['final_reference_kl'], 1.0, 3):>8} "
            f"{fmt(r['max_reference_kl'], 1.0, 3):>8} "
            f"{fmt(r['mean_policy_clip_fraction'], 1.0, 3):>8} "
            f"{fmt(r['final_candidate_delta_m'], 100.0, 3):>10} "
            f"{fmt(r['gain_per_refkl'], 1e6, 2):>14}"
        )

    out = args.root / "advantage_norm_summary.csv"
    keys = list(summaries[0].keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summaries)

    print()
    print("Interpretation:")
    print(
        "  centered greatly lowers advStd and raw grad -> "
        "group std normalization is a major gradient amplifier."
    )
    print(
        "  centered lowers advStd but raw grad stays huge -> "
        "diffusion log-prob Jacobian is the dominant source."
    )
    print(
        "  global keeps gain while lowering grad/KL -> "
        "pooled/global scaling is a promising compromise."
    )
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
