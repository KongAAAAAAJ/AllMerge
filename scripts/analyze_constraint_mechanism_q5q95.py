from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np

METHODS = ("vanilla", "lagrangian", "hard_worst", "soft_active")
CONSTRAINTS = ("collision", "road", "ttc", "background_gap", "teammate_gap")


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fval(row, *keys):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                pass
    return math.nan


def finite(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def mean(values):
    arr = finite(values)
    return float(arr.mean()) if arr.size else math.nan


def maxv(values):
    arr = finite(values)
    return float(arr.max()) if arr.size else math.nan


def parse_step_log(path):
    metrics = []
    kv = re.compile(r"([A-Za-z0-9_./-]+)=([-+0-9.eE]+)")
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("[step "):
            continue
        d = {}
        for key, value in kv.findall(line):
            try:
                d[key] = float(value)
            except ValueError:
                pass
        metrics.append(d)
    return metrics


def summarize(method, root, steps):
    d = root / method
    fixed_path = d / f"{method}_curved{steps}.fixed_validation.csv"
    log_path = d / "train.log"
    q_path = d / "q5q95_diag.csv"

    fixed = read_csv(fixed_path) if fixed_path.exists() else []
    train = parse_step_log(log_path)
    qrows = read_csv(q_path) if q_path.exists() else []
    last_fixed = fixed[-1] if fixed else {}

    def train_col(key):
        return [r.get(key, math.nan) for r in train]

    def q_col(key):
        return [fval(r, key) for r in qrows]

    row = {
        "method": method,
        "final_task_gain": fval(
            last_fixed,
            "fixed_validation/paired_reward_gain_mean",
            "paired_reward_gain_mean",
        ),
        "positive_fraction": fval(
            last_fixed,
            "fixed_validation/positive_fraction",
            "positive_fraction",
        ),
        "selected_task_gain": fval(
            last_fixed,
            "fixed_validation/selected_vehicle_reward_gain_mean",
            "selected_vehicle_reward_gain_mean",
        ),
        "worst_role_gain": fval(
            last_fixed,
            "fixed_validation/worst_vehicle_reward_gain_mean",
            "worst_vehicle_reward_gain_mean",
        ),
        "role_gain_spread": fval(
            last_fixed,
            "fixed_validation/vehicle_reward_gain_spread",
            "vehicle_reward_gain_spread",
        ),
        "feasible_fraction_current": fval(
            last_fixed,
            "fixed_validation/current_constraint_feasible_fraction_mean",
            "current_constraint_feasible_fraction_mean",
        ),
        "feasible_fraction_gain": fval(
            last_fixed,
            "fixed_validation/constraint_feasible_fraction_gain_mean",
            "constraint_feasible_fraction_gain_mean",
        ),
        "max_violation_current": fval(
            last_fixed,
            "fixed_validation/current_constraint_max_violation_mean_mean",
            "current_constraint_max_violation_mean_mean",
            "fixed_validation/current_constraint_max_violation_mean",
            "current_constraint_max_violation_mean",
        ),
        "max_violation_change": fval(
            last_fixed,
            "fixed_validation/constraint_max_violation_change_mean",
            "constraint_max_violation_change_mean",
        ),
        "mean_reference_kl": mean(train_col("reference_kl")),
        "max_reference_kl": maxv(train_col("reference_kl")),
        "mean_ppo_clip_fraction": mean(train_col("clip_fraction")),
        "mean_raw_grad_norm": mean(train_col("grad_norm")),
        "mean_parameter_update_norm": mean(train_col("parameter_update_norm")),
        "mean_advantage_std_train": mean(train_col("advantage_std")),
        "q5q95_postclip_adv_std": mean(q_col("postclip_adv_std")),
        "q5q95_changed_fraction": mean(q_col("clip_changed_fraction")),
    }

    for name in CONSTRAINTS:
        row[f"{name}_violation_change"] = fval(
            last_fixed,
            f"fixed_validation/constraint_{name}_violation_change_mean",
            f"constraint_{name}_violation_change_mean",
        )
    return row


def fmt(x, scale=1.0, digits=4):
    try:
        x = float(x)
    except Exception:
        return "nan"
    if not math.isfinite(x):
        return "nan"
    return f"{x*scale:.{digits}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--steps", type=int, choices=(10, 15), required=True)
    args = p.parse_args()

    summaries = [summarize(m, args.root, args.steps) for m in METHODS]

    out_csv = args.root / f"mechanism_ablation_g48_{args.steps}step_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    print()
    print("=" * 170)
    print(
        f"Q5/Q95 constrained GRPO mechanism ablation — G48 x {args.steps} steps"
    )
    print("=" * 170)
    header = (
        f"{'method':<13} {'taskGain':>10} {'posFrac':>8} "
        f"{'selGain':>10} {'worstRole':>10} "
        f"{'feasGain':>10} {'maxVchg':>10} "
        f"{'refKL':>8} {'maxKL':>8} {'ppoClip':>8} "
        f"{'grad':>9} {'advStd':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in summaries:
        print(
            f"{r['method']:<13} "
            f"{fmt(r['final_task_gain']):>10} "
            f"{fmt(r['positive_fraction'],1,3):>8} "
            f"{fmt(r['selected_task_gain']):>10} "
            f"{fmt(r['worst_role_gain']):>10} "
            f"{fmt(r['feasible_fraction_gain']):>10} "
            f"{fmt(r['max_violation_change']):>10} "
            f"{fmt(r['mean_reference_kl'],1,3):>8} "
            f"{fmt(r['max_reference_kl'],1,3):>8} "
            f"{fmt(r['mean_ppo_clip_fraction'],1,3):>8} "
            f"{fmt(r['mean_raw_grad_norm'],1,1):>9} "
            f"{fmt(r['q5q95_postclip_adv_std'],1,3):>8}"
        )

    print()
    print("Constraint violation change (negative = improvement):")
    ch = f"{'method':<13}" + "".join(
        f"{name:>16}" for name in CONSTRAINTS
    )
    print(ch)
    print("-" * len(ch))
    for r in summaries:
        print(
            f"{r['method']:<13}"
            + "".join(
                f"{fmt(r[f'{name}_violation_change']):>16}"
                for name in CONSTRAINTS
            )
        )

    print()
    print("Interpretation rule:")
    print("  Primary: retain/improve task reward while reducing constraint violation.")
    print("  Better feasibility gain is positive; better violation change is negative.")
    print("  Compare methods on the task-reward vs constraint-satisfaction Pareto front.")
    print("  KL/clip/grad are stability diagnostics, not the final objective.")
    print(f"[OK] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
