#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BASELINE_CONVERGENCE_ANALYZER_500STEP_V1_20261007
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

STEP_RE = re.compile(r"^\[step\s+(\d+)/(\d+)\]\s+(.*)$")
KV_RE = re.compile(r"([A-Za-z0-9_./-]+)=([^\s]+)")

CONSTRAINTS = (
    "collision",
    "road",
    "ttc",
    "background_gap",
    "teammate_gap",
)

FIXED_KEYS = {
    "task": "fixed_validation/paired_reward_gain_mean",
    "selected": "fixed_validation/selected_vehicle_reward_gain_mean",
    "worst": "fixed_validation/worst_vehicle_reward_gain_mean",
    "maxv": "fixed_validation/constraint_max_violation_change_mean",
    "feasible": "fixed_validation/constraint_feasible_fraction_gain_mean",
    "road": "fixed_validation/constraint_road_violation_change_mean",
    "ttc": "fixed_validation/constraint_ttc_violation_change_mean",
    "collision": "fixed_validation/constraint_collision_violation_change_mean",
    "background_gap": "fixed_validation/constraint_background_gap_violation_change_mean",
    "teammate_gap": "fixed_validation/constraint_teammate_gap_violation_change_mean",
    "candidate_delta": "fixed_validation/candidate_delta_m_mean",
}


def finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def fnum(value, default=float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def parse_train_log(path: Path) -> List[Dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP_RE.match(line)
        if not m:
            continue
        row: Dict[str, float] = {
            "step": float(m.group(1)),
            "total_steps": float(m.group(2)),
        }
        for key, raw in KV_RE.findall(m.group(3)):
            value = fnum(raw)
            if finite(value):
                row[key] = value
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no [step ...] rows parsed from {path}")
    return rows


def read_fixed_csv(path: Path) -> List[Dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "step" not in (reader.fieldnames or []):
            raise RuntimeError(f"fixed-validation CSV missing step column: {path}")
        for raw in reader:
            row: Dict[str, float] = {}
            for key, value in raw.items():
                v = fnum(value)
                if finite(v):
                    row[key] = v
            if "step" in row:
                rows.append(row)
    rows.sort(key=lambda r: r["step"])
    if not rows:
        raise RuntimeError(f"no rows parsed from {path}")
    return rows


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if finite(v)]
    return statistics.fmean(vals) if vals else float("nan")


def std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if finite(v)]
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return 0.0
    return statistics.pstdev(vals)


def min_finite(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if finite(v)]
    return min(vals) if vals else float("nan")


def max_finite(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if finite(v)]
    return max(vals) if vals else float("nan")


def slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if finite(x) and finite(y)
    ]
    if len(pairs) < 2:
        return float("nan")
    mx = mean(x for x, _ in pairs)
    my = mean(y for _, y in pairs)
    den = sum((x - mx) ** 2 for x, _ in pairs)
    if den <= 1e-20:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in pairs) / den


def tail_rows(rows: List[Dict[str, float]], tail_steps: int) -> List[Dict[str, float]]:
    max_step = max(r["step"] for r in rows)
    start = max_step - float(tail_steps)
    return [r for r in rows if r["step"] >= start - 1e-9]


def metric_stats(
    rows: List[Dict[str, float]],
    key: str,
    tail_steps: int,
) -> Dict[str, float]:
    tail = tail_rows(rows, tail_steps)
    xs = [r["step"] for r in tail if key in r and finite(r[key])]
    ys = [r[key] for r in tail if key in r and finite(r[key])]
    s = slope(xs, ys)
    span = (max(xs) - min(xs)) if len(xs) >= 2 else 0.0
    return {
        "count": float(len(ys)),
        "mean": mean(ys),
        "std": std(ys),
        "min": min_finite(ys),
        "max": max_finite(ys),
        "slope_per_step": s,
        "slope_drift": abs(s) * span if finite(s) else float("nan"),
        "span_steps": span,
    }


def value_at_final(rows: List[Dict[str, float]], key: str) -> float:
    for row in reversed(rows):
        if key in row and finite(row[key]):
            return float(row[key])
    return float("nan")


def best_step(rows: List[Dict[str, float]], key: str, maximize: bool) -> Tuple[float, float]:
    candidates = [
        (r["step"], r[key])
        for r in rows
        if key in r and finite(r[key])
    ]
    if not candidates:
        return float("nan"), float("nan")
    return (max(candidates, key=lambda p: p[1]) if maximize
            else min(candidates, key=lambda p: p[1]))


def joint_improvement_steps(rows: List[Dict[str, float]]) -> List[int]:
    out = []
    tkey = FIXED_KEYS["task"]
    ckey = FIXED_KEYS["maxv"]
    for r in rows:
        if (
            finite(r.get(tkey, float("nan")))
            and finite(r.get(ckey, float("nan")))
            and r[tkey] > 0.0
            and r[ckey] < 0.0
        ):
            out.append(int(round(r["step"])))
    return out


def summarize_method(
    name: str,
    kind: str,
    run_dir: Path,
    fixed_name: str,
    *,
    expected_steps: int,
    tail_steps: int,
    task_drift_tol: float,
    task_std_tol: float,
    maxv_drift_tol: float,
    maxv_std_tol: float,
    kl_drift_tol: float,
    kl_std_tol: float,
    kl_max: float,
    lambda_drift_tol: float,
    lambda_std_tol: float,
    projection_fallback_max: float,
    projection_after_tol: float,
) -> Dict[str, object]:
    train = parse_train_log(run_dir / "train.log")
    fixed = read_fixed_csv(run_dir / fixed_name)

    train_max_step = int(round(max(r["step"] for r in train)))
    fixed_max_step = int(round(max(r["step"] for r in fixed)))

    task_stats = metric_stats(fixed, FIXED_KEYS["task"], tail_steps)
    selected_stats = metric_stats(fixed, FIXED_KEYS["selected"], tail_steps)
    worst_stats = metric_stats(fixed, FIXED_KEYS["worst"], tail_steps)
    maxv_stats = metric_stats(fixed, FIXED_KEYS["maxv"], tail_steps)
    feasible_stats = metric_stats(fixed, FIXED_KEYS["feasible"], tail_steps)
    kl_stats = metric_stats(train, "reference_kl", tail_steps)
    update_stats = metric_stats(train, "parameter_update_norm", tail_steps)
    grad_stats = metric_stats(train, "grad_norm", tail_steps)

    hard_reasons: List[str] = []
    plateau_reasons: List[str] = []
    mechanism_reasons: List[str] = []

    if train_max_step != expected_steps:
        hard_reasons.append(
            f"train log ends at step {train_max_step}, expected {expected_steps}"
        )
    if fixed_max_step != expected_steps:
        hard_reasons.append(
            f"fixed validation ends at step {fixed_max_step}, expected {expected_steps}"
        )

    ref_kls = [r.get("reference_kl", float("nan")) for r in train]
    max_ref_kl = max_finite(ref_kls)
    if not finite(max_ref_kl) or max_ref_kl > kl_max:
        hard_reasons.append(
            f"max reference KL {max_ref_kl:.6g} exceeds {kl_max:.6g}"
        )

    for label, rows, key in (
        ("task gain", fixed, FIXED_KEYS["task"]),
        ("max violation change", fixed, FIXED_KEYS["maxv"]),
        ("reference KL", train, "reference_kl"),
        ("parameter update norm", train, "parameter_update_norm"),
    ):
        vals = [r.get(key, float("nan")) for r in rows]
        if not vals or any(not finite(v) for v in vals):
            hard_reasons.append(f"{label} contains missing/non-finite values")

    # Prescribed interval=10 + tail_steps=100 => 400,410,...,500.
    if task_stats["count"] < 11 or maxv_stats["count"] < 11:
        plateau_reasons.append(
            "insufficient fixed-validation tail points: "
            f"task={int(task_stats['count'])}, maxV={int(maxv_stats['count'])}; expected >=11"
        )
    if kl_stats["count"] < 80:
        plateau_reasons.append(
            "insufficient train-step tail points for KL trend: "
            f"{int(kl_stats['count'])}; expected >=80"
        )

    if (
        not finite(task_stats["slope_drift"])
        or task_stats["slope_drift"] > task_drift_tol
        or task_stats["std"] > task_std_tol
    ):
        plateau_reasons.append(
            "task tail not flat: "
            f"drift={task_stats['slope_drift']:.3g}, std={task_stats['std']:.3g}"
        )
    if (
        not finite(maxv_stats["slope_drift"])
        or maxv_stats["slope_drift"] > maxv_drift_tol
        or maxv_stats["std"] > maxv_std_tol
    ):
        plateau_reasons.append(
            "max-violation tail not flat: "
            f"drift={maxv_stats['slope_drift']:.3g}, std={maxv_stats['std']:.3g}"
        )
    if (
        not finite(kl_stats["slope_drift"])
        or kl_stats["slope_drift"] > kl_drift_tol
        or kl_stats["std"] > kl_std_tol
    ):
        plateau_reasons.append(
            "reference-KL tail not flat: "
            f"drift={kl_stats['slope_drift']:.3g}, std={kl_stats['std']:.3g}"
        )

    mechanism: Dict[str, float] = {}
    if kind == "lagrangian":
        lm = metric_stats(train, "lagrangian_v3/lambda_mass_before", tail_steps)
        mechanism.update({
            "lambda_mass_tail_mean": lm["mean"],
            "lambda_mass_tail_std": lm["std"],
            "lambda_mass_tail_drift": lm["slope_drift"],
            "lambda_mass_final": value_at_final(
                train, "lagrangian_v3/lambda_mass_before"
            ),
        })
        if (
            not finite(lm["slope_drift"])
            or lm["slope_drift"] > lambda_drift_tol
            or lm["std"] > lambda_std_tol
        ):
            mechanism_reasons.append(
                "lambda mass not stable: "
                f"drift={lm['slope_drift']:.3g}, std={lm['std']:.3g}"
            )
        for cname in CONSTRAINTS:
            mechanism[f"lambda_{cname}_final"] = value_at_final(
                train, f"lagrangian/lambda_{cname}"
            )
            cs = metric_stats(train, f"lagrangian/lambda_{cname}", tail_steps)
            mechanism[f"lambda_{cname}_tail_drift"] = cs["slope_drift"]

    elif kind == "projection":
        tail = tail_rows(train, tail_steps)
        fallback = mean(
            r.get("projection/collect_any_zero_margin_fallback", float("nan"))
            for r in tail
        )
        correction = mean(
            r.get("projection/collect_any_correction", float("nan"))
            for r in tail
        )
        corr_ratio = metric_stats(
            train, "projection/collect_max_correction_ratio", tail_steps
        )
        min_margin_after = min_finite(
            r.get("projection/collect_min_margin_residual_after", float("nan"))
            for r in tail
        )
        max_violated_after = max_finite(
            r.get("projection/collect_max_violated_after_count", float("nan"))
            for r in tail
        )
        max_projected_norm = max_finite(
            r.get("projection/projected_grad_norm", float("nan"))
            for r in tail
        )
        mechanism.update({
            "projection_tail_correction_fraction": correction,
            "projection_tail_correction_ratio_mean": corr_ratio["mean"],
            "projection_tail_correction_ratio_std": corr_ratio["std"],
            "projection_tail_zero_margin_fallback_fraction": fallback,
            "projection_tail_min_margin_residual_after": min_margin_after,
            "projection_tail_max_violated_after_count": max_violated_after,
            "projection_tail_max_projected_grad_norm": max_projected_norm,
        })
        if not finite(fallback) or fallback > projection_fallback_max:
            mechanism_reasons.append(
                f"zero-margin fallback fraction {fallback:.3g} "
                f"> {projection_fallback_max:.3g}"
            )
        if (
            not finite(max_violated_after)
            or max_violated_after > projection_after_tol
        ):
            mechanism_reasons.append(
                "projected gradient violates constraints after projection: "
                f"max count={max_violated_after:.3g}"
            )
        if not finite(min_margin_after) or min_margin_after < -1e-5:
            mechanism_reasons.append(
                "negative post-projection margin residual: "
                f"{min_margin_after:.3g}"
            )
        if finite(max_projected_norm) and max_projected_norm > 5.0005:
            mechanism_reasons.append(
                f"projected grad norm {max_projected_norm:.6g} > 5"
            )
    else:
        raise ValueError(kind)

    hard_stable = not hard_reasons
    plateau = not plateau_reasons
    mechanism_stable = not mechanism_reasons
    ready = hard_stable and plateau and mechanism_stable

    best_task_step, best_task_value = best_step(
        fixed, FIXED_KEYS["task"], maximize=True
    )
    best_safe_step, best_safe_value = best_step(
        fixed, FIXED_KEYS["maxv"], maximize=False
    )
    joint = joint_improvement_steps(fixed)

    return {
        "method": name,
        "kind": kind,
        "run_dir": str(run_dir),
        "expected_steps": expected_steps,
        "train_final_step": train_max_step,
        "fixed_final_step": fixed_max_step,
        "hard_stable": hard_stable,
        "plateau": plateau,
        "mechanism_stable": mechanism_stable,
        "baseline_ready": ready,
        "hard_reasons": hard_reasons,
        "plateau_reasons": plateau_reasons,
        "mechanism_reasons": mechanism_reasons,
        "final_task_gain": value_at_final(fixed, FIXED_KEYS["task"]),
        "final_selected_gain": value_at_final(fixed, FIXED_KEYS["selected"]),
        "final_worst_vehicle_gain": value_at_final(fixed, FIXED_KEYS["worst"]),
        "final_max_violation_change": value_at_final(fixed, FIXED_KEYS["maxv"]),
        "final_feasible_fraction_gain": value_at_final(fixed, FIXED_KEYS["feasible"]),
        "final_road_change": value_at_final(fixed, FIXED_KEYS["road"]),
        "final_ttc_change": value_at_final(fixed, FIXED_KEYS["ttc"]),
        "final_collision_change": value_at_final(fixed, FIXED_KEYS["collision"]),
        "final_background_gap_change": value_at_final(fixed, FIXED_KEYS["background_gap"]),
        "final_teammate_gap_change": value_at_final(fixed, FIXED_KEYS["teammate_gap"]),
        "final_candidate_delta_m": value_at_final(fixed, FIXED_KEYS["candidate_delta"]),
        "max_reference_kl": max_ref_kl,
        "tail_reference_kl_mean": kl_stats["mean"],
        "tail_reference_kl_std": kl_stats["std"],
        "tail_reference_kl_drift": kl_stats["slope_drift"],
        "tail_task_mean": task_stats["mean"],
        "tail_task_std": task_stats["std"],
        "tail_task_drift": task_stats["slope_drift"],
        "tail_selected_mean": selected_stats["mean"],
        "tail_selected_drift": selected_stats["slope_drift"],
        "tail_worst_mean": worst_stats["mean"],
        "tail_worst_drift": worst_stats["slope_drift"],
        "tail_maxv_mean": maxv_stats["mean"],
        "tail_maxv_std": maxv_stats["std"],
        "tail_maxv_drift": maxv_stats["slope_drift"],
        "tail_feasible_mean": feasible_stats["mean"],
        "tail_update_norm_mean": update_stats["mean"],
        "tail_update_norm_std": update_stats["std"],
        "tail_grad_norm_mean": grad_stats["mean"],
        "best_task_step": int(best_task_step) if finite(best_task_step) else None,
        "best_task_gain": best_task_value,
        "best_safety_step": int(best_safe_step) if finite(best_safe_step) else None,
        "best_max_violation_change": best_safe_value,
        "joint_task_safety_improvement_steps": joint,
        **mechanism,
    }


def flatten_for_csv(row: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for k, v in row.items():
        if isinstance(v, list):
            out[k] = ";".join(str(x) for x in v)
        else:
            out[k] = v
    return out


def fmt(v) -> str:
    if isinstance(v, bool):
        return "YES" if v else "NO"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            return "nan"
        return f"{v:.6g}"
    return str(v)


def write_timeseries(
    output_path: Path,
    method: str,
    fixed: List[Dict[str, float]],
    append: bool,
) -> None:
    keys = [
        "step",
        FIXED_KEYS["task"],
        FIXED_KEYS["selected"],
        FIXED_KEYS["worst"],
        FIXED_KEYS["maxv"],
        FIXED_KEYS["feasible"],
        FIXED_KEYS["collision"],
        FIXED_KEYS["road"],
        FIXED_KEYS["ttc"],
        FIXED_KEYS["background_gap"],
        FIXED_KEYS["teammate_gap"],
        FIXED_KEYS["candidate_delta"],
    ]
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", *keys])
        if not append:
            writer.writeheader()
        for r in fixed:
            writer.writerow({
                "method": method,
                **{key: r.get(key, "") for key in keys},
            })


def main() -> int:
    p = argparse.ArgumentParser(
        description="Analyze G48x500 convergence of frozen Lagrangian V3 and Projection V2 baselines."
    )
    p.add_argument("--lagrangian-dir", type=Path, required=True)
    p.add_argument("--projection-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-steps", type=int, default=500)
    p.add_argument("--tail-steps", type=int, default=100)
    p.add_argument("--task-drift-tol", type=float, default=5e-6)
    p.add_argument("--task-std-tol", type=float, default=7.5e-6)
    p.add_argument("--maxv-drift-tol", type=float, default=2.5e-4)
    p.add_argument("--maxv-std-tol", type=float, default=4e-4)
    p.add_argument("--kl-drift-tol", type=float, default=0.05)
    p.add_argument("--kl-std-tol", type=float, default=0.05)
    p.add_argument("--kl-max", type=float, default=0.25)
    p.add_argument("--lambda-drift-tol", type=float, default=0.10)
    p.add_argument("--lambda-std-tol", type=float, default=0.08)
    p.add_argument("--projection-fallback-max", type=float, default=0.05)
    p.add_argument("--projection-after-tol", type=float, default=0.0)
    args = p.parse_args()

    lag_fixed = "lagrangian_v3_curved500.fixed_validation.csv"
    proj_fixed = "projection_v2_curved500.fixed_validation.csv"

    common = dict(
        expected_steps=args.expected_steps,
        tail_steps=args.tail_steps,
        task_drift_tol=args.task_drift_tol,
        task_std_tol=args.task_std_tol,
        maxv_drift_tol=args.maxv_drift_tol,
        maxv_std_tol=args.maxv_std_tol,
        kl_drift_tol=args.kl_drift_tol,
        kl_std_tol=args.kl_std_tol,
        kl_max=args.kl_max,
        lambda_drift_tol=args.lambda_drift_tol,
        lambda_std_tol=args.lambda_std_tol,
        projection_fallback_max=args.projection_fallback_max,
        projection_after_tol=args.projection_after_tol,
    )

    lag = summarize_method(
        "Normalized Lagrangian V3",
        "lagrangian",
        args.lagrangian_dir,
        lag_fixed,
        **common,
    )
    proj = summarize_method(
        "Projection V2 Restorative",
        "projection",
        args.projection_dir,
        proj_fixed,
        **common,
    )
    rows = [lag, proj]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.output_dir / "baseline_convergence_500step_summary.csv"
    csv_rows = [flatten_for_csv(r) for r in rows]
    fieldnames: List[str] = []
    for row in csv_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    summary_json = args.output_dir / "baseline_convergence_500step.json"
    summary_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ts_path = args.output_dir / "baseline_convergence_500step_timeseries.csv"
    lag_fixed_rows = read_fixed_csv(args.lagrangian_dir / lag_fixed)
    proj_fixed_rows = read_fixed_csv(args.projection_dir / proj_fixed)
    write_timeseries(ts_path, "Normalized Lagrangian V3", lag_fixed_rows, False)
    write_timeseries(ts_path, "Projection V2 Restorative", proj_fixed_rows, True)

    report = args.output_dir / "baseline_convergence_500step_report.txt"
    lines: List[str] = []
    lines.append("Lagrangian V3 vs Projection V2 — G48 x 500 convergence analysis")
    lines.append("=" * 88)
    lines.append(
        f"Tail window: last {args.tail_steps} steps | "
        f"KL hard guard <= {args.kl_max}"
    )
    lines.append(
        "Plateau thresholds: "
        f"task drift <= {args.task_drift_tol:g}, task std <= {args.task_std_tol:g}; "
        f"maxV drift <= {args.maxv_drift_tol:g}, maxV std <= {args.maxv_std_tol:g}; "
        f"KL drift/std <= {args.kl_drift_tol:g}/{args.kl_std_tol:g}"
    )
    lines.append("")
    header = (
        f"{'method':28s} {'ready':>7s} {'stable':>7s} {'plateau':>8s} "
        f"{'task':>12s} {'selected':>12s} {'maxV_d':>12s} "
        f"{'maxKL':>9s} {'taskDr':>10s} {'maxVDr':>10s}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"{str(r['method'])[:28]:28s} "
            f"{fmt(r['baseline_ready']):>7s} "
            f"{fmt(r['hard_stable']):>7s} "
            f"{fmt(r['plateau']):>8s} "
            f"{fmt(r['final_task_gain']):>12s} "
            f"{fmt(r['final_selected_gain']):>12s} "
            f"{fmt(r['final_max_violation_change']):>12s} "
            f"{fmt(r['max_reference_kl']):>9s} "
            f"{fmt(r['tail_task_drift']):>10s} "
            f"{fmt(r['tail_maxv_drift']):>10s}"
        )
    lines.append("")
    for r in rows:
        lines.append(str(r["method"]))
        lines.append("-" * len(str(r["method"])))
        lines.append(
            f"status: baseline_ready={r['baseline_ready']}, "
            f"hard_stable={r['hard_stable']}, plateau={r['plateau']}, "
            f"mechanism_stable={r['mechanism_stable']}"
        )
        lines.append(
            f"final: task={fmt(r['final_task_gain'])}, "
            f"selected={fmt(r['final_selected_gain'])}, "
            f"worst={fmt(r['final_worst_vehicle_gain'])}, "
            f"maxV={fmt(r['final_max_violation_change'])}, "
            f"road={fmt(r['final_road_change'])}, "
            f"ttc={fmt(r['final_ttc_change'])}, "
            f"feasible_gain={fmt(r['final_feasible_fraction_gain'])}"
        )
        lines.append(
            f"tail: task mean/std/drift="
            f"{fmt(r['tail_task_mean'])}/{fmt(r['tail_task_std'])}/{fmt(r['tail_task_drift'])}; "
            f"maxV mean/std/drift="
            f"{fmt(r['tail_maxv_mean'])}/{fmt(r['tail_maxv_std'])}/{fmt(r['tail_maxv_drift'])}; "
            f"KL mean/std/drift="
            f"{fmt(r['tail_reference_kl_mean'])}/{fmt(r['tail_reference_kl_std'])}/{fmt(r['tail_reference_kl_drift'])}"
        )
        lines.append(
            f"best task: step {r['best_task_step']} -> {fmt(r['best_task_gain'])}; "
            f"best safety: step {r['best_safety_step']} -> {fmt(r['best_max_violation_change'])}"
        )
        lines.append(
            "joint task+safety improvement steps: "
            + (
                ",".join(map(str, r["joint_task_safety_improvement_steps"]))
                if r["joint_task_safety_improvement_steps"]
                else "none"
            )
        )
        if r["kind"] == "lagrangian":
            lines.append(
                "lagrangian mechanism: "
                f"lambda_mass final={fmt(r.get('lambda_mass_final'))}, "
                f"tail mean/std/drift="
                f"{fmt(r.get('lambda_mass_tail_mean'))}/"
                f"{fmt(r.get('lambda_mass_tail_std'))}/"
                f"{fmt(r.get('lambda_mass_tail_drift'))}"
            )
        else:
            lines.append(
                "projection mechanism: "
                f"correction_fraction={fmt(r.get('projection_tail_correction_fraction'))}, "
                f"correction_ratio mean/std="
                f"{fmt(r.get('projection_tail_correction_ratio_mean'))}/"
                f"{fmt(r.get('projection_tail_correction_ratio_std'))}, "
                f"fallback_fraction={fmt(r.get('projection_tail_zero_margin_fallback_fraction'))}, "
                f"max_violated_after={fmt(r.get('projection_tail_max_violated_after_count'))}, "
                f"min_margin_after={fmt(r.get('projection_tail_min_margin_residual_after'))}"
            )
        reasons = (
            list(r["hard_reasons"])
            + list(r["plateau_reasons"])
            + list(r["mechanism_reasons"])
        )
        if reasons:
            lines.append("not-ready reasons:")
            for reason in reasons:
                lines.append(f"  - {reason}")
        else:
            lines.append("not-ready reasons: none")
        lines.append("")

    both_ready = all(bool(r["baseline_ready"]) for r in rows)
    lines.append(
        "OVERALL: "
        + (
            "BOTH BASELINES READY TO FREEZE"
            if both_ready
            else "AT LEAST ONE BASELINE HAS NOT YET MET THE PRE-DECLARED CONVERGENCE GATE"
        )
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"[OK] {summary_csv}")
    print(f"[OK] {summary_json}")
    print(f"[OK] {ts_path}")
    print(f"[OK] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
