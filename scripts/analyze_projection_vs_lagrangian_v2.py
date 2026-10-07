from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if not (_REPO_ROOT / "highway_env").is_dir():
    raise SystemExit(f"[FAIL] repo root not found from helper script: {_REPO_ROOT}")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PAIR = re.compile(r"([A-Za-z0-9_./-]+)=([-+0-9.eE]+)")


def read_fixed(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty fixed-validation csv: {path}")
    return rows


def f(row, key, default=float("nan")):
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_log(path: Path):
    rows=[]
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[step "):
            rows.append({k:float(v) for k,v in PAIR.findall(line)})
    return rows


def finite_mean(values):
    xs=[x for x in values if math.isfinite(x)]
    return sum(xs)/len(xs) if xs else float("nan")


def summarize(name: str, fixed_path: Path, log_path: Path):
    fixed=read_fixed(fixed_path)
    final=fixed[-1]
    log=parse_log(log_path)
    both=[]
    for row in fixed:
        rg=f(row,"fixed_validation/paired_reward_gain_mean")
        cv=f(row,"fixed_validation/constraint_max_violation_change_mean")
        if rg>0 and cv<0:
            both.append(int(float(row.get("step",0))))
    out={
        "method":name,
        "final_step":int(float(final.get("step",0))),
        "final_task_gain":f(final,"fixed_validation/paired_reward_gain_mean"),
        "final_selected_gain":f(final,"fixed_validation/selected_vehicle_reward_gain_mean"),
        "final_worst_vehicle_gain":f(final,"fixed_validation/worst_vehicle_reward_gain_mean"),
        "final_max_violation_change":f(final,"fixed_validation/constraint_max_violation_change_mean"),
        "final_feasible_fraction_gain":f(final,"fixed_validation/constraint_feasible_fraction_gain_mean"),
        "collision_change":f(final,"fixed_validation/constraint_collision_violation_change_mean"),
        "road_change":f(final,"fixed_validation/constraint_road_violation_change_mean"),
        "ttc_change":f(final,"fixed_validation/constraint_ttc_violation_change_mean"),
        "background_gap_change":f(final,"fixed_validation/constraint_background_gap_violation_change_mean"),
        "teammate_gap_change":f(final,"fixed_validation/constraint_teammate_gap_violation_change_mean"),
        "both_task_and_max_violation_improved_steps":";".join(map(str,both)),
        "mean_reference_kl":finite_mean([r.get("reference_kl",float("nan")) for r in log]),
        "max_reference_kl":max([r.get("reference_kl",float("nan")) for r in log], default=float("nan")),
        "mean_grad_norm":finite_mean([r.get("grad_norm",float("nan")) for r in log]),
        "mean_parameter_update_norm":finite_mean([r.get("parameter_update_norm",float("nan")) for r in log]),
    }
    if name.lower().startswith("projection"):
        out.update({
            "projection_correction_step_fraction":finite_mean([
                r.get("projection/collect_any_correction",float("nan")) for r in log
            ]),
            "projection_mean_max_correction_ratio":finite_mean([
                r.get("projection/collect_max_correction_ratio",float("nan")) for r in log
            ]),
            "projection_min_margin_scale_used":min([
                r.get("projection/collect_min_margin_scale_used",float("nan")) for r in log
                if math.isfinite(r.get("projection/collect_min_margin_scale_used",float("nan")))
            ], default=float("nan")),
            "projection_zero_margin_fallback_fraction":finite_mean([
                r.get("projection/collect_any_zero_margin_fallback",float("nan")) for r in log
            ]),
        })
    return out


def fmt(x):
    if isinstance(x,float):
        return "nan" if not math.isfinite(x) else f"{x:.6g}"
    return str(x)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--projection-dir",type=Path,required=True)
    p.add_argument("--lagrangian-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    rows=[
        summarize(
            "Projection V2 Restorative",
            args.projection_dir/"projection_v2_curved30.fixed_validation.csv",
            args.projection_dir/"train.log",
        ),
        summarize(
            "Normalized Lagrangian V3",
            args.lagrangian_dir/"lagrangian_v3_curved30.fixed_validation.csv",
            args.lagrangian_dir/"train.log",
        ),
    ]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    csv_path=args.output_dir/"projection_vs_lagrangian_30step.csv"
    with csv_path.open("w",encoding="utf-8-sig",newline="") as fobj:
        w=csv.DictWriter(fobj,fieldnames=keys); w.writeheader(); w.writerows(rows)
    json_path=args.output_dir/"projection_vs_lagrangian_30step.json"
    json_path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")

    print("\nProjection V2 vs Normalized Lagrangian V3 (G48 x 30)")
    print("="*108)
    print(f"{'method':30s} {'task_gain':>12s} {'selected':>12s} {'worst':>12s} {'maxV_d':>12s} {'road_d':>12s} {'ttc_d':>12s}")
    for r in rows:
        print(
            f"{r['method'][:30]:30s} "
            f"{fmt(r['final_task_gain']):>12s} {fmt(r['final_selected_gain']):>12s} "
            f"{fmt(r['final_worst_vehicle_gain']):>12s} {fmt(r['final_max_violation_change']):>12s} "
            f"{fmt(r['road_change']):>12s} {fmt(r['ttc_change']):>12s}"
        )
    print("\nDecision guidance:")
    print("  Primary safety: lower max_violation_change; negative is improvement.")
    print("  Task cost: higher paired/selected/worst-vehicle reward gain is better.")
    print("  Feasibility: positive feasible_fraction_gain is preferred.")
    print("  Stability guard: max reference KL should remain <= 0.25.")
    print(f"[OK] {csv_path}")
    print(f"[OK] {json_path}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
