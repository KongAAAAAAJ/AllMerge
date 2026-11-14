from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import numpy as np


def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8-sig")
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in keys:
                value = row.get(key)
                if isinstance(value, (np.generic,)):
                    value = value.item()
                if isinstance(value, (list, tuple, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                clean[key] = value
            writer.writerow(clean)


def summarize_anchor_rows(rows: List[dict], feasibility_rows: List[dict]):
    summary = []
    for method in ("A0", "A1", "A2"):
        values = [r for r in rows if r["method"] == method]
        if not values:
            continue
        safe_num = sum(int(r["safe_anchor_count"]) for r in values)
        valid_num = sum(int(r["valid_anchor_count"]) for r in values)
        diversity = [float(r["diversity_m"]) for r in values if np.isfinite(float(r["diversity_m"]))]
        joint_diversity = [float(r["joint_diversity_normalized"]) for r in values
                           if np.isfinite(float(r["joint_diversity_normalized"]))]
        method_feas = [r for r in feasibility_rows if r["method"] == method]
        feas_rate = (float(np.mean([bool(r["feasible"]) for r in method_feas]))
                     if method_feas else None)
        summary.append({
            "method": method,
            "minADE_m": float(np.mean([r["minade_m"] for r in values])),
            "safe_anchor_ratio_pct": 100.0 * safe_num / max(valid_num, 1),
            "diversity_m": float(np.mean(diversity)) if diversity else float("nan"),
            "feasibility_pct": None if feas_rate is None else 100.0 * feas_rate,
            "num_scenes": len(values),
            "valid_anchor_count": valid_num,
            "safe_anchor_count": safe_num,
            "joint_diversity_normalized": float(np.mean(joint_diversity)) if joint_diversity else float("nan"),
        })
    return summary


def write_latex_table(path: Path, summary: List[dict]):
    lookup = {r["method"]: r for r in summary}
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Adaptive Anchor Ablation Results}",
        r"\label{tab:anchor_ablation}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\hline",
        r"\textbf{Method} & \shortstack{\textbf{minADE}\\\textbf{(m)}} & \shortstack{\textbf{Safe Ratio}\\\textbf{(\%)}} & \shortstack{\textbf{Diversity}\\\textbf{(m)}} & \shortstack{\textbf{Feasibility}\\\textbf{(\%)}} \\",
        r"\hline",
    ]
    for method, label in (("A0", "Fixed Anchors"), ("A1", "Dynamic Anchors"), ("A2", r"\textbf{Ours}")):
        row = lookup.get(method)
        if row is None:
            vals = ("--", "--", "--", "--")
        else:
            feas = "--" if row["feasibility_pct"] is None else f'{row["feasibility_pct"]:.2f}'
            vals = (f'{row["minADE_m"]:.3f}', f'{row["safe_anchor_ratio_pct"]:.2f}',
                    f'{row["diversity_m"]:.3f}', feas)
        lines.append(f"{label} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} " + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")
