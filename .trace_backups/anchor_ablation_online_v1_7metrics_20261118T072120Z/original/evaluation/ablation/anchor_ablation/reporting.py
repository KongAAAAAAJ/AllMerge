from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


PAPER_METRIC_KEYS = (
    "anchor_minade_m",
    "safe_ratio_pct",
    "trajectory_diversity_m",
    "control_diversity_normalized",
    "planning_minade_m",
    "preprojection_feasibility_pct",
    "projection_correction_m",
)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def _clean(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: Sequence[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _clean(row.get(key)) for key in fields})


def _mean(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return float(np.mean(values)) if values else None


def summarize(anchor_rows: Sequence[dict], planning_rows: Sequence[dict],
              methods: Sequence[str] = ("A0", "A1", "A2")):
    """Summarize complete or staged results; unavailable-stage metrics remain None."""
    output = []
    for method in methods:
        anchors = [row for row in anchor_rows if row.get("method") == method]
        plans = [row for row in planning_rows if row.get("method") == method]
        corrections = [
            float(row["projection_correction_m"])
            for row in plans
            if bool(row.get("projection_success"))
            and row.get("projection_correction_m") is not None
            and math.isfinite(float(row["projection_correction_m"]))
        ]
        row = {
            "method": method,
            "anchor_minade_m": _mean(anchors, "anchor_minade_m"),
            "safe_ratio_pct": (100.0 * _mean(anchors, "safe_ratio")) if anchors else None,
            "trajectory_diversity_m": _mean(anchors, "trajectory_diversity_m"),
            "control_diversity_normalized": _mean(anchors, "control_diversity_normalized"),
            "planning_minade_m": _mean(plans, "planning_minade_m"),
            "preprojection_feasibility_pct": (
                100.0 * float(np.mean([bool(x["preprojection_feasible"]) for x in plans]))
                if plans else None
            ),
            # Average correction over successful JTP projections. The success
            # rate and failure count are retained separately to expose selection.
            "projection_correction_m": float(np.mean(corrections)) if corrections else None,
            "projection_success_rate_pct": (
                100.0 * float(np.mean([bool(x["projection_success"]) for x in plans]))
                if plans else None
            ),
            "num_anchor_scenes": len(anchors),
            "num_planning_runs": len(plans),
            "num_projection_success": len(corrections),
            "num_projection_failures": sum(not bool(x.get("projection_success")) for x in plans),
        }
        output.append(row)
    return output


def write_latex_table(path: Path, summary: Sequence[dict]):
    lookup = {row["method"]: row for row in summary}
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Adaptive Anchor Ablation Results}",
        r"\label{tab:anchor_ablation}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\hline",
        r" & \multicolumn{4}{c}{\textbf{Anchor Quality}} & \multicolumn{3}{c}{\textbf{Planning Performance}} \\",
        r"\cline{2-5} \cline{6-8}",
        r"\textbf{Method} & \shortstack{\textbf{minADE}\\(m)$\downarrow$} & \shortstack{\textbf{Safe Ratio}\\(\%)$\uparrow$} & \shortstack{\textbf{Trajectory Div.}\\(m)} & \shortstack{\textbf{Control Div.}\\($-$)} & \shortstack{\textbf{minADE}\\(m)$\downarrow$} & \shortstack{\textbf{Pre-proj. Feas.}\\(\%)$\uparrow$} & \shortstack{\textbf{Proj. Corr.}\\(m)$\downarrow$} \\",
        r"\hline",
    ]
    keys = PAPER_METRIC_KEYS
    for method, label in (("A0", "Fixed Anchors"), ("A1", "Dynamic Anchors"), ("A2", r"\textbf{Ours}")):
        row = lookup.get(method)
        if row is None:
            continue
        values = []
        for key in keys:
            value = row.get(key)
            if value is None or not math.isfinite(float(value)):
                values.append("--")
            else:
                values.append(f"{float(value):.3f}")
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
                    encoding="utf-8")
