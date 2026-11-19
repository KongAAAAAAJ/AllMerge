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
    "collision_rate_pct",
    "drivable_area_compliance_pct",
    "ttc_compliance_pct",
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


def _rate_pct(rows, key):
    values = [row[key] for row in rows if key in row and row[key] is not None]
    return 100.0 * float(np.mean([bool(value) for value in values])) if values else None


def summarize(anchor_rows: Sequence[dict], planning_rows: Sequence[dict],
              methods: Sequence[str] = ("A0", "A1", "A2")):
    """Summarize model metrics and the optional raw expert trajectory reference."""
    output = []
    for method in methods:
        anchors = [row for row in anchor_rows if row.get("method") == method]
        plans = [row for row in planning_rows if row.get("method") == method]
        row = {
            "method": method,
            "anchor_minade_m": _mean(anchors, "anchor_minade_m"),
            "safe_ratio_pct": (100.0 * _mean(anchors, "safe_ratio")) if anchors else None,
            "trajectory_diversity_m": _mean(anchors, "trajectory_diversity_m"),
            "control_diversity_normalized": _mean(anchors, "control_diversity_normalized"),
            "planning_minade_m": _mean(plans, "planning_minade_m"),
            "collision_rate_pct": _rate_pct(plans, "collision"),
            "drivable_area_compliance_pct": _mean(plans, "drivable_area_compliance_pct"),
            "ttc_compliance_pct": _rate_pct(plans, "ttc_compliant"),
            # Diagnostics are retained in CSV/JSON, but are not paper-table columns.
            "mean_min_ttc_s": _mean(plans, "min_ttc_s"),
            "mean_external_min_ttc_s": _mean(plans, "external_min_ttc_s"),
            "num_anchor_scenes": len(anchors),
            "num_planning_runs": len(plans),
            "num_collision_runs": sum(bool(x.get("collision")) for x in plans),
            "num_ttc_noncompliant_runs": sum(
                not bool(x.get("ttc_compliant")) for x in plans if "ttc_compliant" in x
            ),
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
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\hline",
        r" & \multicolumn{4}{c}{\textbf{Anchor Quality}} & \multicolumn{4}{c}{\textbf{Planning Performance}} \\",
        r"\cline{2-5} \cline{6-9}",
        r"\textbf{Method} & \shortstack{\textbf{minADE}\\(m)$\downarrow$} & "
        r"\shortstack{\textbf{Safe Ratio}\\(\%)$\uparrow$} & "
        r"\shortstack{\textbf{Trajectory Div.}\\(m)$\uparrow$} & "
        r"\shortstack{\textbf{Control Div.}\\($-$)$\uparrow$} & "
        r"\shortstack{\textbf{minADE}\\(m)$\downarrow$} & "
        r"\shortstack{\textbf{CR}\\(\%)$\downarrow$} & "
        r"\shortstack{\textbf{DAC}\\(\%)$\uparrow$} & "
        r"\shortstack{\textbf{TTC}\\(\%)$\uparrow$} \\",
        r"\hline",
    ]
    keys = PAPER_METRIC_KEYS
    for method, label in (("A0", "Fixed Anchors"), ("A1", "Dynamic Anchors"),
                          ("A2", r"\textbf{Ours}"), ("Expert", "Expert trajectory")):
        row = lookup.get(method)
        if row is None:
            continue
        values = []
        for key in keys:
            if method == 'Expert' and key in PAPER_METRIC_KEYS[:4]:
                values.append('-')
                continue
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
