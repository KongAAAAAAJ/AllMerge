"""Paired state-level comparison of existing Open-Loop evaluator CSV outputs.

Usage:
 python scripts/compare_ar_diffusion_open_loop.py --diffusion-csv ... --ar-csv ... --output-csv ...
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path

KEY = ("shard_path", "shard_sample_index")
METRICS = (
    "selected_ADE", "selected_FDE", "minADE_at_M_all", "minFDE_at_M_all",
    "execution_sparse_ADE", "execution_sparse_FDE", "execution_dense_ADE",
    "execution_dense_FDE", "base_dense_collision_proxy", "base_dense_offroad_proxy",
    "execution_collision_proxy", "execution_offroad_proxy",
)


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"Empty evaluation CSV: {path}")
    return rows


def as_map(rows):
    # One paired UNIT is an individual recorded ego/sample, not a G48 candidate.
    keys = KEY + ("ego_index",)
    if not all(k in rows[0] for k in keys):
        raise ValueError(f"Evaluation CSV missing sample keys: {keys}")
    mapped = {}
    for r in rows:
        key = tuple(r[k] for k in keys)
        if key in mapped:
            raise ValueError(f"Duplicate sample key in evaluation CSV: {key}")
        mapped[key] = r
    return mapped


def compare(a, b):
    d = as_map(a)
    e = as_map(b)
    if d.keys() != e.keys():
        missing_in_ar = len(d.keys() - e.keys())
        missing_in_diff = len(e.keys() - d.keys())
        raise RuntimeError(
            f"Not a paired evaluation: missing_in_ar={missing_in_ar}; "
            f"missing_in_diffusion={missing_in_diff}"
        )
    result = []
    for metric in METRICS:
        vals = []
        for key in sorted(d):
            if metric not in d[key] or metric not in e[key]:
                continue
            x = float(d[key][metric]); y = float(e[key][metric])
            if math.isfinite(x) and math.isfinite(y):
                vals.append((x, y))
        if not vals:
            continue
        n = len(vals)
        result.append({
            "metric": metric, "paired_states": n,
            "diffusion_mean": sum(x for x, _ in vals) / n,
            "ar_v1_mean": sum(y for _, y in vals) / n,
            "ar_minus_diffusion": sum(y - x for x, y in vals) / n,
        })
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diffusion-csv", required=True)
    p.add_argument("--ar-csv", required=True)
    p.add_argument("--output-csv", required=True)
    a = p.parse_args()
    rows = compare(read(a.diffusion_csv), read(a.ar_csv))
    if not rows:
        raise RuntimeError("No shared numeric evaluation metrics")
    out = Path(a.output_csv)
    out.parent.mkdir(exist_ok=True, parents=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"[PASS] {len(rows)} paired metrics -> {out}")
    for r in rows:
        print(f" {r['metric']}: diffusion={r['diffusion_mean']:.5f} "
              f"AR={r['ar_v1_mean']:.5f} delta={r['ar_minus_diffusion']:+.5f}")


if __name__ == "__main__":
    main()
