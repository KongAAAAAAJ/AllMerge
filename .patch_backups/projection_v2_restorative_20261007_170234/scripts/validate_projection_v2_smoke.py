from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

PAIR = re.compile(r"([A-Za-z0-9_./-]+)=([-+0-9.eE]+)")


def parse_steps(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("[step "):
            continue
        row = {key: float(value) for key, value in PAIR.findall(line)}
        rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--pass-marker", type=Path, required=True)
    args = p.parse_args()

    rows = parse_steps(args.log)
    if len(rows) < 3:
        raise SystemExit(f"[FAIL] expected >=3 training-step rows, got {len(rows)}")

    required = (
        "projection/collect_any_correction",
        "projection/collect_max_correction_ratio",
        "projection/collect_max_violated_before_count",
        "projection/collect_max_violated_after_count",
        "projection/collect_min_margin_scale_used",
        "projection/collect_min_margin_residual_after",
        "projection/collect_any_zero_margin_fallback",
        "projection/projected_grad_norm",
        "reference_kl",
        "epoch1_ratio_mean",
    )
    missing = sorted({key for key in required if any(key not in row for row in rows)})
    if missing:
        raise SystemExit("[FAIL] smoke log missing metrics: " + ", ".join(missing))

    for i, row in enumerate(rows, 1):
        bad = [key for key in required if not math.isfinite(row[key])]
        if bad:
            raise SystemExit(f"[FAIL] step {i} non-finite metrics: {bad}")

    correction_steps = sum(row["projection/collect_any_correction"] > 0.5 for row in rows)
    max_correction = max(row["projection/collect_max_correction_ratio"] for row in rows)
    max_before = max(row["projection/collect_max_violated_before_count"] for row in rows)
    max_after = max(row["projection/collect_max_violated_after_count"] for row in rows)
    min_scale = min(row["projection/collect_min_margin_scale_used"] for row in rows)
    min_residual = min(row["projection/collect_min_margin_residual_after"] for row in rows)
    any_zero_fallback = max(row["projection/collect_any_zero_margin_fallback"] for row in rows)
    max_projected_norm = max(row["projection/projected_grad_norm"] for row in rows)
    max_ref_kl = max(row["reference_kl"] for row in rows)
    max_epoch1_ratio_error = max(abs(row["epoch1_ratio_mean"] - 1.0) for row in rows)

    print(f"[smoke] steps={len(rows)} correction_steps={correction_steps}")
    print(f"[smoke] max_correction_ratio={max_correction:.6g}")
    print(f"[smoke] max_violated_before={max_before:.0f} max_violated_after={max_after:.0f}")
    print(f"[smoke] min_margin_scale_used={min_scale:.6g} min_margin_residual_after={min_residual:.6g}")
    print(f"[smoke] any_zero_margin_fallback={any_zero_fallback:.0f}")
    print(f"[smoke] max_projected_grad_norm={max_projected_norm:.6g} max_ref_kl={max_ref_kl:.6g}")

    failures = []
    if correction_steps < 1 or max_correction <= 1e-4:
        failures.append("restorative projection never produced a material correction")
    if max_before < 1:
        failures.append("no positive-margin half-space was violated before projection")
    if max_after > 0.5:
        failures.append("at least one used projection margin remained violated after projection")
    if min_residual < -1e-4:
        failures.append("post-projection margin residual is below tolerance")
    if min_scale <= 0.0 or any_zero_fallback > 0.5:
        failures.append("smoke required zero-margin fallback; positive restorative intersection not verified")
    if max_projected_norm > 5.005:
        failures.append("projected gradient exceeded max_grad_norm budget")
    if max_ref_kl > 0.25:
        failures.append("reference KL exceeded 0.25 smoke guard")
    if max_epoch1_ratio_error > 1e-3:
        failures.append("epoch-1 ratio is not ~1; replay semantics may be broken")

    if failures:
        for item in failures:
            print("[FAIL]", item)
        raise SystemExit(2)

    args.pass_marker.parent.mkdir(parents=True, exist_ok=True)
    args.pass_marker.write_text(
        "Projection V2 smoke passed\n"
        f"correction_steps={correction_steps}\n"
        f"max_correction_ratio={max_correction}\n"
        f"min_margin_scale_used={min_scale}\n",
        encoding="utf-8",
    )
    print("[PASS] Projection V2 restorative projection is active and feasible")
    print(f"[PASS] marker={args.pass_marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
