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
        rows.append({key: float(value) for key, value in PAIR.findall(line)})
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--pass-marker", type=Path, required=True)
    p.add_argument("--kl-budget", type=float, default=0.10)
    args = p.parse_args()

    rows = parse_steps(args.log)
    if len(rows) < 3:
        raise SystemExit(f"[FAIL] expected >=3 step rows, got {len(rows)}")

    required = (
        "safempo/dual_success",
        "safempo/dual_iterations",
        "safempo/nu",
        "safempo/teacher_kl",
        "safempo/teacher_entropy_normalized",
        "safempo/distill_kl",
        "safempo/student_old_particle_kl",
        "safempo/student_target_l1",
        "constraint/feasible_fraction",
        "constraint/max_violation_mean",
        "reference_kl",
        "grad_norm",
        "epoch1_ratio_mean",
    )
    missing = sorted({key for key in required if any(key not in row for row in rows)})
    if missing:
        raise SystemExit("[FAIL] SafeMPO smoke missing metrics: " + ", ".join(missing))

    for i, row in enumerate(rows, 1):
        bad = [key for key in required if not math.isfinite(row[key])]
        if bad:
            raise SystemExit(f"[FAIL] step {i} non-finite metrics: {bad}")

    max_teacher_kl = max(row["safempo/teacher_kl"] for row in rows)
    min_dual_success = min(row["safempo/dual_success"] for row in rows)
    max_ref_kl = max(row["reference_kl"] for row in rows)
    max_grad = max(row["grad_norm"] for row in rows)
    max_epoch1_ratio_error = max(abs(row["epoch1_ratio_mean"] - 1.0) for row in rows)
    min_entropy = min(row["safempo/teacher_entropy_normalized"] for row in rows)
    max_entropy = max(row["safempo/teacher_entropy_normalized"] for row in rows)

    failures = []
    if min_dual_success < 0.5:
        failures.append("dual solver failed on at least one step")
    if max_teacher_kl > args.kl_budget + 5e-3:
        failures.append("teacher KL exceeded E-step budget")
    if max_ref_kl > 0.25:
        failures.append("reference KL exceeded 0.25 smoke guard")
    if max_grad > 5.005:
        # logged grad_norm is pre-clip in PyTorch; retain only as an informational
        # guard against numerical explosion rather than enforcing == max_norm.
        if max_grad > 1e6:
            failures.append("pre-clip gradient norm exploded")
    if max_epoch1_ratio_error > 1e-3:
        failures.append("epoch-1 ratio is not ~1; replay semantics may be broken")
    if min_entropy < -1e-6 or max_entropy > 1.0001:
        failures.append("teacher normalized entropy is outside [0,1]")

    print(f"[smoke] steps={len(rows)} max_teacher_kl={max_teacher_kl:.6g}")
    print(f"[smoke] dual_success_min={min_dual_success:.0f} max_ref_kl={max_ref_kl:.6g}")
    print(f"[smoke] teacher_entropy_range={min_entropy:.6g}..{max_entropy:.6g}")
    print(f"[smoke] max_preclip_grad_norm={max_grad:.6g}")

    if failures:
        for item in failures:
            print("[FAIL]", item)
        raise SystemExit(2)

    args.pass_marker.parent.mkdir(parents=True, exist_ok=True)
    args.pass_marker.write_text(
        "SafeMPO-Diff V1 smoke passed\n"
        f"steps={len(rows)}\n"
        f"max_teacher_kl={max_teacher_kl}\n"
        f"max_reference_kl={max_ref_kl}\n",
        encoding="utf-8",
    )
    print("[PASS] SafeMPO-Diff V1 E-step + KL distillation smoke passed")
    print(f"[PASS] marker={args.pass_marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
