#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BASELINE_CONVERGENCE_COMPLETION_CHECK_500STEP_V1_20261007
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

STEP_RE = re.compile(r"^\[step\s+(\d+)/(\d+)\]")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--fixed-csv", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--expected-step", type=int, default=500)
    p.add_argument("--marker", type=Path, required=True)
    args = p.parse_args()

    for path in (args.log, args.fixed_csv, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"[FAIL] missing expected run artifact: {path}")

    last_train = None
    total = None
    ok_wrote = False
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP_RE.match(line)
        if m:
            last_train = int(m.group(1))
            total = int(m.group(2))
        if line.startswith("[OK] wrote "):
            ok_wrote = True

    if last_train != args.expected_step or total != args.expected_step:
        raise SystemExit(
            f"[FAIL] train log ends at {last_train}/{total}; "
            f"expected {args.expected_step}/{args.expected_step}"
        )
    if not ok_wrote:
        raise SystemExit("[FAIL] train log has no final [OK] wrote checkpoint marker")

    with args.fixed_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("[FAIL] fixed-validation CSV is empty")
    fixed_last = int(round(float(rows[-1]["step"])))
    if fixed_last != args.expected_step:
        raise SystemExit(
            f"[FAIL] fixed validation ends at {fixed_last}; "
            f"expected {args.expected_step}"
        )

    args.marker.parent.mkdir(parents=True, exist_ok=True)
    args.marker.write_text(
        f"complete_step={args.expected_step}\n"
        f"log={args.log}\n"
        f"fixed_csv={args.fixed_csv}\n"
        f"checkpoint={args.checkpoint}\n",
        encoding="utf-8",
    )
    print(f"[PASS] complete {args.expected_step}-step run")
    print(f"[MARKER] {args.marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
