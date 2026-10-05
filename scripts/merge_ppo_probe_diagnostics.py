#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def read_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return list(reader), list(reader.fieldnames or [])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge PPO V2 critic diagnostics into learning_probe.csv"
    )
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--diag", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.probe.is_file():
        raise FileNotFoundError(args.probe)
    if not args.diag.is_file():
        raise FileNotFoundError(args.diag)

    probe_rows, probe_fields = read_rows(args.probe)
    diag_rows, diag_fields = read_rows(args.diag)
    diag_by_step = {str(int(float(row["step"]))): row for row in diag_rows}

    extra_fields = [f for f in diag_fields if f != "step"]
    output = args.output or args.probe

    if output.resolve() == args.probe.resolve():
        backup = args.probe.with_name(
            args.probe.stem + ".before_ppo_v2_metrics" + args.probe.suffix
        )
        shutil.copy2(args.probe, backup)
        print(f"[backup] {backup}")

    fieldnames = list(probe_fields)
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    temp = output.with_suffix(output.suffix + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temp.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in probe_rows:
            step_key = str(int(float(row["step"])))
            diag = diag_by_step.get(step_key, {})
            merged = dict(row)
            for field in extra_fields:
                merged[field] = diag.get(field, "")
            writer.writerow(merged)

    temp.replace(output)
    print(
        f"[OK] merged PPO diagnostics: probe_rows={len(probe_rows)} "
        f"diag_rows={len(diag_rows)} added_columns={len(extra_fields)}"
    )
    print(f"[OK] output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
