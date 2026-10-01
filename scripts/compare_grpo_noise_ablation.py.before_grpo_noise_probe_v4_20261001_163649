from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path


def _parse_run(value: str) -> RunSpec:
    label, sep, raw = value.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("--run must be LABEL=OUTPUT_DIR")
    label = label.strip()
    path = Path(raw.strip())
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    return RunSpec(label, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine GRPO noise-probe v3 additive/multiplicative ablation summaries."
    )
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/grpo_noise_probe_v3_ablation"),
    )
    args = parser.parse_args()

    overall_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for spec in args.run:
        overall_path = spec.path / "overall_summary.csv"
        selected_path = spec.path / "selected_overall_summary.csv"
        if not overall_path.is_file():
            raise SystemExit(f"missing: {overall_path}")
        if not selected_path.is_file():
            raise SystemExit(f"missing: {selected_path}")

        for row in _read_csv(overall_path):
            overall_rows.append({"run": spec.label, **row})
        for row in _read_csv(selected_path):
            selected_rows.append({"run": spec.label, **row})

    _write_csv(args.output_dir / "overall_ablation_comparison.csv", overall_rows)
    _write_csv(
        args.output_dir / "selected_ablation_comparison.csv",
        selected_rows,
    )

    print("\n=== SELECTED-MODE ABLATION / scenario=all ===")
    for row in selected_rows:
        if row.get("scenario") != "all":
            continue
        print(
            f"{row['run']:>10s} | {row['checkpoint']:<18s} | "
            f"ADE={float(row['mean_selected_pairwise_ADE']):.3f}m | "
            f"dx={float(row['mean_selected_pairwise_abs_dx']):.3f}m | "
            f"dy={float(row['mean_selected_pairwise_abs_dy']):.3f}m | "
            f"end_dy_std={float(row['mean_selected_endpoint_dy_std']):.3f}m | "
            f"Rstd={float(row['mean_selected_reward_std']):.3f} | "
            f"Rrange={float(row['mean_selected_reward_range']):.3f} | "
            f"best-mean={float(row['mean_selected_best_minus_mean']):+.3f}"
        )

    print(f"\n[OK] {args.output_dir / 'overall_ablation_comparison.csv'}")
    print(f"[OK] {args.output_dir / 'selected_ablation_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
