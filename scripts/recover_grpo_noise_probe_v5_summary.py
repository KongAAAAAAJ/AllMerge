from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

IDENTITY_FIELDS = {
    "checkpoint",
    "scenario",
    "sample_index",
    "env_seed",
    "noise_seed",
    "vehicle_role",
    "mode",
    "best_group_id",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _stable_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _aggregate(
    rows: list[dict[str, str]],
    *,
    checkpoint: str,
    scenario: str,
    scope: str,
) -> dict[str, object]:
    subset = [
        row
        for row in rows
        if row.get("checkpoint") == checkpoint
        and (scenario == "all" or row.get("scenario") == scenario)
    ]
    if not subset:
        raise RuntimeError(
            f"no rows for checkpoint={checkpoint!r} scenario={scenario!r} scope={scope!r}"
        )

    out: dict[str, object] = {
        "checkpoint": checkpoint,
        "scenario": scenario,
        "scope": scope,
        "row_count": len(subset),
    }

    keys: list[str] = []
    seen: set[str] = set()
    for row in subset:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    for key in keys:
        if key in IDENTITY_FIELDS:
            continue
        values = []
        for row in subset:
            number = _as_float(row.get(key))
            if number is not None:
                values.append(number)
        if values:
            out[f"mean_{key}"] = float(np.mean(values))
    return out


def _write_union_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("no summary rows")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recover GRPO noise-probe v5 reward_component_overall_summary.csv "
            "from already-written detailed component CSVs after the v5 writer bug."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    state_path = output_dir / "reward_component_state_summary.csv"
    selected_path = output_dir / "reward_component_selected_mode.csv"
    state_rows = _read_csv(state_path)
    selected_rows = _read_csv(selected_path)

    checkpoints = _stable_unique(
        [row.get("checkpoint", "") for row in state_rows]
        + [row.get("checkpoint", "") for row in selected_rows]
    )
    scenarios = _stable_unique(
        [row.get("scenario", "") for row in state_rows]
        + [row.get("scenario", "") for row in selected_rows]
    )
    preferred = ["straight", "curved", "merge_in", "merge_out"]
    scenarios = [s for s in preferred if s in scenarios] + [
        s for s in scenarios if s not in preferred
    ]

    summary_rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        for scope, source_rows in (
            ("all_valid_vehicle_modes", state_rows),
            ("selected_modes", selected_rows),
        ):
            summary_rows.append(
                _aggregate(
                    source_rows,
                    checkpoint=checkpoint,
                    scenario="all",
                    scope=scope,
                )
            )
            for scenario in scenarios:
                if any(
                    row.get("checkpoint") == checkpoint
                    and row.get("scenario") == scenario
                    for row in source_rows
                ):
                    summary_rows.append(
                        _aggregate(
                            source_rows,
                            checkpoint=checkpoint,
                            scenario=scenario,
                            scope=scope,
                        )
                    )

    target = output_dir / "reward_component_overall_summary.csv"
    _write_union_csv(target, summary_rows)

    print(f"[OK] recovered: {target}")
    print(f"[OK] checkpoints: {', '.join(checkpoints)}")
    print(f"[OK] scenarios  : {', '.join(scenarios)}")
    print(f"[OK] summary rows: {len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
