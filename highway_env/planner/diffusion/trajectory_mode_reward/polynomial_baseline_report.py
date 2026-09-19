"""Aggregation and reporting for Polynomial reward validation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolynomialRoleRecord:
    scenario: str
    seed: int
    rollout_step: int
    simulation_time_s: float
    role: int
    selected_mode_idx: int
    selected_mode_name: str
    expected_semantic_group: str
    selected_semantic_group: str
    semantic_match: bool
    reward: float
    unsafe: bool
    collision: bool
    out_of_drivable: bool
    clearance_violation: bool
    progress_score: float
    gap_penalty: float
    ttc_penalty: float
    road_penalty: float
    comfort_penalty: float
    minimum_background_gap_m: float
    minimum_teammate_gap_m: float
    minimum_road_margin_m: float
    minimum_ttc_s: float


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise AssertionError(
            "summary input must be non-empty and finite"
        )
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def aggregate(
    records: list[PolynomialRoleRecord],
) -> dict[str, Any]:
    if not records:
        raise AssertionError("no Polynomial reward records")

    def values(name: str):
        return [
            float(getattr(record, name))
            for record in records
        ]

    def rate(name: str) -> float:
        return float(
            np.mean(
                [
                    bool(getattr(record, name))
                    for record in records
                ]
            )
        )

    mode_histogram = Counter(
        record.selected_mode_name
        for record in records
    )
    return {
        "sample_count": len(records),
        "reward": _stats(values("reward")),
        "rates": {
            "unsafe": rate("unsafe"),
            "collision": rate("collision"),
            "out_of_drivable": rate("out_of_drivable"),
            "clearance_violation": rate(
                "clearance_violation"
            ),
            "semantic_match": rate("semantic_match"),
        },
        "component_mean": {
            name: float(np.mean(values(name)))
            for name in (
                "progress_score",
                "gap_penalty",
                "ttc_penalty",
                "road_penalty",
                "comfort_penalty",
            )
        },
        "minimum_metrics": {
            name: _stats(values(name))
            for name in (
                "minimum_background_gap_m",
                "minimum_teammate_gap_m",
                "minimum_road_margin_m",
                "minimum_ttc_s",
            )
        },
        "selected_mode_histogram": dict(
            sorted(mode_histogram.items())
        ),
    }


def summary_by_scenario(
    records: list[PolynomialRoleRecord],
) -> dict[str, Any]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record.scenario].append(record)
    return {
        scenario: aggregate(group)
        for scenario, group in sorted(grouped.items())
    }


def _markdown(
    overall: dict[str, Any],
    by_scenario: dict[str, Any],
    state_count: int,
    max_parity_error: float,
    requested_steps: int,
    seeds: list[int],
) -> str:
    lines = [
        "# Polynomial Baseline Reward Report",
        "",
        "Source: real aligned Polynomial trajectories evaluated before road.step().",
        "",
        f"- planning states: {state_count}",
        f"- vehicle trajectory samples: {overall['sample_count']}",
        f"- seeds: {seeds}",
        f"- requested steps per reset: {requested_steps}",
        f"- max candidate/reference parity error: {max_parity_error:.3e}",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| reward mean | {overall['reward']['mean']:.6f} |",
        f"| reward std | {overall['reward']['std']:.6f} |",
        f"| reward p05 | {overall['reward']['p05']:.6f} |",
        f"| reward p50 | {overall['reward']['p50']:.6f} |",
        f"| reward p95 | {overall['reward']['p95']:.6f} |",
        f"| unsafe rate | {overall['rates']['unsafe']:.6f} |",
        f"| collision rate | {overall['rates']['collision']:.6f} |",
        f"| out-of-drivable rate | {overall['rates']['out_of_drivable']:.6f} |",
        f"| clearance violation rate | {overall['rates']['clearance_violation']:.6f} |",
        f"| semantic match rate | {overall['rates']['semantic_match']:.6f} |",
        "",
        "## By scenario",
        "",
        "| scenario | n | reward mean | p05 | p50 | p95 | unsafe | collision | offroad | clearance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for scenario, summary in by_scenario.items():
        r = summary["reward"]
        rate = summary["rates"]
        lines.append(
            f"| {scenario} | {summary['sample_count']} | "
            f"{r['mean']:.6f} | {r['p05']:.6f} | "
            f"{r['p50']:.6f} | {r['p95']:.6f} | "
            f"{rate['unsafe']:.6f} | {rate['collision']:.6f} | "
            f"{rate['out_of_drivable']:.6f} | "
            f"{rate['clearance_violation']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Component means",
            "",
            "| scenario | progress | gap | TTC | road | comfort |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario, summary in by_scenario.items():
        c = summary["component_mean"]
        lines.append(
            f"| {scenario} | "
            f"{c['progress_score']:.6f} | "
            f"{c['gap_penalty']:.6f} | "
            f"{c['ttc_penalty']:.6f} | "
            f"{c['road_penalty']:.6f} | "
            f"{c['comfort_penalty']:.6f} |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- The scored path is the Polynomial trajectory, not its nearest anchor.",
            "- nearest_mode_idx is used only as a trajectory_mode storage slot.",
            "- Teammates in this report are the two Polynomial trajectories from the same state.",
            "- GRPO still uses frozen Stage-1 argmax teammate context.",
            "- Non-zero unsafe/collision/offroad rates are reported, not automatically treated as migration failures.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    *,
    records: list[PolynomialRoleRecord],
    overall: dict[str, Any],
    by_scenario: dict[str, Any],
    state_count: int,
    max_parity_error: float,
    requested_steps: int,
    seeds: list[int],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "validation": (
            "trajectory_mode_reward_polynomial_baseline_v1"
        ),
        "planner_source": "real aligned Polynomial output",
        "comparison_unit": "trajectory_mode",
        "teammate_context_for_this_report": (
            "other two Polynomial trajectories from same planning state"
        ),
        "grpo_contract_changed": False,
        "state_count": int(state_count),
        "requested_steps_per_seed": int(requested_steps),
        "seeds": [int(seed) for seed in seeds],
        "max_candidate_reference_parity_error": float(
            max_parity_error
        ),
        "overall": overall,
        "by_scenario": by_scenario,
        "records": [asdict(record) for record in records],
    }
    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(
        _markdown(
            overall,
            by_scenario,
            state_count,
            max_parity_error,
            requested_steps,
            seeds,
        ),
        encoding="utf-8",
    )
