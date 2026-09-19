"""Report helpers for background-prediction fidelity diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .background_prediction_rollout import PairComparison


@dataclass(frozen=True)
class PredictionRecord:
    scenario: str
    seed: int
    rollout_step: int
    simulation_time_s: float
    selected_mode_idx: int
    target_role: int
    actor_id: str
    predicted_collision: bool
    actual_collision: bool
    predicted_clearance: bool
    actual_clearance: bool
    predicted_first_collision_time_s: float | None
    actual_first_collision_time_s: float | None
    predicted_first_clearance_time_s: float | None
    actual_first_clearance_time_s: float | None
    predicted_min_gap_m: float
    actual_min_gap_m: float
    predicted_min_ttc_s: float
    actual_min_ttc_s: float
    endpoint_position_error_m: float
    max_position_error_m: float
    mean_position_error_m: float
    endpoint_longitudinal_error_m: float
    endpoint_lateral_error_m: float
    endpoint_speed_error_mps: float
    max_speed_error_mps: float

    @classmethod
    def from_comparison(
        cls,
        *,
        scenario: str,
        seed: int,
        rollout_step: int,
        simulation_time_s: float,
        selected_mode_idx: int,
        comparison: PairComparison,
    ) -> "PredictionRecord":
        return cls(
            scenario=scenario,
            seed=int(seed),
            rollout_step=int(rollout_step),
            simulation_time_s=float(simulation_time_s),
            selected_mode_idx=int(selected_mode_idx),
            **asdict(comparison),
        )


def _rate(
    records: list[PredictionRecord],
    predicate,
) -> float:
    if not records:
        return 0.0
    return float(
        np.mean([bool(predicate(record)) for record in records])
    )


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def summarize(
    records: list[PredictionRecord],
) -> dict[str, Any]:
    predicted_collision = [
        record
        for record in records
        if record.predicted_collision
    ]
    return {
        "pair_count": len(records),
        "predicted_collision_count": sum(
            record.predicted_collision
            for record in records
        ),
        "actual_collision_count": sum(
            record.actual_collision
            for record in records
        ),
        "predicted_collision_actual_true_count": sum(
            record.predicted_collision
            and record.actual_collision
            for record in records
        ),
        "predicted_collision_actual_false_count": sum(
            record.predicted_collision
            and not record.actual_collision
            for record in records
        ),
        "actual_collision_prediction_false_count": sum(
            record.actual_collision
            and not record.predicted_collision
            for record in records
        ),
        "predicted_clearance_count": sum(
            record.predicted_clearance
            for record in records
        ),
        "actual_clearance_count": sum(
            record.actual_clearance
            for record in records
        ),
        "collision_agreement_rate": _rate(
            records,
            lambda record: (
                record.predicted_collision
                == record.actual_collision
            ),
        ),
        "clearance_agreement_rate": _rate(
            records,
            lambda record: (
                record.predicted_clearance
                == record.actual_clearance
            ),
        ),
        "endpoint_position_error_m": _stats(
            [
                record.endpoint_position_error_m
                for record in records
            ]
        ),
        "max_position_error_m": _stats(
            [
                record.max_position_error_m
                for record in records
            ]
        ),
        "endpoint_speed_error_mps": _stats(
            [
                abs(record.endpoint_speed_error_mps)
                for record in records
            ]
        ),
        "predicted_collision_pairs_endpoint_error_m": _stats(
            [
                record.endpoint_position_error_m
                for record in predicted_collision
            ]
        ),
    }


def _by_scenario(
    records: list[PredictionRecord],
) -> dict[str, Any]:
    return {
        scenario: summarize(
            [
                record
                for record in records
                if record.scenario == scenario
            ]
        )
        for scenario in sorted(
            {record.scenario for record in records}
        )
    }


def _markdown(
    records: list[PredictionRecord],
    state_count: int,
    rollout_mode: str,
) -> str:
    overall = summarize(records)
    lines = [
        "# Background Prediction Diagnostics",
        "",
        (
            "Compares reward state_adapter constant-speed prediction with "
            "native AllMerge simulator background rollout."
        ),
        "",
        f"- planning states: {state_count}",
        f"- target/background pairs: {overall['pair_count']}",
        f"- actual rollout mode: {rollout_mode}",
        (
            "- target collision geometry is always the SAME original "
            "Polynomial planned trajectory"
        ),
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---:|",
        (
            f"| predicted collision | "
            f"{overall['predicted_collision_count']} |"
        ),
        (
            f"| actual collision | "
            f"{overall['actual_collision_count']} |"
        ),
        (
            "| predicted collision -> actual collision | "
            f"{overall['predicted_collision_actual_true_count']} |"
        ),
        (
            "| predicted collision -> actual safe | "
            f"{overall['predicted_collision_actual_false_count']} |"
        ),
        (
            "| predicted safe -> actual collision | "
            f"{overall['actual_collision_prediction_false_count']} |"
        ),
        (
            f"| collision agreement | "
            f"{overall['collision_agreement_rate']:.6f} |"
        ),
        (
            f"| clearance agreement | "
            f"{overall['clearance_agreement_rate']:.6f} |"
        ),
        (
            "| endpoint position error mean / p95 / max (m) | "
            f"{overall['endpoint_position_error_m']['mean']:.3f} / "
            f"{overall['endpoint_position_error_m']['p95']:.3f} / "
            f"{overall['endpoint_position_error_m']['max']:.3f} |"
        ),
        (
            "| endpoint speed error mean / p95 / max (m/s) | "
            f"{overall['endpoint_speed_error_mps']['mean']:.3f} / "
            f"{overall['endpoint_speed_error_mps']['p95']:.3f} / "
            f"{overall['endpoint_speed_error_mps']['max']:.3f} |"
        ),
        "",
        "## Predicted collision pairs",
        "",
        (
            "| scenario | seed | step | role | actor | pred coll t | "
            "actual collision | actual coll t | pred min gap | "
            "actual min gap | endpoint error | speed error |"
        ),
        (
            "|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|"
        ),
    ]

    collision_records = [
        record
        for record in records
        if record.predicted_collision
    ]
    for record in collision_records:
        pred_t = (
            "-"
            if record.predicted_first_collision_time_s is None
            else f"{record.predicted_first_collision_time_s:.2f}"
        )
        actual_t = (
            "-"
            if record.actual_first_collision_time_s is None
            else f"{record.actual_first_collision_time_s:.2f}"
        )
        lines.append(
            f"| {record.scenario} | {record.seed} | "
            f"{record.rollout_step} | {record.target_role} | "
            f"{record.actor_id} | {pred_t} | "
            f"{record.actual_collision} | {actual_t} | "
            f"{record.predicted_min_gap_m:.3f} | "
            f"{record.actual_min_gap_m:.3f} | "
            f"{record.endpoint_position_error_m:.3f} | "
            f"{record.endpoint_speed_error_mps:.3f} |"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            (
                "- predicted_collision=True / actual_collision=False with "
                "large 3-4 s position error indicates predictor-induced "
                "false collision risk."
            ),
            (
                "- predicted_collision=True / actual_collision=True supports "
                "the current reward background-risk signal."
            ),
            (
                "- This is a counterfactual fidelity diagnostic, not a full "
                "closed-loop safety evaluation."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    *,
    records: list[PredictionRecord],
    state_count: int,
    rollout_mode: str,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "validation": (
            "trajectory_mode_reward_background_prediction_diagnostics_v1"
        ),
        "reward_behavior_changed": False,
        "rollout_mode": rollout_mode,
        "target_geometry": (
            "original Polynomial planned trajectory held fixed for both "
            "predicted-background and actual-background collision tests"
        ),
        "state_count": int(state_count),
        "overall": summarize(records),
        "by_scenario": _by_scenario(records),
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
            records,
            state_count,
            rollout_mode,
        ),
        encoding="utf-8",
    )
