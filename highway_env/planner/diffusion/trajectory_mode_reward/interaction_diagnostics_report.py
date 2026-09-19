"""Structures and reports for interaction-source diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InteractionEvent:
    scenario: str
    seed: int
    rollout_step: int
    simulation_time_s: float
    target_role: int
    selected_mode_idx: int
    source_type: str
    source_id: str
    source_role: int | None
    collision: bool
    clearance_violation: bool
    first_collision_dense_index: int | None
    first_collision_time_s: float | None
    first_clearance_dense_index: int | None
    first_clearance_time_s: float | None
    minimum_gap_m: float
    minimum_gap_dense_index: int
    minimum_gap_time_s: float
    minimum_ttc_s: float
    minimum_ttc_dense_index: int
    minimum_ttc_time_s: float
    safe_gap_threshold_m: float
    source_current_x_m: float | None
    source_current_y_m: float | None
    source_current_speed_mps: float | None
    source_lane_index: str | None


@dataclass(frozen=True)
class UnsafeRoleDiagnostic:
    scenario: str
    seed: int
    rollout_step: int
    simulation_time_s: float
    target_role: int
    selected_mode_idx: int
    reward: float
    collision: bool
    clearance_violation: bool
    out_of_drivable: bool
    background_collision: bool
    teammate_collision: bool
    background_clearance: bool
    teammate_clearance: bool
    trigger_event_count: int


def summarize_diagnostics(
    unsafe_records: list[UnsafeRoleDiagnostic],
    events: list[InteractionEvent],
) -> dict[str, Any]:
    def count(attribute: str) -> int:
        return sum(
            bool(getattr(record, attribute))
            for record in unsafe_records
        )

    t0_roles = {
        (
            event.scenario,
            event.seed,
            event.rollout_step,
            event.target_role,
        )
        for event in events
        if (
            event.clearance_violation
            and event.first_clearance_time_s is not None
            and abs(event.first_clearance_time_s) <= 1e-12
        )
    }
    return {
        "unsafe_role_count": len(unsafe_records),
        "collision_role_count": count("collision"),
        "clearance_role_count": count("clearance_violation"),
        "out_of_drivable_role_count": count("out_of_drivable"),
        "background_collision_role_count": count(
            "background_collision"
        ),
        "teammate_collision_role_count": count(
            "teammate_collision"
        ),
        "background_clearance_role_count": count(
            "background_clearance"
        ),
        "teammate_clearance_role_count": count(
            "teammate_clearance"
        ),
        "t0_clearance_role_count": len(t0_roles),
        "triggered_interaction_count": len(events),
        "collision_interaction_count": sum(
            event.collision for event in events
        ),
        "clearance_interaction_count": sum(
            event.clearance_violation for event in events
        ),
    }


def _by_scenario(
    unsafe_records: list[UnsafeRoleDiagnostic],
    events: list[InteractionEvent],
) -> dict[str, Any]:
    scenarios = sorted(
        {record.scenario for record in unsafe_records}
        | {event.scenario for event in events}
    )
    return {
        scenario: summarize_diagnostics(
            [
                record
                for record in unsafe_records
                if record.scenario == scenario
            ],
            [
                event
                for event in events
                if event.scenario == scenario
            ],
        )
        for scenario in scenarios
    }


def _markdown(
    unsafe_records: list[UnsafeRoleDiagnostic],
    events: list[InteractionEvent],
    state_count: int,
    seeds: list[int],
    requested_steps: int,
) -> str:
    overall = summarize_diagnostics(unsafe_records, events)
    by_scenario = _by_scenario(unsafe_records, events)

    lines = [
        "# Reward Interaction Diagnostics",
        "",
        "Diagnostic-only source attribution; reward behavior is unchanged.",
        "",
        f"- planning states: {state_count}",
        f"- seeds: {seeds}",
        f"- requested rollout steps/reset: {requested_steps}",
        f"- unsafe target roles: {overall['unsafe_role_count']}",
        f"- collision target roles: {overall['collision_role_count']}",
        f"- t=0 clearance target roles: {overall['t0_clearance_role_count']}",
        "",
        "## Source summary",
        "",
        (
            "| scenario | unsafe | collision | bg collision | "
            "teammate collision | bg clearance | teammate clearance | "
            "t=0 clearance |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, summary in by_scenario.items():
        lines.append(
            f"| {scenario} | "
            f"{summary['unsafe_role_count']} | "
            f"{summary['collision_role_count']} | "
            f"{summary['background_collision_role_count']} | "
            f"{summary['teammate_collision_role_count']} | "
            f"{summary['background_clearance_role_count']} | "
            f"{summary['teammate_clearance_role_count']} | "
            f"{summary['t0_clearance_role_count']} |"
        )

    merge_collision = [
        event
        for event in events
        if event.scenario == "merge_in" and event.collision
    ]
    lines.extend(
        [
            "",
            "## merge_in collision interactions",
            "",
            (
                "| seed | step | role | source | other | first collision t | "
                "min gap | min-gap t | min TTC |"
            ),
            "|---:|---:|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for event in merge_collision:
        first_t = (
            "-"
            if event.first_collision_time_s is None
            else f"{event.first_collision_time_s:.2f}"
        )
        lines.append(
            f"| {event.seed} | {event.rollout_step} | "
            f"{event.target_role} | {event.source_type} | "
            f"{event.source_id} | {first_t} | "
            f"{event.minimum_gap_m:.3f} | "
            f"{event.minimum_gap_time_s:.2f} | "
            f"{event.minimum_ttc_s:.3f} |"
        )
    if not merge_collision:
        lines.append("| - | - | - | - | none | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Unsafe target roles",
            "",
            (
                "| scenario | seed | step | role | reward | collision | "
                "clearance | offroad | bg coll | teammate coll | "
                "bg clr | teammate clr |"
            ),
            "|---|---:|---:|---:|---:|---|---|---|---|---|---|---|",
        ]
    )
    for record in unsafe_records:
        lines.append(
            f"| {record.scenario} | {record.seed} | "
            f"{record.rollout_step} | {record.target_role} | "
            f"{record.reward:.5f} | {record.collision} | "
            f"{record.clearance_violation} | "
            f"{record.out_of_drivable} | "
            f"{record.background_collision} | "
            f"{record.teammate_collision} | "
            f"{record.background_clearance} | "
            f"{record.teammate_clearance} |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Collision matches reward semantics: t=0 excluded; first checked point is t=0.1 s.",
            "- Clearance matches reward semantics: t=0 is included.",
            "- background_N follows the state_adapter background vehicle ordering.",
            "- Exact interaction times, dense indices, min gaps, TTC, and background metadata are in JSON.",
            "",
        ]
    )
    return "\n".join(lines)


def write_diagnostic_reports(
    *,
    unsafe_records: list[UnsafeRoleDiagnostic],
    events: list[InteractionEvent],
    state_count: int,
    seeds: list[int],
    requested_steps: int,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "validation": (
            "trajectory_mode_reward_interaction_diagnostics_v1"
        ),
        "reward_behavior_changed": False,
        "dense_dt_s": 0.1,
        "collision_excludes_t0": True,
        "clearance_includes_t0": True,
        "state_count": int(state_count),
        "seeds": [int(seed) for seed in seeds],
        "requested_steps_per_reset": int(requested_steps),
        "overall": summarize_diagnostics(
            unsafe_records,
            events,
        ),
        "by_scenario": _by_scenario(
            unsafe_records,
            events,
        ),
        "unsafe_roles": [
            asdict(record) for record in unsafe_records
        ],
        "interaction_events": [
            asdict(event) for event in events
        ],
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
            unsafe_records,
            events,
            state_count,
            seeds,
            requested_steps,
        ),
        encoding="utf-8",
    )
