"""Run interaction-source diagnostics on real Polynomial baseline trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import TrajectoryModeRewardConfig
from .constants import HORIZON_STEPS, NUM_MODES, NUM_VEHICLES
from .counterfactual import TrajectoryModeCounterfactualReward
from .evaluator import evaluate_candidates
from .interaction_diagnostics_geometry import (
    background_metadata,
    build_interaction_event,
    dense_joint_world,
    tracking_dimensions,
)
from .interaction_diagnostics_report import (
    InteractionEvent,
    UnsafeRoleDiagnostic,
    summarize_diagnostics,
    write_diagnostic_reports,
)
from .polynomial_baseline_validation import (
    SCENARIOS,
    _advance_without_replanning,
    _assert_config,
    _extract_alignment,
)


def _diagnose_state(
    env: object,
    *,
    scenario: str,
    seed: int,
    rollout_step: int,
    config: TrajectoryModeRewardConfig,
) -> tuple[list[UnsafeRoleDiagnostic], list[InteractionEvent]]:
    (
        expert_xy,
        selected_idx,
        _selected_name,
        _expected_semantic,
        _selected_semantic,
        _semantic_match,
    ) = _extract_alignment(env)

    all_modes = np.zeros(
        (NUM_VEHICLES, NUM_MODES, HORIZON_STEPS, 2),
        dtype=np.float32,
    )
    selected_valid = np.zeros(
        (NUM_VEHICLES, NUM_MODES),
        dtype=np.bool_,
    )
    for role in range(NUM_VEHICLES):
        mode_idx = int(selected_idx[role])
        all_modes[role, mode_idx] = expert_xy[role]
        selected_valid[role, mode_idx] = True

    result = evaluate_candidates(
        env=env,
        candidates=all_modes[:, :, None],
        frozen_all_mode_trajectories=all_modes,
        frozen_argmax_joint_trajectories=expert_xy,
        valid_mode_mask=selected_valid,
        config=config,
    )

    scorer = TrajectoryModeCounterfactualReward(config=config)
    context = scorer.build_geometry_context(env, expert_xy)
    dense_world, times = dense_joint_world(
        expert_xy,
        context.poses,
        config,
    )
    gap_dimensions = tracking_dimensions(config)
    bg_meta = background_metadata(env)

    unsafe_records = []
    events = []

    for target_role in range(NUM_VEHICLES):
        mode_idx = int(selected_idx[target_role])
        reward_unsafe = bool(
            result.unsafe[target_role, mode_idx, 0]
        )
        if not reward_unsafe:
            continue

        target_world = dense_world[target_role]
        role_events = []

        for actor, predictions in context.backgrounds[
            target_role
        ].items():
            for branch_name, predicted, other_dimensions in predictions:
                if branch_name != actor:
                    continue
                event = build_interaction_event(
                    scenario=scenario,
                    seed=seed,
                    rollout_step=rollout_step,
                    simulation_time_s=float(getattr(env, "time", 0.0)),
                    target_role=target_role,
                    selected_mode_idx=mode_idx,
                    source_type="background",
                    source_id=actor,
                    source_role=None,
                    source_metadata=bg_meta.get(actor),
                    target_world=target_world,
                    other_world=predicted,
                    target_gap_dimensions=gap_dimensions,
                    other_dimensions=other_dimensions,
                    safe_gap_m=config.background_safe_gap_m,
                    times=times,
                    config=config,
                )
                if event is not None:
                    role_events.append(event)

        for other_role in range(NUM_VEHICLES):
            if other_role == target_role:
                continue
            event = build_interaction_event(
                scenario=scenario,
                seed=seed,
                rollout_step=rollout_step,
                simulation_time_s=float(getattr(env, "time", 0.0)),
                target_role=target_role,
                selected_mode_idx=mode_idx,
                source_type="teammate",
                source_id=f"role_{other_role}",
                source_role=other_role,
                source_metadata=None,
                target_world=target_world,
                other_world=dense_world[other_role],
                target_gap_dimensions=gap_dimensions,
                other_dimensions=gap_dimensions,
                safe_gap_m=config.platoon_safe_gap_m,
                times=times,
                config=config,
            )
            if event is not None:
                role_events.append(event)

        reconstructed_collision = any(
            event.collision for event in role_events
        )
        reconstructed_clearance = any(
            event.clearance_violation for event in role_events
        )

        reward_collision = bool(
            result.collision[target_role, mode_idx, 0]
        )
        reward_clearance = bool(
            result.clearance_violation[target_role, mode_idx, 0]
        )
        reward_offroad = bool(
            result.out_of_drivable[target_role, mode_idx, 0]
        )

        if reconstructed_collision != reward_collision:
            raise AssertionError(
                "collision attribution mismatch: "
                f"{scenario} seed={seed} step={rollout_step} "
                f"role={target_role}"
            )
        if reconstructed_clearance != reward_clearance:
            raise AssertionError(
                "clearance attribution mismatch: "
                f"{scenario} seed={seed} step={rollout_step} "
                f"role={target_role}"
            )
        if (
            reconstructed_collision
            or reconstructed_clearance
            or reward_offroad
        ) != reward_unsafe:
            raise AssertionError("unsafe attribution mismatch")

        background_collision = any(
            e.collision and e.source_type == "background"
            for e in role_events
        )
        teammate_collision = any(
            e.collision and e.source_type == "teammate"
            for e in role_events
        )
        background_clearance = any(
            e.clearance_violation and e.source_type == "background"
            for e in role_events
        )
        teammate_clearance = any(
            e.clearance_violation and e.source_type == "teammate"
            for e in role_events
        )

        unsafe_records.append(
            UnsafeRoleDiagnostic(
                scenario=scenario,
                seed=int(seed),
                rollout_step=int(rollout_step),
                simulation_time_s=float(getattr(env, "time", 0.0)),
                target_role=int(target_role),
                selected_mode_idx=mode_idx,
                reward=float(
                    result.rewards[target_role, mode_idx, 0]
                ),
                collision=reward_collision,
                clearance_violation=reward_clearance,
                out_of_drivable=reward_offroad,
                background_collision=background_collision,
                teammate_collision=teammate_collision,
                background_clearance=background_clearance,
                teammate_clearance=teammate_clearance,
                trigger_event_count=len(role_events),
            )
        )
        events.extend(role_events)

    return unsafe_records, events


def _run_seed(
    *,
    scenario: str,
    seed: int,
    steps: int,
    group_action: int,
    config: TrajectoryModeRewardConfig,
):
    env = SCENARIOS[scenario](
        config={
            "show_trajectories": False,
            "show_future_trajectories": False,
        },
        render_mode=None,
    )
    unsafe_records = []
    events = []
    states = 0

    try:
        env.reset(seed=int(seed))
        _assert_config(env)
        for rollout_step in range(int(steps)):
            env.action_type.act(int(group_action))
            state_unsafe, state_events = _diagnose_state(
                env,
                scenario=scenario,
                seed=seed,
                rollout_step=rollout_step,
                config=config,
            )
            unsafe_records.extend(state_unsafe)
            events.extend(state_events)
            states += 1

            _advance_without_replanning(env)
            if env._is_terminated() or env._is_truncated():
                break
    finally:
        env.close()

    return unsafe_records, events, states


def _print_merge_in_collisions(events: list[InteractionEvent]) -> None:
    collisions = [
        event
        for event in events
        if event.scenario == "merge_in" and event.collision
    ]
    print()
    print("merge_in collision interaction events")
    print(
        "seed step role source      other         "
        "first_t  min_gap  gap_t   min_ttc  ttc_t"
    )
    print("-" * 88)
    if not collisions:
        print("(none)")
        return

    for event in collisions:
        first_t = (
            "-"
            if event.first_collision_time_s is None
            else f"{event.first_collision_time_s:.2f}"
        )
        print(
            f"{event.seed:4d} "
            f"{event.rollout_step:4d} "
            f"{event.target_role:4d} "
            f"{event.source_type:10s} "
            f"{event.source_id:12s} "
            f"{first_t:>7s} "
            f"{event.minimum_gap_m:8.3f} "
            f"{event.minimum_gap_time_s:6.2f} "
            f"{event.minimum_ttc_s:8.3f} "
            f"{event.minimum_ttc_time_s:6.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=sorted(SCENARIOS),
        default=list(SCENARIOS),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--group-action", type=int, default=3)
    parser.add_argument(
        "--json-out",
        default=(
            "infos/reward_validation/"
            "interaction_diagnostics.json"
        ),
    )
    parser.add_argument(
        "--markdown-out",
        default=(
            "infos/reward_validation/"
            "INTERACTION_DIAGNOSTICS.md"
        ),
    )
    args = parser.parse_args()

    if args.seed_count <= 0 or args.steps <= 0:
        raise SystemExit("--seed-count and --steps must be positive")
    if args.group_action not in (0, 1, 2, 3):
        raise SystemExit("--group-action must be 0,1,2,3")

    seeds = list(
        range(
            args.seed_start,
            args.seed_start + args.seed_count,
        )
    )
    config = TrajectoryModeRewardConfig()
    unsafe_records = []
    events = []
    state_count = 0

    for scenario in args.scenarios:
        before_unsafe = len(unsafe_records)
        before_events = len(events)
        scenario_states = 0

        for seed in seeds:
            seed_unsafe, seed_events, states = _run_seed(
                scenario=scenario,
                seed=seed,
                steps=args.steps,
                group_action=args.group_action,
                config=config,
            )
            unsafe_records.extend(seed_unsafe)
            events.extend(seed_events)
            state_count += states
            scenario_states += states

        summary = summarize_diagnostics(
            unsafe_records[before_unsafe:],
            events[before_events:],
        )
        print(
            f"[OK] {scenario}: states={scenario_states}, "
            f"unsafe_roles={summary['unsafe_role_count']}, "
            f"collision_roles={summary['collision_role_count']}, "
            f"bg_collision={summary['background_collision_role_count']}, "
            f"teammate_collision="
            f"{summary['teammate_collision_role_count']}, "
            f"bg_clearance={summary['background_clearance_role_count']}, "
            f"teammate_clearance="
            f"{summary['teammate_clearance_role_count']}"
        )

    overall = summarize_diagnostics(unsafe_records, events)
    _print_merge_in_collisions(events)

    write_diagnostic_reports(
        unsafe_records=unsafe_records,
        events=events,
        state_count=state_count,
        seeds=seeds,
        requested_steps=args.steps,
        json_path=Path(args.json_out),
        markdown_path=Path(args.markdown_out),
    )

    print()
    print("[OK] interaction diagnostics passed")
    print(f"[OK] planning states: {state_count}")
    print(f"[OK] unsafe roles: {overall['unsafe_role_count']}")
    print(f"[OK] collision roles: {overall['collision_role_count']}")
    print(
        "[OK] collision source roles background/teammate: "
        f"{overall['background_collision_role_count']} / "
        f"{overall['teammate_collision_role_count']}"
    )
    print(
        "[OK] clearance source roles background/teammate: "
        f"{overall['background_clearance_role_count']} / "
        f"{overall['teammate_clearance_role_count']}"
    )
    print(
        f"[OK] t=0 clearance roles: "
        f"{overall['t0_clearance_role_count']}"
    )
    print(f"[OK] JSON: {args.json_out}")
    print(f"[OK] Markdown: {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
