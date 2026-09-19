"""Compare state_adapter background prediction with native AllMerge rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .background_prediction_report import (
    PredictionRecord,
    summarize,
    write_reports,
)
from .background_prediction_rollout import (
    compare_prediction_to_rollout,
    rollout_backgrounds,
)
from .config import TrajectoryModeRewardConfig
from .constants import NUM_VEHICLES
from .counterfactual import TrajectoryModeCounterfactualReward
from .interaction_diagnostics_geometry import dense_joint_world
from .polynomial_baseline_validation import (
    SCENARIOS,
    _advance_without_replanning,
    _assert_config,
    _extract_alignment,
)


ROLLOUT_MODE = (
    "deepcopy simulator; native background road.act()+road.step(); "
    "current Polynomial controls held; no planner replan"
)


def _state_records(
    env: object,
    *,
    scenario: str,
    seed: int,
    rollout_step: int,
    config: TrajectoryModeRewardConfig,
):
    (
        expert_xy,
        selected_idx,
        _selected_name,
        _expected_semantic,
        _selected_semantic,
        _semantic_match,
    ) = _extract_alignment(env)

    scorer = TrajectoryModeCounterfactualReward(config=config)
    context = scorer.build_geometry_context(
        env,
        expert_xy,
    )
    target_world, times = dense_joint_world(
        expert_xy,
        context.poses,
        config,
    )

    if not np.allclose(
        times,
        np.arange(len(times), dtype=np.float64)
        * config.interpolation_dt_s,
        atol=1e-9,
        rtol=0.0,
    ):
        raise AssertionError("unexpected reward dense time grid")

    actual = rollout_backgrounds(
        env,
        horizon_s=float(times[-1]),
        dt_s=config.interpolation_dt_s,
    )

    records = []
    for target_role in range(NUM_VEHICLES):
        background_map = context.backgrounds[target_role]

        for actor_id, branches in background_map.items():
            canonical = [
                branch
                for branch in branches
                if branch[0] == actor_id
            ]
            if len(canonical) != 1:
                raise AssertionError(
                    f"{actor_id}: expected one canonical prediction branch"
                )

            (
                _branch_name,
                predicted_background,
                background_dimensions,
            ) = canonical[0]

            if actor_id not in actual:
                raise AssertionError(
                    f"{actor_id}: missing from actual simulator rollout"
                )

            comparison = compare_prediction_to_rollout(
                target_role=target_role,
                actor_id=actor_id,
                target_world=target_world[target_role],
                predicted_background=predicted_background,
                actual_background=actual[actor_id],
                background_dimensions=background_dimensions,
                config=config,
            )
            records.append(
                PredictionRecord.from_comparison(
                    scenario=scenario,
                    seed=seed,
                    rollout_step=rollout_step,
                    simulation_time_s=float(
                        getattr(env, "time", 0.0)
                    ),
                    selected_mode_idx=int(
                        selected_idx[target_role]
                    ),
                    comparison=comparison,
                )
            )

    return records


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
    records = []
    states = 0

    try:
        env.reset(seed=int(seed))
        _assert_config(env)

        for rollout_step in range(int(steps)):
            env.action_type.act(int(group_action))

            records.extend(
                _state_records(
                    env,
                    scenario=scenario,
                    seed=seed,
                    rollout_step=rollout_step,
                    config=config,
                )
            )
            states += 1

            # Advance the LIVE validation episode only one policy interval,
            # exactly as Polynomial baseline validation did previously.
            _advance_without_replanning(env)
            if env._is_terminated() or env._is_truncated():
                break
    finally:
        env.close()

    return records, states


def _print_predicted_collisions(records: list[PredictionRecord]) -> None:
    collisions = [
        record
        for record in records
        if record.predicted_collision
    ]
    print()
    print("predicted collision fidelity")
    print(
        "scenario seed step role actor         pred_t actual actual_t "
        "pred_gap actual_gap end_err speed_err"
    )
    print("-" * 105)

    if not collisions:
        print("(none)")
        return

    for record in collisions:
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
        print(
            f"{record.scenario:12s} "
            f"{record.seed:4d} "
            f"{record.rollout_step:4d} "
            f"{record.target_role:4d} "
            f"{record.actor_id:12s} "
            f"{pred_t:>6s} "
            f"{str(record.actual_collision):>6s} "
            f"{actual_t:>8s} "
            f"{record.predicted_min_gap_m:8.3f} "
            f"{record.actual_min_gap_m:10.3f} "
            f"{record.endpoint_position_error_m:7.3f} "
            f"{record.endpoint_speed_error_mps:9.3f}"
        )


def _print_false_negatives(
    records: list[PredictionRecord],
) -> None:
    misses = [
        record
        for record in records
        if (
            record.actual_collision
            and not record.predicted_collision
        )
    ]
    print()
    print("predicted-safe -> actual-collision pairs")
    print(
        "scenario seed step role actor         actual_t "
        "pred_gap actual_gap pred_ttc actual_ttc end_err speed_err"
    )
    print("-" * 112)

    if not misses:
        print("(none)")
        return

    for record in misses:
        actual_t = (
            "-"
            if record.actual_first_collision_time_s is None
            else f"{record.actual_first_collision_time_s:.2f}"
        )
        print(
            f"{record.scenario:12s} "
            f"{record.seed:4d} "
            f"{record.rollout_step:4d} "
            f"{record.target_role:4d} "
            f"{record.actor_id:12s} "
            f"{actual_t:>8s} "
            f"{record.predicted_min_gap_m:8.3f} "
            f"{record.actual_min_gap_m:10.3f} "
            f"{record.predicted_min_ttc_s:8.3f} "
            f"{record.actual_min_ttc_s:10.3f} "
            f"{record.endpoint_position_error_m:7.3f} "
            f"{record.endpoint_speed_error_mps:9.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=sorted(SCENARIOS),
        default=["merge_in"],
        help="default: merge_in",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=2,
        help="default targets previous merge_in collision seed",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--group-action",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--json-out",
        default=(
            "infos/reward_validation/"
            "background_prediction_diagnostics.json"
        ),
    )
    parser.add_argument(
        "--markdown-out",
        default=(
            "infos/reward_validation/"
            "BACKGROUND_PREDICTION_DIAGNOSTICS.md"
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
    all_records = []
    state_count = 0

    for scenario in args.scenarios:
        start = len(all_records)
        scenario_states = 0

        for seed in seeds:
            records, states = _run_seed(
                scenario=scenario,
                seed=seed,
                steps=args.steps,
                group_action=args.group_action,
                config=config,
            )
            all_records.extend(records)
            state_count += states
            scenario_states += states

        summary = summarize(all_records[start:])
        print(
            f"[OK] {scenario}: "
            f"states={scenario_states}, "
            f"pairs={summary['pair_count']}, "
            f"pred_collision={summary['predicted_collision_count']}, "
            f"actual_collision={summary['actual_collision_count']}, "
            f"pred->actual_safe="
            f"{summary['predicted_collision_actual_false_count']}, "
            f"endpoint_err_p95="
            f"{summary['endpoint_position_error_m']['p95']:.3f}m"
        )

    overall = summarize(all_records)
    _print_predicted_collisions(all_records)
    _print_false_negatives(all_records)

    write_reports(
        records=all_records,
        state_count=state_count,
        rollout_mode=ROLLOUT_MODE,
        json_path=Path(args.json_out),
        markdown_path=Path(args.markdown_out),
    )

    print()
    print("[OK] background prediction diagnostics passed")
    print(f"[OK] planning states: {state_count}")
    print(f"[OK] target/background pairs: {overall['pair_count']}")
    print(
        "[OK] predicted/actual collision count: "
        f"{overall['predicted_collision_count']} / "
        f"{overall['actual_collision_count']}"
    )
    print(
        "[OK] predicted collision -> actual safe: "
        f"{overall['predicted_collision_actual_false_count']}"
    )
    print(
        "[OK] predicted safe -> actual collision: "
        f"{overall['actual_collision_prediction_false_count']}"
    )
    print(
        "[OK] collision agreement: "
        f"{overall['collision_agreement_rate']:.2%}"
    )
    print(
        "[OK] collision precision/recall: "
        f"{overall['collision_precision']:.2%} / "
        f"{overall['collision_recall']:.2%}"
    )
    print(
        "[OK] endpoint position error mean/p95/max: "
        f"{overall['endpoint_position_error_m']['mean']:.3f} / "
        f"{overall['endpoint_position_error_m']['p95']:.3f} / "
        f"{overall['endpoint_position_error_m']['max']:.3f} m"
    )
    print(
        "[OK] endpoint speed error mean/p95/max: "
        f"{overall['endpoint_speed_error_mps']['mean']:.3f} / "
        f"{overall['endpoint_speed_error_mps']['p95']:.3f} / "
        f"{overall['endpoint_speed_error_mps']['max']:.3f} m/s"
    )
    print(f"[OK] JSON: {args.json_out}")
    print(f"[OK] Markdown: {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
