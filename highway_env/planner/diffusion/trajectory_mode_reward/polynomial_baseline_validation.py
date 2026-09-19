"""Real Polynomial trajectory reward-distribution validation.

Evaluates aligned Polynomial outputs at the same simulator state at which they
were planned. Reward formulas and GRPO semantics are unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from highway_env.envs.scenarios import (
    CurvedLaneChangeEnv,
    MergeInEnv,
    MergeOutEnv,
    StraightLaneChangeEnv,
)
from highway_env.planner.trajectory import build_planner_time_grid

from .config import TrajectoryModeRewardConfig
from .constants import HORIZON_STEPS, NUM_MODES, NUM_VEHICLES
from .evaluator import evaluate_candidates
from .polynomial_baseline_report import (
    PolynomialRoleRecord,
    aggregate,
    summary_by_scenario,
    write_reports,
)


SCENARIOS = {
    "straight_lane_change": StraightLaneChangeEnv,
    "curved_lane_change": CurvedLaneChangeEnv,
    "merge_in": MergeInEnv,
    "merge_out": MergeOutEnv,
}


@dataclass(frozen=True)
class StateValidation:
    role_records: tuple[PolynomialRoleRecord, ...]
    reward_path_max_abs_error: float


def _float_array(
    value: Any,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise AssertionError(
            f"{name}: expected {shape}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise AssertionError(f"{name}: contains NaN/Inf")
    return np.ascontiguousarray(array)


def _extract_alignment(env: object):
    alignment = getattr(env, "latest_expert_alignment", None)
    if not isinstance(alignment, dict):
        raise AssertionError(
            "Polynomial planner did not publish latest_expert_alignment"
        )

    expert_xy = _float_array(
        alignment.get("expert_trajectory_xy"),
        (NUM_VEHICLES, HORIZON_STEPS, 2),
        "expert_trajectory_xy",
    )
    time_s = _float_array(
        alignment.get("trajectory_time_s"),
        (HORIZON_STEPS,),
        "trajectory_time_s",
    )
    expected_time = build_planner_time_grid(
        horizon_steps=HORIZON_STEPS,
        trajectory_dt=0.5,
    )
    if not np.allclose(
        time_s,
        expected_time,
        atol=1e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "Polynomial time grid is not [0.5,...,4.0]"
        )

    source_valid = np.asarray(
        alignment.get("mode_valid_mask"),
        dtype=np.bool_,
    )
    selected_idx = np.asarray(
        alignment.get("nearest_mode_idx"),
        dtype=np.int64,
    )
    semantic_match = np.asarray(
        alignment.get("semantic_match"),
        dtype=np.bool_,
    )
    selected_name = tuple(
        str(v)
        for v in alignment.get("nearest_mode_name", ())
    )
    expected_semantic = tuple(
        str(v)
        for v in alignment.get("expected_semantic_group", ())
    )
    selected_semantic = tuple(
        str(v)
        for v in alignment.get("nearest_semantic_group", ())
    )

    if source_valid.shape != (NUM_VEHICLES, NUM_MODES):
        raise AssertionError("mode_valid_mask must be [3,10]")
    if selected_idx.shape != (NUM_VEHICLES,):
        raise AssertionError("nearest_mode_idx must be [3]")
    if semantic_match.shape != (NUM_VEHICLES,):
        raise AssertionError("semantic_match must be [3]")
    for name, values in (
        ("nearest_mode_name", selected_name),
        ("expected_semantic_group", expected_semantic),
        ("nearest_semantic_group", selected_semantic),
    ):
        if len(values) != NUM_VEHICLES:
            raise AssertionError(f"{name} must have 3 entries")

    for role, mode_idx in enumerate(selected_idx.tolist()):
        if mode_idx < 0 or mode_idx >= NUM_MODES:
            raise AssertionError(
                f"role={role}: invalid nearest mode {mode_idx}"
            )
        if not source_valid[role, mode_idx]:
            raise AssertionError(
                f"role={role}: selected mode {mode_idx} is invalid"
            )

    return (
        expert_xy,
        selected_idx,
        selected_name,
        expected_semantic,
        selected_semantic,
        semantic_match,
    )


def _evaluate_state(
    env: object,
    scenario: str,
    seed: int,
    rollout_step: int,
    config: TrajectoryModeRewardConfig,
) -> StateValidation:
    (
        expert_xy,
        selected_idx,
        selected_name,
        expected_semantic,
        selected_semantic,
        semantic_match,
    ) = _extract_alignment(env)

    # nearest_mode_idx is only a trajectory_mode slot. The scored path is the
    # real Polynomial trajectory, never its anchor.
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
        candidates=all_modes[:, :, None, :, :],
        frozen_all_mode_trajectories=all_modes,
        frozen_argmax_joint_trajectories=expert_xy,
        valid_mode_mask=selected_valid,
        config=config,
    )

    records = []
    parity_errors = []
    for role in range(NUM_VEHICLES):
        mode_idx = int(selected_idx[role])
        reward = float(result.rewards[role, mode_idx, 0])
        reference_reward = float(
            result.pretrain_rewards[role, mode_idx]
        )
        parity_errors.append(abs(reward - reference_reward))

        c = {
            name: float(values[role, mode_idx, 0])
            for name, values in result.components.items()
        }
        records.append(
            PolynomialRoleRecord(
                scenario=scenario,
                seed=int(seed),
                rollout_step=int(rollout_step),
                simulation_time_s=float(getattr(env, "time", 0.0)),
                role=role,
                selected_mode_idx=mode_idx,
                selected_mode_name=selected_name[role],
                expected_semantic_group=expected_semantic[role],
                selected_semantic_group=selected_semantic[role],
                semantic_match=bool(semantic_match[role]),
                reward=reward,
                unsafe=bool(result.unsafe[role, mode_idx, 0]),
                collision=bool(result.collision[role, mode_idx, 0]),
                out_of_drivable=bool(
                    result.out_of_drivable[role, mode_idx, 0]
                ),
                clearance_violation=bool(
                    result.clearance_violation[role, mode_idx, 0]
                ),
                progress_score=c["progress_score"],
                gap_penalty=c["gap_penalty"],
                ttc_penalty=c["ttc_penalty"],
                road_penalty=c["road_penalty"],
                comfort_penalty=c["comfort_penalty"],
                minimum_background_gap_m=c[
                    "minimum_background_gap_m"
                ],
                minimum_teammate_gap_m=c[
                    "minimum_teammate_gap_m"
                ],
                minimum_road_margin_m=c["minimum_road_margin_m"],
                minimum_ttc_s=c["minimum_ttc_s"],
            )
        )

    max_error = float(max(parity_errors))
    if max_error > 1.0e-6:
        raise AssertionError(
            "candidate/reference reward path mismatch: "
            f"{max_error}"
        )
    return StateValidation(tuple(records), max_error)


def _advance_without_replanning(env: object) -> None:
    sim_hz = int(env.config["simulation_frequency"])
    policy_hz = int(env.config["policy_frequency"])
    if (
        sim_hz <= 0
        or policy_hz <= 0
        or sim_hz % policy_hz != 0
    ):
        raise AssertionError(
            "simulation_frequency/policy_frequency ratio invalid"
        )

    env.time += 1.0 / float(policy_hz)
    dt = 1.0 / float(sim_hz)
    for _ in range(sim_hz // policy_hz):
        # Polynomial controls were already installed by action_type.act().
        env.road.act()
        env.road.step(dt)
        env.steps += 1


def _assert_config(env: object) -> None:
    planner = env.config.get("Planner", {})
    if not planner.get("state", False):
        raise AssertionError("Planner.state must be True")
    if planner.get("type") != "Polynomial":
        raise AssertionError("Planner.type must be Polynomial")
    if planner.get("Polynomial", {}).get("mode") != "aligned":
        raise AssertionError(
            "Planner.Polynomial.mode must be aligned"
        )
    if not planner.get("ExpertAlignment", {}).get(
        "enabled",
        False,
    ):
        raise AssertionError("ExpertAlignment must be enabled")


def _run_seed(
    scenario_name: str,
    seed: int,
    steps: int,
    group_action: int,
    config: TrajectoryModeRewardConfig,
):
    env = SCENARIOS[scenario_name](
        config={
            "show_trajectories": False,
            "show_future_trajectories": False,
        },
        render_mode=None,
    )
    records = []
    states = 0
    max_error = 0.0

    try:
        env.reset(seed=int(seed))
        _assert_config(env)

        for rollout_step in range(int(steps)):
            # Real Polynomial planning + alignment; no road.step yet.
            env.action_type.act(int(group_action))

            state_result = _evaluate_state(
                env,
                scenario_name,
                seed,
                rollout_step,
                config,
            )
            records.extend(state_result.role_records)
            states += 1
            max_error = max(
                max_error,
                state_result.reward_path_max_abs_error,
            )

            _advance_without_replanning(env)
            if env._is_terminated() or env._is_truncated():
                break
    finally:
        env.close()

    return records, states, max_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=sorted(SCENARIOS),
        default=list(SCENARIOS),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--group-action", type=int, default=3)
    parser.add_argument(
        "--json-out",
        default=(
            "infos/reward_validation/"
            "polynomial_baseline.json"
        ),
    )
    parser.add_argument(
        "--markdown-out",
        default=(
            "infos/reward_validation/"
            "POLYNOMIAL_BASELINE_REPORT.md"
        ),
    )
    args = parser.parse_args()

    if args.seed_count <= 0 or args.steps <= 0:
        raise SystemExit(
            "--seed-count and --steps must be positive"
        )
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
    max_parity_error = 0.0

    for scenario_name in args.scenarios:
        begin = len(all_records)
        scenario_states = 0
        for seed in seeds:
            records, states, error = _run_seed(
                scenario_name,
                seed,
                args.steps,
                args.group_action,
                config,
            )
            all_records.extend(records)
            state_count += states
            scenario_states += states
            max_parity_error = max(max_parity_error, error)

        summary = aggregate(all_records[begin:])
        print(
            f"[OK] {scenario_name}: "
            f"states={scenario_states}, "
            f"samples={summary['sample_count']}, "
            f"R_mean={summary['reward']['mean']:.4f}, "
            f"R_p05={summary['reward']['p05']:.4f}, "
            f"unsafe={summary['rates']['unsafe']:.2%}, "
            f"collision={summary['rates']['collision']:.2%}, "
            f"offroad={summary['rates']['out_of_drivable']:.2%}"
        )

    overall = aggregate(all_records)
    by_scenario = summary_by_scenario(all_records)
    write_reports(
        records=all_records,
        overall=overall,
        by_scenario=by_scenario,
        state_count=state_count,
        max_parity_error=max_parity_error,
        requested_steps=args.steps,
        seeds=seeds,
        json_path=Path(args.json_out),
        markdown_path=Path(args.markdown_out),
    )

    print()
    print("[OK] Polynomial baseline reward validation passed")
    print(f"[OK] planning states: {state_count}")
    print(
        f"[OK] vehicle trajectory samples: "
        f"{overall['sample_count']}"
    )
    print(
        f"[OK] reward mean/std: "
        f"{overall['reward']['mean']:.6f} / "
        f"{overall['reward']['std']:.6f}"
    )
    print(
        f"[OK] reward p05/p50/p95: "
        f"{overall['reward']['p05']:.6f} / "
        f"{overall['reward']['p50']:.6f} / "
        f"{overall['reward']['p95']:.6f}"
    )
    print(
        f"[OK] unsafe/collision/offroad/clearance: "
        f"{overall['rates']['unsafe']:.2%} / "
        f"{overall['rates']['collision']:.2%} / "
        f"{overall['rates']['out_of_drivable']:.2%} / "
        f"{overall['rates']['clearance_violation']:.2%}"
    )
    print(
        f"[OK] candidate/reference parity max error: "
        f"{max_parity_error:.3e}"
    )
    print(f"[OK] JSON: {args.json_out}")
    print(f"[OK] Markdown: {args.markdown_out}")
    print(
        "[NEXT] inspect reward/component balance, "
        "then run standalone/GRPO parity."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
