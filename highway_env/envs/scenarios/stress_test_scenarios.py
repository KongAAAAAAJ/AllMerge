from __future__ import annotations

"""
Batch stress test for AllMerge scenario environments.

Default:
    - 4 scenarios
    - 100 seeds per scenario
    - headless rollout
    - Rule + Polynomial
    - group_action=3
    - 5 s maximum duration
    - CSV output with every episode

Recommended:
    python highway_env/envs/scenarios/stress_test_scenarios.py

Examples:
    python highway_env/envs/scenarios/stress_test_scenarios.py --episodes 100 --seed-start 0
    python highway_env/envs/scenarios/stress_test_scenarios.py --scenario merge_in --episodes 500
    python highway_env/envs/scenarios/stress_test_scenarios.py --no-csv
"""

import argparse
import csv
import math
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from highway_env.envs.scenarios.base_env import BaseScenarioEnv
from highway_env.envs.scenarios.curved_lane_change_env import CurvedLaneChangeEnv
from highway_env.envs.scenarios.merge_in_env import MergeInEnv
from highway_env.envs.scenarios.merge_out_env import MergeOutEnv
from highway_env.envs.scenarios.straight_lane_change_env import StraightLaneChangeEnv


SCENARIOS: Dict[str, Type[BaseScenarioEnv]] = {
    "straight": StraightLaneChangeEnv,
    "curved": CurvedLaneChangeEnv,
    "merge_in": MergeInEnv,
    "merge_out": MergeOutEnv,
}

OUTCOMES = (
    "success",
    "collision",
    "offroad",
    "timeout",
    "terminated_unknown",
    "error",
)


@dataclass
class EpisodeResult:
    scenario: str
    seed: int
    outcome: str
    steps: int
    sim_time: float
    terminated: bool
    truncated: bool
    task_success: bool
    leader_spawn_s: Optional[float] = None
    leader_speed: Optional[float] = None
    error: Optional[str] = None


def _check_reset_state(env: BaseScenarioEnv) -> None:
    if env.road is None:
        raise RuntimeError("reset finished but env.road is None")

    expected = int(env.config["controlled_vehicles"])
    actual = len(env.controlled_vehicles)
    if actual != expected:
        raise RuntimeError(
            f"controlled vehicle count mismatch: expected={expected}, actual={actual}"
        )

    for ego_idx, vehicle in enumerate(env.controlled_vehicles):
        if getattr(vehicle, "lane_index", None) is None:
            raise RuntimeError(f"ego[{ego_idx}] has no lane_index")
        if bool(getattr(vehicle, "crashed", False)):
            raise RuntimeError(f"ego[{ego_idx}] crashed immediately after reset")
        if not bool(getattr(vehicle, "on_road", True)):
            raise RuntimeError(f"ego[{ego_idx}] off-road immediately after reset")


def _check_polynomial_output(env: BaseScenarioEnv) -> None:
    planner_cfg = env.config.get("Planner", {})
    if not planner_cfg.get("state", False):
        raise RuntimeError("Planner.state is False")
    if planner_cfg.get("type") != "Polynomial":
        raise RuntimeError(
            f"expected Polynomial planner, got {planner_cfg.get('type')!r}"
        )

    missing = []
    for ego_idx, vehicle in enumerate(env.controlled_vehicles):
        if getattr(vehicle, "latest_planner_trajectory", None) is None:
            missing.append(ego_idx)
    if missing:
        raise RuntimeError(
            "latest_planner_trajectory missing for ego indices "
            f"{missing}"
        )


def _classify_episode(
    env: BaseScenarioEnv,
    info: dict,
    *,
    terminated: bool,
    truncated: bool,
    exhausted_steps: bool,
) -> str:
    # Safety failures intentionally take precedence over task_success.
    if any(bool(v.crashed) for v in env.controlled_vehicles):
        return "collision"

    if any(not bool(v.on_road) for v in env.controlled_vehicles):
        return "offroad"

    if bool(info.get("task_success", False)):
        return "success"

    if truncated or exhausted_steps:
        return "timeout"

    if terminated:
        return "terminated_unknown"

    return "timeout"


def run_episode(
    scenario_name: str,
    env_cls: Type[BaseScenarioEnv],
    *,
    seed: int,
    duration_s: float,
    group_action: int,
) -> EpisodeResult:
    env: Optional[BaseScenarioEnv] = None
    steps = 0
    terminated = False
    truncated = False
    info: dict = {}

    try:
        env = env_cls(
            config={
                "duration": float(duration_s),
                "offscreen_rendering": False,
            },
            render_mode=None,
        )

        _, info = env.reset(seed=int(seed))
        _check_reset_state(env)

        leader_spawn_s = getattr(env, "_platoon_leader_s", None)
        leader_speed = getattr(env, "_platoon_leader_speed", None)

        policy_frequency = float(env.config["policy_frequency"])
        max_steps = int(math.ceil(float(duration_s) * policy_frequency))
        first_step_checked = False

        for step_idx in range(max_steps):
            _, _, terminated, truncated, info = env.step(group_action)
            steps = step_idx + 1

            if not first_step_checked:
                _check_polynomial_output(env)
                first_step_checked = True

            if terminated or truncated:
                break

        if not first_step_checked:
            raise RuntimeError("rollout ended before first planner step")

        exhausted_steps = (
            steps >= max_steps and not terminated and not truncated
        )
        outcome = _classify_episode(
            env,
            info,
            terminated=bool(terminated),
            truncated=bool(truncated),
            exhausted_steps=exhausted_steps,
        )

        return EpisodeResult(
            scenario=scenario_name,
            seed=int(seed),
            outcome=outcome,
            steps=int(steps),
            sim_time=float(env.time),
            terminated=bool(terminated),
            truncated=bool(truncated),
            task_success=bool(info.get("task_success", False)),
            leader_spawn_s=(
                None if leader_spawn_s is None else float(leader_spawn_s)
            ),
            leader_speed=(
                None if leader_speed is None else float(leader_speed)
            ),
        )

    except Exception as exc:
        sim_time = float(env.time) if env is not None else 0.0
        return EpisodeResult(
            scenario=scenario_name,
            seed=int(seed),
            outcome="error",
            steps=int(steps),
            sim_time=sim_time,
            terminated=bool(terminated),
            truncated=bool(truncated),
            task_success=False,
            leader_spawn_s=(
                None
                if env is None or not hasattr(env, "_platoon_leader_s")
                else float(env._platoon_leader_s)
            ),
            leader_speed=(
                None
                if env is None or not hasattr(env, "_platoon_leader_speed")
                else float(env._platoon_leader_speed)
            ),
            error=f"{type(exc).__name__}: {exc}",
        )

    finally:
        if env is not None:
            env.close()


def _rate(count: int, total: int) -> float:
    return 100.0 * float(count) / float(total) if total > 0 else 0.0


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _percentile(values: List[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def _format_float(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.3f}"


def _seed_preview(results: List[EpisodeResult], outcome: str) -> str:
    seeds = [r.seed for r in results if r.outcome == outcome]
    if not seeds:
        return "-"
    shown = seeds[:20]
    text = ",".join(str(seed) for seed in shown)
    if len(seeds) > len(shown):
        text += f",...(+{len(seeds) - len(shown)})"
    return text


def print_scenario_summary(
    scenario_name: str,
    results: List[EpisodeResult],
) -> None:
    counts = Counter(result.outcome for result in results)
    total = len(results)

    success_times = [
        result.sim_time
        for result in results
        if result.outcome == "success"
    ]

    print("\n" + "=" * 100)
    print(f"STRESS SUMMARY | scenario={scenario_name} | episodes={total}")
    print("=" * 100)

    for outcome in OUTCOMES:
        count = counts.get(outcome, 0)
        print(
            f"{outcome:18s}: "
            f"{count:5d}/{total:<5d} "
            f"({_rate(count, total):6.2f}%)"
        )

    print(
        "success completion : "
        f"mean={_format_float(_mean(success_times))}s | "
        f"p50={_format_float(_percentile(success_times, 50.0))}s | "
        f"p95={_format_float(_percentile(success_times, 95.0))}s"
    )

    for outcome in (
        "collision",
        "offroad",
        "timeout",
        "terminated_unknown",
        "error",
    ):
        if counts.get(outcome, 0) > 0:
            print(
                f"{outcome:18s} seeds: "
                f"{_seed_preview(results, outcome)}"
            )


def write_csv(path: Path, results: List[EpisodeResult]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(EpisodeResult.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    print(f"\nCSV saved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch random-seed stress test for the four AllMerge scenario "
            "environments."
        )
    )
    parser.add_argument(
        "--scenario",
        choices=["all", *SCENARIOS.keys()],
        default="all",
        help="scenario to run; default: all",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="number of episodes per selected scenario; default: 100",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="first reset seed; default: 0",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="seed increment between episodes; default: 1",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="maximum rollout duration in seconds; default: 5.0",
    )
    parser.add_argument(
        "--group-action",
        type=int,
        choices=[0, 1, 2, 3],
        default=3,
        help="platoon group action; default: 3",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="print progress every N episodes per scenario; default: 10",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("scenario_stress_results.csv"),
        help=(
            "episode-level CSV output path; default: "
            "scenario_stress_results.csv"
        ),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="do not write the episode-level CSV",
    )
    parser.add_argument(
        "--trace-errors",
        action="store_true",
        help="print Python traceback for episodes classified as error",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.seed_step == 0:
        raise ValueError("--seed-step must be non-zero")
    if args.duration <= 0:
        raise ValueError("--duration must be > 0")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be > 0")

    if args.scenario == "all":
        selected = list(SCENARIOS.items())
    else:
        selected = [(args.scenario, SCENARIOS[args.scenario])]

    seeds = [
        int(args.seed_start + episode_index * args.seed_step)
        for episode_index in range(args.episodes)
    ]

    print("=" * 100)
    print("SCENARIO RANDOM-SEED STRESS TEST")
    print("=" * 100)
    print(
        f"scenarios={[name for name, _ in selected]} | "
        f"episodes_per_scenario={args.episodes} | "
        f"seed_start={args.seed_start} | "
        f"seed_step={args.seed_step} | "
        f"duration={args.duration:.2f}s | "
        f"group_action={args.group_action}"
    )
    print(
        "classification priority: "
        "collision > offroad > success > timeout > terminated_unknown"
    )

    all_results: List[EpisodeResult] = []

    for scenario_name, env_cls in selected:
        print("\n" + "-" * 100)
        print(f"[RUN] scenario={scenario_name}")
        print("-" * 100)

        scenario_results: List[EpisodeResult] = []

        for episode_index, seed in enumerate(seeds, start=1):
            result = run_episode(
                scenario_name,
                env_cls,
                seed=seed,
                duration_s=args.duration,
                group_action=args.group_action,
            )
            scenario_results.append(result)
            all_results.append(result)

            if result.outcome == "error" and args.trace_errors:
                print(
                    f"[ERROR] scenario={scenario_name} "
                    f"seed={seed}: {result.error}"
                )

            if (
                episode_index == 1
                or episode_index % args.progress_interval == 0
                or episode_index == args.episodes
            ):
                counts = Counter(
                    item.outcome for item in scenario_results
                )
                print(
                    f"[{episode_index:4d}/{args.episodes}] "
                    f"success={counts.get('success', 0)} "
                    f"collision={counts.get('collision', 0)} "
                    f"offroad={counts.get('offroad', 0)} "
                    f"timeout={counts.get('timeout', 0)} "
                    f"unknown={counts.get('terminated_unknown', 0)} "
                    f"error={counts.get('error', 0)}"
                )

        print_scenario_summary(scenario_name, scenario_results)

    print("\n" + "=" * 100)
    print("GLOBAL STRESS TEST SUMMARY")
    print("=" * 100)

    global_counts = Counter(result.outcome for result in all_results)
    total = len(all_results)

    for outcome in OUTCOMES:
        count = global_counts.get(outcome, 0)
        print(
            f"{outcome:18s}: "
            f"{count:5d}/{total:<5d} "
            f"({_rate(count, total):6.2f}%)"
        )

    print("\nPer-scenario success rate:")
    grouped: Dict[str, List[EpisodeResult]] = defaultdict(list)
    for result in all_results:
        grouped[result.scenario].append(result)

    for scenario_name, _ in selected:
        scenario_results = grouped[scenario_name]
        success_count = sum(
            result.outcome == "success"
            for result in scenario_results
        )
        success_times = [
            result.sim_time
            for result in scenario_results
            if result.outcome == "success"
        ]
        print(
            f"  {scenario_name:12s}: "
            f"{success_count:4d}/{len(scenario_results):<4d} "
            f"({_rate(success_count, len(scenario_results)):6.2f}%) | "
            f"mean_success_time="
            f"{_format_float(_mean(success_times))}s"
        )

    if not args.no_csv:
        write_csv(args.csv, all_results)

    non_success = [
        result for result in all_results
        if result.outcome != "success"
    ]

    if non_success:
        print(
            f"\nStress test completed with "
            f"{len(non_success)} non-success episode(s)."
        )
        print(
            "Use the reported seed(s) to reproduce individual failures with "
            "test_scenarios.py --seed <seed>."
        )
        return 1

    print("\nAll stress-test episodes succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
