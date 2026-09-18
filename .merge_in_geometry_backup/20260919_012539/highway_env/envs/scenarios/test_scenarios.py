from __future__ import annotations

"""
Smoke test for the new scenario environments.
用于测试scenarios场景

Default behavior:
    - Run all four scenario environments
    - Explicit reset
    - Render every step
    - Roll out for up to 5 seconds at 10 Hz
    - Use Rule + Polynomial through the normal env.step(...) path
    - Print per-scenario diagnostics
    - Exit with non-zero status if any scenario fails

Recommended invocation from the AllMerge project root:

    python -m highway_env.envs.scenarios.smoke_test

Examples:

    # Run only the straight lane-change scene
    python -m highway_env.envs.scenarios.smoke_test --scenario straight

    # Headless render smoke test
    python -m highway_env.envs.scenarios.smoke_test --render-mode rgb_array

    # Exercise another platoon group action
    python -m highway_env.envs.scenarios.smoke_test --group-action 2
"""

import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np


# Allow both:
#   python -m highway_env.envs.scenarios.smoke_test
# and:
#   python highway_env/envs/scenarios/smoke_test.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from highway_env.envs.scenarios.base_env import BaseScenarioEnv
from highway_env.envs.scenarios.curved_lane_change_env import CurvedLaneChangeEnv
from highway_env.envs.scenarios.merge_in_env import MergeInEnv
from highway_env.envs.scenarios.merge_out_env import MergeOutEnv
from highway_env.envs.scenarios.straight_lane_change_env import (
    StraightLaneChangeEnv,
)


SCENARIOS: Dict[str, Type[BaseScenarioEnv]] = {
    "straight": StraightLaneChangeEnv,
    "curved": CurvedLaneChangeEnv,
    "merge_in": MergeInEnv,
    "merge_out": MergeOutEnv,
}


@dataclass
class SmokeResult:
    name: str
    passed: bool
    steps: int = 0
    sim_time: float = 0.0
    terminated: bool = False
    truncated: bool = False
    task_success: bool = False
    error: Optional[str] = None


def _lane_indices(env: BaseScenarioEnv) -> List[tuple]:
    return [tuple(v.lane_index) for v in env.controlled_vehicles]


def _positions(env: BaseScenarioEnv) -> List[List[float]]:
    return [
        [round(float(v.position[0]), 3), round(float(v.position[1]), 3)]
        for v in env.controlled_vehicles
    ]


def _speeds(env: BaseScenarioEnv) -> List[float]:
    return [round(float(v.speed), 3) for v in env.controlled_vehicles]

def _rule_target_lane_indices(env: BaseScenarioEnv) -> List[Optional[tuple]]:
    """Current target_lane_index stored on the controlled vehicles."""
    result: List[Optional[tuple]] = []
    for vehicle in env.controlled_vehicles:
        lane_index = getattr(vehicle, "target_lane_index", None)
        result.append(tuple(lane_index) if lane_index is not None else None)
    return result


def _scenario_target_lane_indices(env: BaseScenarioEnv) -> List[Optional[tuple]]:
    """Resolved scenario targets before the Rule/MOBIL safety gate."""
    # SCENARIO TARGET LANE V1: diagnostics.
    result: List[Optional[tuple]] = []
    for vehicle in env.controlled_vehicles:
        target = env.scenario_target_lane_index(vehicle)
        result.append(tuple(target) if target is not None else None)
    return result



def _check_reset_state(env: BaseScenarioEnv) -> None:
    """Fail early on obvious scene-construction errors."""
    if env.road is None:
        raise RuntimeError("reset finished but env.road is None")

    expected = int(env.config["controlled_vehicles"])
    actual = len(env.controlled_vehicles)
    if actual != expected:
        raise RuntimeError(
            f"controlled vehicle count mismatch: expected={expected}, actual={actual}"
        )

    if len(env.road.vehicles) < actual:
        raise RuntimeError(
            "road.vehicles contains fewer vehicles than controlled_vehicles"
        )

    for ego_idx, vehicle in enumerate(env.controlled_vehicles):
        if vehicle.lane_index is None:
            raise RuntimeError(f"ego[{ego_idx}] has no lane_index")
        # Verify that the referenced lane actually exists.
        env.road.network.get_lane(vehicle.lane_index)

        if vehicle.crashed:
            raise RuntimeError(f"ego[{ego_idx}] is crashed immediately after reset")
        if not vehicle.on_road:
            raise RuntimeError(f"ego[{ego_idx}] is off-road immediately after reset")


def _check_polynomial_output(env: BaseScenarioEnv) -> None:
    """
    After the first env.step(), verify that the normal
    MultiAgentAction -> Rule -> Polynomial path produced planner trajectories.
    """
    planner_cfg = env.config.get("Planner", {})
    if not planner_cfg.get("state", False):
        raise RuntimeError("Planner.state is False; Polynomial was not exercised")

    if planner_cfg.get("type") != "Polynomial":
        raise RuntimeError(
            f"expected Polynomial planner, got {planner_cfg.get('type')!r}"
        )

    missing = []
    for ego_idx, vehicle in enumerate(env.controlled_vehicles):
        trajectory = getattr(vehicle, "latest_planner_trajectory", None)
        if trajectory is None:
            missing.append(ego_idx)

    if missing:
        raise RuntimeError(
            "Polynomial step completed but latest_planner_trajectory is missing "
            f"for ego indices {missing}"
        )


def _render_once(env: BaseScenarioEnv, render_mode: Optional[str]) -> None:
    if render_mode is None:
        return

    frame = env.render()

    if render_mode == "rgb_array":
        if frame is None:
            raise RuntimeError("rgb_array render returned None")
        frame = np.asarray(frame)
        if frame.ndim != 3:
            raise RuntimeError(
                f"unexpected rgb_array frame shape: {frame.shape}"
            )


def run_one_scenario(
    name: str,
    env_cls: Type[BaseScenarioEnv],
    *,
    seed: int,
    duration_s: float,
    group_action: int,
    render_mode: Optional[str],
    print_interval: int,
) -> SmokeResult:
    env: Optional[BaseScenarioEnv] = None

    try:
        print("\n" + "=" * 100)
        print(f"[SMOKE] scenario={name}")
        print("=" * 100)

        env = env_cls(
            config={
                "duration": float(duration_s),
                "offscreen_rendering": render_mode == "rgb_array",
            },
            render_mode=render_mode,
        )

        # Explicit reset is intentional: this test is specifically checking
        # reset -> render -> rollout behavior.
        obs, info = env.reset(seed=seed)
        _check_reset_state(env)
        _render_once(env, render_mode)

        policy_frequency = float(env.config["policy_frequency"])
        max_steps = int(np.ceil(duration_s * policy_frequency))

        print(
            f"reset OK | scenario={info.get('scenario_name')} "
            f"| maneuver={info.get('maneuver')}"
        )
        print(f"controlled={len(env.controlled_vehicles)}")
        print(f"road_vehicles={len(env.road.vehicles)}")
        print(f"lane_index={_lane_indices(env)}")
        print(
            "scenario_target_lane="
            f"{_scenario_target_lane_indices(env)}"
        )
        print(f"position={_positions(env)}")
        print(f"speed={_speeds(env)}")
        print(
            "planner="
            f"{env.config['Planner'].get('type')} "
            f"| polynomial_mode="
            f"{env.config['Planner'].get('Polynomial', {}).get('mode')}"
        )
        print(
            f"rollout target={duration_s:.2f}s "
            f"| policy_frequency={policy_frequency:.1f}Hz "
            f"| max_steps={max_steps} "
            f"| group_action={group_action}"
        )

        terminated = False
        truncated = False
        first_step_checked = False
        steps = 0

        for step_idx in range(max_steps):
            obs, reward, terminated, truncated, info = env.step(group_action)
            steps = step_idx + 1

            if not first_step_checked:
                _check_polynomial_output(env)
                first_step_checked = True
                print("Rule + Polynomial first-step check: OK")

            _render_once(env, render_mode)

            if (
                steps == 1
                or steps % print_interval == 0
                or terminated
                or truncated
            ):
                print(
                    f"step={steps:02d} "
                    f"time={env.time:.2f}s "
                    f"terminated={terminated} "
                    f"truncated={truncated} "
                    f"task_success={info.get('task_success')} "
                    f"lanes={_lane_indices(env)} "
                    f"rule_targets={_rule_target_lane_indices(env)} "
                    f"speed={_speeds(env)}"
                )

            if terminated or truncated:
                break

        if not first_step_checked:
            raise RuntimeError("rollout ended before the first planner step")

        result = SmokeResult(
            name=name,
            passed=True,
            steps=steps,
            sim_time=float(env.time),
            terminated=bool(terminated),
            truncated=bool(truncated),
            task_success=bool(info.get("task_success", False)),
        )

        print(
            f"[PASS] {name}: steps={result.steps}, "
            f"time={result.sim_time:.2f}s, "
            f"terminated={result.terminated}, "
            f"truncated={result.truncated}, "
            f"task_success={result.task_success}"
        )
        return result

    except Exception as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()

        sim_time = float(env.time) if env is not None else 0.0
        return SmokeResult(
            name=name,
            passed=False,
            sim_time=sim_time,
            error=f"{type(exc).__name__}: {exc}",
        )

    finally:
        if env is not None:
            env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="reset + render + 5 s rollout smoke test for scenario envs"
    )
    parser.add_argument(
        "--scenario",
        choices=["all", *SCENARIOS.keys()],
        default="all",
        help="scenario to run; default: all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="environment reset seed; default: 1",
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
        help=(
            "platoon group action sent to env.step(); default: 3. "
            "Use 3 first for stable environment smoke testing."
        ),
    )
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array", "none"],
        default="human",
        help="render mode; default: human",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=10,
        help="print rollout state every N steps; default: 10",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.duration <= 0:
        raise ValueError("--duration must be > 0")
    if args.print_interval <= 0:
        raise ValueError("--print-interval must be > 0")

    render_mode = None if args.render_mode == "none" else args.render_mode

    if args.scenario == "all":
        selected = list(SCENARIOS.items())
    else:
        selected = [(args.scenario, SCENARIOS[args.scenario])]

    results = [
        run_one_scenario(
            name,
            env_cls,
            seed=args.seed,
            duration_s=args.duration,
            group_action=args.group_action,
            render_mode=render_mode,
            print_interval=args.print_interval,
        )
        for name, env_cls in selected
    ]

    print("\n" + "=" * 100)
    print("SMOKE TEST SUMMARY")
    print("=" * 100)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        details = (
            f"steps={result.steps}, time={result.sim_time:.2f}s, "
            f"terminated={result.terminated}, truncated={result.truncated}, "
            f"task_success={result.task_success}"
            if result.passed
            else f"error={result.error}"
        )
        print(f"{status:4s} | {result.name:12s} | {details}")

    failed = [result for result in results if not result.passed]
    if failed:
        print(f"\n{len(failed)} scenario(s) failed.")
        return 1

    print("\nAll selected scenario smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
