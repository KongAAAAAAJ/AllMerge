"""Real AllMerge scenario validation for trajectory-mode reward.

Run:
    python -m highway_env.planner.diffusion.trajectory_mode_reward.real_env_validation

Scope:
- four real scenario classes
- real RoadNetwork
- real controlled/background vehicles
- shared evaluate_candidates(...)
- no change to reward formulas
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from highway_env.envs.scenarios import (
    CurvedLaneChangeEnv,
    MergeInEnv,
    MergeOutEnv,
    StraightLaneChangeEnv,
)

from .config import TrajectoryModeRewardConfig, TrajectoryModeRewardError
from .counterfactual import TrajectoryModeCounterfactualReward
from .evaluator import evaluate_candidates
from .constants import HORIZON_STEPS, NUM_MODES, NUM_VEHICLES


SCENARIOS = {
    "straight_lane_change": StraightLaneChangeEnv,
    "curved_lane_change": CurvedLaneChangeEnv,
    "merge_in": MergeInEnv,
    "merge_out": MergeOutEnv,
}

BASELINE_MODE = 0
LOW_PROGRESS_MODE = 1
OFFROAD_MODE = 2
ZIGZAG_MODE = 3


@dataclass(frozen=True)
class ValidationRecord:
    scenario: str
    seed: int
    background_vehicle_count: int
    baseline_reward_mean: float
    baseline_unsafe_count: int
    baseline_progress_mean: float
    low_progress_mean: float
    baseline_comfort_mean: float
    zigzag_comfort_mean: float
    baseline_road_penalty_mean: float
    offroad_road_penalty_mean: float
    offroad_detected_count: int
    baseline_min_background_gap_m: float
    baseline_min_teammate_gap_m: float
    baseline_min_ttc_s: float


def _vehicle_pose(vehicle: object) -> np.ndarray:
    position = np.asarray(
        getattr(vehicle, "position", ()),
        dtype=np.float64,
    ).reshape(-1)
    heading = float(getattr(vehicle, "heading", np.nan))
    if (
        position.size < 2
        or not np.isfinite(position[:2]).all()
        or not math.isfinite(heading)
    ):
        raise TrajectoryModeRewardError(
            "controlled vehicle must expose finite position and heading"
        )
    return np.asarray(
        [position[0], position[1], heading],
        dtype=np.float64,
    )


def _world_to_local(
    world_xy: np.ndarray,
    origin_pose: np.ndarray,
) -> np.ndarray:
    values = np.asarray(world_xy, dtype=np.float64)
    pose = np.asarray(origin_pose, dtype=np.float64).reshape(3)
    delta = values - pose[None, :2]
    c = math.cos(float(pose[2]))
    s = math.sin(float(pose[2]))
    local = np.empty_like(delta)
    local[:, 0] = c * delta[:, 0] + s * delta[:, 1]
    local[:, 1] = -s * delta[:, 0] + c * delta[:, 1]
    return local


def _lane_follow_reference(
    env: object,
    vehicle: object,
    *,
    dt_s: float,
) -> np.ndarray:
    lane_index = getattr(vehicle, "lane_index", None)
    if lane_index is None:
        raise TrajectoryModeRewardError(
            "vehicle.lane_index is required"
        )
    lane = env.road.network.get_lane(lane_index)
    pose = _vehicle_pose(vehicle)
    s0, lateral = lane.local_coordinates(pose[:2])
    speed = max(float(getattr(vehicle, "speed", 0.0)), 0.0)

    world_xy = np.empty(
        (HORIZON_STEPS, 2),
        dtype=np.float64,
    )
    for index in range(HORIZON_STEPS):
        time_s = (index + 1) * float(dt_s)
        future_s = float(s0) + speed * time_s
        future_s = float(
            np.clip(
                future_s,
                0.0,
                max(float(lane.length) - 1.0e-3, 0.0),
            )
        )
        world_xy[index] = lane.position(
            future_s,
            float(lateral),
        )
    return np.ascontiguousarray(
        _world_to_local(world_xy, pose),
        dtype=np.float32,
    )


def _build_validation_inputs(
    env: object,
    config: TrajectoryModeRewardConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    controlled = list(
        getattr(env, "controlled_vehicles", ()) or ()
    )
    if len(controlled) != NUM_VEHICLES:
        raise TrajectoryModeRewardError(
            f"expected {NUM_VEHICLES} controlled vehicles, "
            f"got {len(controlled)}"
        )

    all_modes = np.zeros(
        (
            NUM_VEHICLES,
            NUM_MODES,
            HORIZON_STEPS,
            2,
        ),
        dtype=np.float32,
    )

    for role, vehicle in enumerate(controlled):
        baseline = _lane_follow_reference(
            env,
            vehicle,
            dt_s=config.trajectory_dt_s,
        )
        all_modes[role] = baseline[None, :, :]

        # Mode 1: very low progress, kept directionally valid.
        all_modes[role, LOW_PROGRESS_MODE] = (
            baseline * np.float32(0.08)
        )

        # Mode 2: large progressive lateral displacement, guaranteed
        # to test real road-margin/out-of-drivable handling.
        offroad = baseline.copy()
        offroad[:, 1] += np.linspace(
            0.0,
            16.0,
            HORIZON_STEPS,
            dtype=np.float32,
        )
        all_modes[role, OFFROAD_MODE] = offroad

        # Mode 3: comfort stress. We only compare the comfort component,
        # not total reward, because real traffic interactions can dominate R.
        zigzag = baseline.copy()
        zigzag[:, 1] += (
            np.asarray(
                [1, -1, 1, -1, 1, -1, 1, -1],
                dtype=np.float32,
            )
            * np.float32(2.5)
        )
        all_modes[role, ZIGZAG_MODE] = zigzag

    frozen_argmax = np.ascontiguousarray(
        all_modes[:, BASELINE_MODE],
        dtype=np.float32,
    )
    valid = np.ones(
        (NUM_VEHICLES, NUM_MODES),
        dtype=np.bool_,
    )
    return all_modes, frozen_argmax, valid


def _validate_geometry_context(
    env: object,
    scorer: TrajectoryModeCounterfactualReward,
    frozen_argmax: np.ndarray,
) -> None:
    context = scorer.build_geometry_context(
        env,
        frozen_argmax,
    )
    if len(context.poses) != NUM_VEHICLES:
        raise AssertionError(
            f"expected {NUM_VEHICLES} poses, got {len(context.poses)}"
        )
    if len(context.backgrounds) != NUM_VEHICLES:
        raise AssertionError(
            "background context must exist for all target roles"
        )
    if context.road is not env.road:
        raise AssertionError(
            "reward context must retain the real scenario road object"
        )

    expected_background = len(
        list(getattr(env, "background_vehicles", ()) or ())
    )
    for role, actor_map in enumerate(context.backgrounds):
        if len(actor_map) != expected_background:
            raise AssertionError(
                f"role={role}: expected {expected_background} background "
                f"actors, got {len(actor_map)}"
            )
        for actor_name, branches in actor_map.items():
            if not branches:
                raise AssertionError(
                    f"role={role}, actor={actor_name}: no prediction branch"
                )
            for _, trajectory, dimensions in branches:
                trajectory = np.asarray(trajectory)
                dimensions = np.asarray(
                    dimensions,
                    dtype=np.float64,
                )
                if (
                    trajectory.ndim != 2
                    or trajectory.shape[1] != 3
                    or trajectory.shape[0] < 2
                    or not np.isfinite(trajectory).all()
                ):
                    raise AssertionError(
                        f"invalid background prediction for {actor_name}: "
                        f"{trajectory.shape}"
                    )
                if (
                    dimensions.shape != (2,)
                    or np.any(dimensions <= 0.0)
                    or not np.isfinite(dimensions).all()
                ):
                    raise AssertionError(
                        f"invalid background dimensions for {actor_name}"
                    )


def validate_one(
    scenario_name: str,
    seed: int,
) -> ValidationRecord:
    env_class = SCENARIOS[scenario_name]
    env = env_class(
        config={
            "show_trajectories": False,
            "show_future_trajectories": False,
        },
        render_mode=None,
    )

    try:
        env.reset(seed=int(seed))
        controlled = list(
            getattr(env, "controlled_vehicles", ()) or ()
        )
        background = list(
            getattr(env, "background_vehicles", ()) or ()
        )
        if len(controlled) != NUM_VEHICLES:
            raise AssertionError(
                f"{scenario_name}: expected 3 controlled vehicles"
            )
        if not background:
            raise AssertionError(
                f"{scenario_name}: expected real background traffic"
            )

        config = TrajectoryModeRewardConfig()
        frozen_all, frozen_argmax, valid = (
            _build_validation_inputs(
                env,
                config,
            )
        )

        scorer = TrajectoryModeCounterfactualReward(
            config=config
        )
        _validate_geometry_context(
            env,
            scorer,
            frozen_argmax,
        )

        # N=1 diagnostic candidate per trajectory mode.
        candidates = frozen_all[:, :, None, :, :]
        result = evaluate_candidates(
            env=env,
            candidates=candidates,
            frozen_all_mode_trajectories=frozen_all,
            frozen_argmax_joint_trajectories=frozen_argmax,
            valid_mode_mask=valid,
            config=config,
        )

        if result.rewards.shape != (
            NUM_VEHICLES,
            NUM_MODES,
            1,
        ):
            raise AssertionError(
                f"unexpected reward shape: {result.rewards.shape}"
            )
        if not np.isfinite(result.rewards).all():
            raise AssertionError(
                "non-finite total reward detected"
            )
        for name, values in result.components.items():
            if not np.isfinite(values).all():
                raise AssertionError(
                    f"non-finite component detected: {name}"
                )

        progress = result.components[
            "progress_score"
        ][:, :, 0]
        comfort = result.components[
            "comfort_penalty"
        ][:, :, 0]
        road = result.components[
            "road_penalty"
        ][:, :, 0]

        baseline_progress = progress[:, BASELINE_MODE]
        low_progress = progress[:, LOW_PROGRESS_MODE]
        baseline_comfort = comfort[:, BASELINE_MODE]
        zigzag_comfort = comfort[:, ZIGZAG_MODE]
        baseline_road = road[:, BASELINE_MODE]
        offroad_road = road[:, OFFROAD_MODE]
        offroad_flag = result.out_of_drivable[
            :,
            OFFROAD_MODE,
            0,
        ]

        if not np.all(
            baseline_progress > low_progress + 1.0e-4
        ):
            raise AssertionError(
                f"{scenario_name}: progress direction check failed: "
                f"base={baseline_progress.tolist()} "
                f"low={low_progress.tolist()}"
            )

        if not np.all(
            zigzag_comfort > baseline_comfort + 1.0e-4
        ):
            raise AssertionError(
                f"{scenario_name}: comfort direction check failed: "
                f"base={baseline_comfort.tolist()} "
                f"zigzag={zigzag_comfort.tolist()}"
            )

        if not np.all(offroad_flag):
            raise AssertionError(
                f"{scenario_name}: forced-offroad not detected for "
                f"all roles: {offroad_flag.tolist()}"
            )

        if not np.all(offroad_road >= baseline_road):
            raise AssertionError(
                f"{scenario_name}: offroad road penalty is lower "
                "than baseline"
            )

        baseline_rewards = result.rewards[
            :,
            BASELINE_MODE,
            0,
        ]
        baseline_unsafe = result.unsafe[
            :,
            BASELINE_MODE,
            0,
        ]

        bg_gap = result.components[
            "minimum_background_gap_m"
        ][:, BASELINE_MODE, 0]
        team_gap = result.components[
            "minimum_teammate_gap_m"
        ][:, BASELINE_MODE, 0]
        ttc = result.components[
            "minimum_ttc_s"
        ][:, BASELINE_MODE, 0]

        return ValidationRecord(
            scenario=scenario_name,
            seed=int(seed),
            background_vehicle_count=len(background),
            baseline_reward_mean=float(
                np.mean(baseline_rewards)
            ),
            baseline_unsafe_count=int(
                np.sum(baseline_unsafe)
            ),
            baseline_progress_mean=float(
                np.mean(baseline_progress)
            ),
            low_progress_mean=float(
                np.mean(low_progress)
            ),
            baseline_comfort_mean=float(
                np.mean(baseline_comfort)
            ),
            zigzag_comfort_mean=float(
                np.mean(zigzag_comfort)
            ),
            baseline_road_penalty_mean=float(
                np.mean(baseline_road)
            ),
            offroad_road_penalty_mean=float(
                np.mean(offroad_road)
            ),
            offroad_detected_count=int(
                np.sum(offroad_flag)
            ),
            baseline_min_background_gap_m=float(
                np.min(bg_gap)
            ),
            baseline_min_teammate_gap_m=float(
                np.min(team_gap)
            ),
            baseline_min_ttc_s=float(
                np.min(ttc)
            ),
        )
    finally:
        env.close()


def _parse_seeds(text: str) -> list[int]:
    seeds = [
        int(part.strip())
        for part in text.split(",")
        if part.strip()
    ]
    if not seeds:
        raise argparse.ArgumentTypeError(
            "at least one integer seed is required"
        )
    return seeds


def _write_report(
    records: list[ValidationRecord],
    output: Path,
) -> None:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    payload = {
        "validation": "trajectory_mode_reward_real_env_v1",
        "scope": (
            "real AllMerge scenario reset state, RoadNetwork, "
            "controlled vehicles and randomized background vehicles"
        ),
        "diagnostic_trajectory_source": (
            "lane-follow probes; not Polynomial planner outputs"
        ),
        "records": [
            asdict(record)
            for record in records
        ],
    }
    output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        nargs="*",
        choices=sorted(SCENARIOS),
        default=list(SCENARIOS),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=[0, 1, 2],
        help="comma-separated seeds, default 0,1,2",
    )
    parser.add_argument(
        "--json-out",
        default=(
            "infos/reward_validation/"
            "real_env_validation.json"
        ),
    )
    args = parser.parse_args()

    records: list[ValidationRecord] = []
    for scenario_name in args.scenarios:
        for seed in args.seeds:
            record = validate_one(
                scenario_name,
                int(seed),
            )
            records.append(record)
            print(
                f"[OK] {scenario_name} seed={seed}: "
                f"R_base={record.baseline_reward_mean:.4f}, "
                f"unsafe={record.baseline_unsafe_count}/3, "
                f"offroad={record.offroad_detected_count}/3"
            )

    print()
    print(
        "scenario                 seed  bg   R_base   unsafe  "
        "progress(base/low) comfort(base/zigzag) offroad"
    )
    print("-" * 108)
    for row in records:
        print(
            f"{row.scenario:24s} "
            f"{row.seed:4d} "
            f"{row.background_vehicle_count:3d} "
            f"{row.baseline_reward_mean:8.4f} "
            f"{row.baseline_unsafe_count:6d} "
            f"{row.baseline_progress_mean:6.3f}/"
            f"{row.low_progress_mean:6.3f} "
            f"{row.baseline_comfort_mean:6.3f}/"
            f"{row.zigzag_comfort_mean:6.3f} "
            f"{row.offroad_detected_count}/3"
        )

    output = Path(args.json_out)
    _write_report(
        records,
        output,
    )

    print()
    print(
        f"[OK] real AllMerge scenario validation passed: "
        f"{len(records)} states"
    )
    print(
        f"[OK] report: {output}"
    )
    print(
        "[NEXT] connect real Polynomial planner outputs for "
        "reward-distribution validation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
