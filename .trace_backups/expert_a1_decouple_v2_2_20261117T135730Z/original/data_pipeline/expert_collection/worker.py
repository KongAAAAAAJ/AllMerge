"""Isolated one-episode worker for the unified TRACE expert data pipeline.

One process owns one episode. This is intentional for native TruckSim DLL
isolation. The batch controller subsequently builds A1 anchors before exposing
the final training dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCENARIO_NAMES = ("straight", "curved", "merge_in", "merge_out")


def _load_scenarios():
    from highway_env.envs.scenarios.straight_lane_change_env import StraightLaneChangeEnv
    from highway_env.envs.scenarios.curved_lane_change_env import CurvedLaneChangeEnv
    from highway_env.envs.scenarios.merge_in_env import MergeInEnv
    from highway_env.envs.scenarios.merge_out_env import MergeOutEnv
    return {"straight": StraightLaneChangeEnv, "curved": CurvedLaneChangeEnv,
            "merge_in": MergeInEnv, "merge_out": MergeOutEnv}

def _extract_polynomial_gt_10hz(
    env,
    expert_trajectory: Dict,
    *,
    dense_dt_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract true Polynomial path at 10 Hz in the planner ego frame.

    The Polynomial planner internally produces a 0.1 s path in world
    coordinates. The supervised target stores aligned 8-point 0.5 s samples.
    The unified dataset also preserves the original path on a fixed
    10 Hz grid and transform it with the SAME planning origin used by
    PlannerTrajectory.from_world_path().
    """
    from highway_env.planner.geometry import world_to_ego_point

    sparse_time_s = np.asarray(
        expert_trajectory["trajectory_time_s"],
        dtype=np.float32,
    ).reshape(-1)
    if sparse_time_s.size == 0:
        raise RuntimeError("Empty expert trajectory_time_s")

    horizon_s = float(sparse_time_s[-1])
    if horizon_s <= 0.0:
        raise RuntimeError(f"Invalid expert horizon: {horizon_s}")
    if dense_dt_s <= 0.0:
        raise ValueError("dense_dt_s must be positive")

    steps = int(round(horizon_s / float(dense_dt_s)))
    dense_time_s = (
        np.arange(steps + 1, dtype=np.float32)
        * np.float32(dense_dt_s)
    )
    # Keep the last timestamp exactly aligned with the sparse planner horizon.
    if not np.isclose(
        float(dense_time_s[-1]),
        horizon_s,
        atol=1e-5,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Planner horizon is not compatible with 10 Hz dense GT: "
            f"horizon={horizon_s:.6f}s"
        )
    dense_time_s[-1] = np.float32(horizon_s)

    dense_ego_xy = []
    for ego_idx, vehicle in enumerate(env.controlled_vehicles):
        path = getattr(vehicle, "path", None)
        planner_trajectory = getattr(
            vehicle,
            "latest_planner_trajectory",
            None,
        )
        if path is None or planner_trajectory is None:
            raise RuntimeError(
                "Polynomial dense GT unavailable for ego "
                f"{ego_idx}: missing vehicle.path/latest_planner_trajectory"
            )

        source_time_s = np.asarray(
            path.get("time_s"),
            dtype=np.float32,
        ).reshape(-1)
        if source_time_s.size < 2:
            raise RuntimeError(
                f"Polynomial dense GT path too short for ego {ego_idx}"
            )

        def _scalar_path(values, name: str) -> np.ndarray:
            values = list(values)
            result = np.asarray(
                [
                    float(np.asarray(value).reshape(-1)[0])
                    for value in values
                ],
                dtype=np.float32,
            )
            if result.shape[0] != source_time_s.shape[0]:
                raise RuntimeError(
                    f"Polynomial {name}/time length mismatch for ego "
                    f"{ego_idx}: {result.shape[0]} vs {source_time_s.shape[0]}"
                )
            return result

        world_x = _scalar_path(path["x"], "x")
        world_y = _scalar_path(path["y"], "y")

        if np.any(np.diff(source_time_s) <= 0.0):
            raise RuntimeError(
                f"Polynomial time grid is not strictly increasing for ego {ego_idx}"
            )
        if float(source_time_s[0]) > 1e-5:
            raise RuntimeError(
                "Polynomial path starts after t=0 for ego "
                f"{ego_idx}: {float(source_time_s[0]):.6f}s"
            )
        if float(source_time_s[-1]) < horizon_s - 1e-5:
            raise RuntimeError(
                "Polynomial path does not cover sparse expert horizon for ego "
                f"{ego_idx}: source_end={float(source_time_s[-1]):.6f}s, "
                f"horizon={horizon_s:.6f}s"
            )

        dense_world_xy = np.stack(
            [
                np.interp(dense_time_s, source_time_s, world_x),
                np.interp(dense_time_s, source_time_s, world_y),
            ],
            axis=-1,
        ).astype(np.float32)

        origin = planner_trajectory.get("ego_position_world")
        heading = planner_trajectory.get("ego_heading_world")
        if origin is None or heading is None:
            raise RuntimeError(
                "PlannerTrajectory is missing planning origin for ego "
                f"{ego_idx}"
            )

        local_xy = world_to_ego_point(
            dense_world_xy,
            np.asarray(origin, dtype=np.float32),
            float(heading),
        ).astype(np.float32)
        dense_ego_xy.append(local_xy)

    return np.stack(dense_ego_xy, axis=0), dense_time_s




DEFAULT_CONFIG = ROOT / "configs/data/expert_collection.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/data/expert_collection.yaml")
    p.add_argument("--scenario", choices=SCENARIO_NAMES, required=True)
    p.add_argument("--episode-index", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--group-action", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--trucksim-model-id", type=str, default=None)
    p.add_argument("--copy-trucksim-dll-instances", action="store_true")
    p.add_argument("--engine-map-cache", type=Path, default=None)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args(argv)
    from data_pipeline.expert_collection.config import load_config, resolve_path
    cfg = load_config(args.config)
    args.config_path = resolve_path(args.config)
    args.trucksim_koopman_config = Path(cfg["koopman_collection_config"])
    args.ego_length_m = float(cfg["vehicle_dimensions"]["length_m"])
    args.ego_width_m = float(cfg["vehicle_dimensions"]["width_m"])
    road = cfg["road_geometry"]
    args.road_forward_min_m = float(road["forward_min_m"])
    args.road_forward_max_m = float(road["forward_max_m"])
    args.road_point_spacing_m = float(road["point_spacing_m"])
    args.road_max_lanes = int(road["max_lanes"])
    args.road_max_points = int(road["max_points_per_lane"])
    args.copy_trucksim_dll_instances = bool(args.copy_trucksim_dll_instances)
    args.output_dir = resolve_path(args.output_dir)
    args.trucksim_koopman_config = resolve_path(args.trucksim_koopman_config)
    args.engine_map_cache = resolve_path(args.engine_map_cache) if args.engine_map_cache else None
    return args


def _extract_one(value, i, n):
    a = np.asarray(value)
    return np.asarray(a[i] if a.ndim and a.shape[0] == n else a).copy()


def _resolve_project_path(value):
    """Resolve CLI/config/output paths relative to the AllMerge root."""
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main(argv=None):
    args = parse_args(argv)
    SCENARIOS = _load_scenarios()
    if args.duration <= 0 or args.group_action not in (0, 1, 2, 3):
        raise ValueError("duration must be positive and group-action must be 0..3")

    # All relative CLI paths are anchored at the all_merge repository root.
    args.output_dir = _resolve_project_path(args.output_dir)
    args.trucksim_koopman_config = _resolve_project_path(
        args.trucksim_koopman_config
    )
    args.engine_map_cache = _resolve_project_path(args.engine_map_cache)
    os.chdir(ROOT)

    trucksim_cfg = case = history = None
    env_config = {"duration": float(args.duration), "offscreen_rendering": False}
    if args.trucksim_koopman_config is not None:
        from data_pipeline.expert_collection.trucksim import (
            load_collection_config, select_case, scenario_trucksim_config,
            HistoricalContexts, check_context,
        )
        trucksim_cfg = load_collection_config(args.trucksim_koopman_config)
        reference_controller = trucksim_cfg.get("reference_controller", {})
        engine_map_path = reference_controller.get("engine_map_path")
        if engine_map_path:
            reference_controller["engine_map_path"] = str(
                _resolve_project_path(engine_map_path)
            )
        if args.trucksim_model_id:
            from data_pipeline.expert_collection.trucksim import physical_cases
            matches = [x for x in physical_cases(trucksim_cfg) if x["model_id"] == args.trucksim_model_id]
            if len(matches) != 1:
                raise ValueError(f"TruckSim model id not found or ambiguous: {args.trucksim_model_id}")
            case = matches[0]
        else:
            case = select_case(trucksim_cfg, args.episode_index)
        # TruckSim model paths in YAML are also interpreted relative to all_merge.
        for path_key in ("simfile", "dll"):
            case[path_key] = str(_resolve_project_path(case[path_key]))
        if args.check_only:
            for name in ("simfile", "dll"):
                if not Path(case[name]).is_file():
                    raise FileNotFoundError(f"Missing TruckSim {name}: {case[name]}")
            print(f"[CHECK-ONLY PASS] {case['model_id']} | scenario={args.scenario} | output={args.output_dir}")
            return 0
        env_config.update(scenario_trucksim_config(
            trucksim_cfg, case, num_egos=3,
            copy_instances=args.copy_trucksim_dll_instances,
        ))
        if args.engine_map_cache is not None:
            env_config["TruckSim"]["engine_map_cache_npy"] = str(args.engine_map_cache)

    # Supply a complete Planner config before environment construction because
    # AbstractEnv performs an initial reset in __init__ and configures TruckSim
    # native objects during scene creation.
    from copy import deepcopy
    planner_cfg = deepcopy(SCENARIOS[args.scenario].default_config().get("Planner", {}))
    planner_cfg.setdefault("ExpertTrajectory", {})["enabled"] = True
    planner_cfg.pop("ExpertAlignment", None)
    features_cfg = planner_cfg.setdefault("features", {})
    features_cfg.setdefault("anchors", {})["enabled"] = False
    features_cfg["road_geometry"] = {
        "enabled": True,
        "max_lanes": int(args.road_max_lanes),
        "max_points_per_lane": int(args.road_max_points),
        "point_spacing_m": float(args.road_point_spacing_m),
        "forward_min_m": float(args.road_forward_min_m),
        "forward_max_m": float(args.road_forward_max_m),
        "prediction_horizon_s": 4.0,
        "reaction_margin_s": 0.8,
        "distance_margin_m": 35.0,
        "backward_range_m": 30.0,
        "lateral_range_m": 32.0,
        "ego_length_m": float(args.ego_length_m),
        "ego_width_m": float(args.ego_width_m),
    }
    env_config["Planner"] = planner_cfg
    if args.check_only:
        print(f"[CHECK-ONLY PASS] kinematic | scenario={args.scenario} | output={args.output_dir}")
        return 0
    env = SCENARIOS[args.scenario](config=env_config, render_mode=None)
    env.reset(seed=int(args.seed))
    # The platoon vehicles use one consistent heavy-truck footprint throughout
    # feature export, road-boundary checks and collision geometry.
    for controlled in env.controlled_vehicles:
        controlled.LENGTH = float(args.ego_length_m)
        controlled.WIDTH = float(args.ego_width_m)

    if case is not None:
        from data_pipeline.expert_collection.trucksim import HistoricalContexts, check_context
        history = HistoricalContexts(
            env.controlled_vehicles,
            compression_sign=trucksim_cfg["units"].get("wheel_compression_sign", 1.0),
        )
        print(f"[TRUCKSIM] case={case['model_id']} payload={case['payload_kg']}kg mu={case['road_mu']}", flush=True)

    samples = []
    frame_count = int(np.ceil(args.duration * float(env.config["policy_frequency"])))
    try:
        for step in range(frame_count):
            pre_context = None
            if history is not None and history.ready():
                pre_context = [history.at_planning_time(i, case) for i in range(len(env.controlled_vehicles))]
            _, _, terminated, truncated, info = env.step(int(args.group_action))
            if history is not None:
                history.append_executed_transition(env.controlled_vehicles)
            if bool(terminated) or bool(truncated):
                break

            features = getattr(env, "latest_planner_features", None)
            road = getattr(env, "latest_planner_road_geometry", None)
            expert_trajectory = getattr(env, "latest_expert_trajectory", None)
            if features is None or road is None or expert_trajectory is None:
                raise RuntimeError(
                    "unified collection requires planner features, full road geometry "
                    "and the aligned Polynomial expert trajectory"
                )
            if any(k in features for k in ("coarse_controls", "coarse_trajectories", "mode_valid_mask")):
                raise RuntimeError("anchor generation was not disabled in unified expert collection")

            dense_xy, dense_t = _extract_polynomial_gt_10hz(
                env, expert_trajectory
            )
            expert_xy = np.asarray(
                expert_trajectory["expert_trajectory_xy"], np.float32
            )
            n_ego = len(env.controlled_vehicles)
            if expert_xy.shape != (n_ego, 8, 2) or dense_xy.shape != (n_ego, 41, 2):
                raise RuntimeError(f"unexpected expert target shapes: sparse={expert_xy.shape}, dense={dense_xy.shape}")
            for ego_i in range(n_ego):
                sample = {}
                for k, v in features.items():
                    sample[k] = _extract_one(v, ego_i, n_ego)
                for k, v in road.items():
                    sample[k] = _extract_one(v, ego_i, n_ego)
                sample["expert_trajectory_xy"] = expert_xy[ego_i].copy()
                sample["future_trajectory_dense"] = np.asarray(
                    expert_trajectory["future_trajectory_dense"][ego_i], np.float32
                ).copy()  # 40 future points at 0.1 s, t=0.1..4.0
                sample["expert_trajectory_10hz_xy"] = dense_xy[ego_i].copy()  # includes t=0
                sample["trajectory_time_s"] = np.asarray(
                    expert_trajectory["trajectory_time_s"], np.float32
                ).copy()
                sample["expert_trajectory_10hz_time_s"] = dense_t.copy()
                sample["dense_dt"] = np.asarray(0.1, np.float32)
                sample["trajectory_horizon_s"] = np.asarray(4.0, np.float32)
                sample["frame_index"] = np.asarray(step, np.int32)
                sample["episode_index"] = np.asarray(args.episode_index, np.int32)
                sample["episode_step"] = np.asarray(step, np.int32)
                sample["ego_index"] = np.asarray(ego_i, np.int8)
                sample["seed"] = np.asarray(args.seed, np.int32)
                sample["scenario"] = np.asarray(args.scenario, dtype="U16")
                sample["expert_type"] = np.asarray("Polynomial", dtype="U16")
                vehicle = env.controlled_vehicles[ego_i]
                sample["ego_pose_world"] = np.asarray(
                    [float(vehicle.position[0]), float(vehicle.position[1]), float(vehicle.heading)],
                    dtype=np.float32,
                )
                sample["ego_dimensions"] = np.asarray(
                    [args.ego_length_m, args.ego_width_m], dtype=np.float32
                )
                if case is not None:
                    ctx = pre_context[ego_i] if pre_context is not None else None
                    if ctx is None:
                        continue
                    sample.update(ctx)
                    sample["koopman_payload_kg"] = np.asarray(case["payload_kg"], np.float32)
                    sample["koopman_road_mu"] = np.asarray(case["road_mu"], np.float32)
                    sample["koopman_model_id"] = np.asarray(case["model_id"], dtype="U32")
                    # Validate the complete sample: check_context also verifies
                    # future_trajectory_dense, which is added above.
                    check_context(sample)
                samples.append(sample)
            if step % 10 == 0:
                print(f"[EXPERT-COLLECT] step={step+1}/{frame_count} samples={len(samples)}", flush=True)
    finally:
        env.close()

    if not samples:
        raise RuntimeError("No valid pre-anchor expert samples collected; check warm-up and episode duration")
    keys = tuple(samples[0])
    for i, sample in enumerate(samples):
        if tuple(sample) != keys:
            raise RuntimeError(f"sample schema mismatch at sample {i}")
        for k, v in sample.items():
            a = np.asarray(v)
            if a.dtype.kind in "f" and not np.isfinite(a).all():
                raise RuntimeError(f"nonfinite values in {k}, sample={i}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard = args.output_dir / f"episode_{args.episode_index:06d}.npz"
    tmp = shard.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **{k: np.stack([s[k] for s in samples], axis=0) for k in keys})
    os.replace(tmp, shard)
    manifest = {
        "schema_version": "trace_expert_preanchor_episode_v2",
        "anchor_generation": "disabled_during_collection",
        "a1_builder": "integrated_into_unified_batch_pipeline",
        "scenario": args.scenario,
        "episode_index": int(args.episode_index),
        "seed": int(args.seed),
        "sample_count": len(samples),
        "fields": {k: {"shape": list(np.asarray(samples[0][k]).shape), "dtype": str(np.asarray(samples[0][k]).dtype)} for k in keys},
        "vehicle_dimensions": {
            "length_m": float(args.ego_length_m),
            "width_m": float(args.ego_width_m),
        },
        "road_geometry": {
            "source": "live road.network lane geometry; center/left-edge/right-edge sampled directly",
            "point_spacing_m": float(args.road_point_spacing_m),
            "max_lanes": int(args.road_max_lanes),
            "max_points_per_lane": int(args.road_max_points),
            "dynamic_forward_preview": True,
        },
        "trucksim_case": None if case is None else {k: case[k] for k in ("model_id", "payload_kg", "road_mu")},
        "episode_file": shard.name,
    }
    (args.output_dir / f"episode_{args.episode_index:06d}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] pre-anchor expert samples={len(samples)} shard={shard}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
