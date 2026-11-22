"""Isolated one-episode worker for the unified TRACE expert data pipeline.

One process owns one episode. This is intentional for native TruckSim DLL
isolation. Raw scenes are consumed directly by online anchor generation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
    args.min_contiguous_control_steps = int(cfg['collection']['min_contiguous_control_steps'])
    args.config_path = resolve_path(args.config)
    args.trucksim_koopman_config = Path(cfg["koopman_collection_config"])
    args.ego_length_m = float(cfg["vehicle_dimensions"]["length_m"])
    args.ego_width_m = float(cfg["vehicle_dimensions"]["width_m"])
    args.ego_wheelbase_m = float(cfg["vehicle_dimensions"]["wheelbase_m"])
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


def longest_control_run(steps):
    """Count consecutive controls recovered from saved history windows."""
    longest = run = 0
    previous = None
    for step in sorted(steps):
        run = run + 1 if previous is not None and step == previous + 1 else 1
        longest = max(longest, run)
        previous = step
    return longest


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
    min_control_steps = args.min_contiguous_control_steps
    env_config = {"duration": float(args.duration), "offscreen_rendering": False,
                  "expert_collection_min_control_steps": min_control_steps}
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
        print(f"[CHECK-ONLY PASS] TruckSim | scenario={args.scenario} | output={args.output_dir}")
        return 0
    env = SCENARIOS[args.scenario](config=env_config, render_mode=None)
    env.reset(seed=int(args.seed))
    # The platoon vehicles use one consistent heavy-truck footprint throughout
    # feature export, road-boundary checks and collision geometry.
    for controlled in env.controlled_vehicles:
        controlled.LENGTH = float(args.ego_length_m)
        controlled.WIDTH = float(args.ego_width_m)
        controlled.WHEELBASE = float(args.ego_wheelbase_m)

    if case is not None:
        from data_pipeline.expert_collection.trucksim import HistoricalContexts, check_context
        history = HistoricalContexts(
            env.controlled_vehicles,
            compression_sign=trucksim_cfg["units"].get("wheel_compression_sign", 1.0),
        )
        print(f"[TRUCKSIM] case={case['model_id']} payload={case['payload_kg']}kg mu={case['road_mu']}", flush=True)

    from data_pipeline.expert_collection.trajectories import finalize_dual_samples
    if not np.isclose(float(env.config['policy_frequency']), 10.0):
        raise ValueError('Dual expert collection requires a 10-Hz policy grid')
    def current_poses():
        return np.asarray([[*v.position, v.heading] for v in env.controlled_vehicles], dtype=np.float64)
    world_poses = [current_poses()]
    samples = []
    recorded_controls = [set() for _ in env.controlled_vehicles]
    env._expert_collection_control_windows_ready = min_control_steps == 0
    frame_count = int(np.ceil(args.duration * float(env.config["policy_frequency"])))
    exit_status = None
    try:
        for step in range(frame_count):
            pre_context = None
            if history is not None and history.ready():
                pre_context = [history.at_planning_time(i, case) for i in range(len(env.controlled_vehicles))]
            _, _, terminated, truncated, info = env.step(int(args.group_action))
            world_poses.append(current_poses())
            if history is not None:
                history.append_executed_transition(env.controlled_vehicles)
            # Only complete future windows count towards the termination gate.
            mature_frame = step + 1 - 40
            for sample in samples:
                if int(sample['frame_index']) == mature_frame:
                    recorded_controls[int(sample['ego_index'])].update(range(mature_frame - 4, mature_frame))
            env._expert_collection_control_windows_ready = all(
                longest_control_run(s) >= min_control_steps for s in recorded_controls)
            if bool(terminated) or bool(truncated):
                exit_status = {
                    'step': step + 1, 'time_s': float(env.time),
                    'terminated': bool(terminated), 'truncated': bool(truncated),
                    'task_success': bool(info.get('task_success', False)),
                    'offroad_terminal': bool(env.config.get('offroad_terminal', False)),
                    'recorded_control_runs': [longest_control_run(s) for s in recorded_controls],
                    'min_contiguous_control_steps': min_control_steps,
                    'vehicles': [dict(ego_index=i, crashed=bool(v.crashed),
                                      on_road=bool(v.on_road), speed_mps=float(v.speed),
                                      position=np.asarray(v.position).tolist(),
                                      lane_index=list(v.lane_index))
                                 for i, v in enumerate(env.controlled_vehicles)],
                }
                print('[EPISODE-END] ' + json.dumps(exit_status), flush=True)
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

            n_ego = len(env.controlled_vehicles)
            for ego_i in range(n_ego):
                sample = {}
                for k, v in features.items():
                    sample[k] = _extract_one(v, ego_i, n_ego)
                for k, v in road.items():
                    sample[k] = _extract_one(v, ego_i, n_ego)
                sample["polynomial_future_xy"] = np.asarray(
                    expert_trajectory["future_trajectory_dense"][ego_i], np.float32).copy()
                sample["dense_dt"] = np.asarray(0.1, np.float32)
                sample["trajectory_horizon_s"] = np.asarray(4.0, np.float32)
                sample["frame_index"] = np.asarray(step, np.int32)
                sample["episode_index"] = np.asarray(args.episode_index, np.int32)
                sample["episode_step"] = np.asarray(step, np.int32)
                sample["ego_index"] = np.asarray(ego_i, np.int8)
                sample["seed"] = np.asarray(args.seed, np.int32)
                sample["scenario"] = np.asarray(args.scenario, dtype="U16")
                sample["expert_type"] = np.asarray("Polynomial", dtype="U16")
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
                    # Context and Polynomial reference share the planning origin.
                    check_context(sample)
                samples.append(sample)
            if step % 10 == 0:
                print(f"[EXPERT-COLLECT] step={step+1}/{frame_count} samples={len(samples)}", flush=True)
    finally:
        env.close()

    samples = finalize_dual_samples(samples, world_poses)
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
        "schema_version": "trace_expert_raw_k2_v3",
        "trajectory_sources": ["polynomial", "trucksim_closed_loop"],
        "execution_target": "future closed-loop execution including replanning",
        "anchor_generation": "disabled_during_collection",
        "a1_builder": "online_runtime",
        "scenario": args.scenario,
        "episode_index": int(args.episode_index),
        "seed": int(args.seed),
        "sample_count": len(samples),
        "exit_status": exit_status,
        "min_contiguous_control_steps": min_control_steps,
        "recorded_control_runs": [longest_control_run(s) for s in recorded_controls],
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
