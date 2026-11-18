"""Batch controller for the single TRACE Polynomial/TruckSim expert pipeline."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.expert_collection.trucksim import load_collection_config, physical_cases

PREANCHOR_REQUIRED = (
    "ego_state", "agent_states", "agent_valid_mask", "map_polylines",
    "map_valid_mask", "target_point", "target_lane_polyline",
    "expert_trajectory_xy", "future_trajectory_dense",
    "expert_trajectory_10hz_xy", "expert_trajectory_10hz_time_s",
    "trajectory_time_s", "dense_dt", "trajectory_horizon_s",
    "road_lane_geometry", "road_lane_point_mask", "road_lane_mask",
    "road_lane_index", "road_lane_meta", "road_lane_line_types",
    "road_lane_connections", "road_current_lane_slot", "road_target_lane_slot",
    "road_current_lane_index", "road_target_lane_index", "road_forward_preview_m",
    "road_backward_preview_m", "road_geometry_version", "ego_pose_world",
    "ego_dimensions", "koopman_state_hist_si", "koopman_control_hist_si",
    "koopman_pose0_si", "koopman_ltr0", "koopman_payload_kg", "koopman_road_mu",
    "koopman_model_id", "koopman_vx_init_kmh", "episode_index", "frame_index",
    "episode_step", "ego_index", "seed", "scenario", "expert_type",
)
PREANCHOR_FORBIDDEN = {"coarse_controls", "coarse_trajectories", "mode_valid_mask", "target_mode", "target_semantic", "target_mode_geometry_valid", "target_mode_assignment_distance"}


def _atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _resolve_case_paths(cases):
    for case in cases:
        for key in ("simfile", "dll"):
            p = Path(case[key]).expanduser()
            case[key] = str((p if p.is_absolute() else ROOT / p).resolve())
    return cases


def _engine_map_cache(dataset: Path, koopman_cfg: dict, *, write: bool = True) -> Path | None:
    reference = koopman_cfg.get("reference_controller", {})
    value = reference.get("engine_map_path")
    source = Path(value) if value else ROOT / "highway_env/vehicle/trucksim_simulation/engine_map_4455kg.csv"
    if not source.is_absolute():
        source = ROOT / source
    if not source.is_file():
        raise FileNotFoundError(f"TruckSim engine map CSV not found: {source}")
    data = np.loadtxt(source, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 12 or not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("Invalid engine map CSV; expected RPM plus 11 throttle columns")
    cache = dataset / "engine_map_cache.npy"
    if cache.exists():
        old = np.load(cache, allow_pickle=False)
        if not np.array_equal(old, data):
            raise RuntimeError("Engine map differs from the existing dataset cache; choose a new dataset name")
    elif write:
        dataset.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".npy.tmp")
        with tmp.open("wb") as f:
            np.save(f, data)
        os.replace(tmp, cache)
    return cache if write or cache.exists() else None


def _validate_preanchor_episode(path: Path):
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        missing = sorted(set(PREANCHOR_REQUIRED) - keys)
        forbidden = sorted((PREANCHOR_FORBIDDEN & keys) | {k for k in keys if k.startswith("a1_")})
        if missing:
            raise ValueError(f"{path.name}: missing pre-anchor fields {missing}")
        if forbidden:
            raise ValueError(f"{path.name}: anchor-dependent fields present before A1 stage: {forbidden}")
        n = int(z["ego_state"].shape[0])
        if n <= 0:
            raise ValueError(f"{path.name}: empty episode")
        shapes = {
            "expert_trajectory_xy": (n, 8, 2),
            "future_trajectory_dense": (n, 40, 2),
            "expert_trajectory_10hz_xy": (n, 41, 2),
            "koopman_state_hist_si": (n, 5, 5),
            "koopman_control_hist_si": (n, 4, 2),
            "koopman_pose0_si": (n, 3),
        }
        for key, shape in shapes.items():
            if z[key].shape != shape:
                raise ValueError(f"{path.name}: {key} shape={z[key].shape}, expected={shape}")
        geometry = z["road_lane_geometry"]
        if geometry.ndim != 4 or geometry.shape[0] != n or geometry.shape[-1] != 9:
            raise ValueError(f"{path.name}: road_lane_geometry must be [N,L,P,9], got {geometry.shape}")
        if z["road_lane_point_mask"].shape != geometry.shape[:3]:
            raise ValueError(f"{path.name}: road_lane_point_mask shape mismatch")
        if z["road_lane_mask"].shape != geometry.shape[:2]:
            raise ValueError(f"{path.name}: road_lane_mask shape mismatch")
        if z["road_lane_connections"].shape != (n, geometry.shape[1], geometry.shape[1]):
            raise ValueError(f"{path.name}: road_lane_connections shape mismatch")
        for key in keys:
            if np.asarray(z[key]).ndim and np.asarray(z[key]).shape[0] != n:
                raise ValueError(f"{path.name}: {key} sample dimension mismatch")
        for key in keys:
            arr = z[key]
            if arr.dtype.kind in "fiu" and not np.isfinite(arr).all():
                raise ValueError(f"{path.name}: nonfinite values in {key}")
        if not np.all(z["road_lane_mask"].astype(bool).any(axis=1)):
            raise ValueError(f"{path.name}: at least one sample has no road lane")
        if not np.all(z["road_current_lane_slot"] >= 0) or not np.all(z["road_target_lane_slot"] >= 0):
            raise ValueError(f"{path.name}: missing current/target lane slot")
        episode_ids = np.unique(np.asarray(z["episode_index"]).reshape(-1).astype(int))
        if episode_ids.size != 1:
            raise ValueError(f"{path.name}: one pre-anchor episode file must contain one episode id")
        return n, int(episode_ids[0])


def _write_audit(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["episode", "case", "scenario", "samples", "returncode", "attempt", "status"])
        w.writerow(row)


def main(args=None):
    if args is None:
        from data_pipeline.expert_collection.config import parse_args
        args = parse_args()
    if args.vehicle_model != "trucksim":
        raise ValueError("Only the unified TruckSim expert pipeline is supported")

    dataset = (args.output_root / args.dataset_name).resolve()
    preanchor_dir = dataset / "episodes"
    logs_dir = dataset / "reports" / "native_logs"
    reports = dataset / "reports"
    if (dataset / "shards").exists() or (dataset / ".collection_work").exists():
        raise RuntimeError(f"{dataset} is an old integrated dataset or staging directory; choose a new raw dataset name")

    koopman_cfg = load_collection_config(args.koopman_config)
    cases = _resolve_case_paths(list(physical_cases(koopman_cfg)))
    if args.trucksim_model_id is not None:
        cases = [case for case in cases if case["model_id"] == args.trucksim_model_id]
    if not cases:
        raise RuntimeError("No verified payload/friction TruckSim cases were found")
    if args.check_only:
        missing = [(c["model_id"], k, c[k]) for c in cases for k in ("simfile", "dll") if not Path(c[k]).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing TruckSim model assets: {missing[:10]}")
        _engine_map_cache(dataset, koopman_cfg, write=False)
        from data_pipeline.expert_collection.config import SCENARIOS
        for name in args.scenarios:
            if name not in SCENARIOS:
                raise ValueError(f"Unsupported scenario {name}")
        print(f"[CHECK-ONLY PASS] unified Polynomial/TruckSim pipeline | cases={len(cases)} "
              f"scenarios={args.scenarios} | output={dataset}")
        return 0

    for d in (preanchor_dir, logs_dir, reports):
        d.mkdir(parents=True, exist_ok=True)
    progress_path = reports / "collection_progress.json"
    existing_data = (any(preanchor_dir.glob("episode_*.npz")) or progress_path.exists() or
                     (dataset / "engine_map_cache.npy").exists())
    if existing_data and not args.resume:
        raise FileExistsError(
            f"Dataset already contains collection state: {dataset}. "
            "Set collection.resume: true or select a new dataset_name."
        )

    # Resume only under the same sampling contract; target size and episode cap may change.
    contract_path = reports / "collection_contract.json"
    contract = {"scenarios": args.scenarios, "cases": cases, "seed": args.start_seed,
                "start_episode_index": args.start_episode_index, "duration": args.duration,
                "group_action": args.group_action,
                "min_contiguous_control_steps": args.min_contiguous_control_steps,
                "vehicle_dimensions_m": [args.ego_length_m, args.ego_width_m],
                "road_geometry": args.road_geometry,
                "split_ratios": [args.train_split_ratio, args.val_split_ratio, args.test_split_ratio],
                "split_seed": args.split_seed,
                "koopman_config_sha256": hashlib.sha256(args.koopman_config.read_bytes()).hexdigest()}
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Raw collection contract differs; choose a new dataset name")
    elif existing_data:
        raise RuntimeError("Existing raw collection has no resume contract; use a new dataset name")
    else:
        _atomic_json(contract_path, contract)
    cache_path = _engine_map_cache(dataset, koopman_cfg)
    preanchor_files = sorted(preanchor_dir.glob("episode_*.npz"))
    preanchor_by_episode = {}
    for p in preanchor_files:
        n, episode = _validate_preanchor_episode(p)
        if episode in preanchor_by_episode:
            raise RuntimeError(f"Duplicate pre-anchor episode index {episode}")
        preanchor_by_episode[episode] = (p, n)

    accepted = sum(v[1] for v in preanchor_by_episode.values())
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    max_seen = max(list(preanchor_by_episode), default=args.start_episode_index - 1)
    next_episode = max(int(progress.get("next_episode_index", args.start_episode_index)), max_seen + 1)
    attempts = 0
    consecutive_empty = 0
    audit_path = reports / "episode_audit.csv"

    if accepted < args.target_samples or len(preanchor_by_episode) < args.min_episodes:
        _atomic_json(reports / "manifest.json", {"schema_version": "trace_expert_raw_k2_v2", "complete": False})

    while (accepted < args.target_samples or len(preanchor_by_episode) < args.min_episodes) and attempts < args.max_episodes:
        episode = next_episode
        case = cases[(episode // len(args.scenarios)) % len(cases)]
        scenario = args.scenarios[episode % len(args.scenarios)]
        preanchor_path = preanchor_dir / f"episode_{episode:06d}.npz"
        if episode in preanchor_by_episode:
            next_episode += 1; attempts += 1
            continue
        status = -1
        accepted_this = False
        last_reason = "failed"
        for attempt in range(1, args.retries + 1):
            cmd = [sys.executable, "-u", "-X", "faulthandler",
                   str(ROOT / "scripts/data/collect_expert_dataset.py"), "--worker",
                   "--config", str(args.config_path),
                   "--scenario", scenario, "--episode-index", str(episode),
                   "--seed", str(args.start_seed + episode), "--duration", str(args.duration),
                   "--group-action", str(args.group_action), "--output-dir", str(preanchor_dir),
                   "--trucksim-model-id", str(case["model_id"]), "--engine-map-cache", str(cache_path)]
            if args.copy_trucksim_dll_instances:
                cmd.append("--copy-trucksim-dll-instances")
            stdout = logs_dir / f"episode_{episode:06d}_attempt{attempt}.stdout.log"
            stderr = logs_dir / f"episode_{episode:06d}_attempt{attempt}.stderr.log"
            print(f"[EPISODE] {episode} | {scenario} | {case['model_id']} | "
                  f"attempt={attempt} | samples={accepted}/{args.target_samples}", flush=True)
            try:
                with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
                    proc = subprocess.run(cmd, cwd=ROOT.parent, stdout=out, stderr=err,
                                          timeout=args.timeout_s, check=False)
                status = proc.returncode
            except subprocess.TimeoutExpired:
                status = -999
            try:
                n, actual_episode = _validate_preanchor_episode(preanchor_path)
                if actual_episode != episode:
                    raise ValueError(f"Worker episode id mismatch: {actual_episode} != {episode}")
                preanchor_by_episode[episode] = (preanchor_path, n)
                accepted += n
                _write_audit(audit_path, [episode, case["model_id"], scenario, n, status, attempt,
                                          "accepted" if status == 0 else f"accepted_after_exit_{status}"])
                accepted_this = True
                break
            except Exception as exc:
                last_reason = f"{type(exc).__name__}: {exc}"
                print(f"[WORKER FAIL] episode={episode} status={status} {last_reason}; {stderr}", flush=True)
        if accepted_this:
            consecutive_empty = 0
        else:
            consecutive_empty += 1
            _write_audit(audit_path, [episode, case["model_id"], scenario, 0, status, args.retries, last_reason])
        next_episode += 1
        attempts += 1
        _atomic_json(progress_path, {
            "schema_version": "trace_expert_raw_k2_v2",
            "next_episode_index": next_episode, "accepted_samples": accepted,
            "target_samples": args.target_samples, "preanchor_episode_count": len(preanchor_by_episode),
            "last_episode": episode,
        })
        if consecutive_empty >= args.max_consecutive_empty_episodes:
            raise RuntimeError(f"{consecutive_empty} consecutive episodes failed validation; see {logs_dir}")

    complete = accepted >= args.target_samples and len(preanchor_by_episode) >= args.min_episodes
    manifest = {
        "schema_version": "trace_expert_raw_k2_v2",
        "pipeline": "single_Polynomial_TruckSim_batch_collector",
        "expert": "Polynomial", "plant": "TruckSim_longitudinal_lateral",
        "anchor_generation": "none; A1 generated online by consumers",
        "sample_unit": "one controlled ego at one valid 10-Hz planning frame",
        "samples": accepted, "target_samples": args.target_samples,
        "episodes": len(preanchor_by_episode), "min_episodes": args.min_episodes,
        "episode_files": [{"name": p.name, "episode_index": e, "samples": n}
                          for e, (p, n) in sorted(preanchor_by_episode.items())],
        "trucksim_physical_cases": len(cases), "scenarios": args.scenarios,
        "vehicle_dimensions_m": [args.ego_length_m, args.ego_width_m],
        "road_geometry": args.road_geometry,
        "trajectory": {"sparse_shape": [8, 2], "dense_future_shape": [40, 2],
                       "visualization_shape": [41, 2], "dt_s": 0.1, "horizon_s": 4.0},
        "split_ratios": contract["split_ratios"], "split_seed": args.split_seed,
        "complete": complete,
    }
    _atomic_json(reports / "manifest.json", manifest)
    _atomic_json(progress_path, {"schema_version": manifest["schema_version"],
                                 "next_episode_index": next_episode,
                                 "accepted_samples": accepted, "target_samples": args.target_samples,
                                 "complete": complete})
    print(f"[RAW DONE] {accepted}/{args.target_samples} samples, {len(preanchor_by_episode)} episodes, complete={complete}", flush=True)
    return 0 if complete else 2
