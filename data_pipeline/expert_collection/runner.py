"""Batch controller for the single TRACE Polynomial/TruckSim expert pipeline."""
from __future__ import annotations

import csv
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
from data_pipeline.expert_dataset import split_shards
MODE_SEMANTIC_IDS = (0, 0, 0, 1, 1, 1, 2, 2, 2, 3)

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
FINAL_REQUIRED = PREANCHOR_REQUIRED + (
    "a1_coarse_controls", "a1_coarse_trajectories", "a1_dense_trajectories",
    "a1_valid_mask", "a1_invalid_reasons", "coarse_controls",
    "coarse_trajectories", "mode_valid_mask", "target_mode", "target_semantic",
    "target_mode_geometry_valid", "target_mode_assignment_distance",
)

PREANCHOR_FORBIDDEN = {"coarse_controls", "coarse_trajectories", "mode_valid_mask", "target_mode", "target_semantic"}


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
        forbidden = sorted(PREANCHOR_FORBIDDEN & keys)
        if missing:
            raise ValueError(f"{path.name}: missing pre-anchor fields {missing}")
        if forbidden:
            raise ValueError(f"{path.name}: anchor-dependent fields present before A1 stage: {forbidden}")
        n = int(z["ego_state"].shape[0])
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


def _validate_final_shard(path: Path):
    with np.load(path, allow_pickle=False) as z:
        missing = sorted(set(FINAL_REQUIRED) - set(z.files))
        if missing:
            raise ValueError(f"{path.name}: missing final TRACE fields {missing}")
        n = int(z["expert_trajectory_xy"].shape[0])
        if z["coarse_controls"].shape != (n, 10, 40, 2):
            raise ValueError(f"{path.name}: coarse_controls must be [N,10,40,2]")
        if z["coarse_trajectories"].shape != (n, 10, 8, 2):
            raise ValueError(f"{path.name}: coarse_trajectories must be [N,10,8,2]")
        if z["mode_valid_mask"].shape != (n, 10) or not z["mode_valid_mask"].astype(bool).any(axis=1).all():
            raise ValueError(f"{path.name}: invalid A1 mode_valid_mask")
        if not np.array_equal(z["mode_valid_mask"], z["a1_valid_mask"]):
            raise ValueError(f"{path.name}: mode_valid_mask and a1_valid_mask differ")
        if not np.allclose(z["coarse_controls"], z["a1_coarse_controls"], atol=0, rtol=0):
            raise ValueError(f"{path.name}: paired A1 control aliases differ")
        if not np.allclose(z["coarse_trajectories"], z["a1_coarse_trajectories"], atol=0, rtol=0):
            raise ValueError(f"{path.name}: paired A1 trajectory aliases differ")
        for key in z.files:
            a = np.asarray(z[key])
            if a.dtype.kind in "fiu" and not np.isfinite(a).all():
                raise ValueError(f"{path.name}: nonfinite values in {key}")
        return n, sorted(set(np.asarray(z["episode_index"]).reshape(-1).astype(int).tolist()))


def _package_episode(preanchor_path: Path, final_path: Path, builder: A1AnchorBuilder):
    with np.load(preanchor_path, allow_pickle=False) as z:
        arrays = {k: np.array(z[k], copy=True) for k in z.files}
    n = len(arrays["ego_state"])
    built = [builder.build({k: v[i] for k, v in arrays.items()}) for i in range(n)]
    for key in built[0]:
        arrays[key] = np.stack([x[key] for x in built], axis=0)
    arrays["coarse_controls"] = arrays["a1_coarse_controls"].astype(np.float32, copy=True)
    arrays["coarse_trajectories"] = arrays["a1_coarse_trajectories"].astype(np.float32, copy=True)
    arrays["mode_valid_mask"] = arrays["a1_valid_mask"].astype(bool, copy=True)

    # Fine-grained A1 mode labels are retained for the trajectory-space trainer
    # and its diagnostics. Labels are nearest valid A1 anchor to the Polynomial GT.
    gt = arrays["expert_trajectory_xy"]
    dist = np.linalg.norm(arrays["coarse_trajectories"] - gt[:, None], axis=-1).mean(axis=-1)
    dist = np.where(arrays["mode_valid_mask"], dist, np.inf)
    if not np.isfinite(dist.min(axis=1)).all():
        bad = np.flatnonzero(~np.isfinite(dist.min(axis=1))).tolist()
        raise RuntimeError(f"A1 builder produced no valid anchor for samples {bad[:10]}")
    mode = dist.argmin(axis=1).astype(np.int64)
    arrays["target_mode"] = mode
    arrays["target_semantic"] = np.asarray(MODE_SEMANTIC_IDS, np.int64)[mode]
    arrays["target_mode_geometry_valid"] = np.ones((n,), dtype=bool)
    arrays["target_mode_assignment_distance"] = dist[np.arange(n), mode].astype(np.float32)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, final_path)
    _validate_final_shard(final_path)
    return n


def _existing_final(dataset: Path):
    result = {}
    for path in sorted((dataset / "shards").glob("shard_*.npz")):
        n, episode_ids = _validate_final_shard(path)
        if len(episode_ids) != 1:
            raise RuntimeError(f"{path.name}: final shard must contain exactly one episode")
        for episode in episode_ids:
            if episode in result:
                raise RuntimeError(f"Episode {episode} appears in multiple final shards")
            result[episode] = (path, n)
    return result


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
    preanchor_dir = dataset / ".collection_work" / "preanchor_episodes"
    logs_dir = dataset / "reports" / "native_logs"
    reports = dataset / "reports"
    shards_dir = dataset / "shards"

    koopman_cfg = load_collection_config(args.koopman_config)
    cases = _resolve_case_paths(list(physical_cases(koopman_cfg)))
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

    for d in (preanchor_dir, logs_dir, reports, shards_dir):
        d.mkdir(parents=True, exist_ok=True)
    progress_path = reports / "collection_progress.json"
    existing_data = (any(shards_dir.glob("shard_*.npz")) or
                     any(preanchor_dir.glob("episode_*.npz")) or progress_path.exists() or
                     (dataset / "engine_map_cache.npy").exists())
    if existing_data and not args.resume:
        raise FileExistsError(
            f"Dataset already contains collection state: {dataset}. "
            "Set collection.resume: true or select a new dataset_name."
        )

    cache_path = _engine_map_cache(dataset, koopman_cfg)
    final_by_episode = _existing_final(dataset)
    preanchor_files = sorted(preanchor_dir.glob("episode_*.npz"))
    preanchor_by_episode = {}
    for p in preanchor_files:
        n, episode = _validate_preanchor_episode(p)
        if episode in final_by_episode:
            continue
        if episode in preanchor_by_episode:
            raise RuntimeError(f"Duplicate pre-anchor episode index {episode}")
        preanchor_by_episode[episode] = (p, n)

    accepted = sum(v[1] for v in final_by_episode.values()) + sum(v[1] for v in preanchor_by_episode.values())
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    max_seen = max([*final_by_episode.keys(), *preanchor_by_episode.keys()], default=args.start_episode_index - 1)
    next_episode = max(int(progress.get("next_episode_index", args.start_episode_index)), max_seen + 1)
    attempts = 0
    consecutive_empty = 0
    audit_path = reports / "episode_audit.csv"

    while accepted < args.target_samples and attempts < args.max_episodes:
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
                n, _ = _validate_preanchor_episode(preanchor_path)
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
            "schema_version": "trace_unified_expert_collection_v2",
            "next_episode_index": next_episode, "accepted_samples": accepted,
            "target_samples": args.target_samples, "preanchor_episode_count": len(preanchor_by_episode),
            "final_shard_count": len(final_by_episode), "last_episode": episode,
        })
        if consecutive_empty >= args.max_consecutive_empty_episodes:
            raise RuntimeError(f"{consecutive_empty} consecutive episodes failed validation; see {logs_dir}")

    # Offline A1 anchor generation is part of this same pipeline. Pre-anchor
    # files remain internal staging data, never a second public dataset.
    from highway_env.planner.a1_anchor_builder import A1AnchorBuilder
    builder = A1AnchorBuilder(args.a1_anchors)
    packaged = 0
    for episode, (preanchor_path, _) in sorted(preanchor_by_episode.items()):
        final_path = shards_dir / f"shard_{episode:06d}.npz"
        if final_path.exists():
            _validate_final_shard(final_path)
            continue
        n = _package_episode(preanchor_path, final_path, builder)
        packaged += n
        print(f"[A1 + SHARD] episode={episode} samples={n} -> {final_path.name}", flush=True)

    final_by_episode = _existing_final(dataset)
    final_samples = sum(v[1] for v in final_by_episode.values())
    if final_by_episode:
        split_result = split_shards(
            dataset_root=dataset,
            train_ratio=args.train_split_ratio,
            val_ratio=args.val_split_ratio,
            test_ratio=args.test_split_ratio,
            seed=args.split_seed,
        )
    else:
        split_result = {"train": [], "val": [], "test": []}

    manifest = {
        "schema_version": "trace_expert_road_geometry_a1_k2_v2",
        "pipeline": "single_unified_Polynomial_TruckSim_batch_collector",
        "expert": "Polynomial",
        "plant": "TruckSim_longitudinal_lateral",
        "anchor_generation": "offline_A1_Lattice_inside_unified_collection_pipeline",
        "sample_unit": "one controlled ego at one valid 10-Hz planning frame",
        "samples": final_samples,
        "target_samples": args.target_samples,
        "episodes": len(final_by_episode),
        "trucksim_physical_cases": len(cases),
        "scenarios": args.scenarios,
        "vehicle_dimensions_m": [args.ego_length_m, args.ego_width_m],
        "road_geometry": args.road_geometry,
        "a1_anchor_config": str(args.a1_config_path),
        "a1_anchor_parameters": args.a1_anchors,
        "trajectory": {"sparse_shape": [8, 2], "dense_future_shape": [40, 2],
                       "visualization_shape": [41, 2], "dt_s": 0.1, "horizon_s": 4.0},
        "koopman_context": {"state_history": [5, 5], "control_history": [4, 2],
                            "pose0": [3], "payload_kg": "scalar", "road_mu": "scalar"},
        "a1_anchor_fields": {"coarse_controls": [10, 40, 2], "coarse_trajectories": [10, 8, 2],
                             "mode_valid_mask": [10]},
        "splits": {k: len(v) for k, v in split_result.items()},
        "complete": final_samples >= args.target_samples,
    }
    _atomic_json(reports / "manifest.json", manifest)
    _atomic_json(progress_path, {"schema_version": manifest["schema_version"],
                                 "next_episode_index": next_episode,
                                 "accepted_samples": final_samples,
                                 "target_samples": args.target_samples,
                                 "complete": manifest["complete"]})

    if manifest["complete"]:
        # Pre-anchor files are an internal staging representation only. Once every
        # episode has been A1-enriched and atomically committed, remove staging.
        import shutil
        shutil.rmtree(preanchor_dir, ignore_errors=True)
        try:
            preanchor_dir.parent.rmdir()
        except OSError:
            pass
    print(f"[DONE] unified TRACE dataset: {final_samples}/{args.target_samples} samples, "
          f"{len(final_by_episode)} episode shards, complete={manifest['complete']}", flush=True)
    return 0 if manifest["complete"] else 2
