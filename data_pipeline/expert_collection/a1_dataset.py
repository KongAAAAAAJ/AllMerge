"""Build the final A1/K2 training dataset from immutable raw expert episodes."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import yaml
from data_pipeline.expert_collection.runner import PREANCHOR_REQUIRED, _validate_preanchor_episode, _atomic_json
from data_pipeline.expert_collection.config import resolve_path
from data_pipeline.expert_dataset import split_shards

MODE_SEMANTIC_IDS = (0, 0, 0, 1, 1, 1, 2, 2, 2, 3)
FINAL_REQUIRED = PREANCHOR_REQUIRED + (
    "a1_coarse_controls", "a1_coarse_trajectories", "a1_dense_trajectories",
    "a1_valid_mask", "a1_invalid_reasons", "coarse_controls",
    "coarse_trajectories", "mode_valid_mask", "target_mode", "target_semantic",
    "target_mode_geometry_valid", "target_mode_assignment_distance",
)

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



def read_source(root):
    manifest = json.loads((root / "reports/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "trace_expert_raw_k2_v2" or not manifest.get("complete"):
        raise RuntimeError("Expected a completed anchor-independent raw collection; resume collection first")
    listed = manifest["episode_files"]
    paths = sorted((root / "episodes").glob("episode_*.npz"))
    if not paths or {p.name for p in paths} != {x["name"] for x in listed} or len(paths) != len(listed):
        raise RuntimeError("Raw episode inventory differs from manifest")
    records = {x["name"]: x for x in listed}
    seen = set(); total = 0
    for path in paths:
        n, eid = _validate_preanchor_episode(path)
        rec = records[path.name]
        if n != rec["samples"] or eid != rec["episode_index"] or eid in seen:
            raise RuntimeError(f"Raw inventory mismatch: {path}")
        seen.add(eid); total += n
    if total != manifest["samples"] or len(seen) != manifest["episodes"]:
        raise RuntimeError("Raw manifest totals mismatch")
    return manifest, paths


def build_dataset(input_dir, output_dir, config_file, *, check_only=False):
    src, dst, config_file = map(resolve_path, (input_dir, output_dir, config_file))
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError("Raw source and final destination must be separate non-nested directories")
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}; use a new output directory for each A1 version")
    manifest, paths = read_source(src)
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))["a1"]
    required = ("control_steps", "dt", "wheelbase_m", "max_accel_mps2", "max_decel_mps2", "max_steer_rad", "lane_change_distance_m", "road_edge_margin_m", "speed_delta_mps", "minimum_lane_change_speed_mps")
    for key in required:
        v = cfg[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or (v < 0 if key in ("road_edge_margin_m", "speed_delta_mps", "minimum_lane_change_speed_mps") else v <= 0):
            raise ValueError(f"Invalid A1 parameter {key}: {v}")
    if cfg["control_steps"] != 40 or not math.isclose(cfg["dt"], 0.1, abs_tol=1e-8):
        raise ValueError("A1 requires 40 controls at 0.1 s")
    cfg = dict(cfg)
    for key, value in zip(("ego_length_m", "ego_width_m"), manifest["vehicle_dimensions_m"]):
        if key in cfg and not math.isclose(float(cfg[key]), value, abs_tol=1e-6):
            raise ValueError(f"A1 {key} differs from captured vehicle geometry")
        cfg[key] = value
    print(f"[A1 INPUT] episodes={len(paths)} samples={manifest['samples']} source={src}", flush=True)
    if check_only:
        print("[A1 CHECK-ONLY PASS] source, configuration and destination checked; no files written")
        return 0
    from highway_env.planner.a1_anchor_builder import A1AnchorBuilder
    builder = A1AnchorBuilder(cfg)
    dst.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=dst.name + ".building-", dir=dst.parent))
    manifest_hash = hashlib.sha256((src / "reports/manifest.json").read_bytes()).hexdigest()
    records = {x["name"]: x for x in manifest["episode_files"]}
    try:
        valid_counts = np.zeros(10, dtype=np.int64); total = 0; inventory = []
        for path in paths:
            eid = int(records[path.name]["episode_index"])
            dest = stage / "shards" / f"shard_{eid:06d}.npz"
            n = _package_episode(path, dest, builder)
            with np.load(dest, allow_pickle=False) as data:
                valid_counts += data["a1_valid_mask"].sum(axis=0)
            total += n
            inventory.append({"source": path.name, "shard": dest.name, "samples": n})
            print(f"[A1] {dest.name} samples={n} total={total}/{manifest['samples']}", flush=True)
        if total != manifest["samples"]:
            raise RuntimeError("Final sample count differs from raw source")
        ratios = manifest["split_ratios"]
        splits = split_shards(stage, train_ratio=ratios[0], val_ratio=ratios[1], test_ratio=ratios[2], seed=manifest["split_seed"])
        final = {**manifest, "schema_version": "trace_expert_road_geometry_a1_k2_v2",
                 "pipeline": "offline_A1_enrichment_of_raw_expert_dataset", "anchor_generation": "offline_A1_Lattice",
                 "source_dataset": str(src), "source_manifest_sha256": hashlib.sha256((src / "reports/manifest.json").read_bytes()).hexdigest(),
                 "a1_anchor_config": str(config_file), "a1_anchor_parameters": cfg,
                 "a1_anchor_fields": {"coarse_controls": [10,40,2], "coarse_trajectories": [10,8,2], "mode_valid_mask": [10]},
                 "valid_count_by_mode": valid_counts.tolist(), "valid_anchors_mean": float(valid_counts.sum()/total),
                 "shard_inventory": inventory, "splits": {k: len(v) for k,v in splits.items()}, "complete": True}
        _atomic_json(stage / "reports/manifest.json", final)
        if hashlib.sha256((src / "reports/manifest.json").read_bytes()).hexdigest() != manifest_hash:
            raise RuntimeError("Raw collection changed during A1 build")
        if dst.exists():
            raise FileExistsError(dst)
        stage.rename(dst)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"[A1 DONE] {total} samples, mean_valid={final['valid_anchors_mean']:.3f}/10, splits={final['splits']} -> {dst}", flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="outputs/expert_dataset/expert_trucksim_k2_raw_100k", help="Raw dataset root (contains episodes/ and reports/manifest.json)")
    p.add_argument("--output-dir", default="outputs/expert_dataset/expert_trucksim_k2_anchor_pair_100k")
    p.add_argument("--config", default="configs/anchors/a1.yaml")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args(argv)
    return build_dataset(args.input_dir, args.output_dir, args.config, check_only=args.check_only)
