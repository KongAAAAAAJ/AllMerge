"""Build the TRACE straight-only raw expert dataset in place.

The filter uses the same signed three-point curvature definition and the same
per-sample mean-curvature threshold as analyze_lateral_bias.py. A synchronized
planning frame is retained only when all expected controlled egos in that frame
are present and classified as straight.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATASET = "outputs/expert_dataset/expert_trucksim_50k_straight"
CURVATURE_THRESHOLD_1PM = 1e-3
EXPECTED_EGOS = 3


def signed_polyline_curvature(polyline: np.ndarray) -> np.ndarray:
    """Same three-point signed curvature implementation as analyze_lateral_bias.py."""
    p = np.asarray(polyline, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2:
        return np.empty(0, dtype=np.float64)
    p = p[:, :2]
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 3:
        return np.empty(0, dtype=np.float64)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-4]
    p = p[keep]
    if len(p) < 3:
        return np.empty(0, dtype=np.float64)
    a = p[1:-1] - p[:-2]
    b = p[2:] - p[1:-1]
    c = p[2:] - p[:-2]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    valid = denom > 1e-8
    out = np.full(len(denom), np.nan, dtype=np.float64)
    out[valid] = 2.0 * cross[valid] / denom[valid]
    return out[np.isfinite(out)]


def mean_signed_curvature(polyline: np.ndarray) -> float:
    values = signed_polyline_curvature(polyline)
    return float(np.mean(values)) if len(values) else float("nan")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def episode_split_map(manifest: dict) -> dict[str, str]:
    """Reproduce RawExpertDataset.raw_episode_paths episode-level split exactly."""
    records = manifest["episode_files"]
    ordered = sorted(records, key=lambda r: f"shard_{int(r['episode_index']):06d}.npz")
    ratios = np.asarray(manifest["split_ratios"], dtype=np.float64)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError("Invalid raw dataset split_ratios")
    ratios /= ratios.sum()
    active = [i for i, ratio in enumerate(ratios) if ratio > 0]
    counts = np.zeros(3, dtype=np.int64)
    if len(ordered) >= len(active):
        counts[active] = 1
        exact = (len(ordered) - len(active)) * ratios
        extras = np.floor(exact).astype(np.int64)
        counts += extras
        order = sorted(active, key=lambda i: exact[i] - extras[i], reverse=True)
        for i in order[:len(ordered) - int(counts.sum())]:
            counts[i] += 1
    else:
        counts[active[:len(ordered)]] = 1
    indices = np.arange(len(ordered))
    np.random.RandomState(int(manifest["split_seed"])).shuffle(indices)
    boundaries = (int(counts[0]), int(counts[0] + counts[1]))
    assignment = {}
    for position, record_index in enumerate(indices):
        split = "train" if position < boundaries[0] else "val" if position < boundaries[1] else "test"
        assignment[ordered[int(record_index)]["name"]] = split
    return assignment


def inspect_episode(path: Path, record: dict, threshold: float, expected_egos: int):
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}

    required = ("target_lane_polyline", "episode_index", "frame_index", "ego_index")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"{path.name}: missing fields {missing}")

    n = int(arrays["target_lane_polyline"].shape[0])
    if arrays["target_lane_polyline"].ndim != 3 or arrays["target_lane_polyline"].shape[-1] < 2:
        raise ValueError(
            f"{path.name}: target_lane_polyline must be [N,P,2+], "
            f"got {arrays['target_lane_polyline'].shape}"
        )
    for key, value in arrays.items():
        if value.ndim == 0 or value.shape[0] != n:
            raise ValueError(f"{path.name}: field {key} is not sample-aligned with N={n}")
    if n != int(record["samples"]):
        raise ValueError(f"{path.name}: manifest samples={record['samples']} but shard has {n}")
    episode_ids = np.unique(arrays["episode_index"].astype(np.int64))
    if episode_ids.size != 1 or int(episode_ids[0]) != int(record["episode_index"]):
        raise ValueError(f"{path.name}: episode_index does not match manifest")

    frame_ids = arrays["frame_index"].astype(np.int64)
    ego_ids = arrays["ego_index"].astype(np.int64)
    sample_curvature = np.asarray(
        [mean_signed_curvature(polyline) for polyline in arrays["target_lane_polyline"]],
        dtype=np.float64,
    )
    sample_is_straight = np.isfinite(sample_curvature) & (np.abs(sample_curvature) <= threshold)

    keep = np.zeros(n, dtype=bool)
    retained_frames = 0
    curved_frames = 0
    invalid_geometry_frames = 0
    incomplete_frames = 0
    removed_straight_peer_samples = 0

    for frame_id in np.unique(frame_ids):
        indices = np.flatnonzero(frame_ids == frame_id)
        frame_egos = ego_ids[indices]
        if len(indices) != expected_egos or len(np.unique(frame_egos)) != expected_egos:
            incomplete_frames += 1
            continue
        frame_straight = sample_is_straight[indices]
        if frame_straight.all():
            keep[indices] = True
            retained_frames += 1
        elif not np.isfinite(sample_curvature[indices]).all():
            invalid_geometry_frames += 1
        else:
            curved_frames += 1
            removed_straight_peer_samples += int(frame_straight.sum())

    filtered = {key: value[keep] for key, value in arrays.items()}
    stats = {
        "episode_index": int(record["episode_index"]),
        "file": path.name,
        "samples_before": n,
        "samples_retained": int(keep.sum()),
        "samples_removed": int(n - keep.sum()),
        "frames_before": int(len(np.unique(frame_ids))),
        "frames_retained": retained_frames,
        "frames_removed_curved": curved_frames,
        "frames_removed_invalid_geometry": invalid_geometry_frames,
        "frames_removed_incomplete_ego_group": incomplete_frames,
        "straight_samples_removed_with_curved_frame": removed_straight_peer_samples,
        "curvature_positive_samples": int(np.sum(sample_curvature > threshold)),
        "curvature_negative_samples": int(np.sum(sample_curvature < -threshold)),
        "curvature_straight_samples": int(np.sum(sample_is_straight)),
        "curvature_invalid_samples": int(np.sum(~np.isfinite(sample_curvature))),
    }
    return filtered, stats


def print_summary(summary: dict):
    print("[STRAIGHT-ONLY DATASET]")
    print(f"  dataset: {summary['dataset_root']}")
    print(f"  curvature threshold: |mean signed curvature| <= {summary['threshold_1pm']:.6g} 1/m")
    print(f"  synchronized frame policy: retain only complete {summary['expected_egos']}-ego frames where every ego is straight")
    print(f"  samples: {summary['samples_before']:,} -> {summary['samples_retained']:,} "
          f"(removed {summary['samples_removed']:,})")
    print(f"  frames: {summary['frames_before']:,} -> {summary['frames_retained']:,} "
          f"(removed {summary['frames_removed']:,})")
    for split, row in summary["splits"].items():
        print(f"  {split}: {row['samples_before']:,} -> {row['samples_retained']:,}; "
              f"frames {row['frames_before']:,} -> {row['frames_retained']:,}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--threshold", type=float, default=CURVATURE_THRESHOLD_1PM)
    parser.add_argument("--expected-egos", type=int, default=EXPECTED_EGOS)
    parser.add_argument("--check-only", action="store_true",
                        help="Scan and report the expected filtering result without modifying files")
    parser.add_argument("--apply", action="store_true",
                        help="Rewrite the copied dataset in place, retaining straight synchronized frames")
    args = parser.parse_args(argv)

    if args.check_only == args.apply:
        parser.error("Specify exactly one of --check-only or --apply")
    if args.threshold <= 0 or args.expected_egos < 1:
        parser.error("--threshold and --expected-egos must be positive")

    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "reports" / "manifest.json"
    episodes_dir = dataset_root / "episodes"

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not manifest.get("complete") or manifest.get("schema_version") != "trace_expert_raw_k2_v2":
        raise ValueError("Expected a completed trace_expert_raw_k2_v2 raw expert dataset")
    if not np.isclose(float(manifest["trajectory"]["dt_s"]), 0.1):
        raise ValueError("Expected the TRACE 0.1 s raw expert dataset")
    records = manifest["episode_files"]
    paths = sorted(episodes_dir.glob("episode_*.npz"))
    if len(paths) != len(records) or {path.name for path in paths} != {record["name"] for record in records}:
        raise ValueError("Episode files do not match reports/manifest.json")
    if "straight_only_filter" in manifest:
        raise ValueError("This dataset already contains a straight_only_filter record")

    stage_dir = dataset_root / ".episodes_straight_only_tmp"
    backup_dir = dataset_root / ".episodes_before_straight_only"
    if args.apply:
        if stage_dir.exists() or backup_dir.exists():
            raise FileExistsError("A straight-only staging or backup directory already exists")
        stage_dir.mkdir()

    split_by_file = episode_split_map(manifest)
    record_by_name = {record["name"]: record for record in records}
    episode_results = []
    for index, path in enumerate(paths, start=1):
        filtered, stats = inspect_episode(
            path, record_by_name[path.name], args.threshold, args.expected_egos
        )
        stats["split"] = split_by_file[path.name]
        episode_results.append(stats)
        if args.apply:
            with (stage_dir / path.name).open("wb") as stream:
                np.savez_compressed(stream, **filtered)
        if index == 1 or index % 100 == 0 or index == len(paths):
            print(f"[SCAN] episodes={index}/{len(paths)}", flush=True)

    split_summary = {}
    for split in ("train", "val", "test"):
        rows = [row for row in episode_results if row["split"] == split]
        split_summary[split] = {
            "episodes": len(rows),
            "samples_before": int(sum(row["samples_before"] for row in rows)),
            "samples_retained": int(sum(row["samples_retained"] for row in rows)),
            "frames_before": int(sum(row["frames_before"] for row in rows)),
            "frames_retained": int(sum(row["frames_retained"] for row in rows)),
            "curved_frames_removed": int(sum(row["frames_removed_curved"] for row in rows)),
            "invalid_geometry_frames_removed": int(sum(row["frames_removed_invalid_geometry"] for row in rows)),
            "incomplete_frames_removed": int(sum(row["frames_removed_incomplete_ego_group"] for row in rows)),
        }
        if split_summary[split]["samples_retained"] == 0:
            if args.apply:
                shutil.rmtree(stage_dir)
            raise RuntimeError(f"Straight-only filtering leaves the {split} split empty")

    total_before = int(sum(row["samples_before"] for row in episode_results))
    total_retained = int(sum(row["samples_retained"] for row in episode_results))
    total_frames_before = int(sum(row["frames_before"] for row in episode_results))
    total_frames_retained = int(sum(row["frames_retained"] for row in episode_results))
    summary = {
        "dataset_root": str(dataset_root),
        "source_manifest_sha256": sha256_bytes(manifest_bytes),
        "threshold_1pm": float(args.threshold),
        "curvature_definition": "mean of finite three-point signed curvature samples from target_lane_polyline; identical to analyze_lateral_bias.py",
        "straight_rule": "abs(mean_signed_curvature_1pm) <= threshold",
        "expected_egos": int(args.expected_egos),
        "frame_retention_rule": "retain a frame only when it contains exactly expected_egos distinct ego_index values and every ego sample is straight",
        "episode_split_policy": "unchanged; original episode file inventory and episode-level split assignment are preserved, including zero-sample episode shards",
        "samples_before": total_before,
        "samples_retained": total_retained,
        "samples_removed": total_before - total_retained,
        "frames_before": total_frames_before,
        "frames_retained": total_frames_retained,
        "frames_removed": total_frames_before - total_frames_retained,
        "splits": split_summary,
        "episode_count": len(records),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    print_summary(summary)

    if args.check_only:
        print("[CHECK-ONLY] No dataset files were modified.")
        return 0

    updated_manifest = dict(manifest)
    updated_manifest["samples"] = total_retained
    updated_records = []
    stats_by_name = {row["file"]: row for row in episode_results}
    for record in records:
        new_record = dict(record)
        new_record["samples"] = stats_by_name[record["name"]]["samples_retained"]
        updated_records.append(new_record)
    updated_manifest["episode_files"] = updated_records
    updated_manifest["straight_only_filter"] = {
        "source_manifest_sha256": summary["source_manifest_sha256"],
        "threshold_mean_signed_curvature_1pm": float(args.threshold),
        "curvature_source": "target_lane_polyline",
        "curvature_definition": summary["curvature_definition"],
        "frame_retention_rule": summary["frame_retention_rule"],
        "expected_egos": int(args.expected_egos),
        "samples_before": total_before,
        "samples_retained": total_retained,
        "samples_removed": total_before - total_retained,
        "frames_before": total_frames_before,
        "frames_retained": total_frames_retained,
        "split_summary": split_summary,
    }

    reports_dir = dataset_root / "reports"
    report_path = reports_dir / "straight_only_filter_report.json"
    csv_path = reports_dir / "straight_only_filter_episode_audit.csv"
    manifest_tmp = reports_dir / "manifest.json.straight_only.tmp"
    report_tmp = reports_dir / "straight_only_filter_report.json.tmp"
    csv_tmp = reports_dir / "straight_only_filter_episode_audit.csv.tmp"
    manifest_tmp.write_text(json.dumps(updated_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report_tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_results[0]))
        writer.writeheader()
        writer.writerows(episode_results)

    os.replace(episodes_dir, backup_dir)
    try:
        os.replace(stage_dir, episodes_dir)
        os.replace(manifest_tmp, manifest_path)
        os.replace(report_tmp, report_path)
        os.replace(csv_tmp, csv_path)
    except Exception:
        if episodes_dir.exists():
            shutil.rmtree(episodes_dir)
        os.replace(backup_dir, episodes_dir)
        manifest_path.write_bytes(manifest_bytes)
        raise
    shutil.rmtree(backup_dir)

    print(f"[APPLY PASS] filtered dataset updated: {dataset_root}")
    print(f"[REPORT] {report_path}")
    print(f"[AUDIT] {csv_path}")
    print("[SPLIT] Original episode-level train/val/test assignment is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
