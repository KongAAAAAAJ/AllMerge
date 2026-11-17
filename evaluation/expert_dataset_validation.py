from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np
from data_pipeline.expert_dataset import FEATURE_KEYS, build_dataset

REQUIRED = set(FEATURE_KEYS) | {
    "coarse_controls", "expert_trajectory_xy", "future_trajectory_dense",
    "expert_trajectory_10hz_xy", "dense_dt", "trajectory_horizon_s",
    "target_mode", "target_semantic", "road_lane_geometry",
    "road_lane_point_mask", "road_lane_mask", "road_lane_connections",
    "road_current_lane_slot", "road_target_lane_slot",
    "koopman_state_hist_si", "koopman_control_hist_si", "koopman_pose0_si",
    "koopman_ltr0", "koopman_payload_kg", "koopman_road_mu", "episode_index",
}


def parse_args():
    p = argparse.ArgumentParser(description="Validate the unified TRACE A1/K2 expert dataset")
    p.add_argument("--dataset-root", type=Path, default=Path("outputs/expert_dataset/expert_trucksim_k2_anchor_pair_100k"))
    p.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-loader-samples", type=int, default=64)
    p.add_argument("--report-path", type=Path, default=None)
    return p.parse_args()


def validate_shard(path: Path, reference_schema=None):
    with np.load(path, allow_pickle=False) as z:
        missing = sorted(REQUIRED - set(z.files))
        if missing:
            raise KeyError(f"{path.name}: missing unified TRACE fields {missing}")
        n = len(z["expert_trajectory_xy"])
        expected = {
            "expert_trajectory_xy": (n, 8, 2),
            "future_trajectory_dense": (n, 40, 2),
            "expert_trajectory_10hz_xy": (n, 41, 2),
            "coarse_trajectories": (n, 10, 8, 2),
            "coarse_controls": (n, 10, 40, 2),
            "mode_valid_mask": (n, 10),
            "koopman_state_hist_si": (n, 5, 5),
            "koopman_control_hist_si": (n, 4, 2),
            "koopman_pose0_si": (n, 3),
        }
        for key, shape in expected.items():
            if z[key].shape != shape:
                raise ValueError(f"{path.name}: {key}={z[key].shape}, expected {shape}")
        if not np.asarray(z["mode_valid_mask"], dtype=bool).any(axis=1).all():
            raise ValueError(f"{path.name}: an A1 anchor set has no valid candidate")
        for key in z.files:
            a = np.asarray(z[key])
            if a.dtype.kind in "fiu" and not np.isfinite(a).all():
                raise ValueError(f"{path.name}: non-finite values in {key}")
        if not np.allclose(z["dense_dt"], 0.1, atol=1e-6):
            raise ValueError(f"{path.name}: dense_dt must be 0.1")
        if not np.allclose(z["trajectory_horizon_s"], 4.0, atol=1e-6):
            raise ValueError(f"{path.name}: trajectory_horizon_s must be 4.0")
        sparse = np.asarray(z["expert_trajectory_xy"])
        dense = np.asarray(z["future_trajectory_dense"])
        if not np.allclose(sparse, dense[:, [4, 9, 14, 19, 24, 29, 34, 39]], atol=1e-4):
            raise ValueError(f"{path.name}: sparse and dense Polynomial targets are inconsistent")
        schema = {k: (list(z[k].shape[1:]), str(z[k].dtype)) for k in z.files}
        if reference_schema is not None and schema != reference_schema:
            raise ValueError(f"{path.name}: shard schema differs from the first shard")
        return schema, {"samples": n, "valid_anchors_mean": float(z["mode_valid_mask"].sum(1).mean()),
                        "episode_ids": sorted(set(np.asarray(z["episode_index"]).reshape(-1).astype(int).tolist()))}


def main():
    args = parse_args()
    root = args.dataset_root.expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    root = root.resolve()
    shards = sorted((root / "shards").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No unified TRACE shards under {root / 'shards'}")
    reference = None
    rows = []
    total = 0
    for path in shards:
        reference, summary = validate_shard(path, reference)
        total += summary["samples"]
        rows.append({"shard": path.name, **summary})
    ds = build_dataset(root, split=args.split, include_metadata=False,
                       max_samples=args.max_loader_samples if args.max_loader_samples > 0 else None)
    if len(ds) == 0:
        raise RuntimeError(f"Dataset split {args.split} is empty")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    features, targets = next(iter(loader))
    report = {
        "schema_version": "trace_expert_road_geometry_a1_k2_v2",
        "dataset_root": str(root), "shard_count": len(shards), "sample_count": total,
        "loader_split": args.split, "loader_samples": len(ds),
        "feature_shapes": {k: list(v.shape) for k, v in features.items()},
        "target_shapes": {k: list(v.shape) for k, v in targets.items() if hasattr(v, "shape")},
        "episodes": rows,
    }
    path = args.report_path or (root / "reports" / "dataset_validation.json")
    if not path.is_absolute():
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
