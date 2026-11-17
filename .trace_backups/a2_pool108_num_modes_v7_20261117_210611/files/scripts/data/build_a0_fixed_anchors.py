"""Build TRACE A0 fixed control anchors with control-space K-means.

The current AllMerge expert shards store the 10 scene-conditioned A1 control
candidates in ``coarse_controls [N,10,40,2]``. This builder pools those
collected 40-step control sequences over the TRAIN split and clusters them
DIRECTLY in normalized control space. K-means centroids are used as the fixed
control anchors; no medoid/sample retrieval or trajectory-to-control matching
is performed.

For 4-D candidate banks, ``coarse_trajectories`` is used only to discard
zero-filled geometrically unavailable slots. ``mode_valid_mask`` is never
read, so A0 does not inherit A1 traffic/collision filtering.

If a dataset provides one 40-step control sequence per sample, pass its field
name through ``--control-key``; [N,40,2] is also supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMPLEMENTATION_ID = "A0_FIXED_CONTROL_KMEANS_CENTROIDS_V3"
CONTROL_STEPS = 40
CONTROL_DIM = 2


def _resolve_dataset_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_shards(dataset_root: Path, split: str = "train") -> list[Path]:
    if split != "train":
        raise ValueError("A0 fixed control anchors must be clustered from split='train' only")
    shard_dir = dataset_root / "shards"
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"Shard directory not found: {shard_dir}")
    available = {p.name: p for p in shard_dir.glob("shard_*.npz")}
    split_file = dataset_root / "splits" / "train.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Training split file not found: {split_file}")
    names = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not names:
        raise RuntimeError(f"Training split is empty: {split_file}")
    missing = [Path(name).name for name in names if Path(name).name not in available]
    if missing:
        raise FileNotFoundError(f"Training split references missing shards: {missing[:5]}")
    return [available[Path(name).name] for name in names]


def _candidate_mask(controls: np.ndarray, trajectories: np.ndarray | None,
                    min_path_length_m: float) -> np.ndarray:
    finite = np.isfinite(controls).all(axis=(-1, -2))
    if trajectories is None:
        return finite
    if trajectories.shape[:-2] != controls.shape[:-2] or trajectories.shape[-2:] != (8, 2):
        raise ValueError(
            f"control/trajectory candidate shapes do not align: {controls.shape} vs {trajectories.shape}"
        )
    finite &= np.isfinite(trajectories).all(axis=(-1, -2))
    segment = np.linalg.norm(np.diff(trajectories, axis=-2), axis=-1)
    path_length = segment.sum(axis=-1)
    # A1 stores unavailable geometry slots as all-zero XY/control arrays.
    # This geometric availability check deliberately does NOT inspect the
    # traffic-filtered mode_valid_mask.
    return finite & (path_length >= float(min_path_length_m))


def load_control_candidates(
    shards: list[Path], *, control_key: str = "coarse_controls",
    trajectory_key: str = "coarse_trajectories", max_samples: int = 0,
    min_path_length_m: float = 0.5,
) -> tuple[np.ndarray, dict]:
    parts: list[np.ndarray] = []
    scene_samples = 0
    candidate_count = 0
    source_candidate_count = 0
    source_shape = None
    used_trajectory_geometry_filter = False
    selected_shards: list[str] = []

    for shard_path in shards:
        with np.load(shard_path, allow_pickle=False) as data:
            if control_key not in data.files:
                raise KeyError(f"{shard_path} missing control field {control_key!r}")
            controls = np.asarray(data[control_key], dtype=np.float32)
            if controls.ndim not in (3, 4) or controls.shape[-2:] != (CONTROL_STEPS, CONTROL_DIM):
                raise ValueError(
                    f"{shard_path}:{control_key} must be [N,40,2] or [N,M,40,2], got {controls.shape}"
                )
            n = len(controls)
            if max_samples > 0:
                remaining = max_samples - scene_samples
                if remaining <= 0:
                    break
                controls = controls[:remaining]
                n = len(controls)
            if n == 0:
                continue
            source_shape = list(np.asarray(data[control_key]).shape[1:])

            if controls.ndim == 4:
                if trajectory_key not in data.files:
                    raise KeyError(
                        f"{shard_path} has multi-candidate {control_key}, but lacks {trajectory_key}; "
                        "the trajectory field is needed only to exclude zero-filled unavailable geometry slots"
                    )
                trajectories = np.asarray(data[trajectory_key][:n], dtype=np.float32)
                if trajectories.shape != (n, controls.shape[1], 8, 2):
                    raise ValueError(
                        f"{shard_path}:{trajectory_key} expected {(n, controls.shape[1], 8, 2)}, "
                        f"got {trajectories.shape}"
                    )
                valid = _candidate_mask(controls, trajectories, min_path_length_m)
                used_trajectory_geometry_filter = True
                flat_controls = controls.reshape(-1, CONTROL_STEPS, CONTROL_DIM)
                flat_valid = valid.reshape(-1)
            else:
                flat_controls = controls
                flat_valid = np.isfinite(flat_controls).all(axis=(-1, -2))

            source_candidate_count += int(flat_controls.shape[0])
            selected = flat_controls[flat_valid]
            if selected.size:
                parts.append(np.ascontiguousarray(selected, dtype=np.float32))
                candidate_count += int(len(selected))
            scene_samples += n
            selected_shards.append(shard_path.name)
            if max_samples > 0 and scene_samples >= max_samples:
                break

    if not parts:
        raise RuntimeError("No valid control sequences were found in the training split")
    candidates = np.concatenate(parts, axis=0)
    if len(candidates) < 10:
        raise RuntimeError(f"Need at least 10 valid control sequences, found {len(candidates)}")
    if not np.isfinite(candidates).all():
        raise RuntimeError("Nonfinite controls remained after candidate filtering")

    audit = {
        "source_split": "train",
        "source_shards": selected_shards,
        "num_scene_samples": int(scene_samples),
        "control_source_key": control_key,
        "control_source_shape_per_sample": source_shape,
        "num_source_control_sequences": int(source_candidate_count),
        "num_valid_control_sequences": int(candidate_count),
        "num_filtered_control_sequences": int(source_candidate_count - candidate_count),
        "trajectory_geometry_filter_used": bool(used_trajectory_geometry_filter),
        "trajectory_geometry_filter_key": trajectory_key if used_trajectory_geometry_filter else None,
        "min_path_length_m": float(min_path_length_m) if used_trajectory_geometry_filter else None,
        "mode_valid_mask_used": False,
    }
    return candidates, audit


def _squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x2 = np.einsum("nd,nd->n", x, x)
    c2 = np.einsum("kd,kd->k", centers, centers)
    d = x2[:, None] + c2[None, :] - 2.0 * (x @ centers.T)
    np.maximum(d, 0.0, out=d)
    return d.astype(np.float32, copy=False)


def _kmeans_plus_plus(x: np.ndarray, k: int, rng: np.random.Generator,
                      init_samples: int = 50000) -> np.ndarray:
    if len(x) < k:
        raise ValueError(f"Need at least {k} controls for K-means, got {len(x)}")
    if len(x) > init_samples:
        subset = x[rng.choice(len(x), size=init_samples, replace=False)]
    else:
        subset = x
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    first = int(rng.integers(len(subset)))
    centers[0] = subset[first]
    closest = _squared_distances(subset, centers[:1])[:, 0]
    for j in range(1, k):
        total = float(closest.sum(dtype=np.float64))
        if total <= 1e-12:
            index = int(np.argmax(closest))
        else:
            index = int(rng.choice(len(subset), p=closest.astype(np.float64) / total))
        centers[j] = subset[index]
        closest = np.minimum(closest, _squared_distances(subset, centers[j:j + 1])[:, 0])
    return centers


def _assign_chunks(x: np.ndarray, centers: np.ndarray, chunk_size: int):
    n = len(x)
    labels = np.empty(n, dtype=np.int64)
    min_dist = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        d = _squared_distances(x[start:stop], centers)
        labels[start:stop] = d.argmin(axis=1)
        min_dist[start:stop] = d.min(axis=1)
    return labels, min_dist


def minibatch_kmeans(
    x: np.ndarray, k: int, *, seed: int = 0, batch_size: int = 8192,
    epochs: int = 10, lloyd_refine: int = 5, chunk_size: int = 16384,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Deterministic mini-batch K-means followed by full-data Lloyd refinement."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2 or len(x) < k or k <= 0:
        raise ValueError(f"K-means input must be [N,D], N>=K>0; got {x.shape}, K={k}")
    rng = np.random.default_rng(seed)
    centers = _kmeans_plus_plus(x, k, rng)
    counts = np.zeros(k, dtype=np.float64)
    iterations = 0

    for _epoch in range(max(1, int(epochs))):
        permutation = rng.permutation(len(x))
        epoch_shift = 0.0
        for start in range(0, len(x), max(1, int(batch_size))):
            batch = x[permutation[start:start + batch_size]]
            labels = _squared_distances(batch, centers).argmin(axis=1)
            previous = centers.copy()
            for cluster_id in range(k):
                members = batch[labels == cluster_id]
                if not len(members):
                    continue
                n_new = float(len(members))
                total = counts[cluster_id] + n_new
                batch_mean = members.mean(axis=0, dtype=np.float64).astype(np.float32)
                centers[cluster_id] += (n_new / total) * (batch_mean - centers[cluster_id])
                counts[cluster_id] = total
            epoch_shift = max(epoch_shift, float(np.linalg.norm(centers - previous, axis=1).max()))
            iterations += 1

    # Full-data Lloyd refinement makes each saved prototype a true K-means
    # centroid of its final assignment, rather than an observed medoid.
    for _ in range(max(0, int(lloyd_refine))):
        sums = np.zeros_like(centers, dtype=np.float64)
        cluster_counts = np.zeros(k, dtype=np.int64)
        farthest_points = []
        for start in range(0, len(x), chunk_size):
            chunk = x[start:start + chunk_size]
            d = _squared_distances(chunk, centers)
            labels = d.argmin(axis=1)
            nearest = d.min(axis=1)
            take = min(k, len(chunk))
            local_top = np.argpartition(nearest, len(nearest) - take)[-take:]
            for local_index in local_top:
                farthest_points.append((float(nearest[local_index]), chunk[local_index].copy()))
            if len(farthest_points) > 2 * k:
                farthest_points = sorted(farthest_points, key=lambda item: item[0], reverse=True)[:k]
            for cluster_id in range(k):
                members = chunk[labels == cluster_id]
                if len(members):
                    sums[cluster_id] += members.sum(axis=0, dtype=np.float64)
                    cluster_counts[cluster_id] += len(members)
        farthest_points = sorted(farthest_points, key=lambda item: item[0], reverse=True)[:k]
        updated = centers.copy()
        empty_ids = []
        for cluster_id in range(k):
            if cluster_counts[cluster_id] > 0:
                updated[cluster_id] = (sums[cluster_id] / cluster_counts[cluster_id]).astype(np.float32)
            else:
                empty_ids.append(cluster_id)
        for cluster_id, (_, point) in zip(empty_ids, farthest_points):
            updated[cluster_id] = point
        shift = float(np.linalg.norm(updated - centers, axis=1).max())
        centers = updated
        iterations += 1
        if shift <= 1e-5:
            break

    labels, min_dist = _assign_chunks(x, centers, chunk_size)
    sizes = np.bincount(labels, minlength=k).astype(np.int64)
    if np.any(sizes == 0):
        raise RuntimeError(f"K-means produced empty clusters: {sizes.tolist()}")
    inertia = float(min_dist.sum(dtype=np.float64))
    return centers, sizes, inertia, iterations


def _content_sha256(arrays: Iterable[np.ndarray], metadata_text: str) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(metadata_text.encode("utf-8"))
    return digest.hexdigest()


def build_fixed_anchor_bank(
    dataset_root: str | Path, output_path: str | Path, *, split: str = "train",
    num_anchors: int = 10, seed: int = 0, batch_size: int = 8192,
    epochs: int = 10, lloyd_refine: int = 5, max_samples: int = 0,
    control_key: str = "coarse_controls", trajectory_key: str = "coarse_trajectories",
    min_path_length_m: float = 0.5,
    max_accel_mps2: float = 2.5, max_steer_rad: float = 0.08,
) -> dict:
    if num_anchors != 10:
        raise ValueError("TRACE A0 currently requires exactly 10 anchors to match control.model.num_modes")
    if max_samples < 0 or min_path_length_m < 0:
        raise ValueError("max_samples and min_path_length_m must be nonnegative")
    if batch_size <= 0 or epochs <= 0 or lloyd_refine < 0:
        raise ValueError("batch_size/epochs must be positive and lloyd_refine nonnegative")
    if max_accel_mps2 <= 0 or max_steer_rad <= 0:
        raise ValueError("Physical control scales must be positive")
    root = _resolve_dataset_root(dataset_root)
    shards = resolve_shards(root, split)
    candidates, audit = load_control_candidates(
        shards, control_key=control_key, trajectory_key=trajectory_key,
        max_samples=max_samples, min_path_length_m=min_path_length_m,
    )

    # Dimensionless control utilization gives acceleration and steering equal
    # physical importance while preserving the complete 40-step sequence.
    control_scale = np.asarray([max_accel_mps2, max_steer_rad], dtype=np.float32)
    physical_min = candidates.min(axis=(0, 1)).astype(np.float32)
    physical_max = candidates.max(axis=(0, 1)).astype(np.float32)
    candidates /= control_scale[None, None, :]
    x = candidates.reshape(len(candidates), -1)

    centers_norm, cluster_sizes, inertia, iterations = minibatch_kmeans(
        x, num_anchors, seed=seed, batch_size=batch_size,
        epochs=epochs, lloyd_refine=lloyd_refine,
    )
    centers_controls = (centers_norm.reshape(num_anchors, CONTROL_STEPS, CONTROL_DIM)
                        * control_scale[None, None, :]).astype(np.float32)
    if not np.isfinite(centers_controls).all():
        raise RuntimeError("K-means generated nonfinite control centroids")
    if (np.abs(centers_controls[..., 0]) > max_accel_mps2 + 1e-4).any():
        raise RuntimeError("A0 control centroid exceeds configured acceleration limits")
    if (np.abs(centers_controls[..., 1]) > max_steer_rad + 1e-5).any():
        raise RuntimeError("A0 control centroid exceeds configured steering limits")

    metadata = {
        "implementation": IMPLEMENTATION_ID,
        "num_anchors": int(num_anchors),
        "dataset_root": str(root),
        "source_split": "train",
        "cluster_space": "flattened_40_step_physical_control_sequence",
        "control_order": ["longitudinal_acceleration_mps2", "front_wheel_steering_rad"],
        "control_shape": [CONTROL_STEPS, CONTROL_DIM],
        "control_source": audit,
        "control_source_semantics": (
            "A1_scene_conditioned_control_candidates_stored_with_expert_samples"
            if control_key == "coarse_controls" else "dataset_control_sequence_field"
        ),
        "kmeans": {
            "algorithm": "deterministic_minibatch_kmeans_plus_plus_with_full_lloyd_refinement",
            "seed": int(seed), "batch_size": int(batch_size),
            "epochs": int(epochs), "lloyd_refine": int(lloyd_refine),
            "normalization": "per_channel_physical_limit_scaling",
            "control_scale": control_scale.astype(float).tolist(),
            "objective": "sum_squared_euclidean_distance_over_normalized_flattened_control_sequences",
            "normalized_inertia": float(inertia),
            "iterations": int(iterations),
        },
        "cluster_sizes": cluster_sizes.astype(int).tolist(),
        "prototype_type": "kmeans_centroid",
        "medoid_selection_used": False,
        "trajectory_to_control_matching_used": False,
        "inverse_dynamics_used": False,
        "mode_valid_mask_used": False,
        "safety_scoring_used": False,
        "adaptive_selection_used": False,
        "control_bank_scene_independent": True,
        "trajectory_anchor_policy": "runtime_kinematic_rollout_from_fixed_control_centroid",
        "source_control_min": physical_min.astype(float).tolist(),
        "source_control_max": physical_max.astype(float).tolist(),
    }
    metadata_text = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    bank_hash = _content_sha256((centers_controls, control_scale), metadata_text)

    output = Path(output_path).expanduser()
    output = output.resolve() if output.is_absolute() else (ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            anchors_controls=centers_controls,
            cluster_centers_controls=centers_controls,
            control_scale=control_scale,
            metadata_json=np.asarray(metadata_text),
        )
    temp.replace(output)

    report = {
        "output": str(output), "bank_sha256": bank_hash,
        "num_anchors": int(num_anchors),
        "num_scene_samples": int(audit["num_scene_samples"]),
        "num_source_control_sequences": int(audit["num_source_control_sequences"]),
        "num_valid_control_sequences": int(audit["num_valid_control_sequences"]),
        "num_filtered_control_sequences": int(audit["num_filtered_control_sequences"]),
        "cluster_sizes": cluster_sizes.astype(int).tolist(),
        "normalized_inertia": float(inertia),
        "source_split": "train", "control_source_key": control_key,
        "medoid_selection_used": False,
        "trajectory_to_control_matching_used": False,
        "mode_valid_mask_used": False,
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[A0 CONTROL K-MEANS BUILT]", output)
    print("[A0 BANK SHA256]", bank_hash)
    print("[A0 CONTROL SOURCE]", control_key, audit["control_source_shape_per_sample"])
    print("[A0 VALID SEQUENCES]", audit["num_valid_control_sequences"])
    print("[A0 CLUSTER SIZES]", cluster_sizes.tolist())
    print("[A0 NORMALIZED INERTIA] %.6f" % inertia)
    print("[A0 REPORT]", report_path)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build A0 fixed anchors by direct control-sequence K-means")
    parser.add_argument("--dataset-root", default="outputs/expert_dataset/expert_trucksim_k2_anchor_pair_100k")
    parser.add_argument("--split", default="train", choices=("train",))
    parser.add_argument("--num-anchors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lloyd-refine", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap on source scenes; 0 uses the complete train split")
    parser.add_argument("--control-key", default="coarse_controls")
    parser.add_argument("--trajectory-key", default="coarse_trajectories")
    parser.add_argument("--min-path-length-m", type=float, default=0.5)
    parser.add_argument("--max-accel-mps2", type=float, default=2.5)
    parser.add_argument("--max-steer-rad", type=float, default=0.08)
    parser.add_argument("--output", default="outputs/anchor_ablation/a0_fixed_control_anchors.npz")
    args = parser.parse_args(argv)
    build_fixed_anchor_bank(
        args.dataset_root, args.output, split=args.split,
        num_anchors=args.num_anchors, seed=args.seed,
        batch_size=args.batch_size, epochs=args.epochs,
        lloyd_refine=args.lloyd_refine, max_samples=args.max_samples,
        control_key=args.control_key, trajectory_key=args.trajectory_key,
        min_path_length_m=args.min_path_length_m,
        max_accel_mps2=args.max_accel_mps2, max_steer_rad=args.max_steer_rad,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
