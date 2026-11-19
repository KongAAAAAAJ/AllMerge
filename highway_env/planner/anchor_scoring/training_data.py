"""Raw-only A2 scorer data path with candidate-count-aware episode cache.

The expensive candidate pool and its deterministic safety teacher labels are
computed once per dataset/configuration and stored as memory-mapped NumPy
arrays. Training epochs only load scene features plus cached supervision.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.planner.adaptive_anchors import AdaptiveCandidateBuilder
from .safety import anchor_safety_cost


CACHE_VERSION = "TRACE_A2_RAW_SCORER_CACHE_V5"
CANDIDATE_ALGORITHM_VERSION = "shared-control-candidate-pool-v3"
FEATURE_KEYS = (*RawExpertDataset.FEATURE_KEYS, *RawExpertDataset.ROAD_KEYS)
BOOL_FEATURE_KEYS = {'agent_valid_mask', 'map_valid_mask', 'road_lane_point_mask',
                     'road_lane_mask', 'road_lane_connections'}


def cache_array_specs(candidate_count: int):
    """Return cache array shapes for the configured candidate pool size."""
    count = int(candidate_count)
    if count <= 0:
        raise ValueError("candidate_count must be positive")
    return {
        "coarse_controls": ("coarse_controls.npy", np.float32, (count, 40, 2)),
        "coarse_trajectories": ("coarse_trajectories.npy", np.float32, (count, 8, 2)),
        "mode_valid_mask": ("mode_valid_mask.npy", np.bool_, (count,)),
        "safety_total": ("safety_total.npy", np.float32, (count,)),
        "safety_collision": ("safety_collision.npy", np.float32, (count,)),
        "safety_road": ("safety_road.npy", np.float32, (count,)),
        "safety_hard_safe": ("safety_hard_safe.npy", np.bool_, (count,)),
        "teacher_target": ("teacher_target.npy", np.float32, (count,)),
    }


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_manifest(dataset: RawExpertDataset):
    records = []
    counts = [0] * len(dataset.shard_paths)
    for shard_idx, _ in dataset._index:
        counts[int(shard_idx)] += 1
    for shard_idx, path in enumerate(dataset.shard_paths):
        stat = path.stat()
        count = counts[shard_idx]
        records.append({
            "name": path.name,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sample_count": int(count),
        })
    return records


def cache_fingerprint(dataset, *, candidate_config, teacher_config, temperature,
                      precompute_device="cpu"):
    """Fingerprint every input that can change cached candidates or labels."""
    payload = {
        "cache_version": CACHE_VERSION,
        "candidate_algorithm": CANDIDATE_ALGORITHM_VERSION,
        "dataset_root": str(dataset.shard_paths[0].parent.parent.resolve()),
        "split_manifest": dataset.split_manifest,
        "num_samples": len(dataset),
        "source_episodes": _source_manifest(dataset),
        "candidate_config": candidate_config,
        "teacher_config": teacher_config,
        "temperature": float(temperature),
        "precompute_device": str(precompute_device),
    }
    return _json_hash(payload), payload


def _group_dataset_indices(dataset):
    groups: Dict[int, List[tuple]] = {}
    for global_idx, (shard_idx, sample_idx) in enumerate(dataset._index):
        groups.setdefault(int(shard_idx), []).append((global_idx, int(sample_idx)))
    return groups


def _feature_batch_from_shard(shard, sample_indices):
    features = {}
    for key in FEATURE_KEYS:
        if key not in shard:
            raise KeyError(f"A2 raw episode is missing required feature {key!r}")
        value = np.asarray(shard[key][sample_indices])
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        features[key] = (tensor.bool() if key in BOOL_FEATURE_KEYS else
                         tensor.long() if key.endswith('_slot') else tensor.float())
    return features


def _cache_is_valid(cache_dir: Path, fingerprint: str, num_samples: int, candidate_count: int) -> bool:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_version") != CACHE_VERSION:
            return False
        if manifest.get("fingerprint") != fingerprint:
            return False
        if int(manifest.get("num_samples", -1)) != int(num_samples):
            return False
        array_specs = cache_array_specs(candidate_count)
        if int(manifest.get("candidate_count", -1)) != int(candidate_count):
            return False
        for key, (filename, _, tail_shape) in array_specs.items():
            path = cache_dir / filename
            if not path.is_file():
                return False
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                if tuple(array.shape) != (int(num_samples),) + tuple(tail_shape):
                    return False
            finally:
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def prepare_split_cache(
    dataset_root,
    split: str,
    cfg: dict,
    cache_root,
    *,
    rebuild: bool = False,
    progress_every_shards: int = 1,
):
    """Build one split cache. Returns its final directory and manifest."""
    dataset_root = Path(dataset_root)
    cache_root = Path(cache_root)
    dataset = RawExpertDataset(dataset_root, split=split, max_samples=cfg.get("max_samples"))
    if not len(dataset):
        raise ValueError('A2 raw training split has no selected samples')
    generator = AdaptiveCandidateBuilder(cfg.get("candidate_generation"))
    effective_candidate_config = generator.resolved_config
    candidate_count = int(generator.num_candidates)
    array_specs = cache_array_specs(candidate_count)
    teacher = dict(cfg.get("teacher", {}))
    temperature = float(cfg.get("temperature", 0.3))
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    device_name = str(cfg.get("precompute_device", "cpu"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    fingerprint, fingerprint_payload = cache_fingerprint(
        dataset,
        candidate_config=effective_candidate_config,
        teacher_config=teacher,
        temperature=temperature,
        precompute_device=str(device),
    )
    final_dir = cache_root / fingerprint / split
    if not rebuild and _cache_is_valid(final_dir, fingerprint, len(dataset), candidate_count):
        print(f"[A2-cache] reuse split={split} samples={len(dataset)} path={final_dir}", flush=True)
        return final_dir, json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = final_dir.with_name(final_dir.name + f".tmp-{os.getpid()}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    arrays = {}
    for key, (filename, dtype, tail_shape) in array_specs.items():
        arrays[key] = np.lib.format.open_memmap(
            temp_dir / filename, mode="w+", dtype=dtype,
            shape=(len(dataset),) + tuple(tail_shape),
        )

    groups = _group_dataset_indices(dataset)
    batch_size = max(1, int(cfg.get("precompute_batch_size", 16)))
    started = time.perf_counter()
    total_done = 0

    try:
        for shard_idx, entries in groups.items():
            shard_path = dataset.shard_paths[shard_idx]
            with np.load(shard_path, allow_pickle=False) as source:
                shard = {key: np.array(source[key], copy=True) for key in FEATURE_KEYS}
            for start in range(0, len(entries), batch_size):
                chunk = entries[start:start + batch_size]
                global_indices = np.asarray([item[0] for item in chunk], dtype=np.int64)
                sample_indices = np.asarray([item[1] for item in chunk], dtype=np.int64)
                features = _feature_batch_from_shard(shard, sample_indices)

                pool = generator.build_batch(features)
                controls_np = np.asarray(pool["coarse_controls"], dtype=np.float32)
                xy_np = np.asarray(pool["coarse_trajectories"], dtype=np.float32)
                valid_np = np.asarray(pool["mode_valid_mask"], dtype=np.bool_)

                safety_features = {
                    name: features[name].to(device)
                    for name in ("agent_states", "agent_valid_mask", "ego_state",
                                 "road_lane_geometry", "road_lane_point_mask", "road_lane_mask")
                }
                xy_tensor = torch.as_tensor(xy_np, dtype=torch.float32, device=device)
                valid_tensor = torch.as_tensor(valid_np, dtype=torch.bool, device=device)
                with torch.inference_mode():
                    labels = anchor_safety_cost(
                        safety_features, xy_tensor, valid=valid_tensor, **teacher
                    )
                    teacher_logits = (-labels["total"] / temperature).masked_fill(
                        ~valid_tensor, -torch.inf
                    )
                    target = torch.softmax(teacher_logits, dim=-1)

                arrays["coarse_controls"][global_indices] = controls_np
                arrays["coarse_trajectories"][global_indices] = xy_np
                arrays["mode_valid_mask"][global_indices] = valid_np
                arrays["safety_total"][global_indices] = labels["total"].cpu().numpy()
                arrays["safety_collision"][global_indices] = labels["collision"].cpu().numpy()
                arrays["safety_road"][global_indices] = labels["road"].cpu().numpy()
                arrays["safety_hard_safe"][global_indices] = labels["hard_safe"].cpu().numpy()
                arrays["teacher_target"][global_indices] = target.cpu().numpy()

                total_done += len(chunk)

            if (len(groups) <= 10 or (shard_idx + 1) % max(1, progress_every_shards) == 0):
                elapsed = max(time.perf_counter() - started, 1e-6)
                rate = total_done / elapsed
                print(
                    f"[A2-cache] split={split} episode={shard_path.name} "
                    f"done={total_done}/{len(dataset)} rate={rate:.1f} samples/s",
                    flush=True,
                )

        for value in arrays.values():
            value.flush()
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
        arrays.clear()

        shard_records = []
        offset = 0
        for shard_idx, path in enumerate(dataset.shard_paths):
            count = len(groups.get(shard_idx, []))
            stat = path.stat()
            shard_records.append({
                "name": path.name, "count": count, "offset": offset,
                "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns),
            })
            offset += count

        manifest = {
            "cache_version": CACHE_VERSION,
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "split": split,
            "num_samples": len(dataset),
            "candidate_count": candidate_count,
            "control_steps": 40,
            "trajectory_steps": 8,
            "arrays": {
                key: {"file": spec[0], "dtype": str(np.dtype(spec[1])),
                      "shape": [len(dataset), *spec[2]]}
                for key, spec in array_specs.items()
            },
            "episodes": shard_records,
            "created_unix": time.time(),
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.replace(final_dir)
        elapsed = max(time.perf_counter() - started, 1e-6)
        print(
            f"[A2-cache] complete split={split} samples={len(dataset)} "
            f"elapsed={elapsed/60:.1f} min rate={len(dataset)/elapsed:.1f} samples/s "
            f"path={final_dir}",
            flush=True,
        )
        return final_dir, manifest
    except Exception:
        for value in arrays.values():
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                try:
                    mmap.close()
                except Exception:
                    pass
        arrays.clear()
        raise


class CachedAnchorScoringDataset(Dataset):
    """Scene features plus memory-mapped candidate/safety targets."""

    def __init__(self, dataset, cache_dir):
        self.base = dataset
        self.cache_dir = Path(cache_dir)
        manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get('cache_version') != CACHE_VERSION:
            raise ValueError('A2 cache format is obsolete; rebuild the raw cache')
        if manifest['fingerprint_payload']['split_manifest'] != dataset.split_manifest:
            raise ValueError('A2 raw cache split manifest differs from source dataset')
        if manifest['fingerprint_payload']['source_episodes'] != _source_manifest(dataset):
            raise ValueError('A2 raw source episodes changed; rebuild the cache')
        if int(manifest["num_samples"]) != len(dataset):
            raise ValueError("A2 cache sample count differs from source dataset")
        expected_names = [p.name for p in dataset.shard_paths]
        cached_names = [x["name"] for x in manifest["episodes"]]
        if expected_names != cached_names:
            raise ValueError("A2 cache episode order differs from source dataset")
        self.offset_by_shard = {
            i: int(record["offset"]) for i, record in enumerate(manifest["episodes"])
        }
        self.candidate_count = int(manifest["candidate_count"])
        self.array_specs = cache_array_specs(self.candidate_count)
        self._arrays = None

    def __len__(self):
        return len(self.base)

    def _load_arrays(self):
        if self._arrays is None:
            self._arrays = {
                key: np.load(self.cache_dir / filename, mmap_mode="r", allow_pickle=False)
                for key, (filename, _, _) in self.array_specs.items()
            }
        return self._arrays

    def __getitem__(self, idx):
        shard_idx, sample_idx = self.base._index[idx]
        raw_features, _, _ = self.base[idx]
        features = {key: raw_features[key] for key in FEATURE_KEYS}

        cache_idx = self.offset_by_shard[int(shard_idx)] + int(sample_idx)
        cached = self._load_arrays()
        cache_item = {
            key: torch.from_numpy(np.array(value[cache_idx], copy=True))
            for key, value in cached.items()
        }
        return features, cache_item


class ShardGroupedBatchSampler(Sampler[List[int]]):
    """Shuffle shard order and sample order while keeping each batch shard-local."""

    def __init__(self, dataset, batch_size: int, *, shuffle: bool, seed: int = 0,
                 drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.groups = _group_dataset_indices(dataset)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        shard_ids = list(self.groups)
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(shard_ids)
        for shard_idx in shard_ids:
            indices = np.asarray([item[0] for item in self.groups[shard_idx]], dtype=np.int64)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch.tolist()

    def __len__(self):
        total = 0
        for entries in self.groups.values():
            q, r = divmod(len(entries), self.batch_size)
            total += q if self.drop_last else q + int(r > 0)
        return total


def build_cached_loader(dataset_root, split, cfg, cache_dir, *, shuffle):
    base = RawExpertDataset(Path(dataset_root), split=split, max_samples=cfg.get("max_samples"))
    cached = CachedAnchorScoringDataset(base, cache_dir)
    sampler = ShardGroupedBatchSampler(
        base,
        int(cfg.get("batch_size", 64)),
        shuffle=shuffle,
        seed=int(cfg.get("seed", 0)),
        drop_last=bool(cfg.get("drop_last", False)) if shuffle else False,
    )
    workers = max(0, int(cfg.get("num_workers", 2)))
    kwargs = {
        "dataset": cached,
        "batch_sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(torch.cuda.is_available()),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = max(1, int(cfg.get("prefetch_factor", 2)))
    return torch.utils.data.DataLoader(**kwargs), sampler
