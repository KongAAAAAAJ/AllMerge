from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_pipeline.raw_expert_dataset import RawExpertDataset


def build_complete_frame_subset(dataset_root: Path, split: str, *, num_frames: int = 0,
                                seed: int = 2026, max_samples: int | None = None):
    """Build a reproducible subset containing complete synchronized ego groups."""
    ds = RawExpertDataset(dataset_root, split=split, max_samples=max_samples)
    by_shard = defaultdict(list)
    for global_i, (shard_idx, sample_i) in enumerate(ds._index):
        by_shard[int(shard_idx)].append((global_i, int(sample_i)))
    groups = defaultdict(list)
    expected_by_key = {}
    expected_sets = defaultdict(set)
    sample_meta = {}
    indexed_samples = 0
    print(f"[DATA] indexing synchronized raw samples: {len(ds)} samples across {len(ds.shard_paths)} episode shards", flush=True)
    for shard_idx, shard_path in enumerate(ds.shard_paths):
        rows = by_shard.get(shard_idx, [])
        if not rows:
            continue
        started = time.perf_counter()
        print(f"[DATA] reading episode={shard_path.name} ({shard_idx + 1}/{len(ds.shard_paths)})", flush=True)
        with np.load(shard_path, allow_pickle=False) as shard:
            required = ("episode_index", "frame_index", "ego_index", "agent_states", "agent_valid_mask")
            missing = [k for k in required if k not in shard.files]
            if missing:
                raise KeyError(f"{shard_path} lacks synchronized frame metadata: {missing}")
            # NpzFile does not cache decompressed arrays. Read each field once
            # per episode instead of decompressing it again for every sample.
            arrays = {key: np.asarray(shard[key]) for key in required}
            seeds = np.asarray(shard['seed']) if 'seed' in shard.files else None
            for global_i, sample_i in rows:
                episode = int(np.asarray(arrays["episode_index"][sample_i]).item())
                frame = int(np.asarray(arrays["frame_index"][sample_i]).item())
                ego_index = int(np.asarray(arrays["ego_index"][sample_i]).item())
                key = (episode, frame)
                agents = arrays["agent_states"][sample_i]
                agent_valid = np.asarray(arrays["agent_valid_mask"][sample_i], dtype=bool)
                controlled = int(np.sum(agent_valid & (agents[..., 10] > 0.5)))
                expected = controlled + 1
                expected_sets[key].add(expected)
                expected_by_key[key] = max(expected_by_key.get(key, 0), expected)
                groups[key].append(global_i)
                indexed_samples += 1
                sample_meta[global_i] = {
                    "episode_index": episode,
                    "frame_index": frame,
                    "ego_index": ego_index,
                    "expected_ego_count": expected,
                    "seed": int(np.asarray(seeds[sample_i]).item()) if seeds is not None else -1,
                }
        elapsed = time.perf_counter() - started
        print(f"[DATA] indexed shards={shard_idx + 1}/{len(ds.shard_paths)} "
              f"samples={indexed_samples}/{len(ds)} elapsed={elapsed:.2f}s", flush=True)

    complete = []
    incomplete = []
    for key, indices in groups.items():
        ego_ids = [sample_meta[i]["ego_index"] for i in indices]
        expected = expected_by_key[key]
        expected_consistent = len(expected_sets[key]) == 1
        if expected_consistent and len(indices) == expected and len(set(ego_ids)) == expected:
            complete.append((key, sorted(indices, key=lambda i: sample_meta[i]["ego_index"])))
        else:
            incomplete.append({"frame_key": key, "observed": len(indices), "expected": expected,
                               "expected_counts": sorted(expected_sets[key]), "ego_indices": ego_ids})
    if not complete:
        raise RuntimeError(
            f"No complete synchronized frames found in split={split}. "
            "Check episode_index/frame_index/ego_index and is_controlled fields."
        )
    complete.sort(key=lambda x: (x[0][0], x[0][1]))
    if num_frames > 0 and num_frames < len(complete):
        rng = random.Random(int(seed))
        chosen = sorted(rng.sample(complete, int(num_frames)), key=lambda x: (x[0][0], x[0][1]))
    else:
        chosen = complete
    selected_indices = [i for _, ids in chosen for i in ids]
    subset = Subset(ds, selected_indices)
    return subset, {
        "split": split,
        "input_format": "raw",
        "episode_split": ds.split_manifest,
        "dataset_samples_in_split": len(ds),
        "complete_frames_available": len(complete),
        "selected_frames": len(chosen),
        "selected_samples": len(selected_indices),
        "incomplete_frames_skipped": len(incomplete),
        "incomplete_frame_examples": incomplete[:20],
        "sample_seed": int(seed),
    }


def collate_anchor_batch(samples):
    features_list, targets_list, metadata_list = zip(*samples)
    feature_keys = features_list[0].keys()
    features = {k: torch.stack([x[k] for x in features_list], dim=0) for k in feature_keys}
    target_keys = set().union(*(x.keys() for x in targets_list))
    targets = {}
    for key in target_keys:
        if key == "diagnostics":
            diag_keys = set().union(*(x.get("diagnostics", {}).keys() for x in targets_list))
            diag = {}
            for dkey in diag_keys:
                vals = [x.get("diagnostics", {}).get(dkey) for x in targets_list]
                if all(isinstance(v, torch.Tensor) for v in vals):
                    diag[dkey] = torch.stack(vals, dim=0)
            targets[key] = diag
            continue
        vals = [x.get(key) for x in targets_list]
        if all(isinstance(v, torch.Tensor) for v in vals):
            targets[key] = torch.stack(vals, dim=0)
    keep_meta = ("episode_index", "frame_index", "ego_index", "seed", "shard_path", "shard_sample_index")
    compact_metadata = [{k: item[k] for k in keep_meta if k in item} for item in metadata_list]
    return features, targets, compact_metadata


def make_loader(subset, *, batch_size: int, num_workers: int = 0):
    workers = int(num_workers)
    worker_options = {'persistent_workers': True, 'prefetch_factor': 1} if workers > 0 else {}
    return DataLoader(subset, batch_size=int(batch_size), shuffle=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      collate_fn=collate_anchor_batch, **worker_options)


def read_subset_metadata(subset):
    """Read IDs for figure selection without loading road geometry or targets."""
    indices = list(range(len(subset)))
    dataset = subset
    while isinstance(dataset, Subset):
        indices = [dataset.indices[i] for i in indices]
        dataset = dataset.dataset
    by_shard = defaultdict(list)
    for position, index in enumerate(indices):
        shard_index, row = dataset._index[index]
        by_shard[shard_index].append((position, row))
    metadata = [None] * len(indices)
    for shard_index, rows in by_shard.items():
        path = dataset.shard_paths[shard_index]
        with np.load(path, allow_pickle=False) as archive:
            keys = ('episode_index', 'frame_index', 'ego_index', 'seed')
            arrays = {key: archive[key] for key in keys if key in archive.files}
            for position, row in rows:
                metadata[position] = {key: int(values[row]) for key, values in arrays.items()}
                metadata[position].update(shard_path=str(path), shard_sample_index=int(row))
    return metadata


def select_figure_samples(metadata, *, count=3, seed=2027):
    """Select distinct frames without model results; prefer distinct episodes."""
    if count < 1:
        raise ValueError('figure scene count must be positive')
    frames = defaultdict(set)
    for meta in metadata:
        key = (int(meta['episode_index']), int(meta['frame_index']))
        frames[key].add(int(meta['ego_index']))
    if not frames:
        raise ValueError('No samples available for anchor figures')
    episodes = defaultdict(list)
    for key in sorted(frames):
        episodes[key[0]].append(key)
    rng = random.Random(int(seed))
    picked = [rng.choice(episodes[episode]) for episode in
              rng.sample(sorted(episodes), min(count, len(episodes)))]
    remaining = sorted(set(frames) - set(picked))
    picked.extend(rng.sample(remaining, min(count - len(picked), len(remaining))))
    return [(*key, rng.choice(sorted(frames[key]))) for key in picked]
