from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_pipeline.expert_dataset import AllMergeExpertDatasetConfig, AllMergeExpertShardDataset


def build_complete_frame_subset(dataset_root: Path, split: str, *, num_frames: int = 0,
                                seed: int = 2026, max_samples: int | None = None):
    """Build a reproducible subset containing complete synchronized ego groups."""
    ds = AllMergeExpertShardDataset(AllMergeExpertDatasetConfig(
        dataset_root=Path(dataset_root), split=split, include_metadata=True,
        include_diagnostics=True, max_samples=max_samples,
    ))
    by_shard = defaultdict(list)
    for global_i, (shard_idx, sample_i) in enumerate(ds._index):
        by_shard[int(shard_idx)].append((global_i, int(sample_i)))
    groups = defaultdict(list)
    expected_by_key = {}
    expected_sets = defaultdict(set)
    sample_meta = {}
    for shard_idx, shard_path in enumerate(ds.shard_paths):
        rows = by_shard.get(shard_idx, [])
        if not rows:
            continue
        with np.load(shard_path, allow_pickle=False) as shard:
            required = ("episode_index", "frame_index", "ego_index", "agent_states", "agent_valid_mask")
            missing = [k for k in required if k not in shard.files]
            if missing:
                raise KeyError(f"{shard_path} lacks synchronized frame metadata: {missing}")
            for global_i, sample_i in rows:
                episode = int(np.asarray(shard["episode_index"][sample_i]).item())
                frame = int(np.asarray(shard["frame_index"][sample_i]).item())
                ego_index = int(np.asarray(shard["ego_index"][sample_i]).item())
                key = (episode, frame)
                agents = np.asarray(shard["agent_states"][sample_i])
                agent_valid = np.asarray(shard["agent_valid_mask"][sample_i], dtype=bool)
                controlled = int(np.sum(agent_valid & (agents[..., 10] > 0.5)))
                expected = controlled + 1
                expected_sets[key].add(expected)
                expected_by_key[key] = max(expected_by_key.get(key, 0), expected)
                groups[key].append(global_i)
                sample_meta[global_i] = {
                    "episode_index": episode,
                    "frame_index": frame,
                    "ego_index": ego_index,
                    "expected_ego_count": expected,
                    "seed": int(np.asarray(shard["seed"][sample_i]).item()) if "seed" in shard.files else -1,
                }

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
    return DataLoader(subset, batch_size=int(batch_size), shuffle=False,
                      num_workers=int(num_workers), pin_memory=torch.cuda.is_available(),
                      collate_fn=collate_anchor_batch, persistent_workers=False)
