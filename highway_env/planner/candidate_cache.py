"""Shared raw-scene candidate pools for A1/A2 diffusion training."""
from pathlib import Path
import json
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset

from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.planner.adaptive_anchors import AdaptiveCandidateBuilder

ALGORITHM_VERSION = "shared-control-candidate-pool-v3"
ARRAYS = {
    "coarse_controls": (np.dtype("float32"), (40, 2)),
    "coarse_trajectories": (np.dtype("float32"), (8, 2)),
    "mode_valid_mask": (np.dtype("bool"), ()),
}
CANDIDATE_KEYS = {
    "coarse_controls": "a2_candidate_controls",
    "coarse_trajectories": "a2_candidate_trajectories",
    "mode_valid_mask": "a2_candidate_valid",
}


def source_identity(dataset):
    return {
        "split_manifest": dataset.split_manifest,
        "episodes": [
            {"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in dataset.shard_paths
        ],
    }


def build_cache(dataset_root, cache_root, candidate_config, splits=("train", "val")):
    builder = AdaptiveCandidateBuilder(candidate_config)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    for split in splits:
        destination = cache_root / split
        if destination.exists():
            raise FileExistsError(destination)
        dataset = RawExpertDataset(dataset_root, split=split)
        count, candidates = len(dataset), builder.num_candidates
        with tempfile.TemporaryDirectory(prefix=f".{split}-", dir=cache_root) as temporary:
            temporary = Path(temporary)
            arrays = {
                key: np.lib.format.open_memmap(
                    temporary / f"{key}.npy", mode="w+", dtype=dtype,
                    shape=(count, candidates, *tail),
                ) for key, (dtype, tail) in ARRAYS.items()
            }
            for index in range(count):
                features, _, _ = dataset[index]
                pool = builder.build_batch({key: value.unsqueeze(0) for key, value in features.items()})
                for key, value in arrays.items():
                    value[index] = pool[key][0]
            for value in arrays.values():
                value.flush()
                value._mmap.close()
            arrays.clear()
            manifest = {
                "algorithm_version": ALGORITHM_VERSION,
                "source": source_identity(dataset),
                "sample_order": dataset._index,
                "candidate_config": builder.resolved_config,
                "candidate_count": candidates,
            }
            (temporary / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            temporary.rename(destination)
        print(f"[candidate-cache] split={split} samples={count} path={destination}", flush=True)


class CachedCandidateDataset(Dataset):
    def __init__(self, dataset, cache_root, split, candidate_config):
        self.dataset = dataset
        self.directory = Path(cache_root) / split
        manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        expected_config = AdaptiveCandidateBuilder(candidate_config).resolved_config
        if (manifest["algorithm_version"] != ALGORITHM_VERSION
                or manifest["source"] != source_identity(dataset)
                or manifest["candidate_config"] != expected_config
                or manifest["sample_order"][:len(dataset)] != [list(row) for row in dataset._index]):
            raise ValueError("Candidate cache does not match dataset/configuration; generate a matching cache")
        count = len(manifest["sample_order"])
        candidates = expected_config["candidate_pool_size"]
        for key, (dtype, tail) in ARRAYS.items():
            array = np.load(self.directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            matches = array.shape == (count, candidates, *tail) and array.dtype == dtype
            array._mmap.close()
            if not matches:
                raise ValueError(f"Invalid candidate cache array: {key}")
        self._arrays = None

    def __len__(self):
        return len(self.dataset)

    def __getstate__(self):
        return {**self.__dict__, "_arrays": None}

    def __getitem__(self, index):
        if self._arrays is None:
            self._arrays = {key: np.load(self.directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                            for key in ARRAYS}
        features, targets, metadata = self.dataset[index]
        features.update({CANDIDATE_KEYS[key]: torch.from_numpy(np.array(value[index], copy=True))
                         for key, value in self._arrays.items()})
        return features, targets, metadata
