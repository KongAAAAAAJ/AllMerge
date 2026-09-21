from __future__ import annotations

from dataclasses import dataclass
import io
import os
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_KEYS: Tuple[str, ...] = (
    "ego_state",
    "agent_states",
    "agent_valid_mask",
    "map_polylines",
    "map_valid_mask",
    "target_point",
    "target_lane_polyline",
    "coarse_trajectories",
    "mode_valid_mask",
)

FLOAT_FEATURE_KEYS = {
    "ego_state",
    "agent_states",
    "map_polylines",
    "target_point",
    "target_lane_polyline",
    "coarse_trajectories",
}

BOOL_FEATURE_KEYS = {
    "agent_valid_mask",
    "map_valid_mask",
    "mode_valid_mask",
}

# STAGE3_DENSE_EXPERT_V2
TARGET_KEYS: Tuple[str, ...] = (
    "expert_trajectory_xy",
    "future_trajectory_dense",
    "dense_dt",
    "trajectory_horizon_s",
    "target_mode",
    "target_semantic",
)

DIAGNOSTIC_TARGET_KEYS: Tuple[str, ...] = (
    "target_mode_traffic_valid",
    "target_mode_geometry_valid",
    "target_mode_assignment_distance",
    "selected_anchor_xy",
    "geometry_semantic_residual_xy",
    "geometry_semantic_ade",
    "geometry_semantic_fde",
    "geometry_semantic_max_abs_dx",
    "geometry_semantic_max_abs_dy",
    "contract_ok",
)

DEFAULT_DATASET_ROOT = Path("outputs/expert_dataset/allmerge_expert")


@dataclass(frozen=True)
class AllMergeExpertDatasetConfig:
    dataset_root: Path = DEFAULT_DATASET_ROOT
    split: str = "all"
    include_metadata: bool = False
    include_diagnostics: bool = False
    max_samples: Optional[int] = None
    shard_paths: Optional[Tuple[Path, ...]] = None



# ALLMERGE_CONCURRENT_NPZ_READ_V1
def _read_npz_fully_resilient(
    shard_path: Path,
) -> Dict[str, np.ndarray]:
    """Read one NPZ shard into process-local memory and close file/ZIP handles."""
    max_retries = max(
        1,
        int(os.environ.get("ALLMERGE_NPZ_READ_RETRIES", "5")),
    )
    retry_delay_s = max(
        0.0,
        float(os.environ.get("ALLMERGE_NPZ_RETRY_DELAY_MS", "50")) / 1000.0,
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # One sequential filesystem read. ZIP parsing/decompression happens
            # after the filesystem handle is already closed.
            payload = shard_path.read_bytes()
            with np.load(io.BytesIO(payload), allow_pickle=False) as raw_shard:
                shard = {
                    key: np.array(raw_shard[key], copy=True)
                    for key in raw_shard.files
                }
            return shard
        except (zipfile.BadZipFile, EOFError, OSError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_delay_s * attempt)

    raise RuntimeError(
        "Failed to read NPZ shard after "
        f"{max_retries} attempts: {shard_path}. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    ) from last_error

class AllMergeExpertShardDataset(Dataset):
    """
    Sharded AllMerge structured-planner expert dataset.

    Each stored item is one controlled ego sample. Therefore DataLoader batching
    directly produces the model contract expected by PlannerTensorAdapter:

        ego_state              [B, D_ego]
        agent_states           [B, N_agent, D_agent]
        agent_valid_mask       [B, N_agent]
        map_polylines          [B, N_map, P, D_map]
        map_valid_mask         [B, N_map]
        target_point           [B, 2]
        target_lane_polyline   [B, P, D_map]
        coarse_trajectories    [B, M, 8, 2]
        mode_valid_mask        [B, M]

    Targets:
        trajectory             [B, 8, 2]
        target_mode            [B]
        target_semantic        [B]
    """

    def __init__(self, config: AllMergeExpertDatasetConfig):
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.shard_paths = self._resolve_shards(config)

        if not self.shard_paths:
            raise FileNotFoundError(
                f"No shard files found under {self.dataset_root}"
            )

        self._index: List[Tuple[int, int]] = []
        self._cached_shard_idx: Optional[int] = None
        self._cached_shard: Optional[Dict[str, np.ndarray]] = None
        self._build_index()

    def _resolve_shards(
        self,
        config: AllMergeExpertDatasetConfig,
    ) -> List[Path]:
        if config.shard_paths is not None:
            return sorted(Path(path) for path in config.shard_paths)

        shard_dir = self.dataset_root / "shards"
        if not shard_dir.is_dir():
            raise FileNotFoundError(
                f"Shard directory does not exist: {shard_dir}"
            )

        shard_paths = sorted(shard_dir.glob("shard_*.npz"))
        if config.split == "all":
            return shard_paths

        split_file = self.dataset_root / "splits" / f"{config.split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"Split file does not exist: {split_file}"
            )

        shard_names = {
            line.strip()
            for line in split_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }

        return [
            path
            for path in shard_paths
            if path.name in shard_names
        ]

    def _build_index(self) -> None:
        total_samples = 0

        for shard_idx, shard_path in enumerate(self.shard_paths):
            with np.load(shard_path, allow_pickle=False) as shard:
                if "expert_trajectory_xy" not in shard.files:
                    raise KeyError(
                        f"{shard_path} missing expert_trajectory_xy"
                    )
                shard_len = int(
                    shard["expert_trajectory_xy"].shape[0]
                )

            for sample_idx in range(shard_len):
                self._index.append(
                    (shard_idx, sample_idx)
                )
                total_samples += 1

                if (
                    self.config.max_samples is not None
                    and total_samples >= self.config.max_samples
                ):
                    return

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        shard_idx, sample_idx = self._index[idx]
        shard = self._load_shard(shard_idx)

        features = {}
        for key in FEATURE_KEYS:
            if key not in shard:
                raise KeyError(
                    f"{self.shard_paths[shard_idx]} missing {key}"
                )

            tensor = torch.from_numpy(
                np.asarray(shard[key][sample_idx])
            )

            if key in BOOL_FEATURE_KEYS:
                tensor = tensor.bool()
            else:
                tensor = tensor.float()

            features[key] = tensor

        targets = {
            "trajectory": torch.from_numpy(
                np.asarray(
                    shard["expert_trajectory_xy"][sample_idx]
                )
            ).float(),
            "target_mode": torch.as_tensor(
                shard["target_mode"][sample_idx],
                dtype=torch.long,
            ),
            "target_semantic": torch.as_tensor(
                shard["target_semantic"][sample_idx],
                dtype=torch.long,
            ),
        }

        # STAGE3_DENSE_EXPERT_V2: optional for backward compatibility with v1 shards.
        if "future_trajectory_dense" in shard:
            required_dense_keys = (
                "dense_dt",
                "trajectory_horizon_s",
            )
            missing_dense = [
                key for key in required_dense_keys if key not in shard
            ]
            if missing_dense:
                raise KeyError(
                    f"{self.shard_paths[shard_idx]} dense target missing "
                    f"{missing_dense}"
                )
            targets["trajectory_dense"] = torch.from_numpy(
                np.asarray(shard["future_trajectory_dense"][sample_idx])
            ).float()
            targets["dense_dt"] = torch.as_tensor(
                shard["dense_dt"][sample_idx], dtype=torch.float32
            )
            targets["trajectory_horizon_s"] = torch.as_tensor(
                shard["trajectory_horizon_s"][sample_idx],
                dtype=torch.float32,
            )

        if self.config.include_diagnostics:
            diagnostics = {}
            for key in DIAGNOSTIC_TARGET_KEYS:
                if key not in shard:
                    continue

                value = np.asarray(
                    shard[key][sample_idx]
                )

                if key in {
                    "target_mode_traffic_valid",
                    "target_mode_geometry_valid",
                    "contract_ok",
                }:
                    diagnostics[key] = torch.as_tensor(
                        value,
                        dtype=torch.bool,
                    )
                else:
                    diagnostics[key] = torch.as_tensor(
                        value,
                        dtype=torch.float32,
                    )

            targets["diagnostics"] = diagnostics

        if not self.config.include_metadata:
            return features, targets

        metadata = {}
        excluded = set(FEATURE_KEYS) | set(TARGET_KEYS) | set(
            DIAGNOSTIC_TARGET_KEYS
        )

        for key in shard.keys():
            if key in excluded:
                continue

            value = shard[key][sample_idx]
            if np.isscalar(value) or getattr(value, "shape", ()) == ():
                metadata[key] = (
                    value.item()
                    if hasattr(value, "item")
                    else value
                )
            else:
                metadata[key] = value

        metadata["shard_path"] = str(
            self.shard_paths[shard_idx]
        )
        metadata["shard_sample_index"] = int(sample_idx)

        return features, targets, metadata

    def _load_shard(
        self,
        shard_idx: int,
    ) -> Dict[str, np.ndarray]:
        if (
            self._cached_shard_idx == shard_idx
            and self._cached_shard is not None
        ):
            return self._cached_shard

        shard_path = self.shard_paths[shard_idx]

        # ALLMERGE_CONCURRENT_NPZ_READ_V1
        # Decode from process-local bytes instead of holding a ZipFile against
        # the shared on-disk archive during decompression.
        shard = _read_npz_fully_resilient(shard_path)

        self._cached_shard_idx = shard_idx
        self._cached_shard = shard
        return shard


def build_dataset(
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    split: str = "all",
    include_metadata: bool = False,
    include_diagnostics: bool = False,
    max_samples: Optional[int] = None,
) -> AllMergeExpertShardDataset:
    return AllMergeExpertShardDataset(
        AllMergeExpertDatasetConfig(
            dataset_root=Path(dataset_root),
            split=split,
            include_metadata=include_metadata,
            include_diagnostics=include_diagnostics,
            max_samples=max_samples,
        )
    )


def split_shards(
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> Dict[str, List[str]]:
    """
    Direct migration of the Diffusion-metadrive shard-level split mechanism,
    adapted only for the AllMerge directory contract.
    """
    dataset_root = Path(dataset_root)
    shard_dir = dataset_root / "shards"
    split_dir = dataset_root / "splits"
    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shard_paths = sorted(
        shard_dir.glob("shard_*.npz")
    )
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard files found under {shard_dir}"
        )

    ratios = np.asarray(
        [
            train_ratio,
            val_ratio,
            test_ratio,
        ],
        dtype=np.float64,
    )

    if np.any(ratios < 0.0):
        raise ValueError(
            "Split ratios must be non-negative, "
            f"got {ratios.tolist()}"
        )

    if float(ratios.sum()) <= 0.0:
        raise ValueError(
            "At least one split ratio must be greater than zero"
        )

    ratios = ratios / ratios.sum()

    rng = np.random.RandomState(seed)
    indices = np.arange(len(shard_paths))
    rng.shuffle(indices)

    total_shards = len(indices)
    train_count = int(
        np.floor(
            total_shards * ratios[0]
        )
    )
    val_count = int(
        np.floor(
            total_shards * ratios[1]
        )
    )
    test_count = (
        total_shards
        - train_count
        - val_count
    )

    if (
        total_shards > 0
        and train_count == 0
    ):
        train_count = 1

        if (
            val_count > test_count
            and val_count > 0
        ):
            val_count -= 1
        elif test_count > 0:
            test_count -= 1

    train_slice = indices[
        :train_count
    ]
    val_slice = indices[
        train_count:
        train_count + val_count
    ]
    test_slice = indices[
        train_count + val_count:
        train_count + val_count + test_count
    ]

    train_names = [
        shard_paths[idx].name
        for idx in train_slice
    ]
    val_names = [
        shard_paths[idx].name
        for idx in val_slice
    ]
    test_names = [
        shard_paths[idx].name
        for idx in test_slice
    ]

    split_map = {
        "train": train_names,
        "val": val_names,
        "test": test_names,
    }

    for name, names in split_map.items():
        text = "\n".join(names)
        if text:
            text += "\n"

        (
            split_dir
            / f"{name}.txt"
        ).write_text(
            text,
            encoding="utf-8",
        )

    return split_map
