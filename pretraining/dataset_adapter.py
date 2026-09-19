"""Thin adapter from the completed W1 expert dataset to W2 training."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader

from expert_dataset import build_dataset


def build_w1_dataloader(
    dataset_root: str | Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    limit_samples: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    include_metadata: bool = False,
) -> DataLoader:
    dataset = build_dataset(
        dataset_root=dataset_root,
        split=split,
        include_metadata=include_metadata,
        max_samples=(int(limit_samples) if limit_samples > 0 else None),
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"W1 dataset split '{split}' is empty under {Path(dataset_root)}"
        )

    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)


def unpack_w1_batch(batch) -> dict:
    if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
        raise TypeError(
            "W1 DataLoader batch must be (features, targets) or "
            "(features, targets, metadata)"
        )
    features, targets = batch[0], batch[1]
    metadata = batch[2] if len(batch) == 3 else None
    if not isinstance(features, Mapping) or not isinstance(targets, Mapping):
        raise TypeError("W1 features and targets must be mappings")
    return {
        "features": features,
        "expert_trajectory": targets["trajectory"],
        "expert_mode": targets["target_mode"],
        "expert_semantic": targets["target_semantic"],
        "metadata": metadata,
    }


def metadata_item(metadata, key: str, index: int, default=None):
    if metadata is None or key not in metadata:
        return default
    value = metadata[key]
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.item() if item.ndim == 0 else item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[index]
    try:
        item = value[index]
    except Exception:
        return value
    return item.item() if hasattr(item, "item") else item
