"""Dependency-light static W2 contract test using a synthetic W1 shard."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from data_pipeline.expert_dataset import make_dataloader
from data_pipeline.feature_schema import FEATURE_KEYS, MASK_KEYS, SCHEMA
from pretraining.contract import validate_training_batch


def make_split(root: Path, split: str) -> None:
    path = root / split
    path.mkdir(parents=True, exist_ok=True)
    n = 4
    data = {}
    for key, shape in SCHEMA.feature_shapes.items():
        if key in MASK_KEYS:
            arr = np.ones((n, *shape), dtype=np.bool_)
        else:
            arr = np.zeros((n, *shape), dtype=np.float32)
        data[key] = arr
    data["expert_trajectory"] = np.zeros((n, SCHEMA.horizon_steps, 2), dtype=np.float32)
    data["expert_mode"] = np.zeros((n,), dtype=np.int64)
    data["expert_semantic"] = np.zeros((n,), dtype=np.int64)
    data["scenario"] = np.asarray(["straight"] * n, dtype="U32")
    data["seed"] = np.arange(n, dtype=np.int64)
    data["episode_id"] = np.asarray([f"e{i}" for i in range(n)], dtype="U32")
    data["frame_id"] = np.zeros((n,), dtype=np.int64)
    data["vehicle_role"] = np.asarray([0, 1, 2, 0], dtype=np.int64)
    data["sim_time"] = np.zeros((n,), dtype=np.float32)
    data["task_success"] = np.ones((n,), dtype=np.bool_)
    np.savez_compressed(path / "shard-00000.npz", **data)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for split in ("train", "val", "test"):
            make_split(root, split)
        loader = make_dataloader(root, split="train", batch_size=4, shuffle=False)
        batch = next(iter(loader))
        size = validate_training_batch(batch)
        assert size == 4
        assert tuple(batch["expert_trajectory"].shape) == (4, 8, 2)
        assert tuple(batch["features"]["coarse_trajectories"].shape) == (4, 10, 8, 2)
    print("PASS: W1 -> W2 batch contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
