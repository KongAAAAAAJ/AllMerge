"""Synthetic smoke test for the actual W1 root-level expert_dataset contract."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from highway_env.planner.diffusion.config import build_structured_diffusion_config
from pretraining.contract import validate_training_batch
from pretraining.dataset_adapter import build_w1_dataloader


def _write_shard(root: Path) -> None:
    cfg = build_structured_diffusion_config()
    n = 4
    shard_dir = root / "shards"
    split_dir = root / "splits"
    shard_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)

    data = {
        "ego_state": np.zeros((n, cfg.ego_dim), np.float32),
        "agent_states": np.zeros((n, cfg.max_agents, cfg.agent_dim), np.float32),
        "agent_valid_mask": np.ones((n, cfg.max_agents), np.bool_),
        "map_polylines": np.zeros(
            (n, cfg.max_map_polylines, cfg.map_points, cfg.map_dim), np.float32
        ),
        "map_valid_mask": np.ones((n, cfg.max_map_polylines), np.bool_),
        "target_point": np.zeros((n, 2), np.float32),
        "target_lane_polyline": np.zeros((n, cfg.map_points, cfg.map_dim), np.float32),
        "coarse_trajectories": np.zeros(
            (n, cfg.num_modes, cfg.horizon_steps, 2), np.float32
        ),
        "mode_valid_mask": np.ones((n, cfg.num_modes), np.bool_),
        "expert_trajectory_xy": np.zeros((n, cfg.horizon_steps, 2), np.float32),
        "target_mode": np.zeros((n,), np.int64),
        "target_semantic": np.zeros((n,), np.int64),
    }
    name = "shard_000000.npz"
    np.savez_compressed(shard_dir / name, **data)
    for split in ("train", "val", "test"):
        (split_dir / f"{split}.txt").write_text(name + "\n", encoding="utf-8")


def main() -> int:
    cfg = build_structured_diffusion_config()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_shard(root)
        loader = build_w1_dataloader(
            root,
            "train",
            batch_size=4,
            num_workers=0,
            shuffle=False,
        )
        batch = next(iter(loader))
        assert validate_training_batch(batch, cfg) == 4
    print("PASS: W1 expert_dataset -> W2 pretraining contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
