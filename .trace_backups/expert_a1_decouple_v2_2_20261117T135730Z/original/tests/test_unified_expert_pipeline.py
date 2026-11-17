from __future__ import annotations

import numpy as np
import pytest

from data_pipeline.expert_collection.config import load_config, collection_args
from data_pipeline.expert_dataset import (
    AllMergeExpertDatasetConfig,
    AllMergeExpertShardDataset,
    FEATURE_KEYS,
    split_shards,
)


def _make_final_shard(path, n=2, episode=0):
    rng = np.random.default_rng(7)
    data = {
        "ego_state": rng.normal(size=(n, 14)).astype(np.float32),
        "agent_states": rng.normal(size=(n, 16, 11)).astype(np.float32),
        "agent_valid_mask": np.ones((n, 16), dtype=bool),
        "map_polylines": rng.normal(size=(n, 8, 32, 10)).astype(np.float32),
        "map_valid_mask": np.ones((n, 8), dtype=bool),
        "target_point": rng.normal(size=(n, 2)).astype(np.float32),
        "target_lane_polyline": rng.normal(size=(n, 32, 10)).astype(np.float32),
        "coarse_trajectories": rng.normal(size=(n, 10, 8, 2)).astype(np.float32),
        "coarse_controls": rng.normal(size=(n, 10, 40, 2)).astype(np.float32),
        "mode_valid_mask": np.ones((n, 10), dtype=bool),
        "expert_trajectory_xy": rng.normal(size=(n, 8, 2)).astype(np.float32),
        "future_trajectory_dense": rng.normal(size=(n, 40, 2)).astype(np.float32),
        "dense_dt": np.full(n, 0.1, dtype=np.float32),
        "trajectory_horizon_s": np.full(n, 4.0, dtype=np.float32),
        "target_mode": np.zeros(n, dtype=np.int64),
        "target_semantic": np.zeros(n, dtype=np.int64),
        "target_mode_geometry_valid": np.ones(n, dtype=bool),
        "target_mode_assignment_distance": np.zeros(n, dtype=np.float32),
        "koopman_state_hist_si": rng.normal(size=(n, 5, 5)).astype(np.float32),
        "koopman_control_hist_si": rng.normal(size=(n, 4, 2)).astype(np.float32),
        "koopman_pose0_si": rng.normal(size=(n, 3)).astype(np.float32),
        "koopman_ltr0": np.zeros(n, dtype=np.float32),
        "koopman_vx_init_kmh": np.full(n, 80, dtype=np.float32),
        "koopman_payload_kg": np.full(n, 3500, dtype=np.float32),
        "koopman_road_mu": np.full(n, 0.8, dtype=np.float32),
        "koopman_vx_init_kmh": np.full(n, 80, dtype=np.float32),
        "expert_trajectory_10hz_xy": rng.normal(size=(n, 41, 2)).astype(np.float32),
        "expert_trajectory_10hz_time_s": np.tile(np.arange(41, dtype=np.float32)[None] / 10.0, (n, 1)),
        "trajectory_time_s": np.tile(np.arange(8, dtype=np.float32)[None] * 0.5 + 0.5, (n, 1)),
        "road_lane_geometry": np.zeros((n, 16, 256, 9), dtype=np.float32),
        "road_lane_point_mask": np.ones((n, 16, 256), dtype=bool),
        "road_lane_mask": np.ones((n, 16), dtype=bool),
        "road_lane_connections": np.tile(np.eye(16, dtype=bool)[None], (n, 1, 1)),
        "road_current_lane_slot": np.zeros(n, dtype=np.int16),
        "road_target_lane_slot": np.zeros(n, dtype=np.int16),
        "episode_index": np.full(n, episode, dtype=np.int32),
        "frame_index": np.arange(n, dtype=np.int32),
        "ego_index": np.zeros(n, dtype=np.int8),
        "seed": np.full(n, 2026, dtype=np.int32),
    }
    assert set(FEATURE_KEYS).issubset(data)
    np.savez_compressed(path, **data)


def test_unified_collection_config_has_single_trucksim_pipeline():
    cfg = load_config()
    args = collection_args(cfg)
    assert args.vehicle_model == "trucksim"
    assert args.dataset_name == "expert_trucksim_k2_anchor_pair_100k"
    assert args.scenarios == ["straight", "curved", "merge_in", "merge_out"]
    assert args.ego_length_m == pytest.approx(8.0)
    assert args.ego_width_m == pytest.approx(2.6)
    assert not cfg.get("raw_expert_collection")


def test_dataset_loader_requires_final_a1_k2_schema(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    _make_final_shard(shard_dir / "shard_000000.npz")
    ds = AllMergeExpertShardDataset(AllMergeExpertDatasetConfig(
        dataset_root=tmp_path, split="all"))
    features, targets = ds[0]
    assert features["coarse_controls"].shape == (10, 40, 2)
    assert features["coarse_trajectories"].shape == (10, 8, 2)
    assert features["mode_valid_mask"].shape == (10,)
    assert targets["trajectory_dense"].shape == (40, 2)
    assert targets["koopman_state_hist_si"].shape == (5, 5)


def test_dataset_loader_rejects_legacy_standard_shards(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.savez_compressed(shard_dir / "shard_000000.npz",
                        expert_trajectory_xy=np.zeros((1, 8, 2), np.float32))
    with pytest.raises(RuntimeError, match="Legacy standard expert datasets are unsupported"):
        AllMergeExpertShardDataset(AllMergeExpertDatasetConfig(
            dataset_root=tmp_path, split="all"))


def test_small_dataset_split_keeps_train_val_test_when_three_episodes_exist(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for episode in range(3):
        _make_final_shard(shard_dir / f"shard_{episode:06d}.npz", n=1, episode=episode)
    splits = split_shards(tmp_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0)
    assert all(len(splits[key]) == 1 for key in ("train", "val", "test"))
    assert len(AllMergeExpertShardDataset(AllMergeExpertDatasetConfig(
        dataset_root=tmp_path, split="val"))) == 1
