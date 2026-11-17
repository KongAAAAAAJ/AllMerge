"""Precompute deterministic A2 candidate pools and safety teacher labels.

Run:
  python -m highway_env.planner.anchor_scoring.prepare_cache --config configs/train/adaptive_anchors.yaml
"""
from __future__ import annotations

import argparse
import yaml

from highway_env.planner.diffusion.config.experiment import resolve_repo_file
from .training_data import prepare_split_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare TRACE A2 scorer candidate/label cache")
    parser.add_argument("--config", default="configs/train/adaptive_anchors.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--precompute-batch-size", type=int, default=None)
    parser.add_argument("--precompute-device", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(resolve_repo_file(args.config).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Scorer config must be a mapping")
    if args.dataset_root:
        cfg["dataset_root"] = args.dataset_root
    if args.max_samples is not None:
        cfg["max_samples"] = args.max_samples
    if args.precompute_batch_size is not None:
        cfg["precompute_batch_size"] = args.precompute_batch_size
    if args.precompute_device:
        cfg["precompute_device"] = args.precompute_device

    dataset_root = resolve_repo_file(cfg["dataset_root"])
    cache_root = resolve_repo_file(
        args.cache_dir or cfg.get("cache_dir", "outputs/anchor_ablation/cache_v3")
    )
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits:
        raise ValueError("--splits must contain at least one split name")

    for split in splits:
        prepare_split_cache(
            dataset_root, split, cfg, cache_root, rebuild=args.rebuild_cache
        )
    print(f"[A2-cache] ready: {cache_root}")


if __name__ == "__main__":
    main()
