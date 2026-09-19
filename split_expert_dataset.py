from __future__ import annotations

import argparse
from pathlib import Path

from expert_dataset import split_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val/test shard lists."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = split_shards(
        dataset_root=args.dataset_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(
        "split complete:",
        ", ".join(
            f"{name}={len(shards)} shards"
            for name, shards in result.items()
        ),
    )


if __name__ == "__main__":
    main()
