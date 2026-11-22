"""Public A2 anchor training entry point; run from the all_merge directory."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from highway_env.planner.anchor_scoring.training import train


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train/evaluate A2 anchors from raw expert scenes")
    parser.add_argument("--config", default="configs/train/a2_scorer.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--precompute-batch-size", type=int, default=None)
    parser.add_argument("--precompute-workers", type=int, default=None)
    parser.add_argument("--precompute-device", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--checkpoint", default=None,
                        help="Warm-start model weights only; optimizer state is reset.")
    parser.add_argument("--resume", default=None,
                        help="Resume a scorer checkpoint including optimizer/history state.")
    return train(parser.parse_args(argv))


if __name__ == "__main__":
    main()
