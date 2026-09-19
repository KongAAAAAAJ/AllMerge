from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make repo-root packages importable when invoked as:
#   python scripts/run_open_loop_eval.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.open_loop_eval import (  # noqa: E402
    evaluate_open_loop,
    print_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AllMerge W5 pretrained Diffusion "
            "open-loop evaluation"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=[
            "all",
            "train",
            "val",
            "test",
        ],
        default="val",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/evaluation/"
            "open_loop_results.csv"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=0,
        help=(
            "Deterministic diffusion sampling seed. "
            "Use the same value for checkpoint comparisons."
        ),
    )
    parser.add_argument(
        "--strict-checkpoint",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = evaluate_open_loop(
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        split=args.split,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
        device=args.device,
        inference_seed=args.inference_seed,
        strict_checkpoint=args.strict_checkpoint,
    )
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
