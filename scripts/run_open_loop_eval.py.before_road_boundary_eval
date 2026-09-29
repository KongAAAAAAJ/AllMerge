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
        default="all_merge\\outputs\\diffusion_pretrain_50k\\run_1\\checkpoints\\best.pt",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default="all_merge\\outputs\\expert_dataset\\allmerge_expert_50k",
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
    # EVAL_VIZ_V2_CLI
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable PNG/summary-table generation.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Default: <output-csv parent>/figures",
    )
    parser.add_argument(
        "--visualization-seed",
        type=int,
        default=None,
        help=(
            "Seed for the random 9-sample gallery. "
            "Default None changes samples each run."
        ),
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help=(
            "Training-start checkpoint for the 1x2 same-noise comparison. "
            "Default: <checkpoint parent>/initial.pt"
        ),
    )
    parser.add_argument(
        "--miss-threshold-m",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--collision-distance-m",
        type=float,
        default=2.5,
        help="Open-loop static-agent collision proxy distance.",
    )
    parser.add_argument(
        "--offroad-distance-m",
        type=float,
        default=2.5,
        help="Open-loop map-polyline off-road proxy distance.",
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
        visualize=not args.no_visualization,
        figure_dir=args.figure_dir,
        visualization_seed=args.visualization_seed,
        initial_checkpoint=args.initial_checkpoint,
        miss_threshold_m=args.miss_threshold_m,
        collision_distance_m=args.collision_distance_m,
        offroad_distance_m=args.offroad_distance_m,
    )
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
