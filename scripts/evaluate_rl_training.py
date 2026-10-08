"""Unified in-repository visual evaluation for GRPO, Lagrangian, Projection and SafeMPO-Diff."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.rl_training_report import (
    parse_train_log, parse_fixed_validation,
    plot_training_curves, evaluate_paired_trajectories,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, help="trained GRPO/Lagrangian/Projection/SafeMPO .pt")
    p.add_argument("--pretrain-checkpoint", type=Path, help="original pretrain .pt, required for paired gallery")
    p.add_argument("--train-log", type=Path, help="one-line [step ...] training log")
    p.add_argument("--fixed-validation-csv", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--curves-only", action="store_true", help="plot existing logs without model inference")
    p.add_argument("--scenario", choices=("all", "straight", "curved", "merge_in", "merge_out"), default="curved")
    p.add_argument("--num-states", type=int, default=16)
    p.add_argument("--gallery-size", type=int, default=9, choices=(9,))
    p.add_argument("--group-size", type=int, default=48)
    p.add_argument("--group-action", type=int, default=3)
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--task-reward-type", choices=("legacy_w4", "progress_comfort"), default="progress_comfort")
    p.add_argument("--constraint-names", default="collision,road,ttc,background_gap,teammate_gap")
    p.add_argument("--constraint-residual-cap", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fixed-validation-seed-offset", type=int, default=10000)
    p.add_argument("--visualization-seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.curves_only and (not args.checkpoint or not args.pretrain_checkpoint):
        raise SystemExit("paired trajectory evaluation requires --checkpoint AND --pretrain-checkpoint")
    if not args.train_log and not args.fixed_validation_csv and args.curves_only:
        raise SystemExit("--curves-only requires --train-log and/or --fixed-validation-csv")
    base = args.checkpoint.parent if args.checkpoint else (args.train_log or args.fixed_validation_csv).parent
    out = args.output_dir or base / "visual_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    train_rows = parse_train_log(args.train_log) if args.train_log else []
    fixed_rows = parse_fixed_validation(args.fixed_validation_csv) if args.fixed_validation_csv else []
    results = {}
    if train_rows or fixed_rows:
        results["training_figures"] = plot_training_curves(train_rows, fixed_rows, out)
        print("[eval] plotted reward/loss/violation/KL curves")
    if not args.curves_only:
        results["paired"] = evaluate_paired_trajectories(
            args.checkpoint, args.pretrain_checkpoint, out,
            scenario=args.scenario, num_states=args.num_states,
            gallery_size=args.gallery_size, group_size=args.group_size,
            group_action=args.group_action, seed=args.seed,
            validation_seed_offset=args.fixed_validation_seed_offset,
            visualization_seed=args.visualization_seed,
            eta=args.eta, device_name=args.device,
            task_reward_type=args.task_reward_type,
            constraint_names=tuple(n.strip() for n in args.constraint_names.split(",") if n.strip()),
            constraint_residual_cap=args.constraint_residual_cap,
        )
        print("[eval] plotted paired trajectory gallery + fresh constraint violation rates")
    (out / "evaluation_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[PASS] evaluation complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
