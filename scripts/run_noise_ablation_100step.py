from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("\n[run] " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _next_run(root: Path) -> Path:
    runs = []
    if root.is_dir():
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("run_"):
                try:
                    runs.append((int(path.name.split("_", 1)[1]), path))
                except ValueError:
                    pass
    if not runs:
        raise RuntimeError(f"No run_* directory found under {root}")
    return max(runs)[1]


def _checkpoint(run_dir: Path) -> Path:
    best = run_dir / "checkpoints" / "best.pt"
    last = run_dir / "checkpoints" / "last.pt"
    if best.is_file():
        return best
    if last.is_file():
        return last
    raise FileNotFoundError(f"No best.pt/last.pt under {run_dir / 'checkpoints'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a matched 100-step AllMerge x0-vs-epsilon diffusion ablation, "
            "then evaluate both checkpoints with the same inference seed."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/expert_dataset/allmerge_expert_50k_dense10hz_v2"),
    )
    parser.add_argument("--train-split", default="train", choices=("all", "train", "val", "test"))
    parser.add_argument("--val-split", default="val", choices=("all", "train", "val", "test"))
    parser.add_argument("--eval-split", default="val", choices=("all", "train", "val", "test"))
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-val-samples", type=int, default=1000)
    parser.add_argument("--limit-eval-samples", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/noise_ablation_100step"))
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    x0_root = args.output_root / "x0"
    eps_root = args.output_root / "epsilon"
    eval_root = args.output_root / "evaluation"

    common = [
        "--dataset-root", str(args.dataset_root),
        "--train-split", args.train_split,
        "--val-split", args.val_split,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--max-steps", str(args.max_steps),
        "--limit-val-samples", str(args.limit_val_samples),
        "--device", args.device,
    ]

    # 1) Historical x0/sample baseline.
    _run([
        python,
        "train_diffusion_pretrain.py",
        "--config", "configs/diffusion_pretrain_x0_compare.json",
        "--output-dir", str(x0_root),
        "--prediction-type", "sample",
        *common,
    ])
    x0_run = _next_run(x0_root)
    shared_initial = x0_run / "checkpoints" / "initial.pt"
    if not shared_initial.is_file():
        raise FileNotFoundError(shared_initial)

    # 2) Epsilon/noise model starts from the exact same weights.
    _run([
        python,
        "train_diffusion_pretrain.py",
        "--config", "configs/diffusion_pretrain_noise.json",
        "--output-dir", str(eps_root),
        "--prediction-type", "epsilon",
        "--init-checkpoint", str(shared_initial),
        "--strict-init-checkpoint", "1",
        *common,
    ])
    eps_run = _next_run(eps_root)

    summary = {
        "x0_run_dir": str(x0_run),
        "epsilon_run_dir": str(eps_run),
        "shared_initial_checkpoint": str(shared_initial),
    }

    if not args.skip_eval:
        eval_root.mkdir(parents=True, exist_ok=True)
        x0_csv = eval_root / "x0_open_loop.csv"
        eps_csv = eval_root / "epsilon_open_loop.csv"
        eval_common = [
            "--dataset-root", str(args.dataset_root),
            "--split", args.eval_split,
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--limit-samples", str(args.limit_eval_samples),
            "--inference-seed", str(args.seed),
            "--visualization-seed", str(args.seed),
        ]
        if args.device != "auto":
            eval_common += ["--device", args.device]

        _run([
            python,
            "scripts/run_open_loop_eval.py",
            "--checkpoint", str(_checkpoint(x0_run)),
            "--output-csv", str(x0_csv),
            *eval_common,
        ])
        _run([
            python,
            "scripts/run_open_loop_eval.py",
            "--checkpoint", str(_checkpoint(eps_run)),
            "--output-csv", str(eps_csv),
            *eval_common,
        ])
        comparison_json = eval_root / "prediction_type_comparison.json"
        _run([
            python,
            "scripts/compare_prediction_type_runs.py",
            "--x0-run-dir", str(x0_run),
            "--epsilon-run-dir", str(eps_run),
            "--x0-eval-csv", str(x0_csv),
            "--epsilon-eval-csv", str(eps_csv),
            "--output-json", str(comparison_json),
        ])
        summary.update({
            "x0_eval_csv": str(x0_csv),
            "epsilon_eval_csv": str(eps_csv),
            "comparison_json": str(comparison_json),
        })

    summary_path = args.output_root / "latest_ablation_paths.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
