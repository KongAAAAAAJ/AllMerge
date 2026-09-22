"""AllMerge diffusion pretraining entrypoint using only repository dependencies."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from pretraining.checkpoint_io import export_runtime_checkpoint
from pretraining.config_io import load_config, resolve_pin_memory
from pretraining.dataset_adapter import build_w1_dataloader
from pretraining.trainer import DiffusionPretrainer, seed_everything

RUN_DIR_PATTERN = re.compile(r"^run_(\d+)$")


def create_next_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in output_root.iterdir():
        match = RUN_DIR_PATTERN.match(path.name) if path.is_dir() else None
        if match:
            indices.append(int(match.group(1)))
    run_dir = output_root / f"run_{max(indices, default=0) + 1}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def validate_dataset_root(dataset_root: Path, splits: tuple[str, ...]) -> None:
    shard_dir = dataset_root / "shards"
    if not shard_dir.is_dir() or not any(shard_dir.glob("shard_*.npz")):
        raise FileNotFoundError(f"No W1 shard_*.npz under: {shard_dir}")
    for split in splits:
        if split == "all":
            continue
        split_file = dataset_root / "splits" / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing W1 split file: {split_file}")
        if not split_file.read_text(encoding="utf-8").strip():
            raise RuntimeError(
                f"W1 split '{split}' is empty. For the one-shard 100-sample "
                "overfit set, use --train-split all --val-split all."
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain AllMerge StructuredDiffusionPlanner.")
    parser.add_argument("--config", default="configs/diffusion_pretrain.json")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-split", choices=("all", "train", "val", "test"), default=None)
    parser.add_argument("--val-split", choices=("all", "train", "val", "test"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--strict-init-checkpoint", type=int, choices=(0, 1), default=1)
    parser.add_argument("--limit-train-samples", type=int, default=0)
    parser.add_argument("--limit-val-samples", type=int, default=0)
    parser.add_argument("--overfit-batches", type=float, default=0.0)
    parser.add_argument("--fast-dev-run", type=int, choices=(0, 1), default=0)
    # STAGED_DENSE_SUPERVISION_V1
    parser.add_argument("--dense-loss-enabled", type=int, choices=(0, 1), default=None)
    parser.add_argument("--dense-loss-lambda-p", type=float, default=None)
    parser.add_argument("--dense-loss-type", choices=("smooth_l1", "l1", "mse"), default=None)
    parser.add_argument("--dense-loss-terminal-timestep", type=int, default=None)
    parser.add_argument("--dense-loss-terminal-weight", type=float, default=None)
    # DENSE_RESIDUAL_SEPARATE_LR_V1
    parser.add_argument("--dense-residual-lr", type=float, default=None)
    # DENSE_RESIDUAL_BOUND_CLI_V1
    parser.add_argument("--dense-residual-max-x-m", type=float, default=None)
    parser.add_argument("--dense-residual-max-y-m", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("training") or {})

    # DENSE_RESIDUAL_BOUND_CLI_V1
    if args.dense_residual_max_x_m is not None:
        if float(args.dense_residual_max_x_m) <= 0.0:
            raise ValueError("--dense-residual-max-x-m must be positive")
        model_cfg["dense_residual_max_x_m"] = float(args.dense_residual_max_x_m)
    if args.dense_residual_max_y_m is not None:
        if float(args.dense_residual_max_y_m) <= 0.0:
            raise ValueError("--dense-residual-max-y-m must be positive")
        model_cfg["dense_residual_max_y_m"] = float(args.dense_residual_max_y_m)

    def pick(cli_value, name, default):
        return cli_value if cli_value is not None else train_cfg.get(name, default)

    dataset_root = Path(pick(args.dataset_root, "dataset_root", "outputs/expert_dataset/allmerge_expert"))
    output_root = Path(pick(args.output_dir, "output_dir", "outputs/diffusion_pretrain"))
    train_split = str(pick(args.train_split, "train_split", "train"))
    val_split = str(pick(args.val_split, "val_split", "val"))
    batch_size = int(pick(args.batch_size, "batch_size", 64))
    num_workers = int(pick(args.num_workers, "num_workers", 8))
    max_epochs = int(pick(args.max_epochs, "max_epochs", 50))
    max_steps = int(pick(args.max_steps, "max_steps", -1))
    learning_rate = float(pick(args.lr, "learning_rate", 1e-4))
    device = str(pick(args.device, "device", "auto"))
    precision = str(pick(args.precision, "precision", "auto"))
    grad_clip = float(pick(args.grad_clip, "grad_clip", 5.0))
    seed = int(train_cfg.get("seed", 0))

    # STAGED_DENSE_SUPERVISION_V1
    dense_loss_enabled = bool(int(pick(args.dense_loss_enabled, "dense_loss_enabled", 0)))
    dense_loss_lambda_p = float(pick(args.dense_loss_lambda_p, "dense_loss_lambda_p", 0.0))
    dense_loss_type = str(pick(args.dense_loss_type, "dense_loss_type", "smooth_l1"))
    dense_loss_terminal_only = bool(train_cfg.get("dense_loss_terminal_only", True))
    dense_loss_weight_mode = str(train_cfg.get("dense_loss_weight_mode", "terminal_constant"))
    dense_loss_terminal_timestep = int(pick(
        args.dense_loss_terminal_timestep, "dense_loss_terminal_timestep", 0
    ))
    dense_loss_terminal_weight = float(pick(
        args.dense_loss_terminal_weight, "dense_loss_terminal_weight", 1.0
    ))
    dense_loss_dense_dt = float(train_cfg.get("dense_loss_dense_dt", 0.1))
    # DENSE_RESIDUAL_SEPARATE_LR_V1
    dense_residual_lr = float(pick(
        args.dense_residual_lr, "dense_residual_learning_rate", learning_rate
    ))

    validate_dataset_root(dataset_root, (train_split, val_split))
    seed_everything(seed)
    loader_kwargs = dict(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=bool(train_cfg.get("persistent_workers", True)),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
        pin_memory=resolve_pin_memory(train_cfg.get("pin_memory", "auto")),
    )
    train_loader = build_w1_dataloader(
        split=train_split,
        shuffle=True,
        limit_samples=int(args.limit_train_samples),
        **loader_kwargs,
    )
    val_loader = build_w1_dataloader(
        split=val_split,
        shuffle=False,
        limit_samples=int(args.limit_val_samples),
        **loader_kwargs,
    )

    run_dir = create_next_run_dir(output_root)
    resolved = {
        "model": model_cfg,
        "training": {
            **train_cfg,
            "dataset_root": str(dataset_root),
            "output_dir": str(output_root),
            "train_split": train_split,
            "val_split": val_split,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "max_epochs": max_epochs,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "device": device,
            "precision": precision,
            "grad_clip": grad_clip,
            # STAGED_DENSE_SUPERVISION_V1
            "dense_loss_enabled": dense_loss_enabled,
            "dense_loss_lambda_p": dense_loss_lambda_p,
            "dense_loss_type": dense_loss_type,
            "dense_loss_terminal_only": dense_loss_terminal_only,
            "dense_loss_weight_mode": dense_loss_weight_mode,
            "dense_loss_terminal_timestep": dense_loss_terminal_timestep,
            "dense_loss_terminal_weight": dense_loss_terminal_weight,
            "dense_loss_dense_dt": dense_loss_dense_dt,
            # DENSE_RESIDUAL_SEPARATE_LR_V1
            "dense_residual_learning_rate": dense_residual_lr,
        },
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2), encoding="utf-8"
    )
    print(f"[pretrain] run_dir={run_dir}")
    print(f"[pretrain] dataset_root={dataset_root}")
    print(
        f"[pretrain] split train={train_split}:{len(train_loader.dataset)} "
        f"val={val_split}:{len(val_loader.dataset)} batch_size={batch_size} workers={num_workers}"
    )
    print(
        f"[pretrain] backend=pure_torch torch={torch.__version__} "
        f"cuda={torch.cuda.is_available()}"
    )
    print(
        "[pretrain] dense_supervision="
        f"enabled={dense_loss_enabled} lambda_p={dense_loss_lambda_p:g} "
        f"type={dense_loss_type} runtime_chain="
        f"{tuple(model_cfg.get('inference_timesteps', (8, 0)))} "
        f"residual_lr={dense_residual_lr:g} "
        f"residual_bound=("
        f"{float(model_cfg.get('dense_residual_max_x_m', 2.0)):g}, "
        f"{float(model_cfg.get('dense_residual_max_y_m', 0.75)):g})"
    )

    trainer = DiffusionPretrainer(
        model_config=model_cfg,
        device=device,
        learning_rate=learning_rate,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        min_lr=float(train_cfg.get("min_lr", 1e-6)),
        warmup_epochs=int(train_cfg.get("warmup_epochs", 3)),
        max_epochs=max_epochs,
        grad_clip=grad_clip,
        precision=precision,
        log_dir=run_dir / "tb",
        init_checkpoint=args.init_checkpoint,
        strict_init_checkpoint=bool(args.strict_init_checkpoint),
        # STAGED_DENSE_SUPERVISION_V1
        dense_loss_enabled=dense_loss_enabled,
        dense_loss_lambda_p=dense_loss_lambda_p,
        dense_loss_type=dense_loss_type,
        dense_loss_terminal_only=dense_loss_terminal_only,
        dense_loss_weight_mode=dense_loss_weight_mode,
        dense_loss_terminal_timestep=dense_loss_terminal_timestep,
        dense_loss_terminal_weight=dense_loss_terminal_weight,
        dense_loss_dense_dt=dense_loss_dense_dt,
        # DENSE_RESIDUAL_SEPARATE_LR_V1
        dense_residual_learning_rate=dense_residual_lr,
    )
    try:
        if args.resume_from_checkpoint:
            trainer.load_resume(args.resume_from_checkpoint)
        result = trainer.fit(
            train_loader,
            val_loader,
            max_epochs=max_epochs,
            max_steps=max_steps,
            overfit_batches=float(args.overfit_batches),
            check_val_every_n_epoch=int(train_cfg.get("check_val_every_n_epoch", 1)),
            log_every_n_steps=int(train_cfg.get("log_every_n_steps", 10)),
            fast_dev_run=bool(args.fast_dev_run),
            checkpoint_dir=run_dir / "checkpoints",
            train_config=resolved["training"],
        )
    finally:
        trainer.close()

    source_checkpoint = result["best_checkpoint"] or result["last_checkpoint"]
    runtime_path = None
    if source_checkpoint:
        runtime_path = run_dir / "runtime" / "best_runtime.pt"
        export_runtime_checkpoint(source_checkpoint, runtime_path)
        print(f"[pretrain] runtime_checkpoint={runtime_path}")

    result.update({
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root),
        "train_split": train_split,
        "val_split": val_split,
        "runtime_checkpoint": str(runtime_path) if runtime_path else None,
    })
    (run_dir / "train_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"[pretrain] result={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
