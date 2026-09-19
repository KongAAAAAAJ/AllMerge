"""AllMerge diffusion pretraining entrypoint migrated from Diffusion-metadrive."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pretraining.checkpoint_io import export_runtime_checkpoint
from pretraining.dataset_adapter import build_w1_dataloader
from pretraining.lightning_module import DiffusionPretrainModule

RUN_DIR_PATTERN = re.compile(r"^run_(\d+)$")


def load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc
    with Path(path).open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def resolve_precision(value: str) -> str:
    return "16-mixed" if value == "auto" and torch.cuda.is_available() else (
        "32-true" if value == "auto" else value
    )


def resolve_pin_memory(value) -> bool:
    if value == "auto":
        return torch.cuda.is_available()
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


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
                f"W1 split '{split}' is empty. For the 100-sample one-shard "
                "overfit set, use --train-split all --val-split all."
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain AllMerge StructuredDiffusionPlanner.")
    parser.add_argument("--config", default="configs/diffusion_pretrain.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-split", choices=("all", "train", "val", "test"), default=None)
    parser.add_argument("--val-split", choices=("all", "train", "val", "test"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--strict-init-checkpoint", type=int, choices=(0, 1), default=1)
    parser.add_argument("--limit-train-samples", type=int, default=0)
    parser.add_argument("--limit-val-samples", type=int, default=0)
    parser.add_argument("--overfit-batches", type=float, default=0.0)
    parser.add_argument("--fast-dev-run", type=int, choices=(0, 1), default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("training") or {})

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
    lr = float(pick(args.lr, "learning_rate", 1e-4))
    precision = resolve_precision(str(pick(args.precision, "precision", "auto")))
    seed = int(train_cfg.get("seed", 0))

    validate_dataset_root(dataset_root, (train_split, val_split))
    pl.seed_everything(seed, workers=True)
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

    module = DiffusionPretrainModule(
        model_config=model_cfg,
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        min_lr=float(train_cfg.get("min_lr", 1e-6)),
        warmup_epochs=int(train_cfg.get("warmup_epochs", 3)),
        max_epochs=max_epochs,
        init_checkpoint=args.init_checkpoint,
        strict_init_checkpoint=bool(args.strict_init_checkpoint),
    )

    run_dir = create_next_run_dir(output_root)
    (run_dir / "resolved_config.json").write_text(
        json.dumps({"model": model_cfg, "training": train_cfg}, indent=2), encoding="utf-8"
    )
    print(f"[pretrain] run_dir={run_dir}")
    print(f"[pretrain] dataset_root={dataset_root}")
    print(
        f"[pretrain] split train={train_split}:{len(train_loader.dataset)} "
        f"val={val_split}:{len(val_loader.dataset)} batch_size={batch_size} workers={num_workers}"
    )

    logger = TensorBoardLogger(save_dir=str(run_dir), name="tb")
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        save_top_k=int(train_cfg.get("save_top_k", 3)),
        save_last=True,
        monitor="val/loss",
        mode="min",
        filename="diffusion-{epoch:02d}",
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        logger=logger,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="epoch")],
        accelerator="auto",
        devices=1,
        precision=precision,
        check_val_every_n_epoch=int(train_cfg.get("check_val_every_n_epoch", 1)),
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 10)),
        deterministic=False,
        overfit_batches=float(args.overfit_batches),
        fast_dev_run=bool(args.fast_dev_run),
    )
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint,
    )

    best_path = checkpoint_cb.best_model_path or checkpoint_cb.last_model_path
    runtime_path = None
    if best_path:
        runtime_path = run_dir / "runtime" / "best_runtime.pt"
        export_runtime_checkpoint(best_path, runtime_path)
        print(f"[pretrain] runtime_checkpoint={runtime_path}")

    metrics = {}
    for key, value in trainer.callback_metrics.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        if isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
    result = {
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root),
        "train_split": train_split,
        "val_split": val_split,
        "best_lightning_checkpoint": best_path or None,
        "runtime_checkpoint": str(runtime_path) if runtime_path else None,
        "metrics": metrics,
    }
    (run_dir / "train_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[pretrain] result={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
