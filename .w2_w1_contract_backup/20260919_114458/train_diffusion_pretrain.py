"""AllMerge diffusion pretraining entrypoint.

Migration target of Diffusion-metadrive's train_transfuser.py.  The outer
Dataset -> batch -> model -> loss -> optimizer -> checkpoint -> validation loop
is retained, while the dataset and model calls are adapted to W1 and
StructuredDiffusionPlanner.forward_train().
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader, Subset

from data_pipeline.expert_dataset import (
    ExpertDataset,
    collate_expert_samples,
)
from pretraining.checkpoint_io import export_runtime_checkpoint
from pretraining.lightning_module import DiffusionPretrainModule

RUN_DIR_PATTERN = re.compile(r"^run_(\d+)$")


def load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for configs/diffusion_pretrain.yaml. "
            "Install it with: pip install pyyaml"
        ) from exc
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def resolve_precision(value: str) -> str:
    if value == "auto":
        return "16-mixed" if torch.cuda.is_available() else "32-true"
    return value


def resolve_pin_memory(value) -> bool:
    if value == "auto":
        return torch.cuda.is_available()
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def create_next_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in output_root.iterdir():
        match = RUN_DIR_PATTERN.match(path.name) if path.is_dir() else None
        if match:
            existing.append(int(match.group(1)))
    index = max(existing, default=0) + 1
    run_dir = output_root / f"run_{index}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def validate_dataset_root(dataset_root: Path) -> None:
    for split in ("train", "val"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing W1 split directory: {split_dir}")
        if not any(split_dir.glob("shard-*.npz")):
            raise FileNotFoundError(f"No W1 shard-*.npz under: {split_dir}")
    schema_path = dataset_root / "feature_schema.json"
    if not schema_path.exists():
        print(
            "[pretrain] warning: dataset feature_schema.json is absent; "
            "runtime Python schema checks will still run"
        )


def build_dataloader(
    dataset_root: Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int,
    pin_memory: bool,
    cache_shards: int,
    limit_samples: int = 0,
    shuffle: bool,
) -> DataLoader:
    dataset = ExpertDataset(
        dataset_root,
        split=split,
        validate_samples=False,
        cache_shards=cache_shards,
    )
    if limit_samples > 0:
        dataset = Subset(dataset, range(min(limit_samples, len(dataset))))

    kwargs = dict(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate_expert_samples,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain AllMerge StructuredDiffusionPlanner.")
    parser.add_argument("--config", default="configs/diffusion_pretrain.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--init-checkpoint", default=None,
                        help="Planner/runtime checkpoint used only for weight initialization.")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Lightning checkpoint used to resume model+optimizer+scheduler+epoch.")
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

    dataset_root = Path(pick(args.dataset_root, "dataset_root", "data/expert/allmerge_planner_v1"))
    output_root = Path(pick(args.output_dir, "output_dir", "outputs/diffusion_pretrain"))
    batch_size = int(pick(args.batch_size, "batch_size", 64))
    num_workers = int(pick(args.num_workers, "num_workers", 8))
    max_epochs = int(pick(args.max_epochs, "max_epochs", 50))
    max_steps = int(pick(args.max_steps, "max_steps", -1))
    lr = float(pick(args.lr, "learning_rate", 1e-4))
    precision = resolve_precision(str(pick(args.precision, "precision", "auto")))
    seed = int(train_cfg.get("seed", 0))

    validate_dataset_root(dataset_root)
    pl.seed_everything(seed, workers=True)

    train_loader = build_dataloader(
        dataset_root,
        "train",
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=bool(train_cfg.get("persistent_workers", True)),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
        pin_memory=resolve_pin_memory(train_cfg.get("pin_memory", "auto")),
        cache_shards=int(train_cfg.get("cache_shards", 2)),
        limit_samples=int(args.limit_train_samples),
        shuffle=True,
    )
    val_loader = build_dataloader(
        dataset_root,
        "val",
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=bool(train_cfg.get("persistent_workers", True)),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
        pin_memory=resolve_pin_memory(train_cfg.get("pin_memory", "auto")),
        cache_shards=int(train_cfg.get("cache_shards", 2)),
        limit_samples=int(args.limit_val_samples),
        shuffle=False,
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
        json.dumps({"model": model_cfg, "training": train_cfg}, indent=2),
        encoding="utf-8",
    )
    print(f"[pretrain] run_dir={run_dir}")
    print(f"[pretrain] dataset_root={dataset_root}")
    print(
        "[pretrain] dataloader "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"batch_size={batch_size} workers={num_workers}"
    )
    print(
        "[pretrain] runtime "
        f"precision={precision} max_epochs={max_epochs} max_steps={max_steps} lr={lr}"
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
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
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

    callback_metrics = {}
    for key, value in trainer.callback_metrics.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        if isinstance(value, (int, float)):
            callback_metrics[str(key)] = float(value)

    result = {
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root),
        "best_lightning_checkpoint": best_path or None,
        "runtime_checkpoint": str(runtime_path) if runtime_path else None,
        "metrics": callback_metrics,
    }
    (run_dir / "train_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(f"[pretrain] result={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
