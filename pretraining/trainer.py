"""Pure-PyTorch trainer for the frozen AllMerge structured diffusion planner.

The trainer owns orchestration only. Target-mode assignment, diffusion targets,
trajectory regression loss and mode classification loss remain inside
StructuredDiffusionPlanner.forward_train().
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from highway_env.planner.diffusion.config import build_structured_diffusion_config
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter

from .checkpoint_io import load_checkpoint_file
from .contract import validate_model_schema, validate_training_batch
from .dataset_adapter import unpack_w1_batch
from .warmup_cos_lr import WarmupCosLR




class _FallbackWriter:
    def __init__(self, log_dir: str | Path) -> None:
        self.path = Path(log_dir) / "scalars.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add_scalar(self, tag, value, step) -> None:
        import json
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({"tag": str(tag), "value": float(value), "step": int(step)}) + "\n")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _build_writer(log_dir: str | Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=str(log_dir))
    except ImportError:
        print("[pretrain] tensorboard unavailable; falling back to scalars.jsonl")
        return _FallbackWriter(log_dir)


METRIC_KEYS = (
    "loss",
    "trajectory_regression_loss",
    "trajectory_classification_loss",
    "target_mode_assignment_distance",
    "target_mode_geometry_valid_fraction",
    "target_mode_traffic_valid_fraction",
    "w1_target_mode_match",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _batch_limit(loader, value: float) -> int:
    total = len(loader)
    if value <= 0:
        return total
    if value <= 1:
        return max(1, int(math.ceil(total * value)))
    return min(total, int(value))


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        "features": {key: value.to(device, non_blocking=True) for key, value in batch["features"].items()},
        "expert_trajectory": batch["expert_trajectory"].to(device, non_blocking=True),
        "expert_mode": batch["expert_mode"].to(device, non_blocking=True),
        "expert_semantic": batch["expert_semantic"].to(device, non_blocking=True),
        "metadata": batch.get("metadata"),
    }


def _metric_values(output: dict, batch: dict) -> Dict[str, float]:
    target_mode_match = (
        output["target_mode"].detach() == batch["expert_mode"]
    ).float().mean()
    values = {
        "loss": output["loss"],
        "trajectory_regression_loss": output["trajectory_regression_loss"],
        "trajectory_classification_loss": output["trajectory_classification_loss"],
        "target_mode_assignment_distance": output["target_mode_assignment_distance"].mean(),
        "target_mode_geometry_valid_fraction": output["target_mode_geometry_valid"].float().mean(),
        "target_mode_traffic_valid_fraction": output["target_mode_traffic_valid"].float().mean(),
        "w1_target_mode_match": target_mode_match,
    }
    return {key: float(value.detach().cpu().item()) for key, value in values.items()}


class DiffusionPretrainer:
    def __init__(
        self,
        *,
        model_config: Optional[dict],
        device: str,
        learning_rate: float,
        weight_decay: float,
        min_lr: float,
        warmup_epochs: int,
        max_epochs: int,
        grad_clip: float,
        precision: str,
        log_dir: str | Path,
        init_checkpoint: Optional[str] = None,
        strict_init_checkpoint: bool = True,
    ) -> None:
        self.model_config_dict = dict(model_config or {})
        self.model_config = build_structured_diffusion_config(**self.model_config_dict)
        validate_model_schema(self.model_config)
        self.device = resolve_device(device)
        self.adapter = PlannerTensorAdapter(self.model_config, self.device)
        self.planner = StructuredDiffusionPlanner(self.model_config, self.adapter).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.planner.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.scheduler = WarmupCosLR(
            optimizer=self.optimizer,
            lr=float(learning_rate),
            min_lr=float(min_lr),
            epochs=int(max_epochs),
            warmup_epochs=int(warmup_epochs),
        )
        self.grad_clip = float(grad_clip)
        precision = str(precision).lower()
        self.use_amp = self.device.type == "cuda" and precision in {
            "auto", "16", "fp16", "16-mixed", "mixed"
        }
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.writer = _build_writer(log_dir)
        self.global_step = 0
        self.start_epoch = 0
        self._contract_checked = False

        if init_checkpoint:
            state_dict, stored_config = load_checkpoint_file(init_checkpoint, map_location="cpu")
            if stored_config and stored_config != self.model_config_dict:
                print("[pretrain] init checkpoint model_config differs; explicit config wins")
            incompatible = self.planner.load_state_dict(
                state_dict,
                strict=bool(strict_init_checkpoint),
            )
            if not strict_init_checkpoint and (
                incompatible.missing_keys or incompatible.unexpected_keys
            ):
                print(
                    "[pretrain] non-strict init: "
                    f"missing={len(incompatible.missing_keys)} "
                    f"unexpected={len(incompatible.unexpected_keys)}"
                )

    def close(self) -> None:
        self.writer.close()

    def load_resume(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "planner_state_dict" not in checkpoint:
            raise RuntimeError("Resume checkpoint is not an AllMerge W2 training checkpoint")
        stored_config = dict(checkpoint.get("allmerge_model_config") or {})
        if stored_config != self.model_config_dict:
            raise RuntimeError("Resume checkpoint model_config differs from current config")
        self.planner.load_state_dict(checkpoint["planner_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        print(
            f"[pretrain] resumed from {path}: "
            f"epoch={self.start_epoch} global_step={self.global_step}"
        )

    def save_checkpoint(self, path: str | Path, *, epoch: int, metrics: dict, train_config: dict) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "planner_state_dict": {
                key: value.detach().cpu() for key, value in self.planner.state_dict().items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "metrics": dict(metrics),
            "train_config": dict(train_config),
            "allmerge_model_config": dict(self.model_config_dict),
            "checkpoint_format": "allmerge_diffusion_pretrain_v2_torch",
        }
        torch.save(payload, path)
        return path

    def _run_epoch(
        self,
        loader,
        *,
        stage: str,
        max_batches: int,
        log_every_n_steps: int,
        max_steps: int,
    ) -> tuple[dict, bool]:
        training = stage == "train"
        self.planner.train(training)
        sums = {key: 0.0 for key in METRIC_KEYS}
        samples = 0
        stop = False

        for batch_idx, raw_batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            if training and max_steps > 0 and self.global_step >= max_steps:
                stop = True
                break
            if not self._contract_checked:
                validate_training_batch(raw_batch, self.model_config)
                self._contract_checked = True

            batch = _move_batch_to_device(unpack_w1_batch(raw_batch), self.device)
            batch_size = int(batch["features"]["ego_state"].shape[0])
            if training:
                self.optimizer.zero_grad(set_to_none=True)

            grad_context = torch.enable_grad() if training else torch.no_grad()
            with grad_context:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    output = self.planner.forward_train(
                        batch["features"],
                        batch["expert_trajectory"],
                        batch["expert_semantic"],
                    )
                    loss = output["loss"]

                if training:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.planner.parameters(), self.grad_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.global_step += 1

            values = _metric_values(output, batch)
            for key, value in values.items():
                sums[key] += value * batch_size
            samples += batch_size

            if training and (
                self.global_step == 1
                or self.global_step % max(int(log_every_n_steps), 1) == 0
            ):
                self.writer.add_scalar("train_step/loss", values["loss"], self.global_step)
                self.writer.add_scalar(
                    "train_step/trajectory_regression_loss",
                    values["trajectory_regression_loss"],
                    self.global_step,
                )
                self.writer.add_scalar(
                    "train_step/trajectory_classification_loss",
                    values["trajectory_classification_loss"],
                    self.global_step,
                )

        if samples == 0:
            raise RuntimeError(f"No {stage} samples were processed")
        return {key: value / samples for key, value in sums.items()}, stop

    def fit(
        self,
        train_loader,
        val_loader,
        *,
        max_epochs: int,
        max_steps: int,
        overfit_batches: float,
        check_val_every_n_epoch: int,
        log_every_n_steps: int,
        fast_dev_run: bool,
        checkpoint_dir: str | Path,
        train_config: dict,
    ) -> dict:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_limit = 1 if fast_dev_run else _batch_limit(train_loader, overfit_batches)
        val_limit = 1 if fast_dev_run else _batch_limit(val_loader, overfit_batches)
        epochs = min(int(max_epochs), 1) if fast_dev_run else int(max_epochs)
        best_loss = float("inf")
        best_path = None
        last_path = None
        last_metrics = {}

        for epoch in range(self.start_epoch, epochs):
            train_metrics, stop = self._run_epoch(
                train_loader,
                stage="train",
                max_batches=train_limit,
                log_every_n_steps=log_every_n_steps,
                max_steps=max_steps,
            )
            metrics = {f"train/{key}": value for key, value in train_metrics.items()}
            do_val = (
                (epoch + 1) % max(int(check_val_every_n_epoch), 1) == 0
                or epoch + 1 == epochs
                or stop
            )
            if do_val:
                val_metrics, _ = self._run_epoch(
                    val_loader,
                    stage="val",
                    max_batches=val_limit,
                    log_every_n_steps=log_every_n_steps,
                    max_steps=-1,
                )
                metrics.update({f"val/{key}": value for key, value in val_metrics.items()})

            lr = float(self.optimizer.param_groups[0]["lr"])
            metrics["lr"] = lr
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)
            self.writer.flush()

            summary = (
                f"[epoch {epoch + 1:03d}] "
                f"train_loss={metrics['train/loss']:.6f}"
            )
            if "val/loss" in metrics:
                summary += (
                    f" val_loss={metrics['val/loss']:.6f} "
                    f"mode_match={metrics['val/w1_target_mode_match']:.4f}"
                )
            summary += f" lr={lr:.3e} step={self.global_step}"
            print(summary)

            # Advance the epoch scheduler before checkpointing so resume starts
            # from the exact next-epoch LR state.
            self.scheduler.step()
            last_path = self.save_checkpoint(
                checkpoint_dir / "last.pt",
                epoch=epoch,
                metrics=metrics,
                train_config=train_config,
            )
            if "val/loss" in metrics and metrics["val/loss"] < best_loss:
                best_loss = metrics["val/loss"]
                best_path = self.save_checkpoint(
                    checkpoint_dir / "best.pt",
                    epoch=epoch,
                    metrics=metrics,
                    train_config=train_config,
                )

            last_metrics = metrics
            if stop:
                break

        return {
            "best_checkpoint": str(best_path) if best_path else None,
            "last_checkpoint": str(last_path) if last_path else None,
            "metrics": last_metrics,
            "global_step": int(self.global_step),
        }
