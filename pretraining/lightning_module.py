"""Lightning wrapper for the already-migrated StructuredDiffusionPlanner.

The wrapper intentionally does not reproduce any target-mode assignment,
diffusion target construction, regression loss, or classification loss.  Those
remain owned by StructuredDiffusionPlanner.forward_train().
"""
from __future__ import annotations

from typing import Dict, Optional

import pytorch_lightning as pl
import torch

from highway_env.planner.diffusion.config import build_structured_diffusion_config
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter

from .checkpoint_io import load_checkpoint_file
from .contract import validate_model_schema, validate_training_batch
from .dataset_adapter import unpack_w1_batch
from .warmup_cos_lr import WarmupCosLR


class DiffusionPretrainModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model_config: Optional[dict] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        min_lr: float = 1e-6,
        warmup_epochs: int = 3,
        max_epochs: int = 50,
        init_checkpoint: Optional[str] = None,
        strict_init_checkpoint: bool = True,
        validate_first_batch: bool = True,
    ) -> None:
        super().__init__()
        model_config = dict(model_config or {})
        self.save_hyperparameters()

        self.model_config_dict = model_config
        self.model_config = build_structured_diffusion_config(**model_config)
        validate_model_schema(self.model_config)

        # PlannerTensorAdapter stores scale tensors as plain attributes rather
        # than nn.Module buffers.  Start on CPU, then synchronize them to the
        # Lightning device before every model call.
        self.adapter = PlannerTensorAdapter(self.model_config, torch.device("cpu"))
        self.planner = StructuredDiffusionPlanner(self.model_config, self.adapter)

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.min_lr = float(min_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        self.validate_first_batch = bool(validate_first_batch)
        self._batch_contract_validated = False

        if init_checkpoint:
            state_dict, stored_config = load_checkpoint_file(init_checkpoint, map_location="cpu")
            if stored_config and stored_config != model_config:
                print(
                    "[DiffusionPretrainModule] checkpoint carries model_config; "
                    "current explicit config remains authoritative"
                )
            missing, unexpected = self.planner.load_state_dict(
                state_dict,
                strict=bool(strict_init_checkpoint),
            )
            if not strict_init_checkpoint and (missing or unexpected):
                print(
                    "[DiffusionPretrainModule] non-strict init checkpoint: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )

    def _sync_adapter_device(self) -> None:
        device = self.device
        for name in (
            "ego_scale",
            "agent_scale",
            "map_scale",
            "target_point_scale",
            "trajectory_scale",
        ):
            value = getattr(self.adapter, name)
            if value.device != device:
                setattr(self.adapter, name, value.to(device=device))

    def forward_train(
        self,
        features: Dict[str, torch.Tensor],
        target_trajectory: torch.Tensor,
        target_semantic: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self._sync_adapter_device()
        return self.planner.forward_train(
            features,
            target_trajectory,
            target_semantic,
        )

    def _shared_step(self, batch, stage: str):
        if self.validate_first_batch and not self._batch_contract_validated:
            validate_training_batch(batch, self.model_config)
            self._batch_contract_validated = True

        batch = unpack_w1_batch(batch)
        features = batch["features"]
        target_trajectory = batch["expert_trajectory"]
        target_semantic = batch["expert_semantic"]

        # IMPORTANT: target_mode and all supervised losses are computed inside
        # StructuredDiffusionPlanner.forward_train().
        output = self.forward_train(
            features,
            target_trajectory,
            target_semantic,
        )

        batch_size = int(features["ego_state"].shape[0])
        log_values = {
            f"{stage}/loss": output["loss"],
            f"{stage}/trajectory_regression_loss": output["trajectory_regression_loss"],
            f"{stage}/trajectory_classification_loss": output["trajectory_classification_loss"],
            f"{stage}/target_mode_assignment_distance": output[
                "target_mode_assignment_distance"
            ].mean(),
            f"{stage}/target_mode_geometry_valid_fraction": output[
                "target_mode_geometry_valid"
            ].float().mean(),
            f"{stage}/target_mode_traffic_valid_fraction": output[
                "target_mode_traffic_valid"
            ].float().mean(),
        }
        self.log_dict(
            log_values,
            prog_bar=(stage == "val"),
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=False,
        )

        # W1 expert_mode is a diagnostic copy of the same semantic-constrained
        # assignment.  Compare only; never use it as a training target here.
        if "expert_mode" in batch:
            diagnostic_match = (
                output["target_mode"].detach() == batch["expert_mode"]
            ).float().mean()
            self.log(
                f"{stage}/w1_target_mode_match",
                diagnostic_match,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        return output["loss"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.planner.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=self.min_lr,
            epochs=self.max_epochs,
            warmup_epochs=self.warmup_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint) -> None:
        # Keep Lightning state_dict untouched so optimizer/scheduler resume works.
        # Add a planner-only copy for deterministic runtime export.
        checkpoint["planner_state_dict"] = {
            key: value.detach().cpu()
            for key, value in self.planner.state_dict().items()
        }
        checkpoint["allmerge_model_config"] = dict(self.model_config_dict)
        checkpoint["checkpoint_format"] = "allmerge_diffusion_pretrain_v1"
