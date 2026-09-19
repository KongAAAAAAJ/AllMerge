"""Warmup + cosine learning-rate scheduler migrated from Diffusion-metadrive."""
from __future__ import annotations

import inspect
import math

from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        min_lr: float,
        lr: float,
        warmup_epochs: int,
        epochs: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.min_lr = float(min_lr)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        signature = inspect.signature(_LRScheduler.__init__)
        if "verbose" in signature.parameters:
            super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
        else:
            super().__init__(optimizer, last_epoch=last_epoch)

    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_init_lr(self):
        return self.lr / max(self.warmup_epochs, 1)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.lr * (self.last_epoch + 1) / max(self.warmup_epochs, 1)
        else:
            denominator = max(self.epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / denominator
                )
            )
        if self.optimizer.param_groups and "lr_scale" in self.optimizer.param_groups[0]:
            return [lr * group["lr_scale"] for group in self.optimizer.param_groups]
        return [lr for _ in self.optimizer.param_groups]
