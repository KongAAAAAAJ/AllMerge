# STAGED_DENSE_SUPERVISION_V1
from __future__ import annotations

import torch
import torch.nn.functional as F


def dense_position_loss(
    pred_dense_future: torch.Tensor,
    expert_dense_future: torch.Tensor,
    *,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Position-only dense supervision in physical ego-local metres."""
    if pred_dense_future.shape != expert_dense_future.shape:
        raise ValueError(
            "Dense trajectory shape mismatch: "
            f"pred={tuple(pred_dense_future.shape)} "
            f"target={tuple(expert_dense_future.shape)}"
        )
    if pred_dense_future.ndim != 3 or pred_dense_future.shape[-1] != 2:
        raise ValueError(
            "Dense trajectories must have shape [B,T,2], got "
            f"{tuple(pred_dense_future.shape)}"
        )
    if not torch.isfinite(pred_dense_future).all():
        raise ValueError("pred_dense_future contains NaN/Inf")
    if not torch.isfinite(expert_dense_future).all():
        raise ValueError("expert_dense_future contains NaN/Inf")

    kind = str(loss_type).strip().lower()
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred_dense_future, expert_dense_future)
    if kind == "l1":
        return F.l1_loss(pred_dense_future, expert_dense_future)
    if kind == "mse":
        return F.mse_loss(pred_dense_future, expert_dense_future)
    raise ValueError(
        f"Unsupported dense_loss_type={loss_type!r}; "
        "expected smooth_l1|l1|mse"
    )


def dense_timestep_weight(
    timesteps: torch.Tensor,
    *,
    mode: str = "terminal_constant",
    terminal_weight: float = 1.0,
) -> torch.Tensor:
    """Stage-D weighting hook. First version intentionally uses a constant."""
    kind = str(mode).strip().lower()
    if kind != "terminal_constant":
        raise ValueError(
            f"Unsupported dense_loss_weight_mode={mode!r}; "
            "Stage D supports only terminal_constant"
        )
    weight = float(terminal_weight)
    if weight < 0.0:
        raise ValueError("dense_loss_terminal_weight must be >= 0")
    return torch.full(
        timesteps.shape,
        weight,
        dtype=torch.float32,
        device=timesteps.device,
    )
