# STAGED_DENSE_SUPERVISION_V1
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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

# DENSE_RESIDUAL_HEAD_V2
class DenseSparseResidualHead(nn.Module):
    """Small MLP that learns bounded execution corrections for sparse points.

    Gradient isolation is enforced inside the module as a second line of
    defense: both diffusion-derived inputs are detached before concatenation.
    The final layer is zero-initialized, so an untrained/disabled head produces
    exactly zero correction and therefore preserves legacy runtime behavior.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        horizon_steps: int,
        hidden_dim: int = 128,
        max_residual_x_m: float = 2.0,
        max_residual_y_m: float = 0.75,
    ) -> None:
        super().__init__()
        self.horizon_steps = int(horizon_steps)
        input_dim = int(feature_dim) + self.horizon_steps * 2
        hidden_dim = int(hidden_dim)
        hidden_dim_2 = max(hidden_dim // 2, 32)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim_2),
            nn.GELU(),
            nn.Linear(hidden_dim_2, self.horizon_steps * 2),
        )

        # Start from identity execution behavior: P_exec == P_base.
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        self.register_buffer(
            "max_residual_xy_m",
            torch.tensor(
                [float(max_residual_x_m), float(max_residual_y_m)],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def forward(
        self,
        mode_feature: torch.Tensor,
        sparse_norm: torch.Tensor,
    ) -> torch.Tensor:
        if mode_feature.ndim != 2:
            raise ValueError(
                f"mode_feature must be [B,D], got {tuple(mode_feature.shape)}"
            )
        if sparse_norm.ndim != 3 or tuple(sparse_norm.shape[-2:]) != (
            self.horizon_steps,
            2,
        ):
            raise ValueError(
                "sparse_norm must be [B,H,2], got "
                f"{tuple(sparse_norm.shape)}"
            )
        if mode_feature.shape[0] != sparse_norm.shape[0]:
            raise ValueError("mode_feature and sparse_norm batch sizes differ")

        # Hard stop-gradient boundary. L_dense cannot enter the diffusion
        # planner through either hidden features or sparse-point coordinates.
        feature = mode_feature.detach()
        sparse = sparse_norm.detach()
        inputs = torch.cat([feature, sparse.flatten(1)], dim=-1)
        raw = self.net(inputs).reshape(-1, self.horizon_steps, 2)
        limits = self.max_residual_xy_m.to(
            device=raw.device,
            dtype=raw.dtype,
        )
        return torch.tanh(raw) * limits.view(1, 1, 2)
