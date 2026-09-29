from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SparseCurvatureGuidanceConfig:
    """Runtime-only sparse-trajectory curvature guidance.

    The guidance configuration is deliberately kept outside
    ``StructuredDiffusionConfig`` so pretrained checkpoints remain fully
    compatible and one checkpoint can be evaluated with guidance ON/OFF.
    """

    enabled: bool = False
    scale: float = 0.05
    curvature_limit: float = 0.02
    grad_clip_norm: float = 1.0
    preserve_endpoint: bool = True
    apply_timesteps: Optional[Tuple[int, ...]] = None
    min_segment_length_m: float = 1.0e-3
    eps: float = 1.0e-6

    @classmethod
    def from_mapping(
        cls,
        value: Optional[Mapping[str, object]],
    ) -> "SparseCurvatureGuidanceConfig":
        data = dict(value or {})
        if "apply_timesteps" in data and data["apply_timesteps"] is not None:
            raw = data["apply_timesteps"]
            if isinstance(raw, str):
                text = raw.strip().lower()
                if text in {"", "all", "none"}:
                    data["apply_timesteps"] = None
                else:
                    data["apply_timesteps"] = tuple(
                        int(item.strip())
                        for item in raw.split(",")
                        if item.strip()
                    )
            else:
                data["apply_timesteps"] = tuple(int(item) for item in raw)
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.scale < 0.0:
            raise ValueError("guidance scale must be >= 0")
        if self.curvature_limit <= 0.0:
            raise ValueError("guidance curvature_limit must be positive")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("guidance grad_clip_norm must be positive")
        if self.min_segment_length_m <= 0.0:
            raise ValueError("guidance min_segment_length_m must be positive")
        if self.eps <= 0.0:
            raise ValueError("guidance eps must be positive")
        if self.apply_timesteps is not None:
            if any(int(t) < 0 for t in self.apply_timesteps):
                raise ValueError("guidance apply_timesteps must be >= 0")

    def applies_to(self, timestep: int) -> bool:
        if not self.enabled or self.scale <= 0.0:
            return False
        if self.apply_timesteps is None:
            return True
        return int(timestep) in self.apply_timesteps


def sparse_discrete_curvature(
    trajectory_xy_m: torch.Tensor,
    *,
    min_segment_length_m: float = 1.0e-3,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Three-point geometric curvature for an ego-centric sparse trajectory.

    ``trajectory_xy_m`` is [..., H, 2] in physical metres.  The ego origin
    [0, 0] is prepended, therefore an H-waypoint trajectory yields H-1
    curvature samples.  The formula is the signed circumcircle curvature:

        kappa = 2 * cross(a, b) / (|a| |b| |a+b|)

    Degenerate triplets are marked invalid instead of producing large
    numerical curvature spikes.
    """
    if trajectory_xy_m.ndim < 2 or trajectory_xy_m.shape[-1] != 2:
        raise ValueError(
            "trajectory_xy_m must end with [H,2], got "
            f"{tuple(trajectory_xy_m.shape)}"
        )
    if trajectory_xy_m.shape[-2] < 2:
        raise ValueError("at least two future waypoints are required")

    origin = torch.zeros_like(trajectory_xy_m[..., :1, :])
    points = torch.cat([origin, trajectory_xy_m], dim=-2)

    edge_prev = points[..., 1:-1, :] - points[..., :-2, :]
    edge_next = points[..., 2:, :] - points[..., 1:-1, :]
    chord = points[..., 2:, :] - points[..., :-2, :]

    cross = (
        edge_prev[..., 0] * edge_next[..., 1]
        - edge_prev[..., 1] * edge_next[..., 0]
    )
    len_prev = torch.linalg.vector_norm(edge_prev, dim=-1)
    len_next = torch.linalg.vector_norm(edge_next, dim=-1)
    len_chord = torch.linalg.vector_norm(chord, dim=-1)

    valid = (
        (len_prev > float(min_segment_length_m))
        & (len_next > float(min_segment_length_m))
        & (len_chord > float(min_segment_length_m))
    )
    denom = len_prev * len_next * len_chord
    curvature = torch.where(
        valid,
        2.0 * cross / denom.clamp_min(float(eps)),
        torch.zeros_like(cross),
    )
    return curvature, valid


def _loss_per_mode(
    trajectory_xy_m: torch.Tensor,
    *,
    curvature_limit: float,
    min_segment_length_m: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    curvature, valid = sparse_discrete_curvature(
        trajectory_xy_m,
        min_segment_length_m=min_segment_length_m,
        eps=eps,
    )
    # Dimensionless excess makes the guidance scale less sensitive to the
    # chosen physical curvature threshold while retaining the same hinge.
    excess_ratio = F.relu(
        curvature.abs() / float(curvature_limit) - 1.0
    )
    valid_f = valid.to(dtype=trajectory_xy_m.dtype)
    count = valid_f.sum(dim=-1).clamp_min(1.0)
    loss = (excess_ratio.square() * valid_f).sum(dim=-1) / count
    return loss, curvature, valid


def sparse_curvature_statistics(
    trajectory_xy_m: torch.Tensor,
    *,
    curvature_limit: float,
    min_segment_length_m: float = 1.0e-3,
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    """Return per-trajectory max |kappa| and threshold violation ratio."""
    curvature, valid = sparse_discrete_curvature(
        trajectory_xy_m,
        min_segment_length_m=min_segment_length_m,
        eps=eps,
    )
    abs_curvature = curvature.abs()
    valid_f = valid.to(dtype=trajectory_xy_m.dtype)
    count = valid_f.sum(dim=-1).clamp_min(1.0)
    max_abs = torch.where(
        valid,
        abs_curvature,
        torch.zeros_like(abs_curvature),
    ).amax(dim=-1)
    violation = (
        (abs_curvature > float(curvature_limit)) & valid
    ).to(dtype=trajectory_xy_m.dtype)
    violation_ratio = violation.sum(dim=-1) / count
    return {
        "max_abs_curvature": max_abs,
        "violation_ratio": violation_ratio,
    }


def apply_sparse_curvature_guidance(
    predicted_x0_norm: torch.Tensor,
    *,
    trajectory_scale: torch.Tensor,
    mode_valid_mask: torch.Tensor,
    config: SparseCurvatureGuidanceConfig,
    timestep: int,
) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
    """Apply one inference-time gradient guidance step to clean x0.

    The denoiser output is detached first.  A tiny local autograd graph is
    created only for physical trajectory coordinates, so model parameters do
    not receive gradients even when the outer inference path is under
    ``torch.no_grad()``.
    """
    if not config.applies_to(int(timestep)):
        return predicted_x0_norm, None
    if predicted_x0_norm.ndim != 4 or predicted_x0_norm.shape[-1] != 2:
        raise ValueError(
            "predicted_x0_norm must be [B,M,H,2], got "
            f"{tuple(predicted_x0_norm.shape)}"
        )
    if tuple(mode_valid_mask.shape) != tuple(predicted_x0_norm.shape[:2]):
        raise ValueError(
            "mode_valid_mask shape mismatch: "
            f"mask={tuple(mode_valid_mask.shape)} "
            f"x0={tuple(predicted_x0_norm.shape)}"
        )

    scale_xy = trajectory_scale.to(
        device=predicted_x0_norm.device,
        dtype=predicted_x0_norm.dtype,
    )
    with torch.enable_grad():
        physical_xy = (
            predicted_x0_norm.detach() * scale_xy
        ).requires_grad_(True)
        loss_per_mode, curvature_before, valid_triplet = _loss_per_mode(
            physical_xy,
            curvature_limit=config.curvature_limit,
            min_segment_length_m=config.min_segment_length_m,
            eps=config.eps,
        )
        valid_mode_f = mode_valid_mask.to(dtype=physical_xy.dtype)
        objective = (loss_per_mode * valid_mode_f).sum()
        gradient = torch.autograd.grad(
            objective,
            physical_xy,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]

        # Invalid modes never participate in guidance.
        gradient = gradient * valid_mode_f[..., None, None]

        if config.preserve_endpoint:
            endpoint_mask = torch.ones_like(gradient)
            endpoint_mask[..., -1, :] = 0.0
            gradient = gradient * endpoint_mask

        flat = gradient.flatten(start_dim=-2)
        grad_norm = torch.linalg.vector_norm(flat, dim=-1)
        clip_factor = (
            float(config.grad_clip_norm)
            / grad_norm.clamp_min(float(config.eps))
        ).clamp(max=1.0)
        gradient = gradient * clip_factor[..., None, None]

        guided_xy = physical_xy - float(config.scale) * gradient
        guided_xy = guided_xy.detach()

    with torch.no_grad():
        loss_after, curvature_after, _ = _loss_per_mode(
            guided_xy,
            curvature_limit=config.curvature_limit,
            min_segment_length_m=config.min_segment_length_m,
            eps=config.eps,
        )
        valid_mode_f = mode_valid_mask.to(dtype=guided_xy.dtype)
        mode_count = valid_mode_f.sum(dim=-1).clamp_min(1.0)

        update_per_mode = torch.linalg.vector_norm(
            (guided_xy - physical_xy.detach()).flatten(start_dim=-2),
            dim=-1,
        )
        loss_before_sample = (
            loss_per_mode.detach() * valid_mode_f
        ).sum(dim=-1) / mode_count
        loss_after_sample = (
            loss_after * valid_mode_f
        ).sum(dim=-1) / mode_count
        update_l2_sample = (
            update_per_mode * valid_mode_f
        ).sum(dim=-1) / mode_count

        # Max curvature over all valid modes/triplets for diagnostics.
        valid_all = valid_triplet & mode_valid_mask[..., None]
        max_before = torch.where(
            valid_all,
            curvature_before.detach().abs(),
            torch.zeros_like(curvature_before),
        ).flatten(start_dim=1).amax(dim=-1)
        max_after = torch.where(
            valid_all,
            curvature_after.abs(),
            torch.zeros_like(curvature_after),
        ).flatten(start_dim=1).amax(dim=-1)

        guided_norm = guided_xy / scale_xy
        diagnostics = {
            "loss_before": loss_before_sample,
            "loss_after": loss_after_sample,
            "max_abs_curvature_before": max_before,
            "max_abs_curvature_after": max_after,
            "update_l2_m": update_l2_sample,
        }
    return guided_norm, diagnostics
