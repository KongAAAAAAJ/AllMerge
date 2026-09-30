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
    guidance_type: str = "sparse_curvature"
    scale: float = 0.05
    curvature_limit: float = 0.02
    grad_clip_norm: float = 1.0
    preserve_endpoint: bool = True
    apply_timesteps: Optional[Tuple[int, ...]] = None
    min_segment_length_m: float = 1.0e-3
    dense_min_speed_mps: float = 0.5
    eps: float = 1.0e-6

    @classmethod
    def from_mapping(
        cls,
        value: Optional[Mapping[str, object]],
    ) -> "SparseCurvatureGuidanceConfig":
        data = dict(value or {})
        # DENSE_CURVATURE_GUIDANCE_V1: public runtime key is `type`.
        if "type" in data and "guidance_type" not in data:
            data["guidance_type"] = data.pop("type")
        if "guidance_type" in data:
            aliases = {
                "sparse": "sparse_curvature",
                "dense": "dense_curvature",
                "off": "none",
                "disabled": "none",
            }
            key = str(data["guidance_type"]).strip().lower()
            data["guidance_type"] = aliases.get(key, key)
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
        if self.guidance_type not in {
            "none",
            "sparse_curvature",
            "dense_curvature",
        }:
            raise ValueError(
                "guidance_type must be one of: none, sparse_curvature, "
                f"dense_curvature; got {self.guidance_type!r}"
            )
        if self.scale < 0.0:
            raise ValueError("guidance scale must be >= 0")
        if self.curvature_limit <= 0.0:
            raise ValueError("guidance curvature_limit must be positive")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("guidance grad_clip_norm must be positive")
        if self.min_segment_length_m <= 0.0:
            raise ValueError("guidance min_segment_length_m must be positive")
        if self.dense_min_speed_mps <= 0.0:
            raise ValueError("guidance dense_min_speed_mps must be positive")
        if self.eps <= 0.0:
            raise ValueError("guidance eps must be positive")
        if self.apply_timesteps is not None:
            if any(int(t) < 0 for t in self.apply_timesteps):
                raise ValueError("guidance apply_timesteps must be >= 0")

    def applies_to(self, timestep: int) -> bool:
        if (
            not self.enabled
            or self.scale <= 0.0
            or self.guidance_type == "none"
        ):
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
    if (
        not config.applies_to(int(timestep))
        or config.guidance_type != "sparse_curvature"
    ):
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


# DENSE_CURVATURE_GUIDANCE_V1
def dense_spline_curvature(
    trajectory_xy_m: torch.Tensor,
    *,
    trajectory_spline,
    start_velocity_xy: torch.Tensor,
    min_speed_mps: float = 0.5,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic curvature of the differentiable 10 Hz clamped cubic spline.

    ``trajectory_xy_m`` contains the 8 future sparse control points and may be
    [B,H,2] or [B,M,H,2].  The existing PyTorch spline prepends ego-local
    start_xy=(0,0), uses ego velocity as the clamped start derivative, and is
    evaluated at t=0.1,...,4.0 s.
    """
    if trajectory_xy_m.ndim not in (3, 4) or trajectory_xy_m.shape[-1] != 2:
        raise ValueError(
            "trajectory_xy_m must be [B,H,2] or [B,M,H,2], got "
            f"{tuple(trajectory_xy_m.shape)}"
        )

    start_xy = torch.zeros_like(trajectory_xy_m[..., 0, :])
    start_velocity = torch.as_tensor(
        start_velocity_xy,
        device=trajectory_xy_m.device,
        dtype=trajectory_xy_m.dtype,
    )
    while start_velocity.ndim < trajectory_xy_m.ndim - 1:
        start_velocity = start_velocity.unsqueeze(-2)

    dense_time = trajectory_spline.dense_time_s.to(
        device=trajectory_xy_m.device,
        dtype=trajectory_xy_m.dtype,
    )
    if dense_time.numel() < 2:
        raise ValueError("trajectory spline dense grid must include future samples")

    _, velocity, acceleration = trajectory_spline.evaluate(
        trajectory_xy_m,
        start_xy=start_xy,
        start_velocity_xy=start_velocity,
        query_time_s=dense_time[1:],
    )
    vx = velocity[..., 0]
    vy = velocity[..., 1]
    ax = acceleration[..., 0]
    ay = acceleration[..., 1]
    speed_sq = vx.square() + vy.square()
    speed = torch.sqrt(speed_sq.clamp_min(float(eps)))
    valid = speed > float(min_speed_mps)
    numerator = vx * ay - vy * ax
    denominator = speed_sq.clamp_min(float(eps)).pow(1.5)
    curvature = torch.where(
        valid,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    return curvature, valid


def _dense_loss_per_mode(
    trajectory_xy_m: torch.Tensor,
    *,
    trajectory_spline,
    start_velocity_xy: torch.Tensor,
    curvature_limit: float,
    min_speed_mps: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    curvature, valid = dense_spline_curvature(
        trajectory_xy_m,
        trajectory_spline=trajectory_spline,
        start_velocity_xy=start_velocity_xy,
        min_speed_mps=min_speed_mps,
        eps=eps,
    )
    excess_ratio = F.relu(
        curvature.abs() / float(curvature_limit) - 1.0
    )
    valid_f = valid.to(dtype=trajectory_xy_m.dtype)
    count = valid_f.sum(dim=-1).clamp_min(1.0)
    loss = (excess_ratio.square() * valid_f).sum(dim=-1) / count
    return loss, curvature, valid


def dense_spline_curvature_statistics(
    trajectory_xy_m: torch.Tensor,
    *,
    trajectory_spline,
    start_velocity_xy: torch.Tensor,
    curvature_limit: float,
    min_speed_mps: float = 0.5,
    eps: float = 1.0e-6,
) -> dict[str, torch.Tensor]:
    curvature, valid = dense_spline_curvature(
        trajectory_xy_m,
        trajectory_spline=trajectory_spline,
        start_velocity_xy=start_velocity_xy,
        min_speed_mps=min_speed_mps,
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
    return {
        "max_abs_curvature": max_abs,
        "violation_ratio": violation.sum(dim=-1) / count,
    }


def apply_dense_curvature_guidance(
    predicted_x0_norm: torch.Tensor,
    *,
    trajectory_scale: torch.Tensor,
    mode_valid_mask: torch.Tensor,
    config: SparseCurvatureGuidanceConfig,
    timestep: int,
    trajectory_spline,
    start_velocity_xy: torch.Tensor,
    mode_features: Optional[torch.Tensor] = None,
    execution_residual_head=None,
) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
    """Guide sparse x0 with curvature of the 10 Hz execution spline.

    If the Dense Residual Head is present, its current correction is included
    before spline decoding.  The residual is intentionally stop-gradient, so
    the guidance gradient updates only Diffusion x0 coordinates.
    """
    if (
        not config.applies_to(int(timestep))
        or config.guidance_type != "dense_curvature"
    ):
        return predicted_x0_norm, None
    if predicted_x0_norm.ndim != 4 or predicted_x0_norm.shape[-1] != 2:
        raise ValueError(
            "predicted_x0_norm must be [B,M,H,2], got "
            f"{tuple(predicted_x0_norm.shape)}"
        )
    if tuple(mode_valid_mask.shape) != tuple(predicted_x0_norm.shape[:2]):
        raise ValueError("mode_valid_mask shape mismatch for dense guidance")

    scale_xy = trajectory_scale.to(
        device=predicted_x0_norm.device,
        dtype=predicted_x0_norm.dtype,
    )
    batch, modes, horizon, _ = predicted_x0_norm.shape

    def execution_points(physical_xy: torch.Tensor) -> torch.Tensor:
        if execution_residual_head is None:
            return physical_xy
        if mode_features is None:
            raise ValueError(
                "mode_features are required when execution_residual_head is used"
            )
        if tuple(mode_features.shape[:2]) != (batch, modes):
            raise ValueError("mode_features shape mismatch for dense guidance")
        flat_feature = mode_features.detach().reshape(batch * modes, -1)
        flat_norm = (
            physical_xy.detach() / scale_xy
        ).reshape(batch * modes, horizon, 2)
        residual = execution_residual_head(
            flat_feature,
            flat_norm,
        ).reshape(batch, modes, horizon, 2)
        return physical_xy + residual.to(
            device=physical_xy.device,
            dtype=physical_xy.dtype,
        )

    with torch.enable_grad():
        physical_xy = (
            predicted_x0_norm.detach() * scale_xy
        ).requires_grad_(True)
        execution_xy = execution_points(physical_xy)
        loss_per_mode, curvature_before, valid_dense = _dense_loss_per_mode(
            execution_xy,
            trajectory_spline=trajectory_spline,
            start_velocity_xy=start_velocity_xy,
            curvature_limit=config.curvature_limit,
            min_speed_mps=config.dense_min_speed_mps,
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
        guided_xy = (
            physical_xy - float(config.scale) * gradient
        ).detach()

    with torch.no_grad():
        guided_execution_xy = execution_points(guided_xy)
        loss_after, curvature_after, valid_after = _dense_loss_per_mode(
            guided_execution_xy,
            trajectory_spline=trajectory_spline,
            start_velocity_xy=start_velocity_xy,
            curvature_limit=config.curvature_limit,
            min_speed_mps=config.dense_min_speed_mps,
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

        valid_before_all = valid_dense & mode_valid_mask[..., None]
        valid_after_all = valid_after & mode_valid_mask[..., None]
        max_before = torch.where(
            valid_before_all,
            curvature_before.detach().abs(),
            torch.zeros_like(curvature_before),
        ).flatten(start_dim=1).amax(dim=-1)
        max_after = torch.where(
            valid_after_all,
            curvature_after.abs(),
            torch.zeros_like(curvature_after),
        ).flatten(start_dim=1).amax(dim=-1)

        diagnostics = {
            "loss_before": loss_before_sample,
            "loss_after": loss_after_sample,
            "max_abs_curvature_before": max_before,
            "max_abs_curvature_after": max_after,
            "update_l2_m": update_l2_sample,
        }
        guided_norm = guided_xy / scale_xy
    return guided_norm, diagnostics


def apply_curvature_guidance(
    predicted_x0_norm: torch.Tensor,
    *,
    trajectory_scale: torch.Tensor,
    mode_valid_mask: torch.Tensor,
    config: SparseCurvatureGuidanceConfig,
    timestep: int,
    trajectory_spline=None,
    start_velocity_xy: Optional[torch.Tensor] = None,
    mode_features: Optional[torch.Tensor] = None,
    execution_residual_head=None,
) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
    if not config.applies_to(int(timestep)):
        return predicted_x0_norm, None
    if config.guidance_type == "sparse_curvature":
        return apply_sparse_curvature_guidance(
            predicted_x0_norm,
            trajectory_scale=trajectory_scale,
            mode_valid_mask=mode_valid_mask,
            config=config,
            timestep=timestep,
        )
    if config.guidance_type == "dense_curvature":
        if trajectory_spline is None or start_velocity_xy is None:
            raise ValueError(
                "dense_curvature guidance requires trajectory_spline and "
                "start_velocity_xy"
            )
        return apply_dense_curvature_guidance(
            predicted_x0_norm,
            trajectory_scale=trajectory_scale,
            mode_valid_mask=mode_valid_mask,
            config=config,
            timestep=timestep,
            trajectory_spline=trajectory_spline,
            start_velocity_xy=start_velocity_xy,
            mode_features=mode_features,
            execution_residual_head=execution_residual_head,
        )
    return predicted_x0_norm, None
