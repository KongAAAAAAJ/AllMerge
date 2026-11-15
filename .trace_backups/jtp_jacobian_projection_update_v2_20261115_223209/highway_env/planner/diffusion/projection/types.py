"""Typed road geometry and result containers for TRACE JTP."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class RoadGeometry:
    """World-frame road corridors represented by oriented centerline polylines."""
    centerlines_xy: torch.Tensor
    left_width_m: torch.Tensor | float
    right_width_m: torch.Tensor | float

    def tensors(self, *, device, dtype):
        center = torch.as_tensor(self.centerlines_xy, device=device, dtype=dtype)
        if center.ndim == 2:
            center = center.unsqueeze(0)
        if center.ndim != 3 or center.shape[-1] != 2 or center.shape[1] < 2:
            raise ValueError("Road centerlines must be [P>=2,2] or [L,P>=2,2]")
        lanes, points = center.shape[:2]

        def expand_width(value, name):
            width = torch.as_tensor(value, device=device, dtype=dtype)
            if width.ndim == 0:
                width = width.expand(lanes, points)
            elif width.ndim == 1:
                if width.numel() == lanes:
                    width = width[:, None].expand(lanes, points)
                elif width.numel() == points and lanes == 1:
                    width = width[None, :]
                else:
                    raise ValueError(f"{name} 1-D shape must be [L] or [P] for one lane")
            elif width.ndim != 2 or width.shape != (lanes, points):
                raise ValueError(f"{name} must be scalar, [L], [P] (single lane), or [L,P]")
            if not torch.isfinite(width).all() or (width <= 0).any():
                raise ValueError(f"{name} must contain finite positive widths")
            return width

        if not torch.isfinite(center).all():
            raise ValueError("Road centerlines contain NaN/Inf")
        lengths = torch.linalg.vector_norm(center[:, 1:] - center[:, :-1], dim=-1)
        if (lengths <= 1.0e-6).any():
            raise ValueError("Road centerlines must not contain duplicate consecutive points")
        return center, expand_width(self.left_width_m, "left_width_m"), expand_width(self.right_width_m, "right_width_m")


@dataclass
class JTPResult:
    controls: torch.Tensor
    trajectories: torch.Tensor
    states: torch.Tensor
    ltr: torch.Tensor
    success: bool
    status: str
    iterations: int
    diagnostics: dict
