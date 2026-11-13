"""In-repository MATP-inspired multi-active violation-weighted projection.

Migrated from:
  all_merge-guideDistill/highway_env/planner/diffusion/guide_distill/matp_w1.py

Important provenance: the source file explicitly identifies this as the
in-repository W6 port and states that numerical parity with the historical
MATP V4.3 implementation is unverified.  The implementation below preserves
that algorithm and its diagnostic version label.  Its active constraint set
is the dense-spline curvature bound; it does not implement the later TRACE
Koopman yaw/sideslip/roll/LTR, road, collision, or control-envelope joint QP.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .spline import evaluate_dense_spline


@dataclass(frozen=True)
class MatpW1Config:
    curvature_limit: float = 0.02
    active_threshold: float = 0.015
    tightening: float = 0.0005
    iters: int = 6
    max_active: int = 8
    regularization: float = 1.0e-4
    step_max_m: float = 0.15
    max_trust_m: float = 1.0
    initial_trust_m: float = 0.30
    beta: float = 2.0
    tolerance: float = 1.0e-5
    preserve_endpoint: bool = True



def curvature_series(
    xy: torch.Tensor,
    config,
    initial_velocity_xy=None,
) -> torch.Tensor:
    dense, _, _ = evaluate_dense_spline(
        xy, config, initial_velocity_xy=initial_velocity_xy
    )
    step = dense[1:] - dense[:-1]
    heading = torch.atan2(step[:, 1], step[:, 0])
    delta_heading = torch.atan2(
        torch.sin(heading[1:] - heading[:-1]),
        torch.cos(heading[1:] - heading[:-1]),
    )
    scale = 0.5 * (
        torch.linalg.vector_norm(step[:-1], dim=-1)
        + torch.linalg.vector_norm(step[1:], dim=-1)
    ).clamp_min(0.25)
    return delta_heading / scale


def evaluate_w1_curvature_feasibility(
    xy: torch.Tensor,
    reward_config,
    *,
    initial_velocity_xy=None,
    projector_config: MatpW1Config = MatpW1Config(),
):
    """Evaluate the W1 curvature constraint before applying projection.

    This is deliberately named *curvature feasibility*: it must not be
    reported as full TRACE/MATP feasibility because this W1 port contains no
    Koopman chassis-stability, road-boundary, collision, or actuator checks.
    Input shape is [N,8,2].
    """
    if xy.ndim != 3 or tuple(xy.shape[-2:]) != (8, 2):
        raise ValueError("MATP feasibility needs [N,8,2]")
    if not torch.isfinite(xy).all():
        raise ValueError("Nonfinite input")
    values = []
    for index in range(len(xy)):
        v0 = None if initial_velocity_xy is None else initial_velocity_xy[index]
        values.append(curvature_series(xy[index], reward_config, v0).abs().max())
    peak = torch.stack(values) if values else xy.new_empty((0,))
    feasible = peak <= projector_config.curvature_limit + projector_config.tolerance
    return {
        "max_abs_curvature_before": peak,
        "curvature_feasible_before": feasible,
        "curvature_limit": float(projector_config.curvature_limit),
        "constraint_scope": "dense_spline_curvature_only",
        "projector_variant": "in_repo_matp_weighted_joint_v6",
        "parity_with_frozen_v43_w1": "UNVERIFIED_original_source_unavailable",
    }


def project_w1_local(
    xy: torch.Tensor,
    reward_config,
    *,
    initial_velocity_xy=None,
    projector_config: MatpW1Config = MatpW1Config(),
):
    if xy.ndim != 3 or tuple(xy.shape[-2:]) != (8, 2):
        raise ValueError("MATP needs [N,8,2]")
    if not torch.isfinite(xy).all():
        raise ValueError("Nonfinite input")

    cfg = projector_config
    anchor = xy.detach()
    output = anchor.clone()
    peaks = []
    accepted = []
    trust = cfg.initial_trust_m

    for batch_index in range(len(output)):
        x = output[batch_index].detach().clone()
        velocity = (
            None if initial_velocity_xy is None else initial_velocity_xy[batch_index]
        )
        current_peak = float(curvature_series(x, reward_config, velocity).abs().max())
        best = x.clone()
        best_peak = current_peak

        for _iteration in range(cfg.iters):
            with torch.enable_grad():
                z = x.detach().clone().requires_grad_(True)
                curvature = curvature_series(z, reward_config, velocity)
                violation = curvature.abs() - (cfg.curvature_limit - cfg.tightening)
                top_ids = torch.topk(
                    violation, k=min(cfg.max_active, len(violation))
                ).indices
                active = top_ids[
                    violation[top_ids] > -max(0.0, cfg.curvature_limit - cfg.active_threshold)
                ]
                if len(active) == 0:
                    break
                values = curvature[active]
                signs = values.detach().sign().clamp(min=-1.0, max=1.0)
                rhs = (
                    values.abs() - (cfg.curvature_limit - cfg.tightening)
                ).clamp_min(0.0).detach()
                if not bool((rhs > cfg.tolerance).any()):
                    break
                jacobian = torch.stack([
                    torch.autograd.grad(
                        curvature[j], z, retain_graph=True
                    )[0].reshape(-1)
                    for j in active
                ])

            max_violation = rhs.max().clamp_min(1.0e-8)
            weights = (1.0 + cfg.beta * rhs / max_violation).sqrt()
            weighted_jacobian = signs[:, None] * jacobian * weights[:, None]
            weighted_rhs = rhs * weights
            if cfg.preserve_endpoint:
                weighted_jacobian = weighted_jacobian[:, :-2]
            regularizer = torch.eye(
                len(rhs), device=x.device, dtype=x.dtype
            ) * cfg.regularization
            correction = -(
                weighted_jacobian.T
                @ torch.linalg.solve(
                    weighted_jacobian @ weighted_jacobian.T + regularizer,
                    weighted_rhs,
                )
            )
            delta = torch.zeros(16, device=x.device, dtype=x.dtype)
            delta[:len(correction)] = correction
            delta = delta.reshape(8, 2)
            magnitude = torch.linalg.vector_norm(delta.reshape(-1)).clamp_min(1.0e-9)
            delta = delta * min(1.0, cfg.step_max_m / float(magnitude))
            candidate = x + delta
            distance = torch.linalg.vector_norm(candidate - anchor[batch_index]).clamp_min(1.0e-9)
            candidate = anchor[batch_index] + (
                candidate - anchor[batch_index]
            ) * min(1.0, trust / float(distance))
            if cfg.preserve_endpoint:
                candidate[-1] = anchor[batch_index, -1]
            new_peak = float(
                curvature_series(candidate, reward_config, velocity).abs().max()
            )
            if new_peak + 1.0e-8 < best_peak:
                x = candidate.detach()
                best = x.clone()
                best_peak = new_peak
            else:
                break

        output[batch_index] = best
        peaks.append(best_peak)
        accepted.append(best_peak <= cfg.curvature_limit + cfg.tolerance)

    return output, {
        "max_abs_curvature_after": torch.tensor(
            peaks, device=xy.device, dtype=xy.dtype
        ),
        "feasible": torch.tensor(
            accepted, device=xy.device, dtype=torch.bool
        ),
        "projector_variant": "in_repo_matp_weighted_joint_v6",
        "parity_with_frozen_v43_w1": "UNVERIFIED_original_source_unavailable",
        "constraint_scope": "dense_spline_curvature_only",
    }
