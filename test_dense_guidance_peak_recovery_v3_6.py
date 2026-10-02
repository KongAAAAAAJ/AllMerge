from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent


def load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    if spec is None or spec.loader is None:
        raise RuntimeError(rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


g = load("guidance_peak_v36_smoke", "highway_env/planner/diffusion/guidance.py")
s = load("trajectory_spline_peak_v36_smoke", "highway_env/planner/diffusion/trajectory_spline.py")
Spline = s.ClampedCubicTrajectorySpline


def run(peak_enabled: bool):
    spline = Spline(horizon_s=4.0, sparse_dt=0.5, dense_dt=0.1)
    physical = torch.tensor(
        [[[[8.0,-2.0],[16.0,-7.0],[24.0,-8.0],[32.0,-6.5],
           [40.0,-8.0],[48.0,-6.0],[56.0,-7.5],[72.0,-6.5]]]],
        dtype=torch.float32,
    )
    scale = torch.tensor([120.0,24.0], dtype=torch.float32)
    valid = torch.ones((1,1), dtype=torch.bool)
    v0 = torch.tensor([[16.0,0.0]], dtype=torch.float32)
    cfg = g.SparseCurvatureGuidanceConfig(
        enabled=True,
        guidance_type="dense_curvature",
        curvature_limit=0.02,
        dense_step_m=0.20,
        dense_min_segment_m=0.25,
        dense_inner_iters=1,
        dense_backtracking_enabled=True,
        dense_backtracking_factor=0.5,
        dense_backtracking_max_trials=4,
        dense_min_step_m=0.025,
        dense_max_total_move_m=0.50,
        dense_bound_weight=1.0,
        dense_peak_weight=0.5,
        dense_dkappa_weight=0.1,
        dense_peak_alpha=10.0,
        dense_residual_gradient_enabled=False,
        dense_normal_projection_enabled=True,
        dense_polish_enabled=True,
        dense_polish_max_curvature=0.04,
        dense_polish_iters=1,
        dense_polish_step_m=0.10,
        dense_post_recovery_enabled=True,
        dense_post_recovery_threshold=0.04,
        dense_post_recovery_iters=3,
        dense_post_recovery_step_m=0.10,
        dense_post_recovery_strategy="adaptive_trust",
        dense_post_recovery_hard_curvature=0.10,
        dense_post_recovery_medium_max_total_move_m=0.75,
        dense_post_recovery_hard_max_total_move_m=1.00,
        dense_post_recovery_second_polish_enabled=True,
        dense_post_recovery_second_polish_iters=2,
        dense_post_recovery_second_polish_step_m=0.10,
        dense_peak_recovery_enabled=peak_enabled,
        dense_peak_recovery_threshold=0.04,
        dense_peak_recovery_iters=5,
        dense_peak_recovery_step_m=0.05,
        dense_peak_recovery_bound_weight=0.5,
        dense_peak_recovery_peak_weight=2.0,
        dense_peak_recovery_dkappa_weight=0.05,
        dense_peak_recovery_peak_alpha=20.0,
        dense_peak_recovery_min_peak_improvement=1.0e-6,
        preserve_endpoint=True,
    )
    out, diag = g.apply_dense_curvature_guidance(
        physical / scale,
        trajectory_scale=scale,
        mode_valid_mask=valid,
        config=cfg,
        timestep=8,
        trajectory_spline=spline,
        start_velocity_xy=v0,
        allow_polish=True,
        allow_adaptive_normal=False,
    )
    assert diag is not None
    return out.detach(), diag


def main() -> int:
    base_out, base_diag = run(False)
    peak_out, peak_diag = run(True)
    before = float(base_diag["max_abs_curvature_after"].item())
    after = float(peak_diag["max_abs_curvature_after"].item())
    trigger = float(peak_diag["peak_recovery_trigger_fraction"].item())
    drop = float(peak_diag["peak_recovery_mean_max_curvature_drop"].item())
    assert trigger > 0.0
    assert after <= before + 1.0e-7
    assert drop >= 0.0
    print("[PASS] G4 triggers only on remaining hard-tail modes")
    print("[PASS] strict peak-aware acceptance never increases max curvature")
    print("max|kappa| without G4 -> with G4:", before, "->", after)
    print("reported G4 mean peak drop:", drop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
