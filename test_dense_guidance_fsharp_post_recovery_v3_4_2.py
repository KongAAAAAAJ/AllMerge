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


g = load("guidance_fsharp_v342_smoke", "highway_env/planner/diffusion/guidance.py")
s = load("trajectory_spline_fsharp_v342_smoke", "highway_env/planner/diffusion/trajectory_spline.py")
Spline = s.ClampedCubicTrajectorySpline


def run(enabled: bool, extra_iters: int):
    spline = Spline(horizon_s=4.0, sparse_dt=0.5, dense_dt=0.1)
    physical = torch.tensor(
        [[[[8.0,-1.0],[16.0,-3.8],[24.0,-4.0],[32.0,-3.7],
           [40.0,-4.1],[48.0,-3.6],[56.0,-4.0],[72.0,-3.9]]]],
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
        dense_residual_gradient_enabled=True,
        dense_normal_projection_enabled=True,
        dense_polish_enabled=False,
        dense_polish_max_curvature=0.04,
        dense_polish_iters=3,
        dense_polish_step_m=0.10,
        dense_post_recovery_enabled=enabled,
        dense_post_recovery_threshold=0.04,
        dense_post_recovery_iters=extra_iters,
        dense_post_recovery_step_m=0.10,
        dense_post_recovery_max_total_move_m=0.50,
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
    )
    assert diag is not None
    stats = g.dense_spline_curvature_statistics(
        out * scale,
        trajectory_spline=spline,
        start_velocity_xy=v0,
        curvature_limit=0.02,
        min_segment_m=0.25,
    )
    return out.detach(), float(stats["max_abs_curvature"].item()), diag


def main() -> int:
    base, k_base, d_base = run(False, 3)
    f3, k_f3, d_f3 = run(True, 3)
    f5, k_f5, d_f5 = run(True, 5)

    assert float(d_base["post_recovery_trigger_fraction"].item()) == 0.0
    assert float(d_f3["post_recovery_trigger_fraction"].item()) > 0.0
    assert float(d_f3["post_recovery_iterations_used"].item()) >= 1.0
    assert k_f3 <= k_base + 1.0e-6
    assert k_f5 <= k_f3 + 1.0e-6

    print("[PASS] F# disabled preserves baseline behavior")
    print("[PASS] final hard-tail mask triggers post-baseline recovery")
    print("[PASS] extra normal-only iterations do not increase max curvature")
    print("max|kappa| baseline -> F#3 -> F#5:", k_base, "->", k_f3, "->", k_f5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
