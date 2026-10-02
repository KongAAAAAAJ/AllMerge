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


g = load("guidance_cascade_v35_smoke", "highway_env/planner/diffusion/guidance.py")
s = load("trajectory_spline_cascade_v35_smoke", "highway_env/planner/diffusion/trajectory_spline.py")
Spline = s.ClampedCubicTrajectorySpline


def run(*, enabled: bool, strategy: str, second_polish: bool, hard_amp: bool = False):
    spline = Spline(horizon_s=4.0, sparse_dt=0.5, dense_dt=0.1)
    if hard_amp:
        physical = torch.tensor(
            [[[[8.0,-2.0],[16.0,-7.0],[24.0,-8.0],[32.0,-6.5],
               [40.0,-8.0],[48.0,-6.0],[56.0,-7.5],[72.0,-6.5]]]],
            dtype=torch.float32,
        )
    else:
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
        dense_polish_iters=8,
        dense_polish_step_m=0.10,
        dense_post_recovery_enabled=enabled,
        dense_post_recovery_threshold=0.04,
        dense_post_recovery_iters=5,
        dense_post_recovery_step_m=0.10,
        dense_post_recovery_max_total_move_m=0.50,
        dense_post_recovery_strategy=strategy,
        dense_post_recovery_hard_curvature=0.10,
        dense_post_recovery_medium_tangent_weight=0.30,
        dense_post_recovery_hard_tangent_weight=0.70,
        dense_post_recovery_medium_max_total_move_m=0.75,
        dense_post_recovery_hard_max_total_move_m=1.00,
        dense_post_recovery_second_polish_enabled=second_polish,
        dense_post_recovery_second_polish_iters=8,
        dense_post_recovery_second_polish_step_m=0.10,
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
    base, k_base, d_base = run(enabled=False, strategy="normal", second_polish=False)
    c1, k_c1, d_c1 = run(enabled=True, strategy="normal", second_polish=True)
    g2, k_g2, d_g2 = run(enabled=True, strategy="adaptive_tangent", second_polish=True, hard_amp=True)
    g3, k_g3, d_g3 = run(enabled=True, strategy="adaptive_trust", second_polish=True, hard_amp=True)

    assert float(d_base["post_recovery_trigger_fraction"].item()) == 0.0
    assert float(d_c1["post_recovery_trigger_fraction"].item()) > 0.0
    assert k_c1 <= k_base + 1.0e-6

    assert float(d_g2["post_recovery_mean_tangent_weight"].item()) > 0.0
    assert float(d_g2["post_recovery_mean_trust_limit_m"].item()) <= 0.5001

    assert float(d_g3["post_recovery_mean_tangent_weight"].item()) == 0.0
    assert float(d_g3["post_recovery_mean_trust_limit_m"].item()) >= 0.749

    for diag in (d_c1, d_g2, d_g3):
        assert "post_recovery_to_polish_fraction" in diag
        assert "second_polish_trigger_fraction" in diag
        assert "second_polish_iterations_used" in diag

    print("[PASS] V3.5 disabled preserves baseline")
    print("[PASS] C1 normal recovery + second polishing is wired")
    print("[PASS] G2.2 uses post-baseline tangent freedom at 0.50 m trust")
    print("[PASS] G3 uses normal-only recovery with enlarged adaptive trust")
    print("C1 max|kappa|:", k_base, "->", k_c1)
    print("G2.2 tangent/trust:", float(d_g2["post_recovery_mean_tangent_weight"].item()), float(d_g2["post_recovery_mean_trust_limit_m"].item()))
    print("G3 tangent/trust:", float(d_g3["post_recovery_mean_tangent_weight"].item()), float(d_g3["post_recovery_mean_trust_limit_m"].item()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
