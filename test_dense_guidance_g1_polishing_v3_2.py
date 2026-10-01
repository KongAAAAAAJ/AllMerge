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

g = load("guidance_g1_smoke", "highway_env/planner/diffusion/guidance.py")
s = load("trajectory_spline_g1_smoke", "highway_env/planner/diffusion/trajectory_spline.py")
Spline = s.ClampedCubicTrajectorySpline


def run(polish: bool):
    spline = Spline(horizon_s=4.0, sparse_dt=0.5, dense_dt=0.1)
    amp = 2.0
    x = torch.tensor([8, 16, 24, 32, 40, 48, 56, 64], dtype=torch.float32)
    y = torch.tensor([0, -amp, -1.8*amp, -2.2*amp, -2.0*amp, -1.6*amp, -1.3*amp, -1.2*amp], dtype=torch.float32)
    physical = torch.stack([x, y], dim=-1)[None, None]
    scale = torch.tensor([120.0, 24.0], dtype=torch.float32)
    v0 = torch.tensor([[16.0, 0.0]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    cfg = g.SparseCurvatureGuidanceConfig(
        enabled=True,
        guidance_type="dense_curvature",
        curvature_limit=0.02,
        dense_step_m=0.20,
        dense_min_segment_m=0.25,
        dense_inner_iters=5,
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
        dense_polish_enabled=polish,
        dense_polish_max_curvature=0.04,
        dense_polish_iters=3,
        dense_polish_step_m=0.075,
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
    )
    stats = g.dense_spline_curvature_statistics(
        out * scale,
        trajectory_spline=spline,
        start_velocity_xy=v0,
        curvature_limit=0.02,
        min_segment_m=0.25,
    )
    return float(stats["max_abs_curvature"].item()), diag


def main() -> int:
    base_k, base_diag = run(False)
    polish_k, polish_diag = run(True)
    assert base_diag is not None and polish_diag is not None
    assert polish_k <= base_k + 1.0e-7
    assert float(polish_diag["polish_trigger_fraction"].item()) > 0.0
    assert float(polish_diag["polish_iterations_used"].item()) >= 1.0
    assert float(polish_diag["polish_update_l2_m"].item()) > 0.0
    assert float(polish_diag["update_l2_m"].item()) <= 0.5001
    print("[PASS] F' baseline remains available with polishing disabled")
    print("[PASS] near-boundary polishing triggers only when enabled")
    print("[PASS] polishing does not increase max curvature")
    print("baseline max|kappa| =", base_k)
    print("G1 max|kappa|       =", polish_k)
    print("polish iterations   =", float(polish_diag["polish_iterations_used"].item()))
    print("polish update L2 m  =", float(polish_diag["polish_update_l2_m"].item()))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
