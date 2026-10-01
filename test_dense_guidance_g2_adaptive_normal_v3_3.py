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

g = load("guidance_g2_smoke", "highway_env/planner/diffusion/guidance.py")
s = load("trajectory_spline_g2_smoke", "highway_env/planner/diffusion/trajectory_spline.py")
Spline = s.ClampedCubicTrajectorySpline

def run(amp: float, adaptive: bool):
    spline = Spline(horizon_s=4.0, sparse_dt=0.5, dense_dt=0.1)
    x = torch.tensor([8,16,24,32,40,48,56,64], dtype=torch.float32)
    y = torch.tensor(
        [0,-amp,-1.8*amp,-2.2*amp,-2.0*amp,-1.6*amp,-1.3*amp,-1.2*amp],
        dtype=torch.float32,
    )
    physical = torch.stack([x,y], dim=-1)[None,None]
    scale = torch.tensor([120.0,24.0], dtype=torch.float32)
    v0 = torch.tensor([[16.0,0.0]], dtype=torch.float32)
    valid = torch.ones((1,1), dtype=torch.bool)
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
        dense_adaptive_normal_enabled=adaptive,
        dense_adaptive_normal_medium_curvature=0.04,
        dense_adaptive_normal_hard_curvature=0.10,
        dense_adaptive_normal_medium_tangent_weight=0.20,
        dense_adaptive_normal_hard_tangent_weight=0.50,
        dense_polish_enabled=True,
        dense_polish_max_curvature=0.04,
        dense_polish_iters=3,
        dense_polish_step_m=0.10,
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
        allow_polish=False,
    )
    stats = g.dense_spline_curvature_statistics(
        out * scale,
        trajectory_spline=spline,
        start_velocity_xy=v0,
        curvature_limit=0.02,
        min_segment_m=0.25,
    )
    assert diag is not None
    return out.detach(), float(stats["max_abs_curvature"].item()), diag

def main() -> int:
    low_base, _, _ = run(0.8, False)
    low_g2, _, low_diag = run(0.8, True)
    assert torch.equal(low_base, low_g2), "G2 must not alter <=0.04 modes"
    assert float(low_diag["adaptive_normal_trigger_fraction"].item()) == 0.0

    med_base, med_k0, _ = run(2.0, False)
    med_g2, med_k1, med_diag = run(2.0, True)
    assert float(med_diag["adaptive_normal_trigger_fraction"].item()) > 0.0
    assert float(med_diag["adaptive_normal_mean_tangent_weight"].item()) > 0.0
    assert not torch.equal(med_base, med_g2)

    hard_base, hard_k0, _ = run(5.0, False)
    hard_g2, hard_k1, hard_diag = run(5.0, True)
    assert float(hard_diag["adaptive_normal_trigger_fraction"].item()) > 0.0
    assert float(hard_diag["adaptive_normal_mean_tangent_weight"].item()) > 0.0
    assert not torch.equal(hard_base, hard_g2)
    assert torch.isfinite(hard_g2).all()

    print("[PASS] <=0.04 remains bit-identical to F'+G1.1")
    print("[PASS] medium/hard modes receive tangent freedom")
    print("medium max|kappa| baseline -> G2:", med_k0, "->", med_k1)
    print("hard max|kappa| baseline -> G2:", hard_k0, "->", hard_k1)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
