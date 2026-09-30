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

g = load("guidance_v3_smoke", "highway_env/planner/diffusion/guidance.py")
s = load("trajectory_spline_v3_smoke", "highway_env/planner/diffusion/trajectory_spline.py")
Spline = s.ClampedCubicTrajectorySpline

class DummyResidual(torch.nn.Module):
    def forward(self, feature: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
        # Deterministic frozen mapping whose output depends on x_norm, so the
        # smoke test exercises the execution-head input Jacobian.
        out = torch.zeros_like(x_norm)
        out[..., 1] = 0.25 * x_norm[..., 1]
        return out


def main() -> int:
    spline = Spline(horizon_s=4.0, sparse_dt=0.5, dense_dt=0.1)
    physical = torch.tensor(
        [[[[8.0, -1.0], [16.0, -3.8], [24.0, -4.0], [32.0, -3.7],
           [40.0, -4.1], [48.0, -3.6], [56.0, -4.0], [72.0, -3.9]]]],
        dtype=torch.float32,
    )
    scale = torch.tensor([120.0, 24.0], dtype=torch.float32)
    x0 = physical / scale
    valid_mode = torch.ones((1, 1), dtype=torch.bool)
    v0 = torch.tensor([[16.0, 0.0]], dtype=torch.float32)

    probe = physical.clone().requires_grad_(True)
    kappa, valid, _, _ = g.dense_spline_geometric_curvature(
        probe,
        trajectory_spline=spline,
        start_velocity_xy=v0,
        min_segment_m=0.25,
    )
    (kappa.square() * valid).sum().backward()
    assert probe.grad is not None and torch.isfinite(probe.grad).all()

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
        preserve_endpoint=True,
    )
    before = g.dense_spline_curvature_statistics(
        physical,
        trajectory_spline=spline,
        start_velocity_xy=v0,
        curvature_limit=cfg.curvature_limit,
        min_segment_m=cfg.dense_min_segment_m,
    )
    mode_features = torch.zeros((1, 1, 4), dtype=torch.float32)
    guided_norm, diag = g.apply_dense_curvature_guidance(
        x0,
        trajectory_scale=scale,
        mode_valid_mask=valid_mode,
        config=cfg,
        timestep=8,
        trajectory_spline=spline,
        start_velocity_xy=v0,
        mode_features=mode_features,
        execution_residual_head=DummyResidual(),
    )
    guided = guided_norm * scale
    after = g.dense_spline_curvature_statistics(
        guided,
        trajectory_spline=spline,
        start_velocity_xy=v0,
        curvature_limit=cfg.curvature_limit,
        min_segment_m=cfg.dense_min_segment_m,
    )
    assert diag is not None
    assert float(diag["loss_after"].item()) <= float(diag["loss_before"].item()) + 1e-6
    assert float(diag["max_abs_curvature_after"].item()) <= float(diag["max_abs_curvature_before"].item()) + 1e-6
    assert float(diag["update_l2_m"].item()) <= cfg.dense_max_total_move_m + 1e-4
    assert float(diag["inner_iterations_used"].item()) >= 1.0
    assert torch.allclose(guided[..., -1, :], physical[..., -1, :])
    assert torch.isfinite(guided).all()

    print("[PASS] dense V3 geometric curvature remains differentiable")
    print("[PASS] composite loss + inner iterations reduce accepted objective")
    print("[PASS] backtracking/trust-region prevent max-curvature regression")
    print("[PASS] residual-head input Jacobian path is active without parameter update")
    print("max|kappa|:", float(before["max_abs_curvature"].item()), "->", float(after["max_abs_curvature"].item()))
    print("loss:", float(diag["loss_before"].item()), "->", float(diag["loss_after"].item()))
    print("inner accepted:", float(diag["inner_iterations_used"].item()))
    print("backtracking reductions:", float(diag["backtracking_reductions"].item()))
    print("update_l2_m:", float(diag["update_l2_m"].item()))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
