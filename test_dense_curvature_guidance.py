from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
MOD = ROOT / "highway_env/planner/diffusion"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def main():
    g = load("dense_guidance_repair_test", MOD / "guidance.py")
    s = load("dense_spline_repair_test", MOD / "trajectory_spline.py")
    spline = s.ClampedCubicTrajectorySpline(horizon_s=4.0, sparse_dt=0.5, dense_dt=0.1)
    p = torch.tensor([[[5.,0.],[10.,1.5],[15.,-1.5],[20.,1.5],[25.,-1.5],[30.,1.5],[35.,-1.5],[40.,0.]]], requires_grad=True)
    dense = spline(p, start_xy=torch.zeros((1,2)), start_velocity_xy=torch.tensor([[10.,0.]]))
    dense.square().mean().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0

    physical = p.detach().unsqueeze(1)
    scale = torch.tensor([120.,24.])
    x0 = physical / scale
    mask = torch.ones((1,1), dtype=torch.bool)
    vel = torch.tensor([[10.,0.]])
    cfg = g.SparseCurvatureGuidanceConfig(enabled=True, guidance_type="dense_curvature", scale=0.1, curvature_limit=0.02, grad_clip_norm=1.0, preserve_endpoint=True, dense_min_speed_mps=0.5)
    before = g.dense_spline_curvature_statistics(physical[:,0], trajectory_spline=spline, start_velocity_xy=vel, curvature_limit=0.02, min_speed_mps=0.5)
    guided, diag = g.apply_curvature_guidance(x0, trajectory_scale=scale, mode_valid_mask=mask, config=cfg, timestep=8, trajectory_spline=spline, start_velocity_xy=vel)
    after = g.dense_spline_curvature_statistics((guided*scale)[:,0], trajectory_spline=spline, start_velocity_xy=vel, curvature_limit=0.02, min_speed_mps=0.5)
    assert diag is not None
    assert float(diag["loss_after"]) < float(diag["loss_before"])
    assert float(after["max_abs_curvature"]) < float(before["max_abs_curvature"])
    print("[PASS] differentiable PyTorch spline gradient")
    print("[PASS] dense_curvature guidance reduces spline curvature")
    print("dense max|kappa|:", float(before["max_abs_curvature"]), "->", float(after["max_abs_curvature"]))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
