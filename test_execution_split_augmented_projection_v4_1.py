from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent

def load(name, rel):
    spec=importlib.util.spec_from_file_location(name, ROOT/rel)
    if spec is None or spec.loader is None:
        raise RuntimeError(rel)
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m

g=load('g_split_v41_smoke','highway_env/planner/diffusion/guidance.py')
s=load('s_split_v41_smoke','highway_env/planner/diffusion/trajectory_spline.py')
Spline=s.ClampedCubicTrajectorySpline

def run(mode):
    spline=Spline(horizon_s=4.0,sparse_dt=0.5,dense_dt=0.1)
    physical=torch.tensor([[[[8.,-2.],[16.,-7.],[24.,-8.],[32.,-6.5],[40.,-8.],[48.,-6.],[56.,-7.5],[72.,-6.5]]]],dtype=torch.float32)
    scale=torch.tensor([120.,24.],dtype=torch.float32)
    cfg=g.SparseCurvatureGuidanceConfig(
        enabled=True,guidance_type='dense_curvature',curvature_limit=0.02,
        dense_step_m=0.20,dense_min_segment_m=0.25,dense_inner_iters=1,
        dense_backtracking_enabled=True,dense_backtracking_factor=0.5,
        dense_backtracking_max_trials=4,dense_min_step_m=0.025,
        dense_max_total_move_m=0.50,dense_bound_weight=1.0,
        dense_peak_weight=0.5,dense_dkappa_weight=0.1,dense_peak_alpha=10.0,
        dense_residual_gradient_enabled=False,dense_normal_projection_enabled=True,
        dense_polish_enabled=False,dense_post_recovery_enabled=False,
        dense_peak_recovery_enabled=False,dense_constraint_recovery_mode=mode,
        dense_constraint_recovery_threshold=0.02,
        dense_constraint_recovery_max_total_move_m=0.50,
        dense_projection_iters=8,dense_projection_max_step_m=0.15,
        dense_split_outer_iters=4,dense_split_primal_iters=4,
        dense_split_primal_step_m=0.08,dense_split_rho=5.0,
        dense_split_rho_max=40.0,dense_split_rho_growth=2.0,
        dense_split_proximity_weight=0.05,dense_split_dual_clip_m=0.50,
        preserve_endpoint=True,
    )
    _,diag=g.apply_dense_curvature_guidance(
        physical/scale,trajectory_scale=scale,
        mode_valid_mask=torch.ones((1,1),dtype=torch.bool),
        config=cfg,timestep=8,trajectory_spline=spline,
        start_velocity_xy=torch.tensor([[16.,0.]]),
        allow_polish=True,allow_adaptive_normal=False,
    )
    assert diag is not None
    return float(diag['max_abs_curvature_after'].item()), diag

def main():
    b0,db0=run('none')
    split,ds=run('split_projection')
    assert split < b0, (b0, split)
    assert float(ds['split_outer_iterations_used'].item()) > 0.0
    assert float(ds['split_projection_iterations_used'].item()) > 0.0
    assert float(ds['split_primal_iterations_used'].item()) > 0.0
    assert float(ds['split_update_l2_m'].item()) <= 0.5001
    print('[PASS] split path reduces execution-domain max curvature')
    print('[PASS] projection and augmented planning-consistency stages both executed')
    print('[PASS] shared planner trust budget <= 0.50 m')
    print('synthetic max|kappa| B0 / split:', b0, split)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
