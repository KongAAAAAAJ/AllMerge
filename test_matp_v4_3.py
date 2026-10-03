from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent

def load(name, rel):
    spec=importlib.util.spec_from_file_location(name, ROOT/rel)
    if spec is None or spec.loader is None: raise RuntimeError(rel)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

g=load('g_matp_v43_smoke','highway_env/planner/diffusion/guidance.py')
s=load('s_matp_v43_smoke','highway_env/planner/diffusion/trajectory_spline.py')
Spline=s.ClampedCubicTrajectorySpline

def run(*, weighting=False, residual=False, tightening=0.0005, active=0.015, beta=2.0, stall_peak=2e-4, stall_frac=0.03, max_trust=1.0):
    spline=Spline(horizon_s=4.0,sparse_dt=0.5,dense_dt=0.1)
    physical=torch.tensor([[[[8.,-2.],[16.,-7.],[24.,-8.],[32.,-6.5],[40.,-8.],[48.,-6.],[56.,-7.5],[72.,-6.5]]]],dtype=torch.float32)
    scale=torch.tensor([120.,24.],dtype=torch.float32)
    cfg=g.SparseCurvatureGuidanceConfig(
        enabled=True,guidance_type='dense_curvature',curvature_limit=0.02,
        dense_step_m=0.20,dense_min_segment_m=0.25,dense_inner_iters=1,
        dense_backtracking_enabled=True,dense_backtracking_factor=0.5,dense_backtracking_max_trials=4,
        dense_min_step_m=0.025,dense_max_total_move_m=0.50,dense_bound_weight=1.0,dense_peak_weight=0.5,
        dense_dkappa_weight=0.1,dense_peak_alpha=10.0,dense_residual_gradient_enabled=False,
        dense_normal_projection_enabled=True,dense_polish_enabled=False,dense_post_recovery_enabled=False,
        dense_peak_recovery_enabled=False,dense_constraint_recovery_mode='matp',
        dense_constraint_recovery_threshold=0.02,dense_constraint_recovery_max_total_move_m=0.50,
        dense_matp_iters=6,dense_matp_max_active_constraints=8,dense_matp_active_threshold=active,
        dense_matp_regularization=1e-4,dense_matp_max_step_m=0.15,dense_matp_tightening=tightening,
        dense_matp_safety_factor=1.05,dense_matp_adaptive_trust_enabled=True,
        dense_matp_initial_trust_m=0.30,dense_matp_trust_level2_m=0.50,dense_matp_trust_level3_m=0.75,
        dense_matp_max_trust_m=max_trust,dense_matp_filter_acceptance_enabled=True,dense_matp_archive_enabled=True,
        dense_matp_violation_weighting_enabled=weighting,dense_matp_violation_weight_beta=beta,
        dense_matp_residual_trust_enabled=residual,dense_matp_stall_peak_improvement=stall_peak,
        dense_matp_stall_violation_fraction=stall_frac,preserve_endpoint=True,
    )
    _,diag=g.apply_dense_curvature_guidance(
        physical/scale,trajectory_scale=scale,mode_valid_mask=torch.ones((1,1),dtype=torch.bool),
        config=cfg,timestep=8,trajectory_spline=spline,start_velocity_xy=torch.tensor([[16.,0.]]),
        allow_polish=True,allow_adaptive_normal=False,
    )
    return float(diag['max_abs_curvature_after'].item()),diag

def main():
    ref,d0=run()
    weighted,d1=run(weighting=True)
    # Force a permissive stall criterion here only to ensure the residual-triggered
    # trust branch is exercised in smoke testing.
    residual,d2=run(residual=True,stall_peak=1.0,stall_frac=1.0,max_trust=1.25)
    full,d3=run(weighting=True,residual=True,stall_peak=1.0,stall_frac=1.0,max_trust=1.25)
    assert float(d1['matp_weight_mean'].item()) > 1.0
    assert float(d2['matp_stall_triggers'].item()) > 0.0
    assert all(v < 0.15 for v in [ref,weighted,residual,full])
    print('[PASS] V4.2 reference MATP remains executable')
    print('[PASS] V4.3 violation-weighted joint projection executes')
    print('[PASS] V4.3 residual-triggered trust expansion executes')
    print('synthetic max|kappa| ref/weighted/residual/full:',ref,weighted,residual,full)
    return 0
if __name__=='__main__': raise SystemExit(main())
