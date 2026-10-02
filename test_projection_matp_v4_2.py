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

g=load('g_matp_v42_smoke','highway_env/planner/diffusion/guidance.py')
s=load('s_matp_v42_smoke','highway_env/planner/diffusion/trajectory_spline.py')
Spline=s.ClampedCubicTrajectorySpline

def run(*, tightening=0.0, adaptive=False, filt=False, archive=False):
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
        dense_peak_recovery_enabled=False,dense_constraint_recovery_mode='matp',
        dense_constraint_recovery_threshold=0.02,
        dense_constraint_recovery_max_total_move_m=0.50,
        dense_matp_iters=6,dense_matp_max_active_constraints=8,
        dense_matp_active_threshold=0.015,dense_matp_regularization=1e-4,
        dense_matp_max_step_m=0.15,dense_matp_tightening=tightening,
        dense_matp_safety_factor=1.05,dense_matp_adaptive_trust_enabled=adaptive,
        dense_matp_initial_trust_m=0.30,dense_matp_trust_level2_m=0.50,
        dense_matp_trust_level3_m=0.75,dense_matp_max_trust_m=1.00,
        dense_matp_filter_acceptance_enabled=filt,dense_matp_archive_enabled=archive,
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
    p1,d1=run()
    p2,d2=run(tightening=0.0005)
    p3,d3=run(tightening=0.0005,adaptive=True,filt=True,archive=True)
    assert float(d1['matp_iterations_used'].item()) > 0.0
    assert float(d1['matp_active_constraints_sum'].item()) > 1.0
    assert p1 < 0.15 and p2 < 0.15 and p3 < 0.15
    assert float(d1['constraint_recovery_update_l2_m'].item()) <= 0.5001
    assert float(d2['constraint_recovery_update_l2_m'].item()) <= 0.5001
    assert float(d3['constraint_recovery_update_l2_m'].item()) <= 1.0001
    print('[PASS] P1 executes multi-active joint projection')
    print('[PASS] P2 executes tightened multi-active projection')
    print('[PASS] P3 executes adaptive-trust/filter/archive MATP')
    print('synthetic max|kappa| P1/P2/P3:', p1,p2,p3)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
