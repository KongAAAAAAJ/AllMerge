"""Call the ORIGINAL MATP V4.3 W1 from the sibling all_merge-guidance worktree.

No inexact fallback and no reward import from that worktree. This wrapper
copies the frozen W1 configuration used in MATP_V4_3_Optimization.py smoke.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import torch


def load_matp(guidance_root:Path):
    source=Path(guidance_root)/'highway_env/planner/diffusion/guidance.py'
    if not source.is_file():
        raise FileNotFoundError(f'Frozen W1 source not found: {source}. Supply --guidance-root all_merge-guidance; do not substitute another projector.')
    code=source.read_text(encoding='utf-8')
    if 'MATP_PARAMETER_MECHANISM_OPTIMIZATION_V4_3' not in code or 'MULTI_ACTIVE_TRUST_PROJECTION_V4_2' not in code:
        raise RuntimeError(f'Guidance module lacks frozen MATP V4.3 W1 markers: {source}')
    spec=importlib.util.spec_from_file_location('allmerge_matp_v43_w1_readonly',source)
    if spec is None or spec.loader is None:raise ImportError('Cannot import MATP source '+str(source))
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    if not hasattr(module,'apply_dense_curvature_guidance') or not hasattr(module,'SparseCurvatureGuidanceConfig'):
        raise RuntimeError('Missing original guidance API in '+str(source))
    return module


def project_w1(xy:torch.Tensor, *,module, reward_config):
    """Apply frozen MATP-W1 guidance to [N,8,2] metric trajectories."""
    from highway_env.planner.diffusion.trajectory_spline import ClampedCubicTrajectorySpline
    cfg=module.SparseCurvatureGuidanceConfig(
        enabled=True,guidance_type='dense_curvature',curvature_limit=0.02,
        dense_step_m=0.20,dense_min_segment_m=0.25,dense_inner_iters=1,
        dense_backtracking_enabled=True,dense_backtracking_factor=0.5,dense_backtracking_max_trials=4,
        dense_min_step_m=0.025,dense_max_total_move_m=0.50,dense_bound_weight=1.0,dense_peak_weight=0.5,
        dense_dkappa_weight=0.1,dense_peak_alpha=10.0,dense_residual_gradient_enabled=False,
        dense_normal_projection_enabled=True,dense_polish_enabled=False,dense_post_recovery_enabled=False,
        dense_peak_recovery_enabled=False,dense_constraint_recovery_mode='matp',
        dense_constraint_recovery_threshold=0.02,dense_constraint_recovery_max_total_move_m=0.50,
        dense_matp_iters=6,dense_matp_max_active_constraints=8,dense_matp_active_threshold=0.015,
        dense_matp_regularization=1e-4,dense_matp_max_step_m=0.15,dense_matp_tightening=0.0005,
        dense_matp_safety_factor=1.05,dense_matp_adaptive_trust_enabled=True,
        dense_matp_initial_trust_m=0.30,dense_matp_trust_level2_m=0.50,dense_matp_trust_level3_m=0.75,
        dense_matp_max_trust_m=1.0,dense_matp_filter_acceptance_enabled=True,dense_matp_archive_enabled=True,
        dense_matp_violation_weighting_enabled=True,dense_matp_violation_weight_beta=2.0,
        dense_matp_residual_trust_enabled=False,preserve_endpoint=True,
    )
    spline=ClampedCubicTrajectorySpline(
        horizon_s=8*float(reward_config.trajectory_dt_s),
        sparse_dt=float(reward_config.trajectory_dt_s),
        dense_dt=float(reward_config.interpolation_dt_s)).to(xy.device)
    scale=torch.tensor([120.,24.],device=xy.device,dtype=xy.dtype)
    v0=torch.stack((xy[:,0,0].clamp_min(0.)/float(reward_config.trajectory_dt_s),
                    torch.zeros_like(xy[:,0,0])),dim=-1)
    # Treat N independent samples as a batch of one mode each.
    with torch.enable_grad():
        result,diag=module.apply_dense_curvature_guidance(
            xy[:,None,:,:]/scale,trajectory_scale=scale,
            mode_valid_mask=torch.ones(xy.shape[0],1,device=xy.device,dtype=torch.bool),
            config=cfg,timestep=8,trajectory_spline=spline,start_velocity_xy=v0,
            allow_polish=True,allow_adaptive_normal=False)
    physical=(result[:,0,:,:]*scale).detach()
    if physical.shape!=xy.shape or not torch.isfinite(physical).all():
        raise RuntimeError('MATP output invalid')
    return physical,{k:v.detach() if torch.is_tensor(v) else v for k,v in diag.items()}
