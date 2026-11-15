"""Independent paper-aligned nonlinear JTP ablation verification.

Do NOT use JointTrajectoryProjector.success to compute paper feasibility:
the current active-set implementation omits collision and control-rate families.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from highway_env.planner.diffusion.projection.constraints import stability_limits
from highway_env.planner.diffusion.projection.geometry import geometry_residual
from .data import TrafficActor


def validate_evaluation_config(config: dict, jtp_config: dict):
    ev = config.get('limits', {})
    for key in ('collision_clearance_m', 'acceleration_rate_max_mps3', 'steering_rate_max_rad_s'):
        value = ev.get(key)
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f'Paper JTP independent evaluation needs a documented nonnegative limits.{key}; '
                             'do not silently substitute an invented threshold')
    for key in ('stability_rad', 'road_m', 'curvature_per_m2', 'collision_m', 'control_si'):
        v = config.get('tolerances', {}).get(key)
        if v is None or float(v) < 0 or not math.isfinite(float(v)):
            raise ValueError(f'Invalid or missing tolerances.{key}')
    if jtp_config['execution']['horizon_steps'] != jtp_config['execution']['control_steps']:
        raise ValueError('Current diffusion JTP ablation requires K=H with identity interpolation')


def _circle_centers(pose: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.asarray(pose[..., :2])[..., None, :] + np.stack(
        (np.cos(pose[..., 2]), np.sin(pose[..., 2])), axis=-1)[..., None, :] * offsets[None, :, None]


def _actor_circles(actor: TrafficActor):
    radius = max(actor.width_m / 2., 1e-3)
    start = -actor.length_m / 2. + radius
    stop = actor.length_m / 2. - radius
    if stop < start:
        offsets = np.asarray([0.])
    else:
        count = max(2, int(np.ceil((stop - start) / (2. * radius))) + 1)
        offsets = np.linspace(start, stop, count)
    return _circle_centers(actor.pose, offsets), radius


def evaluate_group(controls: torch.Tensor, trajectories: torch.Tensor, states: torch.Tensor,
                   ltr: torch.Tensor, context: dict, roads: Sequence,
                   actors: Sequence[TrafficActor], jtp_cfg: dict, evaluation_cfg: dict) -> list[dict]:
    """Evaluate EVERY paper constraint on actual world-frame K2 rollout.

    Results are one row per controlled vehicle, with inter-truck collision
    propagated to every vehicle involved. Rates include the previous actual
    control from K2 history at the first node.
    """
    if controls.ndim != 3 or controls.shape[-1] != 2:
        raise ValueError('Expected controls [N,H,2]')
    N, H, _ = controls.shape
    if trajectories.shape != (N, H+1, 3) or states.shape != (N, H+1, 5) or ltr.shape != (N, H+1):
        raise ValueError('Inconsistent K2 rollout shapes')
    if len(roads) != N:
        raise ValueError('Expected one valid world road geometry per truck')
    tol = evaluation_cfg['tolerances']; limit = evaluation_cfg['limits']
    cfg_ctl = jtp_cfg['control']
    dt = float(jtp_cfg['execution']['dt_s'])
    u = controls.detach().cpu().numpy().astype(np.float64)
    poses = trajectories.detach().cpu().numpy().astype(np.float64)
    s = states.detach().cpu().numpy().astype(np.float64)
    L = ltr.detach().cpu().numpy().astype(np.float64)
    offsets = np.asarray(jtp_cfg['geometry']['footprint']['circle_offsets_m'], dtype=np.float64)
    radius = float(jtp_cfg['geometry']['footprint']['circle_radius_m'])
    clearance = float(limit['collision_clearance_m'])
    results = []
    for i in range(N):
        payload = context['koopman_payload_kg'][i]
        mu = context['koopman_road_mu'][i]
        with torch.no_grad():
            v = states[i, 1:, 0]
            bounds = stability_limits(v, payload, mu, jtp_cfg).detach().cpu().numpy().astype(np.float64)
            raw, meta = geometry_residual(trajectories[i], roads[i], jtp_cfg)
            geo = raw.detach().cpu().numpy()
        beta = np.arctan2(s[i, 1:, 1], s[i, 1:, 0])
        Q = np.stack((s[i, 1:, 2], beta, s[i, 1:, 3], L[i, 1:]), axis=-1)
        raw_stability = np.abs(Q) - bounds
        violations = raw_stability > float(tol['stability_rad'])
        # LTR is dimensionless and uses a separate numerical tolerance.
        violations[:, 3] = raw_stability[:, 3] > float(tol.get('ltr', tol['stability_rad']))
        stab_bad = bool(np.any(violations))
        road_bad = bool(np.any(geo[:meta['road_count']] > float(tol['road_m'])))
        curve_bad = bool(np.any(geo[meta['road_count']:] > float(tol['curvature_per_m2'])))
        prev = context['koopman_control_hist_si'][i, -1].detach().cpu().numpy()
        du = np.diff(np.concatenate((prev[None], u[i]), axis=0), axis=0) / dt
        accel_bad = bool(np.any(u[i,:,0] < float(cfg_ctl['acceleration_min_mps2'])-tol['control_si']) or
                         np.any(u[i,:,0] > float(cfg_ctl['acceleration_max_mps2'])+tol['control_si']))
        steer_bad = bool(np.any(u[i,:,1] < float(cfg_ctl['steering_min_rad'])-tol['control_si']) or
                         np.any(u[i,:,1] > float(cfg_ctl['steering_max_rad'])+tol['control_si']))
        rate_accel_bad = bool(np.any(np.abs(du[:,0]) > float(limit['acceleration_rate_max_mps3']) + tol['control_si']))
        rate_steer_bad = bool(np.any(np.abs(du[:,1]) > float(limit['steering_rate_max_rad_s']) + tol['control_si']))
        kappa = np.sqrt(np.clip(geo[meta['road_count']:] +
                    float(jtp_cfg['geometry']['curvature_limit_per_m'])**2,0.,None))
        results.append({
            'stability_violation': stab_bad,
            'yaw_violation': bool(np.any(violations[:, 0])),
            'sideslip_violation': bool(np.any(violations[:, 1])),
            'roll_violation': bool(np.any(violations[:, 2])),
            'ltr_violation': bool(np.any(violations[:, 3])),
            'road_violation': road_bad,
            'curvature_violation': curve_bad,
            'geometric_violation': road_bad or curve_bad,
            'acceleration_violation': accel_bad,
            'steering_violation': steer_bad,
            'acceleration_rate_violation': rate_accel_bad,
            'steering_rate_violation': rate_steer_bad,
            'control_violation': accel_bad or steer_bad or rate_accel_bad or rate_steer_bad,
            'external_collision_violation': False,
            'intertruck_collision_violation': False,
            'max_abs_yaw_rate_rad_s': float(np.max(np.abs(Q[:, 0]))),
            'max_abs_beta_rad': float(np.max(np.abs(Q[:, 1]))),
            'max_abs_roll_rad': float(np.max(np.abs(Q[:, 2]))),
            'max_abs_ltr': float(np.max(np.abs(Q[:, 3]))),
            'max_abs_curvature_per_m': float(np.max(kappa)) if len(kappa) else 0.,
            'max_road_residual_m': float(np.max(geo[:meta['road_count']])) if meta['road_count'] else 0.,
        })
    truck_centers = [_circle_centers(poses[i, 1:], offsets) for i in range(N)]
    for i in range(N):
        for j in range(i+1,N):
            d = truck_centers[i][:,:,None,:] - truck_centers[j][:,None,:,:]
            d2 = np.sum(d*d, axis=-1)
            if np.any(d2 < (2*radius + clearance - tol['collision_m'])**2):
                results[i]['intertruck_collision_violation'] = True
                results[j]['intertruck_collision_violation'] = True
    for actor in actors:
        ext_centers, ext_r = _actor_circles(actor)
        ext_centers = ext_centers[1:]
        for i in range(N):
            d = truck_centers[i][:,:,None,:] - ext_centers[:,None,:,:]
            if np.any(np.sum(d*d,axis=-1) < (radius+ext_r+clearance-tol['collision_m'])**2):
                results[i]['external_collision_violation'] = True
    for row in results:
        row['collision_violation'] = (row['external_collision_violation'] or row['intertruck_collision_violation'])
        row['joint_feasible'] = not (row['stability_violation'] or row['geometric_violation'] or
                                    row['collision_violation'] or row['control_violation'])
    return results


def trajectory_deviation_m(nominal_world: np.ndarray, projected_world: np.ndarray) -> np.ndarray:
    """Per-vehicle mean over t=1..H of XY Euclidean distance, in metres."""
    delta = np.asarray(projected_world)[:,1:,:2] - np.asarray(nominal_world)[:,1:,:2]
    return np.linalg.norm(delta, axis=-1).mean(axis=-1)
