"""Independent full-horizon evaluation of JTP projection outputs.

The paper-level joint-feasibility metric is computed from the complete external
constraint set. The projector's internal constraint flag is diagnostic only.
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
            raise ValueError(
                f'JTP independent evaluation needs a documented nonnegative limits.{key}; '
                'do not silently substitute an invented threshold')
    for key in ('stability_rad', 'road_m', 'curvature_per_m2', 'collision_m', 'control_si'):
        value = config.get('tolerances', {}).get(key)
        if value is None or float(value) < 0 or not math.isfinite(float(value)):
            raise ValueError(f'Invalid or missing tolerances.{key}')
    shared_limits = {
        'collision_clearance_m': float(jtp_config['collision']['clearance_m']),
        'acceleration_rate_max_mps3': float(jtp_config['control']['acceleration_rate_max_mps3']),
        'steering_rate_max_rad_s': float(jtp_config['control']['steering_rate_max_rad_s']),
    }
    for key, expected in shared_limits.items():
        if not math.isclose(float(ev[key]), expected, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f'Independent evaluation limits.{key}={ev[key]} must match the '
                f'JTP projection value {expected}; keep projection and evaluation constraints identical')
    if not math.isclose(float(config.get('tolerances', {}).get('control_si', -1)),
                        float(jtp_config['control']['rate_tolerance_si']),
                        rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError('tolerances.control_si must match JTP control.rate_tolerance_si')
    runtime_collision = jtp_config.get('runtime', {}).get('collision', {})
    for key in ('max_external_actors', 'max_external_distance_m'):
        if key not in config or key not in runtime_collision or not math.isclose(
                float(config[key]), float(runtime_collision[key]), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(f'{key} must match runtime.collision.{key} so online/offline actor sets use the same limits')
    if jtp_config['execution']['horizon_steps'] != jtp_config['execution']['control_steps']:
        raise ValueError('Current diffusion JTP evaluation requires K=H with identity interpolation')
    if config.get('invalid_map_policy', 'error') != 'error':
        raise ValueError('JTP evaluation does not skip invalid-map frames; use invalid_map_policy: error')


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


def _normalized_positive_excess(residual, tolerance, scale):
    residual = np.asarray(residual, dtype=np.float64)
    if residual.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.maximum(residual - float(tolerance), 0.0) / max(float(scale), 1e-8)


def evaluate_group(controls: torch.Tensor, trajectories: torch.Tensor, states: torch.Tensor,
                   ltr: torch.Tensor, context: dict, roads: Sequence,
                   actors: Sequence[TrafficActor], jtp_cfg: dict, evaluation_cfg: dict) -> list[dict]:
    """Evaluate every full-horizon constraint on the actual world-frame K2 rollout.

    Results are one row per controlled vehicle. Inter-truck collision violations
    are propagated to both vehicles involved. Control-rate checks include the
    previous observed control at the first prediction node.
    """
    if controls.ndim != 3 or controls.shape[-1] != 2:
        raise ValueError('Expected controls [N,H,2]')
    N, H, _ = controls.shape
    if trajectories.shape != (N, H + 1, 3) or states.shape != (N, H + 1, 5) or ltr.shape != (N, H + 1):
        raise ValueError('Inconsistent K2 rollout shapes')
    if len(roads) != N:
        raise ValueError('Expected one valid world road geometry per truck')

    tol = evaluation_cfg['tolerances']
    limit = evaluation_cfg['limits']
    cfg_ctl = jtp_cfg['control']
    dt = float(jtp_cfg['execution']['dt_s'])
    u = controls.detach().cpu().numpy().astype(np.float64)
    poses = trajectories.detach().cpu().numpy().astype(np.float64)
    s = states.detach().cpu().numpy().astype(np.float64)
    L = ltr.detach().cpu().numpy().astype(np.float64)
    offsets = np.asarray(jtp_cfg['geometry']['footprint']['circle_offsets_m'], dtype=np.float64)
    radius = float(jtp_cfg['geometry']['footprint']['circle_radius_m'])
    clearance = float(limit['collision_clearance_m'])
    collision_tol = float(tol['collision_m'])
    stability_tol = np.asarray([
        float(tol['stability_rad']), float(tol['stability_rad']),
        float(tol['stability_rad']), float(tol.get('ltr', tol['stability_rad']))
    ], dtype=np.float64)

    results = []
    for i in range(N):
        payload = context['koopman_payload_kg'][i]
        mu = context['koopman_road_mu'][i]
        with torch.no_grad():
            vx = states[i, 1:, 0]
            bounds = stability_limits(vx, payload, mu, jtp_cfg).detach().cpu().numpy().astype(np.float64)
            raw_geometry, geometry_meta = geometry_residual(trajectories[i], roads[i], jtp_cfg)
            geometry = raw_geometry.detach().cpu().numpy().astype(np.float64)

        beta = np.arctan2(s[i, 1:, 1], s[i, 1:, 0])
        q = np.stack((s[i, 1:, 2], beta, s[i, 1:, 3], L[i, 1:]), axis=-1)
        raw_stability = np.abs(q) - bounds
        stability_excess = np.maximum(raw_stability - stability_tol[None, :], 0.0) / np.maximum(bounds, 1e-8)
        stability_bad = bool(np.any(raw_stability > stability_tol[None, :]))

        road_count = int(geometry_meta['road_count'])
        road_excess = _normalized_positive_excess(
            geometry[:road_count], tol['road_m'], jtp_cfg['geometry']['road_clearance_m'])
        curvature_limit = float(jtp_cfg['geometry']['curvature_limit_per_m'])
        curvature_excess = _normalized_positive_excess(
            geometry[road_count:], tol['curvature_per_m2'], curvature_limit ** 2)
        geometry_excess = np.concatenate((road_excess, curvature_excess))
        road_bad = bool(np.any(geometry[:road_count] > float(tol['road_m'])))
        curvature_bad = bool(np.any(geometry[road_count:] > float(tol['curvature_per_m2'])))
        geometric_bad = road_bad or curvature_bad

        previous_control = context['koopman_control_hist_si'][i, -1].detach().cpu().numpy()
        control_rate = np.diff(np.concatenate((previous_control[None], u[i]), axis=0), axis=0) / dt

        acceleration_min = float(cfg_ctl['acceleration_min_mps2'])
        acceleration_max = float(cfg_ctl['acceleration_max_mps2'])
        steering_min = float(cfg_ctl['steering_min_rad'])
        steering_max = float(cfg_ctl['steering_max_rad'])
        acceleration_scale = max(acceleration_max - acceleration_min, 1e-8)
        steering_scale = max(steering_max - steering_min, 1e-8)

        acceleration_excess = np.concatenate((
            np.maximum(acceleration_min - u[i, :, 0] - float(tol['control_si']), 0.0) / acceleration_scale,
            np.maximum(u[i, :, 0] - acceleration_max - float(tol['control_si']), 0.0) / acceleration_scale,
        ))
        steering_excess = np.concatenate((
            np.maximum(steering_min - u[i, :, 1] - float(tol['control_si']), 0.0) / steering_scale,
            np.maximum(u[i, :, 1] - steering_max - float(tol['control_si']), 0.0) / steering_scale,
        ))
        control_magnitude_excess = np.concatenate((acceleration_excess, steering_excess))

        acceleration_rate_limit = float(limit['acceleration_rate_max_mps3'])
        steering_rate_limit = float(limit['steering_rate_max_rad_s'])
        acceleration_rate_excess = (
            np.maximum(np.abs(control_rate[:, 0]) - acceleration_rate_limit - float(tol['control_si']), 0.0)
            / max(acceleration_rate_limit, 1e-8))
        steering_rate_excess = (
            np.maximum(np.abs(control_rate[:, 1]) - steering_rate_limit - float(tol['control_si']), 0.0)
            / max(steering_rate_limit, 1e-8))
        control_rate_excess = np.concatenate((acceleration_rate_excess, steering_rate_excess))

        acceleration_bad = bool(np.any(u[i, :, 0] < acceleration_min - tol['control_si']) or
                                np.any(u[i, :, 0] > acceleration_max + tol['control_si']))
        steering_bad = bool(np.any(u[i, :, 1] < steering_min - tol['control_si']) or
                            np.any(u[i, :, 1] > steering_max + tol['control_si']))
        acceleration_rate_bad = bool(np.any(np.abs(control_rate[:, 0]) >
                                            acceleration_rate_limit + tol['control_si']))
        steering_rate_bad = bool(np.any(np.abs(control_rate[:, 1]) >
                                        steering_rate_limit + tol['control_si']))

        curvature = np.sqrt(np.clip(
            geometry[road_count:] + curvature_limit ** 2, 0.0, None))
        results.append({
            'stability_violation': stability_bad,
            'yaw_violation': bool(np.any(raw_stability[:, 0] > stability_tol[0])),
            'sideslip_violation': bool(np.any(raw_stability[:, 1] > stability_tol[1])),
            'roll_violation': bool(np.any(raw_stability[:, 2] > stability_tol[2])),
            'ltr_violation': bool(np.any(raw_stability[:, 3] > stability_tol[3])),
            'road_violation': road_bad,
            'curvature_violation': curvature_bad,
            'geometric_violation': geometric_bad,
            'acceleration_violation': acceleration_bad,
            'steering_violation': steering_bad,
            'acceleration_rate_violation': acceleration_rate_bad,
            'steering_rate_violation': steering_rate_bad,
            'control_magnitude_violation': acceleration_bad or steering_bad,
            'control_rate_violation': acceleration_rate_bad or steering_rate_bad,
            'control_violation': acceleration_bad or steering_bad or acceleration_rate_bad or steering_rate_bad,
            'external_collision_violation': False,
            'intertruck_collision_violation': False,
            'stability_excess_mean_norm': float(stability_excess.mean()) if stability_excess.size else 0.0,
            'stability_excess_max_norm': float(stability_excess.max()) if stability_excess.size else 0.0,
            'road_excess_mean_norm': float(road_excess.mean()) if road_excess.size else 0.0,
            'curvature_excess_mean_norm': float(curvature_excess.mean()) if curvature_excess.size else 0.0,
            'geometry_excess_mean_norm': float(geometry_excess.mean()) if geometry_excess.size else 0.0,
            'geometry_excess_max_norm': float(geometry_excess.max()) if geometry_excess.size else 0.0,
            'control_magnitude_excess_mean_norm': float(control_magnitude_excess.mean()) if control_magnitude_excess.size else 0.0,
            'control_magnitude_excess_max_norm': float(control_magnitude_excess.max()) if control_magnitude_excess.size else 0.0,
            'control_rate_excess_mean_norm': float(control_rate_excess.mean()) if control_rate_excess.size else 0.0,
            'control_rate_excess_max_norm': float(control_rate_excess.max()) if control_rate_excess.size else 0.0,
            'max_abs_yaw_rate_rad_s': float(np.max(np.abs(q[:, 0]))),
            'max_abs_beta_rad': float(np.max(np.abs(q[:, 1]))),
            'max_abs_roll_rad': float(np.max(np.abs(q[:, 2]))),
            'max_abs_ltr': float(np.max(np.abs(q[:, 3]))),
            'max_abs_curvature_per_m': float(np.max(curvature)) if len(curvature) else 0.0,
            'max_road_residual_m': float(np.max(geometry[:road_count])) if road_count else 0.0,
        })

    truck_centers = [_circle_centers(poses[i, 1:], offsets) for i in range(N)]
    collision_samples = [[] for _ in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            delta = truck_centers[i][:, :, None, :] - truck_centers[j][:, None, :, :]
            distances = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 1e-12))
            penetration = np.maximum(2.0 * radius + clearance - distances - collision_tol, 0.0)
            collided = bool(np.any(penetration > 0.0))
            if collided:
                results[i]['intertruck_collision_violation'] = True
                results[j]['intertruck_collision_violation'] = True
            collision_samples[i].extend(penetration.reshape(-1).tolist())
            collision_samples[j].extend(penetration.reshape(-1).tolist())

    for actor in actors:
        external_centers, external_radius = _actor_circles(actor)
        external_centers = external_centers[1:]
        for i in range(N):
            delta = truck_centers[i][:, :, None, :] - external_centers[:, None, :, :]
            distances = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 1e-12))
            penetration = np.maximum(radius + external_radius + clearance - distances - collision_tol, 0.0)
            if np.any(penetration > 0.0):
                results[i]['external_collision_violation'] = True
            collision_samples[i].extend(penetration.reshape(-1).tolist())

    for i, row in enumerate(results):
        row['collision_violation'] = (
            row['external_collision_violation'] or row['intertruck_collision_violation'])
        row['collision_penetration_mean_m'] = (
            float(np.mean(collision_samples[i])) if collision_samples[i] else 0.0)
        row['collision_penetration_max_m'] = (
            float(np.max(collision_samples[i])) if collision_samples[i] else 0.0)
        row['joint_feasible'] = not (
            row['stability_violation'] or row['geometric_violation'] or
            row['collision_violation'] or row['control_violation'])
    return results


def trajectory_deviation_m(nominal_world: np.ndarray, projected_world: np.ndarray) -> np.ndarray:
    """Per-vehicle mean over t=1..H of XY Euclidean deviation, in metres."""
    delta = np.asarray(projected_world)[:, 1:, :2] - np.asarray(nominal_world)[:, 1:, :2]
    return np.linalg.norm(delta, axis=-1).mean(axis=-1)
