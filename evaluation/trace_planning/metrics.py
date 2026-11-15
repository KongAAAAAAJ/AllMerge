"""Physical-unit metrics for TRACE. No outcome proxies are labelled closed-loop.

Closed-loop stability is evaluated using the *executed* TruckSim state. Open-loop
stability uses frozen Koopman K2 predictions and has a separate metric name.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import torch


def stability_flags(states: torch.Tensor, ltr: torch.Tensor,
                    payload: torch.Tensor, mu: torch.Tensor,
                    jtp: dict) -> dict[str, torch.Tensor]:
    """Compare [vx,vy,r,roll,roll_rate] to the exact JTP four limits.

    Returns per-vehicle per-timestep boolean tensors with dimensions [B,H].
    The initial state t=0 is deliberately excluded, as in JTP stability_residual.
    """
    from highway_env.planner.diffusion.projection.constraints import stability_limits
    if states.ndim != 3 or states.shape[-1] != 5 or ltr.shape != states.shape[:2]:
        raise ValueError('states must be [B,H+1,5], LTR must be [B,H+1]')
    flags = {k: [] for k in ('yaw_rate', 'sideslip', 'roll', 'ltr')}
    for idx in range(len(states)):
        x = states[idx, 1:]
        bounds = stability_limits(x[:, 0], payload[idx], mu[idx], jtp)
        components = torch.stack((x[:, 2], torch.atan2(x[:, 1], x[:, 0]),
                                  x[:, 3], ltr[idx, 1:]), dim=-1)
        failed = torch.abs(components) > bounds
        for col, name in enumerate(flags):
            flags[name].append(failed[:, col])
    result = {name: torch.stack(v, dim=0) for name, v in flags.items()}
    result['any'] = torch.stack(list(result.values()), dim=-1).any(dim=-1)
    return result


def score_open_loop(pred: dict[str, torch.Tensor], truth: torch.Tensor,
                    features: dict[str, torch.Tensor],
                    context: dict[str, torch.Tensor],
                    *, dt: float, fuel_parameters: Any, stability_config: dict,
                    require_k2: bool = True) -> dict[str, torch.Tensor]:
    """One score per expert sample; candidate minima use all 10 selected anchors.

    Note: minADE and minFDE are computed independently across candidates.
    No oracle trajectory is used to pick the deployment mode.
    """
    b, h, two = truth.shape
    if (h, two) != (40, 2):
        raise ValueError('TRACE evaluation requires expert dense XY [B,40,2]')
    xy = pred['trajectory_dense']
    candidates = pred['trajectory_candidates']
    if xy.shape != truth.shape or candidates.shape[0] != b or candidates.shape[-2:] != (8,2):
        raise ValueError('Incorrect selected/candidate TRACE trajectory dimensions')
    err = torch.linalg.vector_norm(xy - truth, dim=-1)
    sparse_gt = truth[:, 4::5]
    sparse_err = torch.linalg.vector_norm(pred['trajectory'] - sparse_gt, dim=-1)
    candidates_err = torch.linalg.vector_norm(candidates - sparse_gt[:, None], dim=-1)
    valid = pred.get('trajectory_candidate_valid_mask', features['mode_valid_mask']).bool()
    if valid.shape != candidates_err.shape[:2] or not bool(valid.any(dim=1).all()):
        raise ValueError('Invalid candidate mask or no valid modes')
    candidate_ade = candidates_err.mean(dim=-1).masked_fill(~valid, float('inf'))
    candidate_fde = candidates_err[..., -1].masked_fill(~valid, float('inf'))
    res = {
        'ade_m': err.mean(dim=-1), 'fde_m': err[:, -1],
        'sparse_ade_m': sparse_err.mean(dim=-1),
        'sparse_fde_m': sparse_err[:, -1],
        'minade8_m': candidate_ade.min(dim=-1).values,
        'minfde8_m': candidate_fde.min(dim=-1).values,
        'goal_error_m': torch.linalg.vector_norm(
            pred['pose'][:, -1, :2] - features['target_point'], dim=-1),
    }
    poses = pred['pose']
    distance = torch.linalg.vector_norm(poses[:, 1:, :2] - poses[:, :-1, :2], dim=-1).sum(dim=-1)
    res['distance_m'] = distance
    res['selected_mode'] = pred['trajectory_mode_idx']
    if require_k2 and not all(k in pred for k in ('states', 'ltr')):
        raise RuntimeError('Paper stability requires frozen Koopman K2; kinematic fallback is not valid')
    if all(k in pred for k in ('states', 'ltr')):
        p = context['koopman_payload_kg'].reshape(-1)
        mu = context['koopman_road_mu'].reshape(-1)
        flags = stability_flags(pred['states'], pred['ltr'], p, mu, stability_config)
        res['k2_stability_violated'] = flags['any'].any(dim=-1).float()
        res['k2_stability_step_fraction'] = flags['any'].float().mean(dim=-1)
        for name in ('yaw_rate', 'sideslip', 'roll', 'ltr'):
            res[f'k2_{name}_step_fraction'] = flags[name].float().mean(dim=-1)
        from highway_env.planner.diffusion.common.fuel import vt_cpfm_from_states
        fuel = vt_cpfm_from_states(pred['speeds'], p, fuel_parameters, dt=dt)
        res['fuel_l_4s'] = fuel['fuel_total']
        res['fuel_l_per_100km'] = torch.where(
            distance > 1.0, 100000.0 * fuel['fuel_total'] / distance.clamp_min(1e-6),
            torch.full_like(distance, float('nan')))
    for name, v in res.items():
        if name == 'fuel_l_per_100km':
            if torch.isinf(v).any():
                raise FloatingPointError(f'Infinite metric: {name}')
        elif not bool(torch.isfinite(v).all()):
            raise FloatingPointError(f'Non-finite metric: {name}')
    return res


def center_spacing_error(positions: np.ndarray, headings: np.ndarray,
                         target_m: float) -> np.ndarray:
    """Per timestep/pair center-to-center longitudinal spacing error [T,N-1].

    Project onto the preceding vehicle's heading; this is a transparent local
    approximation to curvilinear distance, suitable for shared straight segments.
    """
    p = np.asarray(positions, dtype=float)
    hd = np.asarray(headings, dtype=float)
    if p.ndim != 3 or p.shape[-1] != 2 or hd.shape != p.shape[:2]:
        raise ValueError('Positions [T,N,2], heading [T,N] required')
    tangent = np.stack((np.cos(hd[:, :-1]), np.sin(hd[:, :-1])), axis=-1)
    return np.abs(np.sum((p[:, :-1] - p[:, 1:]) * tangent, axis=-1) - target_m)


def summarize_open_loop_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('No open-loop evaluation rows')
    metrics = ('ade_m','fde_m','sparse_ade_m','sparse_fde_m','minade8_m','minfde8_m',
               'goal_error_m','fuel_l_4s','k2_stability_violated',
               'k2_stability_step_fraction','k2_yaw_rate_step_fraction',
               'k2_sideslip_step_fraction','k2_roll_step_fraction',
               'k2_ltr_step_fraction')
    out = {'num_samples': len(rows)}
    for key in metrics:
        values = [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]
        if values:
            out[key] = float(np.mean(values))
    if 'k2_stability_violated' in out:
        out['k2_stability_horizon_rate_pct'] = 100.0 * out['k2_stability_violated']
        out['k2_stability_step_rate_pct'] = 100.0 * out['k2_stability_step_fraction']
    valid = [r for r in rows if 'fuel_l_4s' in r and r['distance_m'] > 1.0]
    if valid:
        out['fuel_l_per_100km'] = 100000.0 * sum(float(r['fuel_l_4s']) for r in valid) / sum(float(r['distance_m']) for r in valid)
        out['mean_fuel_l_4s'] = float(np.mean([r['fuel_l_4s'] for r in valid]))
    return out


def summarize_closed_loop_rows(rows: list[dict]) -> dict:
    """Episode-level Success/Collision, vehicle-step-level stability and spacing."""
    if not rows:
        raise ValueError('No closed-loop episodes')
    failures = [r for r in rows if r.get('status') != 'complete']
    if failures:
        raise RuntimeError(f'{len(failures)} episodes failed; cannot publish a paper aggregate')
    total_stability = sum(int(r['stability_steps']) for r in rows)
    total_obs = sum(int(r['vehicle_steps']) for r in rows)
    total_gap_abs = sum(float(r['spacing_error_sum_m']) for r in rows)
    total_gap_n = sum(int(r['spacing_observations']) for r in rows)
    total_fuel = sum(float(r['fuel_l']) for r in rows)
    total_distance = sum(float(r['distance_m']) for r in rows)
    times = [float(r['planning_time_ms']) for r in rows]
    total_time_ms = sum(float(r.get('planning_time_sum_ms',r['planning_time_ms'])) for r in rows)
    total_time_count = sum(int(r.get('planning_time_count',1)) for r in rows)
    if total_obs <= 0 or total_gap_n <= 0 or total_distance <= 1.0 or not np.isfinite(times).all():
        raise ValueError('Insufficient real executed samples for six paper metrics')
    return {
        'episodes':len(rows),
        'success_pct':100.*sum(bool(r['success']) for r in rows)/len(rows),
        'collision_pct':100.*sum(bool(r['collision']) for r in rows)/len(rows),
        'stability_violation_pct':100.*total_stability/total_obs,
        'spacing_error_m':total_gap_abs/total_gap_n,
        'fuel_l_per_100km':100000.*total_fuel/total_distance,
        'planning_time_ms':total_time_ms/max(1,total_time_count),
        'jtp_execution_fraction':float(sum(int(r.get('jtp_executed_steps',0)) for r in rows)/
                                       max(1,sum(int(r.get('planning_steps',0)) for r in rows))),
    }
