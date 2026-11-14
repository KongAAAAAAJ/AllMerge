"""Per-trajectory physical fuel, progress and imitation metrics."""
from __future__ import annotations
import torch
from highway_env.planner.diffusion.common.fuel import FuelParameters, vt_cpfm_from_states
from highway_env.planner.diffusion.common.progress import centerline_progress

def sample_metrics(prediction: dict, truth_xy: torch.Tensor, features: dict,
                   payload_kg: torch.Tensor, fuel_params: FuelParameters,
                   dt: float) -> dict[str, torch.Tensor]:
    """Compute batch metrics, no oracle mode selection or GT future used in selection."""
    poses = prediction['pose']
    speeds = prediction['speeds']
    xy = prediction['trajectory_dense']
    if poses.ndim != 3 or poses.shape[-1] != 3 or poses.shape[1] != 41:
        raise ValueError('Expected selected K2 poses [B,41,3]')
    if xy.shape != truth_xy.shape or xy.shape[1:] != (40, 2):
        raise ValueError('Expected predicted and expert dense XY [B,40,2]')
    if speeds.shape != (len(xy), 41):
        raise ValueError('Expected selected K2 longitudinal speeds [B,41]')
    if payload_kg.reshape(-1).shape[0] != len(xy):
        raise ValueError('Missing/invalid per-sample physical payload_kg')

    fuel = vt_cpfm_from_states(speeds, payload_kg.reshape(-1), fuel_params, dt=dt)
    path_segments = torch.linalg.vector_norm(poses[:, 1:, :2] - poses[:, :-1, :2], dim=-1)
    distance_m = path_segments.sum(dim=1)
    fuel_l = fuel['fuel_total']
    if bool((distance_m <= 1.0).any()):
        raise ValueError('Predicted executed distance <=1m; fuel per 100km is undefined')
    ade_m = torch.linalg.vector_norm(xy - truth_xy, dim=-1).mean(dim=1)
    progress_m = centerline_progress(poses[:, -1], features['target_lane_polyline'])
    fuel_rate = fuel['fuel_rate']
    cumulative = torch.cat((torch.zeros_like(fuel_l[:, None]),
                            torch.cumsum(fuel_rate, dim=1) * dt), dim=1)
    values = dict(fuel_l=fuel_l, distance_m=distance_m,
                  fuel_l_per_100km=100000.0 * fuel_l / distance_m,
                  progress_m=progress_m, ade_m=ade_m,
                  goal_terminal_error_m=torch.linalg.vector_norm(
                      poses[:, -1, :2] - features['target_point'], dim=-1),
                  selected_mode=prediction['trajectory_mode_idx'],
                  speeds_kmh=speeds * 3.6,
                  cumulative_fuel_l=cumulative)
    for key, value in values.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f'Non-finite open-loop evaluation values: {key}')
    return values

def evaluate_sample_metrics(prediction, targets, features, context, *, fuel_params, dt, progress_threshold_m):
    truth = targets['trajectory_dense']
    base = sample_metrics(prediction, truth, features, context['koopman_payload_kg'], fuel_params, dt)
    xy = prediction['trajectory_dense']
    fde = torch.linalg.vector_norm(xy[:, -1] - truth[:, -1], dim=-1)
    pvr = (base['progress_m'] < progress_threshold_m).to(torch.float32)
    # K2's predicted vx is longitudinal; finite-difference GT XY gives planar
    # speed magnitude, so this is an explicitly labelled auxiliary proxy only.
    # Expert dense XY is ego-local: prepend the origin, not world-space pose0.
    gt_poses = torch.cat((torch.zeros_like(truth[:, :1, :]), truth), dim=1)
    gt_speed = torch.linalg.vector_norm(gt_poses[:, 1:] - gt_poses[:, :-1], dim=-1) / dt
    speed_mae_proxy = (prediction['speeds'][:, 1:] - gt_speed).abs().mean(dim=1) * 3.6
    accel = (prediction['speeds'][:, 1:] - prediction['speeds'][:, :-1]) / dt
    jerk = (accel[:, 1:] - accel[:, :-1]) / dt
    rms_jerk = jerk.square().mean(dim=1).sqrt()
    result = dict(base, fde_m=fde, pvr=pvr,
                  speed_mae_proxy_kmh=speed_mae_proxy,
                  rms_longitudinal_jerk_m_s3=rms_jerk)
    for name, value in result.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f'Non-finite fuel evaluation metric: {name}')
    return result
