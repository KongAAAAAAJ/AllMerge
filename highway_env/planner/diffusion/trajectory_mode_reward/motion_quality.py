"""Geometric smoothness and continuous kinematic-feasibility scores.

The sparse 8x2 trajectory is converted to a 10-Hz C2 clamped cubic spline
through t=0 and all eight predicted waypoints.  This scoring-only spline does
not change the existing collision/road/TTC reward geometry or the planner.
The initial speed is estimated from the first longitudinal waypoint; runtime
has the actual ego speed, which is unavailable to this scoring API.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def _bounded_ratio(value: np.ndarray, scale: float) -> np.ndarray:
    ratio = np.maximum(np.asarray(value, dtype=np.float64), 0.0) / float(scale)
    return ratio / (1.0 + ratio)


def spline_motion_quality(sparse_xy: np.ndarray, config: object) -> dict[str, np.ndarray]:
    """For [G,8,2] return [G] float32 metrics; larger kin score is better.

    Measures sudden changes in curvature, lateral-acceleration jerk, and
    differences of sparse-point curvatures; the last prevents cubic smoothing
    from hiding sharp corner geometry at the original waypoints.
    """
    sparse = np.asarray(sparse_xy, dtype=np.float64)
    if sparse.ndim != 3 or sparse.shape[1:] != (8, 2) or not np.isfinite(sparse).all():
        raise ValueError('sparse_xy must be finite [G,8,2]')
    group = sparse.shape[0]
    times = np.arange(9, dtype=np.float64) * float(config.trajectory_dt_s)
    dense_t = np.linspace(0.0, times[-1], int(round(times[-1] / float(config.interpolation_dt_s))) + 1)
    knots = np.concatenate((np.zeros((group, 1, 2), dtype=np.float64), sparse), axis=1)
    initial_v = np.stack((np.maximum(sparse[:, 0, 0], 0.0) / float(config.trajectory_dt_s),
                          np.zeros(group, dtype=np.float64)), axis=1)
    ending_v = (knots[:, -1] - knots[:, -2]) / float(config.trajectory_dt_s)
    # SciPy and the runtime decoder solve the same clamped cubic spline system.
    cs = CubicSpline(times, knots, axis=1, bc_type=((1, initial_v), (1, ending_v)))
    vel = cs(dense_t, 1)
    acc = cs(dense_t, 2)
    speed = np.linalg.norm(vel, axis=-1)
    safe_speed = np.maximum(speed, 0.5)
    cross = vel[..., 0] * acc[..., 1] - vel[..., 1] * acc[..., 0]
    curvature = cross / (safe_speed ** 3)
    lat_acc = speed * speed * curvature
    step_s = 0.5 * (speed[:, 1:] + speed[:, :-1]) * float(config.interpolation_dt_s)
    valid_segment = step_s > 0.05
    # Integrated curvature TV (1/m): zero for uniform curvature, large for corners.
    curvature_tv = np.sum(np.where(valid_segment, np.abs(np.diff(curvature, axis=1)), 0.0), axis=1)
    # Real, physically scaled lateral jerk derived from the interpolated trajectory.
    lateral_jerk = np.abs(np.diff(lat_acc, axis=1)) / float(config.interpolation_dt_s)
    lateral_jerk_mean = np.mean(np.where(valid_segment, lateral_jerk, 0.0), axis=1)
    # Sparse curvature total variation ensures a corner remains penalized even
    # though the interpolation creates a mathematically C2 curve.
    segment = np.diff(knots, axis=1)
    seg_len = np.linalg.norm(segment, axis=-1)
    a, b = segment[:, :-1], segment[:, 1:]
    turn = np.arctan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
                      np.sum(a * b, axis=-1))
    effective_ds = np.maximum(0.5 * (seg_len[:, :-1] + seg_len[:, 1:]), 0.25)
    sparse_curvature = turn / effective_ds
    sparse_tv = np.sum(np.abs(np.diff(sparse_curvature, axis=1)), axis=1)

    curvature_component = _bounded_ratio(curvature_tv, config.smooth_curvature_tv_scale)
    jerk_component = _bounded_ratio(lateral_jerk_mean, config.smooth_lateral_jerk_scale)
    sparse_component = _bounded_ratio(sparse_tv, config.smooth_sparse_curvature_tv_scale)
    smoothness_penalty = (0.5 * curvature_component + 0.3 * jerk_component
                          + 0.2 * sparse_component)

    delta = np.arctan(float(config.kinematic_wheelbase_m) * curvature)
    steering_speed = np.abs(np.diff(delta, axis=1)) / float(config.interpolation_dt_s)
    excess_curvature = np.maximum(np.abs(curvature) / float(config.kinematic_max_curvature) - 1.0, 0.0)
    excess_steer_rate = np.maximum(steering_speed / float(config.kinematic_max_steer_rate) - 1.0, 0.0)
    excess_lateral_acc = np.maximum(np.abs(lat_acc) / float(config.kinematic_max_lateral_acc) - 1.0, 0.0)
    # Blend average violation with peak violation so a single severe event
    # still affects the reward; do NOT clamp positive excess values.
    def severity(x: np.ndarray) -> np.ndarray:
        return 0.7 * np.mean(x, axis=1) + 0.3 * np.max(x, axis=1)
    kin_excess = (0.3 * severity(excess_curvature)
                  + 0.3 * severity(excess_steer_rate)
                  + 0.4 * severity(excess_lateral_acc))
    kin_score = 1.0 / (1.0 + kin_excess)
    values = {
        'smoothness_penalty': smoothness_penalty,
        'smoothness_curvature_tv': curvature_tv,
        'smoothness_lateral_jerk': lateral_jerk_mean,
        'smoothness_sparse_kink': sparse_tv,
        'kinematic_score': kin_score,
        'kinematic_excess': kin_excess,
    }
    result = {}
    for name, metric in values.items():
        if not np.isfinite(metric).all():
            raise ValueError(f'{name} has NaN/Inf in spline motion quality')
        result[name] = metric.astype(np.float32)
    return result
