"""Independent evaluation of stability and control magnitude only."""
import math
import numpy as np
from .distributions import collect, KEYS, FLAGS
from highway_env.planner.diffusion.projection.constraints import stability_limits


def validate_evaluation_config(config, jtp_config):
    for key in ('stability_rad', 'ltr', 'control_si'):
        value = float(config['tolerances'][key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid tolerances.{key}')


def evaluate_group(controls, trajectories, states, ltr, context, jtp_cfg, evaluation_cfg):
    values, bad = collect(controls, dict(trajectory=trajectories, states=states, ltr=ltr), context, jtp_cfg, evaluation_cfg)
    result = []
    for i in range(len(controls)):
        row = {flag: bool(bad[key][i].any()) for key, flag in zip(KEYS, FLAGS)}
        bounds = stability_limits(states[i,1:,0], context['koopman_payload_kg'][i],
                                  context['koopman_road_mu'][i], jtp_cfg).detach().cpu().numpy()
        stability_excess = np.column_stack([
            np.maximum(np.abs(values[key][i])-bounds[:,j]-evaluation_cfg['tolerances']['ltr' if key=='ltr' else 'stability_rad'],0.) / np.maximum(bounds[:,j],1e-8)
            for j,key in enumerate(KEYS[:4])])
        magnitude = []
        for key,unit in (('acceleration','mps2'),('steering','rad')):
            lo,hi = (jtp_cfg['control'][f'{key}_{side}_{unit}'] for side in ('min','max'))
            magnitude.append(np.maximum(np.maximum(lo-values[key][i], values[key][i]-hi)-evaluation_cfg['tolerances']['control_si'],0.)/(hi-lo))
        row.update(stability_excess_mean_norm=float(stability_excess.mean()),
                   stability_excess_max_norm=float(stability_excess.max()),
                   control_magnitude_excess_mean_norm=float(np.mean(magnitude)),
                   control_magnitude_excess_max_norm=float(np.max(magnitude)))
        row['stability_violation'] = any(row[flag] for flag in FLAGS[:4])
        row['control_magnitude_violation'] = any(row[flag] for flag in FLAGS[4:])
        row['control_violation'] = row['control_magnitude_violation']
        row['joint_feasible'] = not (row['stability_violation'] or row['control_violation'])
        result.append(row)
    return result


def trajectory_deviation_m(nominal_world: np.ndarray, projected_world: np.ndarray) -> np.ndarray:
    """Per-vehicle mean over t=1..H of XY Euclidean deviation, in metres."""
    delta = np.asarray(projected_world)[:, 1:, :2] - np.asarray(nominal_world)[:, 1:, :2]
    return np.linalg.norm(delta, axis=-1).mean(axis=-1)
