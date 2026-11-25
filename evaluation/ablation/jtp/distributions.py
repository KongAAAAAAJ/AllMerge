"""Full-horizon physical constraint distributions, independent of the solver."""
from pathlib import Path
import json
import numpy as np
import torch

from .metrics import _circle_centers, _actor_circles
from highway_env.planner.diffusion.projection.constraints import stability_limits
from highway_env.planner.diffusion.projection.geometry import geometry_residual, wrap_angle

KEYS = ('road', 'curvature', 'yaw', 'beta', 'roll', 'ltr',
        'acceleration_rate', 'steering_rate', 'intertruck', 'external', 'acceleration', 'steering')
LABELS = ('Road safety margin (m)', 'Curvature (1/m)', 'Yaw rate (rad/s)',
          'Sideslip (rad)', 'Roll (rad)', 'LTR', 'Acceleration rate (m/s³)',
          'Steering rate (rad/s)', 'Inter-truck envelope gap (m)',
          'External envelope gap (m)', 'Acceleration (m/s²)', 'Steering (rad)')
FLAGS = ('road_violation', 'curvature_violation', 'yaw_violation', 'sideslip_violation',
         'roll_violation', 'ltr_violation', 'acceleration_rate_violation',
         'steering_rate_violation', 'intertruck_collision_violation',
         'external_collision_violation', 'acceleration_violation', 'steering_violation')


def specifications(cfg, ev):
    c, g, t = cfg['control'], cfg['geometry'], ev['tolerances']
    limits = [[0.], [-g['curvature_limit_per_m'], g['curvature_limit_per_m']],
              [], [], [], [-cfg['stability']['ltr_limit'], cfg['stability']['ltr_limit']]]
    for name in ('acceleration_rate_max_mps3', 'steering_rate_max_rad_s'):
        v = ev['limits'][name]
        limits.append([-v, v])
    limits += [[ev['limits']['collision_clearance_m']]] * 2
    limits += [[c['acceleration_min_mps2'], c['acceleration_max_mps2']],
               [c['steering_min_rad'], c['steering_max_rad']]]
    return {'version': 1, 'methods': ['Baseline', 'Ours'],
            'id_columns': ['seed', 'episode', 'frame', 'ego', 'future_step'],
            'definition': 'All future steps; road minimum margin over circles/sides; collision minimum envelope gap per controlled vehicle. Missing collision objects omitted. Curvature uses H-1 future intervals. Rates include previous observed control.',
            'tolerances': t, 'evaluation_limits': ev['limits'],
            'projection_config': cfg,
            'constraints': {k: {'label': label, 'thresholds': lim, 'trajectory_flag': flag,
                                'dynamic_threshold': k in ('yaw', 'beta', 'roll')}
                            for k, label, lim, flag in zip(KEYS, LABELS, limits, FLAGS)}}


def collect(controls, rollout, context, roads, actors, cfg, ev):
    """Return per-vehicle/per-step values and exact evaluator violation flags."""
    def array(x):
        return x.detach().cpu().numpy().astype(np.float64)
    u = array(controls)
    pose = array(rollout['trajectory'])
    states = array(rollout['states'])
    ltr = array(rollout['ltr'])
    n, h, _ = u.shape
    tol, lim = ev['tolerances'], ev['limits']
    rate = np.diff(np.concatenate((array(context['koopman_control_hist_si'][:, -1:]), u), axis=1), axis=1) / cfg['execution']['dt_s']
    values = {k: [] for k in KEYS}
    bad = {k: [] for k in KEYS}
    footprint = cfg['geometry']['footprint']
    radius = footprint['circle_radius_m']
    centers = [_circle_centers(pose[i, 1:], np.asarray(footprint['circle_offsets_m'])) for i in range(n)]
    for i in range(n):
        with torch.no_grad():
            bounds = array(stability_limits(rollout['states'][i, 1:, 0], context['koopman_payload_kg'][i], context['koopman_road_mu'][i], cfg))
            geom, meta = geometry_residual(rollout['trajectory'][i], roads[i], cfg)
            geom = array(geom)
        road_raw = geom[:meta['road_count']].reshape(h, -1).max(axis=1)
        delta = rollout['trajectory'][i, 2:, :2] - rollout['trajectory'][i, 1:-1, :2]
        curvature = array(wrap_angle(rollout['trajectory'][i, 2:, 2] - rollout['trajectory'][i, 1:-1, 2]) / torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-4))
        row = {'road': -road_raw, 'curvature': curvature,
               'yaw': states[i, 1:, 2], 'beta': np.arctan2(states[i, 1:, 1], states[i, 1:, 0]),
               'roll': states[i, 1:, 3], 'ltr': ltr[i, 1:],
               'acceleration_rate': rate[i, :, 0], 'steering_rate': rate[i, :, 1],
               'acceleration': u[i, :, 0], 'steering': u[i, :, 1]}
        flags = {'road': road_raw > tol['road_m'],
                 'curvature': geom[meta['road_count']:] > tol['curvature_per_m2']}
        for j, k in enumerate(('yaw', 'beta', 'roll', 'ltr')):
            tolerance = tol.get('ltr', tol['stability_rad']) if k == 'ltr' else tol['stability_rad']
            flags[k] = np.abs(row[k]) - bounds[:, j] > tolerance
        for k, name in (('acceleration_rate', 'acceleration_rate_max_mps3'), ('steering_rate', 'steering_rate_max_rad_s')):
            flags[k] = np.abs(row[k]) > lim[name] + tol['control_si']
        for k, unit in (('acceleration', 'mps2'), ('steering', 'rad')):
            flags[k] = (row[k] < cfg['control'][f'{k}_min_{unit}'] - tol['control_si']) | (row[k] > cfg['control'][f'{k}_max_{unit}'] + tol['control_si'])
        for k in ('intertruck', 'external'):
            gaps = []
            others = [(centers[j], radius) for j in range(n) if j != i] if k == 'intertruck' else [(_actor_circles(a)[0][1:], _actor_circles(a)[1]) for a in actors]
            for other, r in others:
                distance = np.sqrt(np.maximum(np.sum((centers[i][:, :, None] - other[:, None]) ** 2, axis=-1), 1e-12))
                gaps.append((distance - radius - r).min(axis=(1, 2)))
            row[k] = np.min(gaps, axis=0) if gaps else np.empty(0)
            flags[k] = lim['collision_clearance_m'] - row[k] > tol['collision_m']
        for k in KEYS:
            if not np.isfinite(row[k]).all():
                raise ValueError(f'Nonfinite distribution values: {k}')
            values[k].append(row[k])
            bad[k].append(flags[k])
    return values, bad


class DistributionRecorder:
    def __init__(self, cfg, ev):
        self.metadata = specifications(cfg, ev)
        self.parts = {}

    def add(self, method, collected, seed, episode, frame, ego_ids, expected):
        values, bad = collected
        for k, flag in zip(KEYS, FLAGS):
            for i, ego in enumerate(ego_ids):
                v, b = values[k][i], bad[k][i]
                if bool(np.any(b)) != bool(expected[i][flag]):
                    raise ValueError(f'Distribution/evaluator mismatch: {method}/{k}/{ego}')
                ids = np.array([[seed, episode, frame, ego, t] for t in range(1, len(v)+1)], dtype=np.int64).reshape(-1, 5)
                for suffix, item in (('values', v), ('bad', b), ('ids', ids)):
                    self.parts.setdefault(f'{method}_{k}_{suffix}', []).append(item)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.concatenate(v) for k, v in self.parts.items()}
        for k in KEYS:
            if not np.array_equal(arrays[f'Baseline_{k}_ids'], arrays[f'Ours_{k}_ids']):
                raise ValueError(f'Unpaired distribution samples: {k}')
        path = directory / 'jtp_constraint_distributions.npz'
        np.savez_compressed(path, **arrays)
        path.with_suffix('.json').write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        return path


def draw(data_path, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path, output = Path(data_path), Path(output_dir)
    if not path.is_file() or not path.with_suffix('.json').is_file():
        raise ValueError('Missing full distribution data/metadata; rerun run_jtp.py with --constraint-distributions. Summary CSV cannot reconstruct these distributions.')
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=False) as data:
        for name, keys in (('geometry_stability', KEYS[:6]), ('control_safety', KEYS[6:])):
            fig, axes = plt.subplots(3, 2, figsize=(12, 12))
            for ax, k in zip(axes.flat, keys):
                spec = metadata['constraints'][k]
                combined = np.concatenate([data[f'{m}_{k}_values'] for m in metadata['methods']])
                thresholds = spec['thresholds']
                if len(combined):
                    low, high = min(combined.min(), min(thresholds, default=combined.min())), max(combined.max(), max(thresholds, default=combined.max()))
                    pad = max((high-low)*.03, max(abs(low), abs(high), 1.)*1e-6)
                    bins = np.linspace(low-pad, high+pad, 61)
                    notes = []
                    for m, color, style in zip(metadata['methods'], ('tab:blue', 'tab:orange'), ('-', '--')):
                        v, b, ids = (data[f'{m}_{k}_{s}'] for s in ('values', 'bad', 'ids'))
                        if len(v):
                            ax.hist(v, bins=bins, density=True, histtype='step', color=color, linestyle=style, linewidth=1.6, label=m)
                            _, inverse = np.unique(ids[:, :4], axis=0, return_inverse=True)
                            trajectory_bad = np.zeros(inverse.max()+1, dtype=bool)
                            np.logical_or.at(trajectory_bad, inverse, b)
                            notes.append(f'{m}: n={len(v):,}; step {100*b.mean():.2f}%; traj {100*trajectory_bad.mean():.2f}%')
                    ax.text(.02, .98, '\n'.join(notes), transform=ax.transAxes, va='top', fontsize=8,
                            bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
                    ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1, .82))
                else:
                    ax.text(.5, .5, 'No eligible collision objects\nn=0; violation rates N/A', ha='center', transform=ax.transAxes)
                for threshold in thresholds:
                    ax.axvline(threshold, color='black', linestyle='--', linewidth=1)
                if spec['dynamic_threshold']:
                    ax.text(.02, .04, 'Threshold varies with operating conditions', transform=ax.transAxes, fontsize=8)
                ax.set(title=spec['label'], xlabel=spec['label'], ylabel='Density')
                ax.grid(alpha=.2)
            fig.tight_layout()
            for extension in ('pdf', 'png'):
                fig.savefig(output / f'jtp_distributions_{name}.{extension}', dpi=300)
            plt.close(fig)
