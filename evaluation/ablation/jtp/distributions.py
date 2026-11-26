"""Full-horizon physical constraint distributions, independent of the solver."""
from pathlib import Path
import json
import numpy as np
import torch

from .metrics import _circle_centers, _actor_circles, prediction_control_rates
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
    return {'version': 2, 'methods': ['Baseline', 'Ours'],
            'control_rate_scope': 'prediction_steps_2_to_H; history-to-step-1 excluded; projector still includes it',
            'id_columns': ['seed', 'episode', 'frame', 'ego', 'future_step'],
            'definition': 'All future steps; road minimum margin over circles/sides; collision minimum envelope gap per controlled vehicle. Missing collision objects omitted. Curvature uses H-1 future intervals. Control rates use prediction steps 2..H only.',
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
    rate = prediction_control_rates(u, cfg['execution']['dt_s'])
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
                start = 2 if k in ('acceleration_rate', 'steering_rate') else 1
                ids = np.array([[seed, episode, frame, ego, t] for t in range(start, start+len(v))], dtype=np.int64).reshape(-1, 5)
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
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
    path, output = Path(data_path), Path(output_dir)
    if not path.is_file() or not path.with_suffix('.json').is_file():
        raise ValueError('Missing full distribution data/metadata; rerun run_jtp.py with --constraint-distributions. Summary CSV cannot reconstruct these distributions.')
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)

    # 绘图字体配置：
    with matplotlib.rc_context({
        'font.family': 'Times New Roman',
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
    }), np.load(path, allow_pickle=False) as data:
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
                    for m, color in zip(metadata['methods'], ('#0072BD', '#EDB120')):
                        v, b = (data[f'{m}_{k}_{s}'] for s in ('values', 'bad'))
                        if len(v):
                            density, _ = np.histogram(v, bins=bins, density=True)
                            widths = np.diff(bins)
                            centers = (bins[:-1] + bins[1:]) / 2
                            ax.bar(centers, density, width=widths*.9, color=color,
                                   alpha=.5, linewidth=0, label=m, zorder=2)
                            notes.append(TextArea(f'Violations: step {100*b.mean():.2f}%',
                                                  textprops={'color': color, 'fontsize': 14}))
                    annotation = AnchoredOffsetbox(loc='upper left',
                        child=VPacker(children=notes, align='left', pad=0, sep=2),
                        pad=.3, borderpad=.5, frameon=True)
                    annotation.patch.set(facecolor='lightgray', alpha=.5, edgecolor='none')
                    ax.add_artist(annotation)
                else:
                    message = 'No eligible collision objects' if k in ('intertruck', 'external') else 'No evaluable prediction intervals'
                    ax.text(.5, .5, message+'\nn=0; violation rates N/A', ha='center', transform=ax.transAxes)
                for threshold in thresholds:
                    ax.axvline(threshold, color='black', linestyle='--', linewidth=1, zorder=3)
                ax.set(xlabel=spec['label'],
                       ylabel='Density' if ax.get_subplotspec().is_first_col() else '')
                ax.set_axisbelow(True)
                ax.grid(alpha=.2)


                # # 手动设置子图的x-y显示范围
                # if k == 'acceleration_rate':
                #     ax.set_xlim(-10, 10)   # 横坐标范围
                #     ax.set_ylim(0, 0.8)    # 纵坐标范围

                # elif k == 'steering_rate':
                #     ax.set_xlim(-1.5, 1.5)
                #     ax.set_ylim(0, 4.5)

                # elif k == 'intertruck':
                #     ax.set_xlim(-2, 20)
                #     ax.set_ylim(0, 1.1)

                # elif k == 'external':
                #     ax.set_xlim(-5, 50)
                #     ax.set_ylim(0, 0.13)

                # elif k == 'acceleration':
                #     ax.set_xlim(-4.5, 3)
                #     ax.set_ylim(0, 6.5)

                # elif k == 'steering':
                #     ax.set_xlim(-0.09, 0.09)
                #     ax.set_ylim(0, 40)


            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center',
                       bbox_to_anchor=(.5, .01), ncol=2, frameon=False)
            fig.tight_layout(rect=(0, .05, 1, 1))
            for extension in ('pdf', 'png'):
                fig.savefig(output / f'jtp_distributions_{name}.{extension}', dpi=300)
            plt.close(fig)
