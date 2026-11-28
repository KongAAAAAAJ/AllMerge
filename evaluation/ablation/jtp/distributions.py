"""Six-constraint independent observations and paired distributions."""
from pathlib import Path
import json
import numpy as np
from highway_env.planner.diffusion.projection.constraints import stability_limits

KEYS = ('yaw', 'beta', 'roll', 'ltr', 'acceleration', 'steering')
FLAGS = ('yaw_violation', 'sideslip_violation', 'roll_violation', 'ltr_violation',
         'acceleration_violation', 'steering_violation')
LABELS = ('Yaw rate (rad/s)', 'Sideslip angle (rad)', 'Roll angle (rad)',
          'LTR', 'Acceleration (m/s?)', 'Steering (rad)')


def specifications(cfg, ev):
    return dict(version=4, methods=['Baseline', 'Ours'],
                id_columns=['seed', 'episode', 'frame', 'ego', 'future_step'],
                definition='Six constraints at future steps 1..H; no geometry, collision or rate constraints',
                projection_config=cfg, tolerances=ev['tolerances'], keys=list(KEYS))


def collect(controls, rollout, context, cfg, ev):
    def array(x):
        return x.detach().cpu().numpy().astype(np.float64)
    states, u, ltr = array(rollout['states'])[:, 1:], array(controls), array(rollout['ltr'])[:, 1:]
    tolerance = ev['tolerances']
    values = dict(yaw=states[:, :, 2], beta=np.arctan2(states[:, :, 1], states[:, :, 0]),
                  roll=states[:, :, 3], ltr=ltr, acceleration=u[:, :, 0], steering=u[:, :, 1])
    bounds = np.stack([array(stability_limits(rollout['states'][i, 1:, 0],
                       context['koopman_payload_kg'][i], context['koopman_road_mu'][i], cfg))
                       for i in range(len(u))])
    bad = {key: abs(values[key]) - bounds[:, :, j] > tolerance['ltr' if key == 'ltr' else 'stability_rad']
           for j, key in enumerate(KEYS[:4])}
    for key, unit in (('acceleration', 'mps2'), ('steering', 'rad')):
        bad[key] = ((values[key] < cfg['control'][f'{key}_min_{unit}'] - tolerance['control_si']) |
                    (values[key] > cfg['control'][f'{key}_max_{unit}'] + tolerance['control_si']))
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
                start = 1
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
    from matplotlib.patches import FancyBboxPatch

    path, output = Path(data_path), Path(output_dir)
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    cfg = metadata['projection_config']
    thresholds = {'ltr': [-cfg['stability']['ltr_limit'], cfg['stability']['ltr_limit']],
                  'acceleration': [cfg['control']['acceleration_min_mps2'], cfg['control']['acceleration_max_mps2']],
                  'steering': [cfg['control']['steering_min_rad'], cfg['control']['steering_max_rad']]}
    y_limits = {'yaw': (0, 5), 'beta': (0, 30), 'roll': (0, 40),
                'ltr': (0, 2), 'acceleration': (0, 5), 'steering': (0, 42)}
    output.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({
        'font.family': 'Times New Roman', 'font.size': 14,
        'axes.labelsize': 16, 'xtick.labelsize': 14,
        'ytick.labelsize': 14, 'legend.fontsize': 16,
    }), np.load(path, allow_pickle=False) as data:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        for ax, key, label in zip(axes.flat, KEYS, LABELS):
            combined = np.concatenate([data[f'{method}_{key}_values'] for method in metadata['methods']])
            low, high = np.quantile(combined, [.005, .995])
            pad = max((high-low)*.03, max(abs(low), abs(high), 1.)*1e-6)
            bins = np.linspace(low-pad, high+pad, 41)
            widths = np.diff(bins)
            centers = (bins[:-1]+bins[1:])/2
            violation_rates = []
            for method, color in zip(metadata['methods'], ('#0072BD', '#EDB120')):
                values = data[f'{method}_{key}_values']
                bad = data[f'{method}_{key}_bad']
                density, _ = np.histogram(values, bins=bins, density=True)
                ax.bar(centers, density, width=widths*.9, color=color,
                       alpha=.5, linewidth=0, label=method, zorder=2)
                violation_rates.append((100*bad.mean(), color))
            annotation = ax.inset_axes([.02, .77, .38, .20])
            annotation.set_axis_off()
            annotation.add_patch(FancyBboxPatch(
                (-.03, -.03), 1.06, 1.06, transform=annotation.transAxes,
                boxstyle='round,pad=0.015,rounding_size=0.06',
                facecolor='lightgray', alpha=.1, edgecolor='none', clip_on=False, zorder=0))
            annotation.set_xlim(0, 1.65)
            annotation.set_ylim(-.5, 2.1)
            annotation.text(.825, 2.05, 'Violations', ha='center', va='top', fontsize=14)
            maximum_rate = max(rate for rate, _ in violation_rates)
            for y, (rate, color) in zip((1, 0), violation_rates):
                length = rate/maximum_rate if maximum_rate else 0
                annotation.barh(y, length, height=.35, color=color, alpha=.5, linewidth=0)
                annotation.text(length+.04, y, f'{rate:.2f}%', va='center', fontsize=14)
            for threshold in thresholds.get(key, ()):
                ax.axvline(threshold, color='black', linestyle='--', linewidth=1, zorder=3)
            ax.set(xlabel=label, ylabel='Density' if ax.get_subplotspec().is_first_col() else '')
            ax.set_ylim(*y_limits[key])
            ax.set_axisbelow(True)
            ax.grid(alpha=.2)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center',
                   bbox_to_anchor=(.5, .01), ncol=2, frameon=False)
        fig.tight_layout(rect=(0, .05, 1, 1))
        for suffix in ('pdf', 'png'):
            fig.savefig(output/f'jtp_distributions_stability_control.{suffix}', dpi=300)
        plt.close(fig)
