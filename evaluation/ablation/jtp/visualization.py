"""Stability and control PDF case-study figures for TRACE joint trajectory projection."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from .payload import PAYLOAD_GROUPS, PAYLOAD_LABELS
from .cases import STABILITY_KEYS


def draw_projection_figures(output_dir: Path, cases: dict):
    """Draw independently selected stability and control case-study PDFs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind in dict.fromkeys('control' if key == 'control' else 'stability' for key in cases):
        # Figure typography (sizes in points).
        with matplotlib.rc_context({
            'font.family': 'Times New Roman',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 15,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
        }):
            columns = 4 if kind == 'stability' else 2
            fig, axes = plt.subplots(3, columns, figsize=(columns*6, 12))
            for row, group in enumerate(PAYLOAD_GROUPS):
                primary, second = '#2563eb', '#ea580c'
                if kind == 'stability':
                    labels = (('(a) Yaw rate', '|r| (rad/s)'),
                              ('(b) Sideslip angle', '|β| (rad)'),
                              ('(c) Roll angle', '|φ| (rad)'),
                              ('(d) Lateral load transfer ratio', '|LTR|'))
                    for column, key in enumerate(STABILITY_KEYS):
                        figure = cases[key][group]
                        exemplar = int(figure['exemplar_index'])
                        sn = np.asarray(figure['nominal_states'])[exemplar]
                        sp = np.asarray(figure['projected_states'])[exemplar]
                        series_n, series_p = (
                            (np.abs(state[1:, 2]) if key == 'yaw' else
                             np.abs(np.arctan2(state[1:, 1], state[1:, 0])) if key == 'beta' else
                             np.abs(state[1:, 3]) if key == 'roll' else
                             np.abs(np.asarray(figure[ltr_key])[exemplar, 1:]))
                            for state, ltr_key in ((sn, 'nominal_ltr'), (sp, 'projected_ltr')))
                        limit_n = np.asarray(figure['nominal_stability_bounds'])[exemplar, :, column]
                        limit_p = np.asarray(figure['projected_stability_bounds'])[exemplar, :, column]
                        times = np.arange(1, len(sn)) * float(figure['dt_s'])
                        ax = axes[row, column]
                        title, ylabel = labels[column]
                        ax.plot(times, series_n, c=primary, ls='--', lw=1.45, label='Baseline')
                        ax.plot(times, series_p, c=second, ls='-', lw=1.5, label='JTP')
                        ax.plot(times, limit_n, c='#333333', ls=':', lw=1.0, label='Baseline limit')
                        if not np.allclose(limit_n, limit_p, rtol=1e-5, atol=1e-8):
                            ax.plot(times, limit_p, c='#777777', ls='-.', lw=1.0, label='JTP limit')
                        ax.fill_between(times, limit_n, series_n, where=series_n > limit_n,
                                        color=primary, alpha=0.10, interpolate=True)
                        ax.fill_between(times, limit_p, series_p, where=series_p > limit_p,
                                        color=second, alpha=0.12, interpolate=True)
                        status = {'remains_feasible': 'Originally feasible',
                                  'least_degradation': 'Not improved'}.get(str(figure['selection_grade']), '')
                        identity = f'Ep {int(figure["episode_index"])}, frame {int(figure["frame_index"])}, truck {int(figure["ego_index"])}'
                        heading = f'{PAYLOAD_LABELS[group]} ({float(figure["payload_kg"]):g} kg)'
                        if status:
                            heading += f' — {status}'
                        ax.set(title=f'{heading}\n{identity}\n{title}', xlabel='Time (s)', ylabel=ylabel)
                        ax.grid(alpha=0.2)
                else:
                    figure = cases['control'][group]
                    exemplar = int(figure['exemplar_index'])
                    times = np.arange(1, np.asarray(figure['nominal_world']).shape[1]) * float(figure['dt_s'])
                    controls_n = np.asarray(figure['nominal_controls'])[exemplar]
                    controls_p = np.asarray(figure['projected_controls'])[exemplar]
                    lower = np.asarray(figure['control_lower'])
                    upper = np.asarray(figure['control_upper'])
                    control_specs = (
                        (times, controls_n[:, 0], controls_p[:, 0], lower[0], upper[0],
                         '(a) Acceleration', 'Acceleration (m/s²)'),
                        (times, controls_n[:, 1], controls_p[:, 1], lower[1], upper[1],
                         '(b) Front-wheel steering angle', 'Steering (rad)'),
                    )
                    for ax, (t, base, post, low, high, title, ylabel) in zip(axes[row], control_specs):
                        ax.plot(t, base, c=primary, ls='--', lw=1.45, label='Baseline')
                        ax.plot(t, post, c=second, lw=1.5, label='JTP')
                        ax.axhline(low, color='#333333', ls=':', lw=1, label='Limits')
                        ax.axhline(high, color='#333333', ls=':', lw=1)
                        for series, color in ((base, primary), (post, second)):
                            ax.fill_between(t, low, series, where=series < low, color=color, alpha=.1, interpolate=True)
                            ax.fill_between(t, high, series, where=series > high, color=color, alpha=.1, interpolate=True)
                        ax.set(title=f'{PAYLOAD_LABELS[group]} ({float(figure["payload_kg"]):g} kg)\n{title}', xlabel='Time (s)', ylabel=ylabel)
                        ax.grid(alpha=.2)
            handles = {}
            for ax in axes.flat:
                h, labels = ax.get_legend_handles_labels()
                handles.update(zip(labels, h))
            fig.legend(handles.values(), handles.keys(), loc='lower center',
                       ncol=4 if kind == 'stability' else 3, frameon=False)
            fig.tight_layout(rect=(0, .06, 1, 1))
            fig.savefig(output_dir/f'projection_ablation_{kind}.pdf', bbox_inches='tight')
            plt.close(fig)
