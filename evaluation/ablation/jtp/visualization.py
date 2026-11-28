"""Stability and control PDF case-study figures for TRACE joint trajectory projection."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def draw_projection_figures(output_dir: Path, cases: dict):
    """Draw independently selected stability and control case-study PDFs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind, figure in cases.items():
        original = np.asarray(figure['nominal_world'])
        nominal_states = np.asarray(figure['nominal_states'])
        projected_states = np.asarray(figure['projected_states'])
        nominal_ltr = np.asarray(figure['nominal_ltr'])
        projected_ltr = np.asarray(figure['projected_ltr'])
        bounds_nominal = np.asarray(figure['nominal_stability_bounds'])
        bounds_projected = np.asarray(figure['projected_stability_bounds'])
        dt = float(figure['dt_s'])
        _, steps, _ = original.shape

        exemplar = int(figure['exemplar_index'])
        times = np.arange(1, steps) * float(dt)
        primary, second = '#2563eb', '#ea580c'

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
            if kind == 'stability':
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                yaw_ax, beta_ax, roll_ax, ltr_ax = axes.flat
                # (d)-(g) Absolute chassis responses and the matching per-step limits.
                sn, sp = nominal_states[exemplar], projected_states[exemplar]
                ln, lp = nominal_ltr[exemplar], projected_ltr[exemplar]
                bn, bp = bounds_nominal[exemplar], bounds_projected[exemplar]
                beta_n = np.abs(np.arctan2(sn[1:, 1], sn[1:, 0]))
                beta_p = np.abs(np.arctan2(sp[1:, 1], sp[1:, 0]))
                panel_specs = (
                    (yaw_ax, np.abs(sn[1:, 2]), np.abs(sp[1:, 2]), bn[:, 0], bp[:, 0],
                     '(a) Yaw rate', '|r| (rad/s)'),
                    (beta_ax, beta_n, beta_p, bn[:, 1], bp[:, 1],
                     '(b) Sideslip angle', '|β| (rad)'),
                    (roll_ax, np.abs(sn[1:, 3]), np.abs(sp[1:, 3]), bn[:, 2], bp[:, 2],
                     '(c) Roll angle', '|φ| (rad)'),
                    (ltr_ax, np.abs(ln[1:]), np.abs(lp[1:]), bn[:, 3], bp[:, 3],
                     '(d) Lateral load transfer ratio', '|LTR|'),
                )
                for ax, series_n, series_p, limit_n, limit_p, title, ylabel in panel_specs:
                    ax.plot(times, series_n, c=primary, ls='--', lw=1.45, label='Baseline')
                    ax.plot(times, series_p, c=second, ls='-', lw=1.5, label='JTP')
                    ax.plot(times, limit_n, c='#333333', ls=':', lw=1.0, label='Baseline limit')
                    if not np.allclose(limit_n, limit_p, rtol=1e-5, atol=1e-8):
                        ax.plot(times, limit_p, c='#777777', ls='-.', lw=1.0, label='JTP limit')
                    ax.fill_between(times, limit_n, series_n, where=series_n > limit_n,
                                    color=primary, alpha=0.10, interpolate=True)
                    ax.fill_between(times, limit_p, series_p, where=series_p > limit_p,
                                    color=second, alpha=0.12, interpolate=True)
                    ax.set(title=title, xlabel='Time (s)', ylabel=ylabel)
                    ax.grid(alpha=0.2)


                handles = {}
                for ax in axes.flat:
                    h, labels = ax.get_legend_handles_labels()
                    handles.update(zip(labels, h))
                fig.legend(handles.values(), handles.keys(), loc='lower center', ncol=4, frameon=False)
                fig.tight_layout(rect=(0, .08, 1, 1))
                fig.savefig(output_dir/'projection_ablation_stability.pdf', bbox_inches='tight')
                plt.close(fig)

            else:
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
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                for ax, (t, base, post, low, high, title, ylabel) in zip(axes.flat, control_specs):
                    ax.plot(t, base, c=primary, ls='--', lw=1.45, label='Baseline')
                    ax.plot(t, post, c=second, lw=1.5, label='JTP')
                    ax.axhline(low, color='#333333', ls=':', lw=1, label='Limits')
                    ax.axhline(high, color='#333333', ls=':', lw=1)
                    for series, color in ((base, primary), (post, second)):
                        ax.fill_between(t, low, series, where=series < low, color=color, alpha=.1, interpolate=True)
                        ax.fill_between(t, high, series, where=series > high, color=color, alpha=.1, interpolate=True)
                    ax.set(title=title, xlabel='Time (s)', ylabel=ylabel)
                    ax.grid(alpha=.2)
                fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False)
                fig.tight_layout(rect=(0, .08, 1, 1))
                fig.savefig(output_dir/'projection_ablation_control.pdf', bbox_inches='tight')
                plt.close(fig)
