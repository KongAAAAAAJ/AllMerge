"""Three PDF case-study figures for TRACE joint trajectory projection."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def _road_pair(edges, vehicle_index: int, vehicle_count: int):
    """Return the selected vehicle's left/right road edge from the flattened list."""
    if edges is None or len(edges) < 2 * vehicle_count:
        return None, None
    return np.asarray(edges[vehicle_index]), np.asarray(edges[vehicle_count + vehicle_index])


def _plot_direction_arrow(ax, path, index, color, *, alpha=0.9, lw=1.0, zorder=5):
    if index < 0 or index + 1 >= len(path):
        return
    p0, p1 = path[index, :2], path[index + 1, :2]
    if np.linalg.norm(p1 - p0) < 1.0e-8:
        return
    ax.annotate(
        '', xy=p1, xytext=p0,
        arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                        mutation_scale=8, alpha=alpha),
        zorder=zorder,
    )


def draw_projection_figures(output_dir: Path, cases: dict):
    """Draw independently selected geometry, stability and control case-study PDFs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind, figure in cases.items():
        original = np.asarray(figure['nominal_world'])
        projected = np.asarray(figure['projected_world'])
        nominal_states = np.asarray(figure['nominal_states'])
        projected_states = np.asarray(figure['projected_states'])
        nominal_ltr = np.asarray(figure['nominal_ltr'])
        projected_ltr = np.asarray(figure['projected_ltr'])
        bounds_nominal = np.asarray(figure['nominal_stability_bounds'])
        bounds_projected = np.asarray(figure['projected_stability_bounds'])
        dt = float(figure['dt_s'])
        n_vehicles, steps, _ = original.shape

        pointwise_delta = np.linalg.norm(original[:, :, :2] - projected[:, :, :2], axis=-1)
        exemplar = int(figure['exemplar_index'])
        a, b = original[exemplar], projected[exemplar]
        differences = pointwise_delta[exemplar]
        focus = int(figure['focus_step']) if kind == 'geometry' else 0
        displacement = float(differences[focus])
        times = np.arange(1, steps) * float(dt)
        primary, second = '#2563eb', '#ea580c'
        light_primary, light_second = '#91add2', '#efb99d'

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
            if kind == 'geometry':
                fig, overall = plt.subplots(figsize=(12, 6.5))
                zoom = overall.inset_axes([.56, .52, .42, .43])

                # (a) Draw the road corridor and its actual configured safety inset.
                road_edges = figure.get('road_edge_world')
                safety_edges = figure.get('road_safety_edge_world')
                left_edge, right_edge = _road_pair(road_edges, exemplar, n_vehicles)
                safe_left, safe_right = _road_pair(safety_edges, exemplar, n_vehicles)
                if left_edge is not None and right_edge is not None:
                    overall.plot(left_edge[:, 0], left_edge[:, 1], color='#6b7280', lw=1.0,
                                 ls='-', alpha=0.8, zorder=0)
                    overall.plot(right_edge[:, 0], right_edge[:, 1], color='#6b7280', lw=1.0,
                                 ls='-', alpha=0.8, zorder=0)
                if safe_left is not None and safe_right is not None:
                    valid_safe = (np.isfinite(safe_left).all() and np.isfinite(safe_right).all())
                    if valid_safe:
                        polygon_x = np.r_[safe_left[:, 0], safe_right[::-1, 0]]
                        polygon_y = np.r_[safe_left[:, 1], safe_right[::-1, 1]]
                        overall.fill(polygon_x, polygon_y, color='#16a34a', alpha=0.055,
                                     zorder=0, label='Road safety envelope')
                        overall.plot(safe_left[:, 0], safe_left[:, 1], color='#15803d',
                                     ls='--', lw=0.85, alpha=0.75, zorder=1)
                        overall.plot(safe_right[:, 0], safe_right[:, 1], color='#15803d',
                                     ls='--', lw=0.85, alpha=0.75, zorder=1)

                for i in range(n_vehicles):
                    is_exemplar = i == exemplar
                    base_color = primary if is_exemplar else light_primary
                    jtp_color = second if is_exemplar else light_second
                    overall.plot(original[i, :, 0], original[i, :, 1], ls='--', c=base_color,
                                 alpha=0.95 if is_exemplar else 0.78, lw=1.65 if is_exemplar else 1.1,
                                 label='Baseline' if is_exemplar else None, zorder=3)
                    overall.plot(projected[i, :, 0], projected[i, :, 1], ls='-', c=jtp_color,
                                 alpha=0.98 if is_exemplar else 0.82, lw=1.8 if is_exemplar else 1.15,
                                 label='JTP' if is_exemplar else None, zorder=4)

                    # Start and terminal markers distinguish both paired predictions.
                    overall.scatter(*original[i, 0, :2], marker='o', s=24,
                                     facecolors='white', edgecolors=base_color, linewidths=1.0,
                                     zorder=6)
                    overall.scatter(*original[i, -1, :2], marker='x', s=30,
                                     c=base_color, linewidths=1.2, zorder=6)
                    overall.scatter(*projected[i, -1, :2], marker='s', s=22,
                                     facecolors='white', edgecolors=jtp_color, linewidths=1.0,
                                     zorder=6)

                    # Direction arrows are placed along the JTP path; they do not alter data.
                    for idx in (max(1, steps // 3), max(1, 2 * steps // 3)):
                        _plot_direction_arrow(overall, projected[i], idx, jtp_color,
                                              alpha=0.9 if is_exemplar else 0.65,
                                              lw=1.05 if is_exemplar else 0.8)

                for actor in figure.get('external_actors', []):
                    actor = np.asarray(actor)
                    if actor.ndim == 2 and actor.shape[-1] >= 2:
                        overall.plot(actor[:, 0], actor[:, 1], ls='-.', c='#737373',
                                     alpha=0.42, lw=0.9, zorder=2)

                overall.add_patch(plt.Rectangle(
                    (min(a[focus, 0], b[focus, 0]) - 1.0,
                     min(a[focus, 1], b[focus, 1]) - 0.65),
                    abs(a[focus, 0] - b[focus, 0]) + 2.0,
                    abs(a[focus, 1] - b[focus, 1]) + 1.3,
                    fill=False, ls=':', lw=1.15, ec='#111827', zorder=7))
                overall.set(title='Trajectory and road constraints', xlabel='World X (m)', ylabel='World Y (m)')
                overall.grid(alpha=0.2)
                fig.legend(handles=[
                    Line2D([0], [0], color=primary, ls='--', lw=1.6, label='Baseline'),
                    Line2D([0], [0], color=second, ls='-', lw=1.7, label='JTP'),
                    Line2D([0], [0], color='#6b7280', lw=1.0, label='Road boundary'),
                    Line2D([0], [0], color='#15803d', ls='--', lw=0.9, label='Safety inset'),
                    Line2D([0], [0], color='#15803d', marker='>', lw=0, label='Motion direction'),
                    Line2D([0], [0], color='#111827', marker='o', markerfacecolor='white', lw=0, label='Start'),
                    Line2D([0], [0], color='#111827', marker='x', lw=0, label='Baseline end'),
                    Line2D([0], [0], color='#111827', marker='s', markerfacecolor='white', lw=0, label='JTP end'),
                ], loc='lower center', fontsize=11, ncol=4, frameon=False)

                # (b) Tight local temporal window with several true pointwise correction vectors.
                window_radius = 2
                lo = max(0, focus - window_radius)
                hi = min(steps - 1, focus + window_radius)
                local_ids = np.arange(lo, hi + 1, dtype=int)
                zoom.plot(a[local_ids, 0], a[local_ids, 1], ls='--', c=primary,
                          lw=1.7, marker='.', ms=4, label='Baseline')
                zoom.plot(b[local_ids, 0], b[local_ids, 1], ls='-', c=second,
                          lw=1.8, marker='.', ms=4, label='JTP')
                arrow_ids = np.unique(np.r_[np.linspace(lo, hi, min(3, len(local_ids)), dtype=int), focus])
                for idx in arrow_ids:
                    if differences[idx] <= 1.0e-5:
                        continue
                    zoom.annotate('', xy=b[idx, :2], xytext=a[idx, :2],
                                  arrowprops=dict(arrowstyle='->', color='#15803d',
                                                  lw=1.25, mutation_scale=10, alpha=0.9),
                                  zorder=6)
                    if idx == focus:
                        zoom.scatter([a[idx, 0]], [a[idx, 1]], s=25, c=primary, zorder=7)
                        zoom.scatter([b[idx, 0]], [b[idx, 1]], s=25, c=second, zorder=7)
                zoom.text(.03, .03, f'Road residual: {figure["nominal_road_residual"][focus-1]:.3f} ? {figure["projected_road_residual"][focus-1]:.3f} m\nDisplacement = {displacement:.3f} m; t = {focus * dt:.1f} s',
                          transform=zoom.transAxes, fontsize=10, color='#166534', va='bottom')
                local_xy = np.concatenate((a[local_ids, :2], b[local_ids, :2]), axis=0)
                x_range = max(float(np.ptp(local_xy[:, 0])), 0.2)
                y_range = max(float(np.ptp(local_xy[:, 1])), 0.08)
                x_margin = max(0.35, 0.10 * x_range)
                y_margin = max(0.18, 0.22 * y_range)
                zoom.set_xlim(float(local_xy[:, 0].min() - x_margin), float(local_xy[:, 0].max() + x_margin))
                zoom.set_ylim(float(local_xy[:, 1].min() - y_margin), float(local_xy[:, 1].max() + y_margin))
                zoom.set(title=f'Local refinement (truck {exemplar})',
                         xlabel='World X (m)', ylabel='World Y (m)')
                zoom.grid(alpha=0.22)
                zoom.tick_params(labelsize=10)
                zoom.title.set_fontsize(12)
                zoom.xaxis.label.set_size(10)
                zoom.yaxis.label.set_size(10)

                fig.subplots_adjust(bottom=.20)
                fig.savefig(output_dir/'projection_ablation_geometry.pdf', bbox_inches='tight')
                plt.close(fig)

            elif kind == 'stability':
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
                rates_n = np.diff(controls_n, axis=0)/dt
                rates_p = np.diff(controls_p, axis=0)/dt
                lower = np.asarray(figure['control_lower'])
                upper = np.asarray(figure['control_upper'])
                rate_limits = np.asarray(figure['control_rate_limits'])
                control_specs = (
                    (times, controls_n[:, 0], controls_p[:, 0], lower[0], upper[0],
                     '(a) Acceleration', 'Acceleration (m/s²)'),
                    (times, controls_n[:, 1], controls_p[:, 1], lower[1], upper[1],
                     '(b) Front-wheel steering angle', 'Steering (rad)'),
                    (times[1:], rates_n[:, 0], rates_p[:, 0], -rate_limits[0], rate_limits[0],
                     '(c) Acceleration rate', 'Acceleration rate (m/s³)'),
                    (times[1:], rates_n[:, 1], rates_p[:, 1], -rate_limits[1], rate_limits[1],
                     '(d) Steering rate', 'Steering rate (rad/s)'),
                )
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
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
                fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False)
                fig.tight_layout(rect=(0, .08, 1, 1))
                fig.savefig(output_dir/'projection_ablation_control.pdf', bbox_inches='tight')
                plt.close(fig)
