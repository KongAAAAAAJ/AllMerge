"""Three PDF case-study figures for TRACE joint trajectory projection."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def _road_pair(edges, vehicle_index: int, vehicle_count: int):
    """Return the selected vehicle's left/right road edge from the flattened list."""
    if edges is None or len(edges) < 2 * vehicle_count:
        return None, None
    return np.asarray(edges[vehicle_index]), np.asarray(edges[vehicle_count + vehicle_index])


def _trajectory_band(ax, path, width, color):
    """Draw a vehicle-width ribbon with fading future segments in world metres."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgba

    path = np.asarray(path)
    normal = np.column_stack((-np.sin(path[:, 2]), np.cos(path[:, 2])))
    left = path[:, :2] + normal * width / 2
    right = path[:, :2] - normal * width / 2
    polygons = np.stack((left[:-1], right[:-1], right[1:], left[1:]), axis=1)
    colors = np.tile(to_rgba(color), (len(polygons), 1))
    colors[:, 3] = np.linspace(.45, .04, len(polygons))
    ax.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors='none'))


def _vehicle_rectangle(ax, pose, size, color):
    from matplotlib.patches import Polygon

    length, width = size
    corners = np.array([[-length, -width], [length, -width],
                        [length, width], [-length, width]]) / 2
    c, s = np.cos(pose[2]), np.sin(pose[2])
    corners = corners @ np.array([[c, s], [-s, c]]) + pose[:2]
    ax.add_patch(Polygon(corners, facecolor=color, edgecolor=color, alpha=.50, zorder=4))


def _draw_geometry(output_dir, figures):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    orange, blue, grey = '#ea580c', '#2563eb', '#737373'
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for panel, (ax, data) in enumerate(zip(axes, figures)):
        original = np.asarray(data['nominal_world'])
        projected = np.asarray(data['projected_world'])
        exemplar = int(data['exemplar_index'])
        left, right = _road_pair(data['road_edge_world'], exemplar, len(original))
        safe_left, safe_right = _road_pair(data['road_safety_edge_world'], exemplar, len(original))
        ax.plot(left[:, 0], left[:, 1], color='#6b7280', lw=1., alpha=.8)
        ax.plot(right[:, 0], right[:, 1], color='#6b7280', lw=1., alpha=.8)
        if np.isfinite(safe_left).all() and np.isfinite(safe_right).all():
            ax.fill(np.r_[safe_left[:, 0], safe_right[::-1, 0]],
                    np.r_[safe_left[:, 1], safe_right[::-1, 1]],
                    color='#16a34a', alpha=.055, zorder=0)
            for edge in (safe_left, safe_right):
                ax.plot(edge[:, 0], edge[:, 1], color='#15803d',
                        ls='--', lw=.85, alpha=.75)
        sizes = np.asarray(data['vehicle_sizes_m'])
        _trajectory_band(ax, original[exemplar], sizes[exemplar, 1], grey)
        for path, size in zip(projected, sizes):
            _trajectory_band(ax, path, size[1], orange)
        for path, size in zip(data['external_actors'], data['external_actor_sizes_m']):
            _trajectory_band(ax, path, size[1], blue)
        for path, size in zip(projected, sizes):
            _vehicle_rectangle(ax, path[0], size, orange)
        for path, size in zip(data['external_actors'], data['external_actor_sizes_m']):
            _vehicle_rectangle(ax, path[0], size, blue)
        ax.set(title=f'({chr(97 + panel)}) Episode {data["episode_index"]}, '
                     f'frame {data["frame_index"]}, truck {data["ego_index"]}',
               xlabel='World X (m)', ylabel='World Y (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.autoscale_view()
        ax.margins(.05)
        ax.grid(alpha=.2)
    fig.legend(handles=[Patch(facecolor=orange, alpha=.45, label='Platoon / JTP'),
                        Patch(facecolor=blue, alpha=.45, label='Background / CV prediction'),
                        Patch(facecolor=grey, alpha=.45, label='Selected truck baseline'),
                        Line2D([0], [0], color='#6b7280', label='Road boundary'),
                        Line2D([0], [0], color='#15803d', ls='--', label='Safety inset')],
               loc='lower center', ncol=5, frameon=False)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(output_dir / 'projection_ablation_geometry.pdf', bbox_inches='tight')
    plt.close(fig)


def draw_projection_figures(output_dir: Path, cases: dict):
    """Draw independently selected geometry, stability and control case-study PDFs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind, figure in cases.items():
        if kind == 'geometry':
            with matplotlib.rc_context({'font.family': 'Times New Roman', 'font.size': 12}):
                _draw_geometry(output_dir, figure)
            continue
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
