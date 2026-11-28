"""Three PDF case-study figures for TRACE joint trajectory projection."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def _road_pair(edges, vehicle_index: int, vehicle_count: int):
    """Return the selected vehicle's left/right road edge from the flattened list."""
    if edges is None or len(edges) < 2 * vehicle_count:
        return None, None
    return np.asarray(edges[vehicle_index]), np.asarray(edges[vehicle_count + vehicle_index])


def _trajectory_band(ax, path, width, color, *, alpha=(.45, .04), zorder=1):
    """Draw one seamless RGBA ribbon at 300 dpi, with width in world metres."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.colors import to_rgb
    from matplotlib.path import Path as PlotPath
    from matplotlib.transforms import Affine2D

    path = np.asarray(path)
    normal = np.column_stack((-np.sin(path[:, 2]), np.cos(path[:, 2])))
    left = path[:, :2] + normal * width / 2
    right = path[:, :2] - normal * width / 2
    boundary = np.vstack((left, right[::-1]))
    low, high = boundary.min(axis=0), boundary.max(axis=0)
    pixel_size = np.ptp(ax.get_xlim()) / (ax.get_position().width * ax.figure.get_figwidth() * 300)
    nx, ny = np.ceil((high - low) / pixel_size).astype(int)
    renderer = RendererAgg(int(nx), int(ny), 300)
    transform = Affine2D().translate(*-low).scale(nx / (high[0] - low[0]),
                                                ny / (high[1] - low[1]))
    gc = renderer.new_gc()
    gc.set_linewidth(0)
    gc.set_foreground((0, 0, 0, 0))
    outline = PlotPath(np.vstack((boundary, boundary[0])), closed=True)
    renderer.draw_path(gc, outline, transform, rgbFace=(1, 1, 1, 1))
    mask = np.asarray(renderer.buffer_rgba())[::-1, :, 3] / 255.
    gc.restore()

    x = low[0] + (np.arange(nx) + .5) * (high[0] - low[0]) / nx
    y = low[1] + (np.arange(ny) + .5) * (high[1] - low[1]) / ny
    xx, yy = np.meshgrid(x, y)
    nearest = np.full((ny, nx), np.inf)
    progress = np.zeros((ny, nx))
    for i, (start, end) in enumerate(zip(path[:-1, :2], path[1:, :2])):
        direction = end - start
        fraction = np.clip(((xx - start[0]) * direction[0] +
                            (yy - start[1]) * direction[1]) / np.dot(direction, direction), 0, 1)
        distance = (xx - start[0] - fraction * direction[0]) ** 2 + \
                   (yy - start[1] - fraction * direction[1]) ** 2
        closer = distance < nearest
        progress[closer] = (i + fraction[closer]) / (len(path) - 1)
        nearest[closer] = distance[closer]
    rgba = np.empty((ny, nx, 4), dtype=np.float32)
    rgba[:, :, :3] = to_rgb(color)
    rgba[:, :, 3] = mask * (alpha[0] + (alpha[1] - alpha[0]) * progress)
    ax.imshow(rgba, origin='lower', extent=(low[0], high[0], low[1], high[1]),
              interpolation='bilinear', zorder=zorder)


def _vehicle_corners(pose, size):
    length, width = size
    corners = np.array([[-length, -width], [length, -width],
                        [length, width], [-length, width]]) / 2
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return corners @ np.array([[c, s], [-s, c]]) + pose[:2]


def _vehicle_rectangle(ax, pose, size, color):
    from matplotlib.patches import Polygon

    ax.add_patch(Polygon(_vehicle_corners(pose, size), facecolor=color,
                         edgecolor=color, alpha=.50, zorder=4))


def _draw_geometry(output_dir, figures):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if len(figures) != 3:
        raise ValueError('Geometry figure requires three distinct scene frames')
    orange, blue, grey = '#ea580c', '#2563eb', '#737373'
    bounds = []
    for data in figures:
        i = int(data['exemplar_index'])
        size = np.asarray(data['vehicle_sizes_m'])[i]
        paths = np.concatenate((data['nominal_world'][i], data['projected_world'][i]))
        normal = np.column_stack((-np.sin(paths[:, 2]), np.cos(paths[:, 2])))
        points = np.vstack((paths[:, :2] - normal * size[1] / 2,
                            paths[:, :2] + normal * size[1] / 2,
                            _vehicle_corners(data['projected_world'][i, 0], size)))
        bounds.append((points.min(axis=0), points.max(axis=0)))
    span = np.max([high - low for low, high in bounds], axis=0) + [8, 30]  # 横纵向坐标轴留白
    plot_width = 2.95
    vertical_scale = 0.2
    plot_height = plot_width * span[1] / span[0] * vertical_scale
    bottom, gap, top = .95, .60, .28
    height = bottom + 3 * plot_height + 2 * gap + top
    fig = plt.figure(figsize=(3.5, height))
    for panel, (data, (low, high)) in enumerate(zip(figures, bounds)):
        ax = fig.add_axes([.43 / 3.5,
                           (bottom + (2 - panel) * (plot_height + gap)) / height,
                           plot_width / 3.5, plot_height / height])
        center = (low + high) / 2
        ax.set_xlim(center[0] - span[0] / 2, center[0] + span[0] / 2)
        ax.set_ylim(center[1] - span[1] / 2, center[1] + span[1] / 2)
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
        _trajectory_band(ax, original[exemplar], sizes[exemplar, 1], blue,
                         alpha=(.60, .20), zorder=2.5)
        _trajectory_band(ax, projected[exemplar], sizes[exemplar, 1], orange,
                         alpha=(.60, .20), zorder=3)
        for i, (path, size) in enumerate(zip(projected, sizes)):
            if i != exemplar:
                _vehicle_rectangle(ax, path[0], size, grey)
        for path, size in zip(data['external_actors'], data['external_actor_sizes_m']):
            _vehicle_rectangle(ax, path[0], size, grey)
        _vehicle_rectangle(ax, projected[exemplar, 0], sizes[exemplar], orange)
        ax.set(title=f'({chr(97 + panel)}) Episode {data["episode_index"]}, '
                     f'frame {data["frame_index"]}, truck {data["ego_index"]}',
               xlabel='World X (m)', ylabel='World Y (m)')
        ax.set_aspect(vertical_scale, adjustable='box')
        ax.tick_params(labelsize=7, pad=2)
        ax.title.set_fontsize(8)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        ax.grid(alpha=.2)
    fig.legend(handles=[Patch(facecolor=orange, alpha=.60, label='Selected truck JTP'),
                        Patch(facecolor=blue, alpha=.60, label='Selected truck baseline'),
                        Patch(facecolor=grey, alpha=.45, label='Other vehicles'),
                        Line2D([0], [0], color='#6b7280', label='Road boundary'),
                        Line2D([0], [0], color='#15803d', ls='--', label='Safety inset')],
               loc='lower center', ncol=2, fontsize=7, frameon=False,
               handlelength=1.4, columnspacing=.8)
    fig.savefig(output_dir / 'projection_ablation_geometry.pdf', dpi=300)
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
