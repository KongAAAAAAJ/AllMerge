"""Paper 7-panel figure for TRACE joint trajectory projection ablation."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def _as_array(value, *, name: str, ndim: int | None = None):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f'{name} must have {ndim} dimensions, got {array.shape}')
    return array


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


def draw_projection_figure(path: Path, figure: dict, *, dt: float, kappa_lim: float):
    """Draw overall/local trajectories, geometry, and all four chassis responses.

    Road-boundary residual is positive when the footprint violates the configured
    road-clearance constraint and negative when clearance remains. The stability
    panels plot absolute responses against the per-step, per-method K2-derived limits.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    original = _as_array(figure['nominal_world'], name='nominal_world', ndim=3)
    projected = _as_array(figure['projected_world'], name='projected_world', ndim=3)
    if original.shape != projected.shape or original.shape[-1] != 3:
        raise ValueError('Expected same-shaped [N,H+1,3] world trajectories')
    n_vehicles, steps, _ = original.shape
    if steps < 4:
        raise ValueError('At least four trajectory states are required for the figure')
    if not np.isfinite(original).all() or not np.isfinite(projected).all():
        raise ValueError('Trajectory figure data contains NaN/Inf')

    nominal_states = _as_array(figure['nominal_states'], name='nominal_states', ndim=3)
    projected_states = _as_array(figure['projected_states'], name='projected_states', ndim=3)
    nominal_ltr = _as_array(figure['nominal_ltr'], name='nominal_ltr', ndim=2)
    projected_ltr = _as_array(figure['projected_ltr'], name='projected_ltr', ndim=2)
    bounds_nominal = _as_array(figure['nominal_stability_bounds'], name='nominal_stability_bounds', ndim=3)
    bounds_projected = _as_array(figure['projected_stability_bounds'], name='projected_stability_bounds', ndim=3)
    expected_horizon = steps - 1
    for name, value, tail in (
        ('nominal_states', nominal_states, (steps, 5)),
        ('projected_states', projected_states, (steps, 5)),
        ('nominal_ltr', nominal_ltr, (steps,)),
        ('projected_ltr', projected_ltr, (steps,)),
        ('nominal_stability_bounds', bounds_nominal, (expected_horizon, 4)),
        ('projected_stability_bounds', bounds_projected, (expected_horizon, 4)),
    ):
        if value.shape != (n_vehicles, *tail):
            raise ValueError(f'{name} must have shape {(n_vehicles, *tail)}, got {value.shape}')

    # Select the vehicle with the largest pointwise displacement in this fixed,
    # preselected synchronized frame. This is a visualization rule, not a metric.
    pointwise_delta = np.linalg.norm(original[:, :, :2] - projected[:, :, :2], axis=-1)
    vehicle_max_delta = pointwise_delta.max(axis=1)
    exemplar = int(np.argmax(vehicle_max_delta))
    a, b = original[exemplar], projected[exemplar]
    differences = pointwise_delta[exemplar]
    focus = int(np.argmax(differences))
    max_deviation = float(differences[focus])
    times = np.arange(1, steps) * float(dt)
    primary, second = '#2563eb', '#ea580c'
    light_primary, light_second = '#91add2', '#efb99d'

    # Layout: (a,b), (c spanning two columns), then the four chassis responses.
    fig = plt.figure(figsize=(13.8, 14.6), constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=(1.05, 0.92, 0.88, 0.88))
    overall = fig.add_subplot(grid[0, 0])
    zoom = fig.add_subplot(grid[0, 1])
    geometry_ax = fig.add_subplot(grid[1, :])
    yaw_ax = fig.add_subplot(grid[2, 0])
    beta_ax = fig.add_subplot(grid[2, 1])
    roll_ax = fig.add_subplot(grid[3, 0])
    ltr_ax = fig.add_subplot(grid[3, 1])

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
    overall.set(title='(a) Overall trajectories', xlabel='World X (m)', ylabel='World Y (m)')
    overall.grid(alpha=0.2)
    overall.legend(handles=[
        Line2D([0], [0], color=primary, ls='--', lw=1.6, label='Baseline'),
        Line2D([0], [0], color=second, ls='-', lw=1.7, label='JTP'),
        Line2D([0], [0], color='#6b7280', lw=1.0, label='Road boundary'),
        Line2D([0], [0], color='#15803d', ls='--', lw=0.9, label='Safety inset'),
        Line2D([0], [0], color='#15803d', marker='>', lw=0, label='Motion direction'),
        Line2D([0], [0], color='#111827', marker='o', markerfacecolor='white', lw=0, label='Start'),
        Line2D([0], [0], color='#111827', marker='x', lw=0, label='Baseline end'),
        Line2D([0], [0], color='#111827', marker='s', markerfacecolor='white', lw=0, label='JTP end'),
    ], loc='upper left', fontsize=7, ncol=2, framealpha=0.92)

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
    zoom.annotate(f'Max. displacement = {max_deviation:.3f} m\nt = {focus * dt:.1f} s',
                  xy=b[focus, :2], xytext=(8, 10), textcoords='offset points',
                  fontsize=8, color='#166534',
                  bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#86a98e', alpha=0.92))
    local_xy = np.concatenate((a[local_ids, :2], b[local_ids, :2]), axis=0)
    x_range = max(float(np.ptp(local_xy[:, 0])), 0.2)
    y_range = max(float(np.ptp(local_xy[:, 1])), 0.08)
    x_margin = max(0.35, 0.10 * x_range)
    y_margin = max(0.18, 0.22 * y_range)
    zoom.set_xlim(float(local_xy[:, 0].min() - x_margin), float(local_xy[:, 0].max() + x_margin))
    zoom.set_ylim(float(local_xy[:, 1].min() - y_margin), float(local_xy[:, 1].max() + y_margin))
    zoom.set(title=f'(b) Local refinement (truck {exemplar})',
             xlabel='World X (m)', ylabel='World Y (m)')
    zoom.grid(alpha=0.22)
    zoom.legend(loc='best', fontsize=8)

    # (c) Curvature and worst footprint road-boundary residual.
    def curvature(path_xyh):
        delta_yaw = np.arctan2(
            np.sin(path_xyh[2:, 2] - path_xyh[1:-1, 2]),
            np.cos(path_xyh[2:, 2] - path_xyh[1:-1, 2]),
        )
        ds = np.linalg.norm(path_xyh[2:, :2] - path_xyh[1:-1, :2], axis=-1)
        return delta_yaw / np.maximum(ds, 1.0e-4)

    curvature_times = np.arange(2, steps) * float(dt)
    curvature_base = curvature(a)
    curvature_jtp = curvature(b)
    geometry_ax.plot(curvature_times, curvature_base, c=primary, ls='--', lw=1.5,
                     label='Baseline curvature')
    geometry_ax.plot(curvature_times, curvature_jtp, c=second, ls='-', lw=1.6,
                     label='JTP curvature')
    geometry_ax.axhline(kappa_lim, color='#333333', ls=':', lw=1.0,
                        label='Curvature limit')
    geometry_ax.axhline(-kappa_lim, color='#333333', ls=':', lw=1.0)
    geometry_ax.set_ylim(-1.55 * float(kappa_lim), 1.55 * float(kappa_lim))
    geometry_ax.set_ylabel('Curvature (1/m)')
    geometry_ax.set_xlabel('Time (s)')
    geometry_ax.set_title('(c) Geometric constraints: curvature and road-boundary residual')
    geometry_ax.grid(alpha=0.2)

    road_base = _as_array(figure.get('nominal_road_residual'),
                          name='nominal_road_residual', ndim=2)
    road_jtp = _as_array(figure.get('projected_road_residual'),
                         name='projected_road_residual', ndim=2)
    if road_base is not None and road_jtp is not None:
        if road_base.shape != (n_vehicles, expected_horizon) or road_jtp.shape != road_base.shape:
            raise ValueError('Road residual arrays must have shape [N,H]')
        road_ax = geometry_ax.twinx()
        road_times = np.arange(1, steps) * float(dt)
        road_ax.plot(road_times, road_base[exemplar], color=primary, ls=':', lw=1.15,
                     alpha=0.78, label='Baseline road residual')
        road_ax.plot(road_times, road_jtp[exemplar], color=second, ls='-.', lw=1.2,
                     alpha=0.82, label='JTP road residual')
        road_ax.axhline(0.0, color='#15803d', ls='--', lw=0.9,
                        label='Road-feasibility boundary')
        road_min = float(min(np.min(road_base[exemplar]), np.min(road_jtp[exemplar]), 0.0))
        road_max = float(max(np.max(road_base[exemplar]), np.max(road_jtp[exemplar]), 0.0))
        road_span = max(road_max - road_min, 0.05)
        road_ax.set_ylim(road_min - 0.08 * road_span, road_max + 0.22 * road_span)
        road_ax.set_ylabel('Worst road residual (m; positive = violation)')
        h1, l1 = geometry_ax.get_legend_handles_labels()
        h2, l2 = road_ax.get_legend_handles_labels()
        geometry_ax.legend(h1 + h2, l1 + l2, fontsize=7.0, ncol=3,
                           loc='upper center', bbox_to_anchor=(0.5, 0.995),
                           framealpha=0.9)
    else:
        geometry_ax.legend(fontsize=8, ncol=3, loc='upper center')
        geometry_ax.text(0.01, 0.04, 'Road residual unavailable in saved figure data',
                         transform=geometry_ax.transAxes, fontsize=8, color='#9a3412')

    # (d)-(g) Absolute chassis responses and the matching per-step limits.
    sn, sp = nominal_states[exemplar], projected_states[exemplar]
    ln, lp = nominal_ltr[exemplar], projected_ltr[exemplar]
    bn, bp = bounds_nominal[exemplar], bounds_projected[exemplar]
    beta_n = np.abs(np.arctan2(sn[1:, 1], sn[1:, 0]))
    beta_p = np.abs(np.arctan2(sp[1:, 1], sp[1:, 0]))
    panel_specs = (
        (yaw_ax, np.abs(sn[1:, 2]), np.abs(sp[1:, 2]), bn[:, 0], bp[:, 0],
         '(d) Yaw rate', '|r| (rad/s)'),
        (beta_ax, beta_n, beta_p, bn[:, 1], bp[:, 1],
         '(e) Sideslip angle', '|β| (rad)'),
        (roll_ax, np.abs(sn[1:, 3]), np.abs(sp[1:, 3]), bn[:, 2], bp[:, 2],
         '(f) Roll angle', '|φ| (rad)'),
        (ltr_ax, np.abs(ln[1:]), np.abs(lp[1:]), bn[:, 3], bp[:, 3],
         '(g) Lateral load transfer ratio', '|LTR|'),
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
        ax.legend(fontsize=7.5, ncol=2, loc='best')

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
