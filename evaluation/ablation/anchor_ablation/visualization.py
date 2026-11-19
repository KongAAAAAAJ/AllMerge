from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


EGO_LENGTH_M = 10.0
EGO_WIDTH_M = 2.4
VEHICLE_ALPHA = 0.35
EXPERT_LINEWIDTH = 0.9
UNSAFE_ANCHOR_COLOR = '#d62728'
UNSAFE_ANCHOR_ALPHA = 0.45
METHOD_LABEL_POSITION = (0.025, 0.94)  # Axes fractions: left, bottom.
METHOD_LABEL_ALPHA = 0.55
METHOD_LABEL_COLORS = {'A0': '#EAD8C4', 'A1': '#CCDDE8', 'A2': '#D4E5D5'}
ANCHOR_INSET_DEFAULTS = {
    'enabled': True,
    'bounds': (0.055, 0.15, 0.48, 0.48),  # left, bottom, width, height; axes fractions
}
# Zero-based scene rows. None selects the shared A1/A2 trajectory range automatically.
ANCHOR_INSET_ROW_LIMITS = {
    0: {'xlim': None, 'ylim': None},
    1: {'xlim': None, 'ylim': None},
    2: {'xlim': None, 'ylim': None},
}
# Example: (0, 'A1'): {'bounds': (0.03, 0.08, 0.50, 0.50)}
ANCHOR_INSET_OVERRIDES = {}
_METHODS = (("A0", "Fixed Anchors"), ("A1", "Dynamic Anchors"), ("A2", "Ours"))


def _draw_vehicle(ax, x, y, heading, length, width, color, label=None):
    from matplotlib.patches import Polygon

    corners = np.array([
        [-length / 2, -width / 2], [length / 2, -width / 2],
        [length / 2, width / 2], [-length / 2, width / 2],
    ])
    c, s = np.cos(heading), np.sin(heading)
    corners = corners @ np.array([[c, s], [-s, c]]) + [x, y]
    ax.add_patch(Polygon(
        corners, closed=True, facecolor=color, edgecolor=color,
        linewidth=0.8, alpha=VEHICLE_ALPHA, label=label, zorder=3,
    ))


def _draw_scene(ax, item):
    features = item["features"]
    maps = np.asarray(features["map_polylines"])
    map_mask = np.asarray(features["map_valid_mask"], dtype=bool)
    for lane in maps[map_mask]:
        ax.plot(lane[:, 0], lane[:, 1], color="0.55", linestyle="--",
                linewidth=0.8, alpha=0.55, zorder=0)

    agents = np.asarray(features["agent_states"])
    agent_mask = np.asarray(features["agent_valid_mask"], dtype=bool)
    other_label_used = False
    for agent in agents[agent_mask]:
        heading = np.arctan2(agent[6], agent[7])
        length = float(agent[8]) if agent[8] > 0 else 5.0
        width = float(agent[9]) if agent[9] > 0 else 2.0
        _draw_vehicle(
            ax, float(agent[0]), float(agent[1]), float(heading),
            length, width, "#4c78a8",
            label=None if other_label_used else "Other vehicles",
        )
        other_label_used = True

    expert = np.asarray(item["expert_xy"])
    ax.plot(np.r_[0.0, expert[:, 0]], np.r_[0.0, expert[:, 1]],
            color="#222222", linestyle="--", linewidth=EXPERT_LINEWIDTH,
            label="Expert trajectory", zorder=5)
    _draw_vehicle(ax, 0.0, 0.0, 0.0, EGO_LENGTH_M, EGO_WIDTH_M,
                  "#f28e2b", label="Ego")
    ax.set_xlabel("Longitudinal position (m)")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")


def _draw_anchor_curves(ax, item, method):
    import matplotlib.pyplot as plt

    xy = np.asarray(item['anchors_xy'])
    valid = np.asarray(item.get('valid', np.ones(len(xy), dtype=bool)), dtype=bool)
    hard_safe = np.ones(len(xy), dtype=bool)
    if method in ('A0', 'A1'):
        if 'hard_safe' not in item:
            raise ValueError(f'{method} figure requires per-anchor hard_safe labels')
        hard_safe = np.asarray(item['hard_safe'], dtype=bool)
        if hard_safe.shape != (len(xy),):
            raise ValueError(f'{method} hard_safe labels must match the anchor count')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    seen = set()
    for index in np.flatnonzero(valid):
        trajectory = xy[index]
        unsafe = not hard_safe[index]
        label = 'Unsafe anchors (A0/A1)' if unsafe else 'Anchor trajectories'
        ax.plot(np.r_[0.0, trajectory[:, 0]], np.r_[0.0, trajectory[:, 1]],
                color=UNSAFE_ANCHOR_COLOR if unsafe else colors[index % len(colors)],
                linestyle='--' if unsafe else '-', linewidth=1.0,
                alpha=UNSAFE_ANCHOR_ALPHA if unsafe else 0.68,
                label=label if label not in seen else None)
        seen.add(label)


def _anchor_inset_limits(scene, row):
    trajectories = []
    for method in ('A1', 'A2'):
        if method in scene:
            item = scene[method]
            xy = np.asarray(item['anchors_xy'])
            valid = np.asarray(item.get('valid', np.ones(len(xy), dtype=bool)), dtype=bool)
            trajectories.extend(xy[valid])
            trajectories.append(np.asarray(item['expert_xy']))
    if not trajectories:
        return None
    points = np.concatenate(trajectories)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return None
    settings = ANCHOR_INSET_ROW_LIMITS.get(row, {})
    forward = max(float(points[:, 0].max()), 1.0)
    xlim = settings.get('xlim') or (0.45 * forward, 0.85 * forward)
    section = points[(points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1])]
    visible_y = [section[:, 1]]
    # Include segment intersections with both window edges so coarse 8-point
    # paths are not clipped between their sampled nodes.
    for trajectory in trajectories:
        start, end = trajectory[:-1], trajectory[1:]
        dx = end[:, 0] - start[:, 0]
        for boundary in xlim:
            crosses = ((start[:, 0] - boundary) * (end[:, 0] - boundary) <= 0) & (np.abs(dx) > 1e-8)
            visible_y.append(start[crosses, 1] + (boundary - start[crosses, 0])
                             / dx[crosses] * (end[crosses, 1] - start[crosses, 1]))
    visible_y = np.concatenate(visible_y)
    visible_y = visible_y[np.isfinite(visible_y)]
    if not len(visible_y):
        visible_y = points[:, 1]
    lo, hi = visible_y.min(), visible_y.max()
    padding = max(0.8, 0.15 * float(hi - lo))
    ylim = settings.get('ylim') or (float(lo - padding), float(hi + padding))
    if xlim[1] <= xlim[0] or ylim[1] <= ylim[0]:
        raise ValueError(f'Inset limits for row {row} must be increasing')
    return xlim, ylim


def _add_anchor_insets(grid, scenes, methods):
    from matplotlib.patches import ConnectionPatch, Rectangle
    from matplotlib.ticker import MaxNLocator

    for row, scene in enumerate(scenes):
        limits = _anchor_inset_limits(scene, row)
        if limits is None:
            continue
        for col, (method, _) in enumerate(methods):
            if method not in ('A1', 'A2'):
                continue
            settings = {**ANCHOR_INSET_DEFAULTS, **ANCHOR_INSET_OVERRIDES.get((row, method), {})}
            if not settings['enabled']:
                continue
            ax = grid[row, col]
            original_xlim, original_ylim = ax.get_xlim(), ax.get_ylim()
            inset = ax.inset_axes(settings['bounds'], zorder=9)
            _draw_scene(inset, scene[method])
            _draw_anchor_curves(inset, scene[method], method)
            inset.set_aspect('auto')
            inset.set_xlim(limits[0])
            inset.set_ylim(limits[1])
            inset.set_xlabel('')
            inset.set_ylabel('')
            inset.tick_params(labelsize=6, length=2, pad=1)
            inset.xaxis.set_major_locator(MaxNLocator(nbins=3))
            inset.yaxis.set_major_locator(MaxNLocator(nbins=3))
            for spine in inset.spines.values():
                spine.set_linewidth(0.6)
            (xlo, xhi), (ylo, yhi) = limits
            style = {'edgecolor': '#777777', 'linewidth': 0.45, 'alpha': 0.45}
            ax.add_patch(Rectangle((xlo, ylo), xhi-xlo, yhi-ylo, fill=False, **style))
            for y, corner in ((ylo, (1, 0)), (yhi, (1, 1))):
                ax.add_artist(ConnectionPatch(
                    xyA=(xlo, y), coordsA=ax.transData, axesA=ax,
                    xyB=corner, coordsB=inset.transAxes, axesB=inset,
                    color='#777777', linewidth=0.45, alpha=0.45, zorder=8))
            ax.set_xlim(original_xlim)
            ax.set_ylim(original_ylim)


def _method_items(payload: Mapping[str, dict]):
    methods = [(method, title) for method, title in _METHODS if method in payload]
    if not methods:
        raise ValueError("No A0/A1/A2 figure payloads were supplied")
    return methods


def _finish_figure(fig, axes, path: Path, sample_label: str, title: str):
    axes[0].set_ylabel("Lateral position (m)")
    unique = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label not in unique:
                unique[label] = handle
    if unique:
        fig.legend(unique.values(), unique.keys(), loc="lower center",
                   ncol=min(6, len(unique)), frameon=False, fontsize=8)
    if sample_label:
        fig.suptitle(f"{title} ({sample_label})")
    else:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _finish_anchor_grid(fig, grid, path, sample_label):
    """Size the canvas around equal-scale road panels, using physical inch gaps."""
    rows, cols = grid.shape
    panel_width = 4.05
    column_gap = 0.12
    row_gap = 0.44
    left, right, top, bottom = 0.58, 0.12, 0.35, 0.62
    fig.canvas.draw()  # Resolve shared limits before measuring each row's aspect.
    heights = []
    for axes in grid:
        xlim, ylim = axes[0].get_xlim(), axes[0].get_ylim()
        ratio = abs((ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))
        heights.append(panel_width * ratio)
    width = left + cols * panel_width + (cols - 1) * column_gap + right
    height = top + sum(heights) + (rows - 1) * row_gap + bottom
    fig.set_size_inches(width, height, forward=True)
    y_top = height - top
    for row, axes in enumerate(grid):
        y_bottom = y_top - heights[row]
        for col, ax in enumerate(axes):
            ax.set_position([(left + col * (panel_width + column_gap)) / width,
                             y_bottom / height, panel_width / width,
                             heights[row] / height])
            ax.tick_params(labelsize=8, pad=2, labelbottom=col == 0, labelleft=col == 0)
            ax.set_xlabel('')
            ax.set_ylabel('')
        y_top = y_bottom - row_gap
    fig.supxlabel('Longitudinal position (m)', x=0.5, y=0.23 / height, fontsize=9, va='bottom')
    fig.supylabel('Lateral position (m)', x=0.05 / width, fontsize=9)
    unique = {}
    for ax in grid.ravel():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
    fig.legend(unique.values(), unique.keys(), loc='lower center',
               bbox_to_anchor=(0.5, 0.025 / height), ncol=min(6, len(unique)),
               frameon=False, fontsize=8, borderaxespad=0,
               columnspacing=1.3, handletextpad=0.5)
    from matplotlib.text import Text
    import matplotlib
    for text in fig.findobj(Text):
        text.set_fontfamily('Times New Roman')
    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig.savefig(path, bbox_inches='tight', pad_inches=0.05)
    import matplotlib.pyplot as plt
    plt.close(fig)


def draw_anchor_ablation(path: Path, figure_payload, *, sample_label: str = "",
                         scene_labels=None):
    """Rows compare independently sampled scenes; columns compare methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    scenes = [figure_payload] if isinstance(figure_payload, Mapping) else list(figure_payload)
    if not scenes:
        raise ValueError('No anchor scenes supplied')
    methods = _method_items(scenes[0])
    if any(_method_items(scene) != methods for scene in scenes):
        raise ValueError('All scenes must contain the same methods')
    if scene_labels is not None and len(scene_labels) != len(scenes):
        raise ValueError('Each scene must have one label')
    fig, grid = plt.subplots(
        len(scenes), len(methods), figsize=(5.2 * len(methods), 3.0 * len(scenes) + 0.8),
        sharex='row', sharey='row', squeeze=False,
    )
    for row, scene in enumerate(scenes):
        for col, (method, title) in enumerate(methods):
            ax = grid[row, col]
            item = scene[method]
            _draw_scene(ax, item)
            _draw_anchor_curves(ax, item, method)
            ax.text(*METHOD_LABEL_POSITION, title, transform=ax.transAxes,
                    ha='left', va='top', fontsize=9, zorder=10,
                    bbox={'boxstyle': 'square,pad=0.30',
                          'facecolor': to_rgba(METHOD_LABEL_COLORS[method], METHOD_LABEL_ALPHA),
                          'edgecolor': '#555555', 'linewidth': 0.65})
            label = scene_labels[row] if scene_labels is not None else f'Scene {row + 1}'
            ax.set_title(label if col == 0 else '', fontsize=10)
    _add_anchor_insets(grid, scenes, methods)
    _finish_anchor_grid(fig, grid, path, sample_label)



def draw_anchor_distribution(path: Path, payload: Mapping[str, dict], sample_label=""):
    """Compatibility alias used by the paper-aligned evaluator."""
    return draw_anchor_ablation(path, payload, sample_label=sample_label)


def draw_diffusion_planning(path: Path, figure_payload: Mapping[str, dict], *,
                            sample_label: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = _method_items(figure_payload)
    fig, grid = plt.subplots(
        1, len(methods), figsize=(5.4 * len(methods), 5.0),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = grid[0]
    for ax, (method, title) in zip(axes, methods):
        item = figure_payload[method]
        _draw_scene(ax, item)

        anchors = np.asarray(item["anchors_xy"])
        anchor_valid = np.asarray(item.get("valid", np.ones(len(anchors), dtype=bool)), dtype=bool)
        first_anchor = True
        for trajectory in anchors[anchor_valid]:
            ax.plot(np.r_[0.0, trajectory[:, 0]], np.r_[0.0, trajectory[:, 1]],
                    color="#4c78a8", linewidth=0.8, alpha=0.28,
                    label="Anchors" if first_anchor else None)
            first_anchor = False

        candidates = np.asarray(item["diffusion_candidates"])
        candidate_valid = np.asarray(
            item.get("diffusion_valid", np.ones(len(candidates), dtype=bool)), dtype=bool
        )
        first_candidate = True
        for trajectory in candidates[candidate_valid]:
            ax.plot(np.r_[0.0, trajectory[:, 0]], np.r_[0.0, trajectory[:, 1]],
                    color="#f2a541", linewidth=0.85, alpha=0.36,
                    label="Diffusion candidates" if first_candidate else None)
            first_candidate = False

        raw = np.asarray(item["selected_raw_xy"])
        ax.plot(np.r_[0.0, raw[:, 0]], np.r_[0.0, raw[:, 1]],
                color="#d62728", linewidth=2.0, linestyle=":",
                label="Selected diffusion trajectory", zorder=7)
        ax.set_title(title)

    _finish_figure(
        fig, axes, path, sample_label,
        "Anchor-conditioned diffusion planning before projection",
    )
