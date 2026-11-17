from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


EGO_LENGTH_M = 10.0
EGO_WIDTH_M = 2.4
VEHICLE_ALPHA = 0.35
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
            color="#222222", linestyle="--", linewidth=1.8,
            label="Expert trajectory", zorder=5)
    _draw_vehicle(ax, 0.0, 0.0, 0.0, EGO_LENGTH_M, EGO_WIDTH_M,
                  "#f28e2b", label="Ego")
    target = np.asarray(features["target_point"])
    ax.scatter([target[0]], [target[1]], marker="*", s=55,
               color="#d62728", label="Target", zorder=6)
    ax.set_xlabel("Longitudinal position (m)")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")


def _method_items(payload: Mapping[str, dict]):
    methods = [(method, title) for method, title in _METHODS if method in payload]
    if not methods:
        raise ValueError("No A0/A1/A2 figure payloads were supplied")
    return methods


def _finish_figure(fig, axes, path: Path, sample_label: str, title: str):
    axes[0].set_ylabel("Lateral position (m)")
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = {}
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


def draw_anchor_ablation(path: Path, figure_payload: Mapping[str, dict], *,
                         sample_label: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = _method_items(figure_payload)
    fig, grid = plt.subplots(
        1, len(methods), figsize=(5.2 * len(methods), 4.8),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = grid[0]
    for ax, (method, title) in zip(axes, methods):
        item = figure_payload[method]
        _draw_scene(ax, item)
        xy = np.asarray(item["anchors_xy"])
        valid = np.asarray(item.get("valid", np.ones(len(xy), dtype=bool)), dtype=bool)
        for idx in np.flatnonzero(valid):
            ax.plot(np.r_[0.0, xy[idx, :, 0]], np.r_[0.0, xy[idx, :, 1]],
                    linewidth=1.0, alpha=0.68,
                    label="Anchor trajectories" if idx == np.flatnonzero(valid)[0] else None)
        ax.set_title(title)
    _finish_figure(
        fig, axes, path, sample_label,
        "Anchor distributions in the same highway scene",
    )


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
        projected = np.asarray(item["selected_projected_xy"])
        ax.plot(np.r_[0.0, raw[:, 0]], np.r_[0.0, raw[:, 1]],
                color="#d62728", linewidth=2.0, linestyle=":",
                label="Selected before JTP", zorder=7)
        ax.plot(np.r_[0.0, projected[:, 0]], np.r_[0.0, projected[:, 1]],
                color="#2ca02c", linewidth=2.0,
                label="Selected after JTP", zorder=8)
        ax.set_title(title)

    _finish_figure(
        fig, axes, path, sample_label,
        "Anchor-conditioned diffusion planning and JTP correction",
    )
