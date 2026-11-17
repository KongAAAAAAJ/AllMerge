from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


# Ego dimensions are not included in ego_state; set the drawing size here (m).
EGO_LENGTH_M = 10.0
EGO_WIDTH_M = 2.4
VEHICLE_ALPHA = 0.35


def _draw_vehicle(ax, x, y, heading, length, width, color, label=None):
    from matplotlib.patches import Polygon

    corners = np.array([[-length / 2, -width / 2], [length / 2, -width / 2],
                        [length / 2, width / 2], [-length / 2, width / 2]])
    c, s = np.cos(heading), np.sin(heading)
    corners = corners @ np.array([[c, s], [-s, c]]) + [x, y]
    ax.add_patch(Polygon(corners, closed=True, facecolor=color, edgecolor=color,
                         linewidth=0.8, alpha=VEHICLE_ALPHA, label=label, zorder=3))


def draw_anchor_ablation(path: Path, figure_payload: Mapping[str, dict], *, sample_label: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [(m, title) for m, title in (("A0", "Fixed Anchors"), ("A1", "Dynamic Anchors"), ("A2", "Ours")) if m in figure_payload]
    fig, grid = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.8), sharex=True, sharey=True, squeeze=False)
    axes = grid[0]
    for ax, (method, title) in zip(axes, methods):
        item = figure_payload[method]
        features = item["features"]
        maps = np.asarray(features["map_polylines"])
        map_mask = np.asarray(features["map_valid_mask"], dtype=bool)
        for lane in maps[map_mask]:
            ax.plot(lane[:, 0], lane[:, 1], linestyle="--", linewidth=0.8, alpha=0.45)
        agents = np.asarray(features["agent_states"])
        agent_mask = np.asarray(features["agent_valid_mask"], dtype=bool)
        for index, agent in enumerate(agents[agent_mask]):
            heading = np.arctan2(agent[6], agent[7])
            length = float(agent[8]) if agent[8] > 0 else 5.0
            width = float(agent[9]) if agent[9] > 0 else 2.0
            _draw_vehicle(ax, agent[0], agent[1], heading, length, width,
                          '#1f77b4', label='Other vehicles' if index == 0 else None)
        gt = np.asarray(item["expert_xy"])
        ax.plot(np.r_[0.0, gt[:, 0]], np.r_[0.0, gt[:, 1]], linestyle="--", linewidth=1.8, label="Expert")
        xy = np.asarray(item["anchors_xy"])
        valid = np.asarray(item["valid"], dtype=bool)
        for idx in np.flatnonzero(valid):
            ax.plot(np.r_[0.0, xy[idx, :, 0]], np.r_[0.0, xy[idx, :, 1]], linewidth=1.0, alpha=0.65)
        target = np.asarray(features["target_point"])
        _draw_vehicle(ax, 0.0, 0.0, 0.0, EGO_LENGTH_M, EGO_WIDTH_M,
                      '#ff7f0e', label='Ego')
        ax.scatter([target[0]], [target[1]], marker="*", s=55, label="Target")
        ax.set_title(title)
        ax.set_xlabel("Longitudinal position (m)")
        ax.grid(True, linewidth=0.35, alpha=0.35)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Lateral position (m)")
    axes[-1].legend(loc="best", fontsize=7)
    if sample_label:
        fig.suptitle(f"Anchor distributions in the same highway scene ({sample_label})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
