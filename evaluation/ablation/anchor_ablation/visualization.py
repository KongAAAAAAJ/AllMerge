from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def draw_anchor_ablation(path: Path, figure_payload: Mapping[str, dict], *, sample_label: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = (("A0", "Fixed Anchors"), ("A1", "Dynamic Anchors"), ("A2", "Ours"))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
    for ax, (method, title) in zip(axes, methods):
        item = figure_payload[method]
        features = item["features"]
        maps = np.asarray(features["map_polylines"])
        map_mask = np.asarray(features["map_valid_mask"], dtype=bool)
        for lane in maps[map_mask]:
            ax.plot(lane[:, 0], lane[:, 1], linestyle="--", linewidth=0.8, alpha=0.45)
        agents = np.asarray(features["agent_states"])
        agent_mask = np.asarray(features["agent_valid_mask"], dtype=bool)
        if agent_mask.any():
            ax.scatter(agents[agent_mask, 0], agents[agent_mask, 1], marker="x", s=20, alpha=0.65)
        gt = np.asarray(item["expert_xy"])
        ax.plot(np.r_[0.0, gt[:, 0]], np.r_[0.0, gt[:, 1]], linestyle="--", linewidth=1.8, label="Expert")
        xy = np.asarray(item["anchors_xy"])
        valid = np.asarray(item["valid"], dtype=bool)
        for idx in np.flatnonzero(valid):
            ax.plot(np.r_[0.0, xy[idx, :, 0]], np.r_[0.0, xy[idx, :, 1]], linewidth=1.0, alpha=0.65)
        target = np.asarray(features["target_point"])
        ax.scatter([0.0], [0.0], marker="o", s=28, label="Ego")
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
