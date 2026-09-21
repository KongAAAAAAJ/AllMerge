from __future__ import annotations

import copy
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class TrajectoryVisualizationSample:
    sample_index: object
    expert_trajectory: np.ndarray
    predicted_trajectory: np.ndarray
    selected_ade: float
    selected_fde: float
    target_mode: int
    selected_mode: int
    trajectory_time_s: Optional[object] = None
    expert_trajectory_10hz: Optional[object] = None
    expert_trajectory_10hz_time_s: Optional[object] = None
    features_cpu: Optional[Dict[str, torch.Tensor]] = None


# OPEN_LOOP_DENSE_VIZ_V1: visualization
def _clamped_cubic_spline_10hz(
    sparse_xy: np.ndarray,
    sparse_time_s: Optional[object],
    *,
    sample_dt_s: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate sparse planner points with a clamped cubic spline.

    The diffusion output is defined at t=[0.5, ..., 4.0] s. For execution
    visualization we prepend the current ego origin (0, 0) at t=0 and use
    first/last secants as the clamped endpoint derivatives. The spline passes
    through every sparse point and is sampled at 10 Hz.
    """
    from scipy.interpolate import CubicSpline

    xy = np.asarray(sparse_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"sparse_xy must be [T,2+], got {xy.shape}")
    xy = xy[:, :2]

    if sparse_time_s is None:
        time_s = (
            np.arange(1, xy.shape[0] + 1, dtype=np.float64)
            * 0.5
        )
    else:
        time_s = np.asarray(sparse_time_s, dtype=np.float64).reshape(-1)
        if time_s.shape[0] != xy.shape[0]:
            raise ValueError(
                "sparse trajectory/time length mismatch: "
                f"xy={xy.shape[0]} time={time_s.shape[0]}"
            )

    if xy.shape[0] < 2:
        raise ValueError("Need at least two sparse trajectory points")
    if not np.isfinite(xy).all() or not np.isfinite(time_s).all():
        raise ValueError("Spline input contains NaN/Inf")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("sparse_time_s must be strictly increasing")

    # Include the current ego pose so the executable curve starts at t=0.
    if float(time_s[0]) > 1e-8:
        time_s = np.concatenate(([0.0], time_s))
        xy = np.concatenate((np.zeros((1, 2), dtype=np.float64), xy), axis=0)

    dt_start = float(time_s[1] - time_s[0])
    dt_end = float(time_s[-1] - time_s[-2])
    start_derivative = (xy[1] - xy[0]) / dt_start
    end_derivative = (xy[-1] - xy[-2]) / dt_end

    spline = CubicSpline(
        time_s,
        xy,
        axis=0,
        bc_type=(
            (1, start_derivative),
            (1, end_derivative),
        ),
    )

    start_s = float(time_s[0])
    end_s = float(time_s[-1])
    count = int(round((end_s - start_s) / float(sample_dt_s)))
    dense_time_s = (
        start_s
        + np.arange(count + 1, dtype=np.float64) * float(sample_dt_s)
    )
    dense_time_s[-1] = end_s
    dense_xy = np.asarray(spline(dense_time_s), dtype=np.float64)
    return dense_time_s, dense_xy


def _optional_xy(value: Optional[object]) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        return None
    array = array[:, :2]
    if not np.isfinite(array).all():
        return None
    return array


def compute_open_loop_safety_proxies(
    selected_trajectory: torch.Tensor,
    features: Mapping[str, torch.Tensor],
    batch_index: int,
    *,
    collision_distance_m: float = 2.5,
    offroad_distance_m: float = 2.5,
) -> Tuple[Optional[bool], Optional[bool]]:
    """
    Lightweight open-loop geometry proxies.

    Collision proxy:
        any selected trajectory point is closer than collision_distance_m to
        the CURRENT position of any valid background agent.

    Off-road proxy:
        any selected trajectory point is farther than offroad_distance_m from
        every valid map-polyline point.

    These are deliberately named/used as proxies because W1 open-loop samples
    do not provide future background trajectories or an exact drivable-area
    polygon. Closed-loop collision/off-road metrics should remain authoritative.
    """
    trajectory = selected_trajectory.detach().float()[..., :2]

    collision: Optional[bool] = None
    agent_states = features.get("agent_states")
    agent_valid_mask = features.get("agent_valid_mask")
    if agent_valid_mask is None:
        agent_valid_mask = features.get("valid_mask")
    if (
        isinstance(agent_states, torch.Tensor)
        and isinstance(agent_valid_mask, torch.Tensor)
        and agent_states.ndim >= 3
        and agent_states.shape[-1] >= 2
    ):
        positions = agent_states[batch_index, ..., :2].reshape(-1, 2)
        valid = agent_valid_mask[batch_index].bool().reshape(-1)
        if valid.numel() == positions.shape[0]:
            positions = positions[valid]
        positions = positions[torch.isfinite(positions).all(dim=-1)]
        if positions.numel() > 0:
            distances = torch.cdist(
                trajectory.to(device=positions.device),
                positions,
            )
            collision = bool(
                (distances.min(dim=1).values < float(collision_distance_m))
                .any()
                .item()
            )
        else:
            collision = False

    offroad: Optional[bool] = None
    map_polylines = features.get("map_polylines")
    map_valid_mask = features.get("map_valid_mask")
    if (
        isinstance(map_polylines, torch.Tensor)
        and isinstance(map_valid_mask, torch.Tensor)
        and map_polylines.ndim >= 4
        and map_polylines.shape[-1] >= 2
    ):
        points = map_polylines[batch_index, ..., :2]
        valid_map = map_valid_mask[batch_index].bool()

        if valid_map.ndim == 1 and points.ndim == 3:
            if valid_map.numel() == points.shape[0]:
                points = points[valid_map].reshape(-1, 2)
            else:
                points = points.reshape(-1, 2)
        elif valid_map.shape == points.shape[:-1]:
            points = points[valid_map]
        else:
            points = points.reshape(-1, 2)

        points = points[torch.isfinite(points).all(dim=-1)]
        if points.numel() > 0:
            distances = torch.cdist(
                trajectory.to(device=points.device),
                points,
            )
            min_distance = distances.min(dim=1).values
            offroad = bool(
                (min_distance > float(offroad_distance_m)).any().item()
            )
        else:
            offroad = None

    return collision, offroad


def plot_ade_fde_boxplot(
    rows: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ade = np.asarray(
        [float(row["selected_ADE"]) for row in rows],
        dtype=np.float64,
    )
    fde = np.asarray(
        [float(row["selected_FDE"]) for row in rows],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.boxplot(
        [ade, fde],
        tick_labels=["ADE", "FDE"],
        showmeans=True,
        showfliers=True,
    )
    ax.set_ylabel("Displacement error [m]")
    ax.set_title("Selected trajectory accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.01,
        0.01,
        f"N={len(rows)}",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.75,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_random_trajectory_gallery(
    samples: Sequence[TrajectoryVisualizationSample],
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 11.0))
    flat_axes = axes.reshape(-1)

    for axis_index, ax in enumerate(flat_axes):
        if axis_index >= len(samples):
            ax.axis("off")
            continue

        sample = samples[axis_index]
        gt_sparse = np.asarray(sample.expert_trajectory, dtype=np.float64)[:, :2]
        pred_sparse = np.asarray(sample.predicted_trajectory, dtype=np.float64)[:, :2]

        points = min(len(gt_sparse), len(pred_sparse))
        gt_sparse = gt_sparse[:points]
        pred_sparse = pred_sparse[:points]

        sparse_time_s = sample.trajectory_time_s
        if sparse_time_s is not None:
            sparse_time_s = np.asarray(
                sparse_time_s,
                dtype=np.float64,
            ).reshape(-1)[:points]

        # True Polynomial GT line. This exists only in datasets recollected
        # after OPEN_LOOP_DENSE_VIZ_V1; old shards are intentionally not faked.
        gt_dense = _optional_xy(sample.expert_trajectory_10hz)
        gt_dense_time = sample.expert_trajectory_10hz_time_s
        if gt_dense is not None:
            if gt_dense_time is not None:
                gt_dense_time = np.asarray(
                    gt_dense_time,
                    dtype=np.float64,
                ).reshape(-1)
                if gt_dense_time.shape[0] != gt_dense.shape[0]:
                    gt_dense = None

        if gt_dense is not None:
            ax.plot(
                gt_dense[:, 0],
                gt_dense[:, 1],
                linewidth=2.2,
                label="GT Polynomial (10 Hz)",
            )
        else:
            ax.text(
                0.02,
                0.04,
                "GT Polynomial 10 Hz not stored\nin this dataset",
                transform=ax.transAxes,
                fontsize=7.5,
                alpha=0.68,
                va="bottom",
            )

        # Predicted execution curve: clamped cubic interpolation through all
        # sparse diffusion outputs, sampled at the same 10 Hz visualization rate.
        _, pred_spline = _clamped_cubic_spline_10hz(
            pred_sparse,
            sparse_time_s,
            sample_dt_s=0.1,
        )
        ax.plot(
            pred_spline[:, 0],
            pred_spline[:, 1],
            linewidth=1.9,
            label="Clamped cubic spline (10 Hz)",
        )

        # Sparse points are MARKERS ONLY: never connect them directly.
        ax.scatter(
            gt_sparse[:, 0],
            gt_sparse[:, 1],
            s=16,
            marker="o",
            label="GT sparse (0.5 s)",
            zorder=3,
        )
        ax.scatter(
            pred_sparse[:, 0],
            pred_sparse[:, 1],
            s=23,
            marker="x",
            label="Prediction sparse (0.5 s)",
            zorder=4,
        )
        ax.scatter(
            [0.0],
            [0.0],
            s=20,
            marker="s",
            label="Ego",
            zorder=5,
        )

        ax.set_title(
            f"sample={sample.sample_index}  "
            f"ADE={sample.selected_ade:.2f} m  "
            f"FDE={sample.selected_fde:.2f} m",
            fontsize=9.5,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.margins(y=0.06)
        ax.set_aspect("auto")
        ax.grid(alpha=0.22)
        if axis_index == 0:
            ax.legend(fontsize=7.5, loc="best")

    fig.suptitle(
        "Random 9-sample: sparse predictions and 10 Hz trajectories",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _fmt_metric(value: Optional[float], *, percentage: bool = False) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(value):
        return "N/A"
    if percentage:
        return f"{100.0 * value:.2f}%"
    return f"{value:.3f}"


def plot_metrics_table(
    *,
    min_ade: float,
    min_fde: float,
    miss_rate: float,
    collision_rate: Optional[float],
    offroad_rate: Optional[float],
    miss_threshold_m: float,
    collision_distance_m: float,
    offroad_distance_m: float,
    output_path: str | Path,
    output_csv: Optional[str | Path] = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "minADE [m] ↓",
        "minFDE [m] ↓",
        f"Miss Rate@{miss_threshold_m:g}m ↓",
        "Collision Rate* ↓",
        "Off-road Rate* ↓",
    ]
    values = [
        _fmt_metric(min_ade),
        _fmt_metric(min_fde),
        _fmt_metric(miss_rate, percentage=True),
        _fmt_metric(collision_rate, percentage=True),
        _fmt_metric(offroad_rate, percentage=True),
    ]

    fig, ax = plt.subplots(figsize=(12.6, 2.7))
    ax.axis("off")
    table = ax.table(
        cellText=[values],
        colLabels=headers,
        rowLabels=["Diffusion"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.65)
    ax.set_title("Open-loop trajectory evaluation summary", pad=16)
    ax.text(
        0.0,
        -0.06,
        (
            "* Collision/off-road are open-loop geometry proxies: "
            f"agent distance < {collision_distance_m:g} m; "
            f"map-polyline distance > {offroad_distance_m:g} m. "
            "Use closed-loop simulation metrics as the authoritative safety result."
        ),
        transform=ax.transAxes,
        fontsize=8.5,
        alpha=0.78,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["minADE", "minFDE", "Miss Rate", "collision rate", "off-road rate"]
            )
            writer.writerow(
                [
                    f"{float(min_ade):.6f}",
                    f"{float(min_fde):.6f}",
                    f"{float(miss_rate):.6f}",
                    "" if collision_rate is None else f"{float(collision_rate):.6f}",
                    "" if offroad_rate is None else f"{float(offroad_rate):.6f}",
                ]
            )

    return output_path


def _extract_checkpoint_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, dict):
        if isinstance(payload, Mapping):
            return payload
        raise RuntimeError("Unsupported checkpoint payload")
    if "planner_state_dict" in payload:
        return payload["planner_state_dict"]
    if "state_dict" in payload:
        return payload["state_dict"]
    return payload


def plot_initial_vs_final_denoising(
    *,
    final_model: torch.nn.Module,
    initial_checkpoint: str | Path,
    sample: TrajectoryVisualizationSample,
    output_path: str | Path,
    seed: int = 0,
    strict_checkpoint: bool = False,
) -> Path:
    """
    Compare the same sample, same Gaussian noise and same GT-assigned mode.

    This intentionally removes mode-selection differences from the comparison:
    both panels visualize the candidate at sample.target_mode.
    """
    import matplotlib.pyplot as plt

    if sample.features_cpu is None:
        raise ValueError("Denoising comparison requires features_cpu")

    initial_checkpoint = Path(initial_checkpoint).expanduser().resolve()
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(
            f"Initial checkpoint does not exist: {initial_checkpoint}"
        )

    final_model = final_model.to("cpu").eval()
    initial_model = copy.deepcopy(final_model).to("cpu").eval()

    payload = torch.load(initial_checkpoint, map_location="cpu")
    state_dict = _extract_checkpoint_state_dict(payload)
    incompatible = initial_model.load_state_dict(
        state_dict,
        strict=bool(strict_checkpoint),
    )
    if not strict_checkpoint and (
        incompatible.missing_keys or incompatible.unexpected_keys
    ):
        print(
            "[evaluation/visualization] initial checkpoint loaded non-strictly; "
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    features = {
        key: value.to("cpu")
        for key, value in sample.features_cpu.items()
    }

    def infer(model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        with torch.no_grad():
            return model.infer_multimodal(
                features,
                generator=generator,
            )

    initial_output = infer(initial_model)
    final_output = infer(final_model)

    target_mode = int(sample.target_mode)
    gt = np.asarray(sample.expert_trajectory, dtype=np.float64)

    panels = []
    for name, output in [
        ("Training start", initial_output),
        ("Training end", final_output),
    ]:
        if "trajectory_noisy_initial" not in output:
            raise RuntimeError(
                "Model output lacks trajectory_noisy_initial. "
                "Apply the structured_model.py visualization patch first."
            )
        noisy = (
            output["trajectory_noisy_initial"][0, target_mode]
            .detach()
            .cpu()
            .numpy()
        )
        pred = (
            output["trajectory_candidates"][0, target_mode]
            .detach()
            .cpu()
            .numpy()
        )
        ade = float(
            np.linalg.norm(pred[..., :2] - gt[..., :2], axis=-1).mean()
        )
        fde = float(
            np.linalg.norm(pred[-1, :2] - gt[-1, :2])
        )
        panels.append((name, noisy, pred, ade, fde))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for ax, (name, noisy, pred, ade, fde) in zip(axes, panels):
        ax.plot(
            noisy[:, 0],
            noisy[:, 1],
            marker=".",
            linewidth=1.2,
            alpha=0.7,
            label="Noisy trajectory",
        )
        ax.plot(
            pred[:, 0],
            pred[:, 1],
            marker="x",
            linewidth=2.0,
            label="Predicted trajectory",
        )
        ax.plot(
            gt[:, 0],
            gt[:, 1],
            marker="o",
            markersize=3.0,
            linewidth=2.0,
            label="Expert",
        )
        ax.scatter([0.0], [0.0], s=20, marker="s", label="Ego")
        ax.set_title(
            f"{name}\nGT mode={target_mode}, ADE={ade:.2f} m, FDE={fde:.2f} m"
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(alpha=0.22)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Same sample + same noise: denoising before vs after training",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    del initial_model
    return output_path
