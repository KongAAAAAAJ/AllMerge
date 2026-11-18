from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

A1_REQUIRED = (
    "ego_state",
    "road_lane_geometry",
    "road_lane_point_mask",
    "road_lane_mask",
    "road_current_lane_slot",
    "road_target_lane_slot",
    "road_lane_meta",
    "road_lane_connections",
    "road_lane_index",
    "episode_index",
    "frame_index",
    "ego_index",
    "a1_coarse_controls",
    "a1_coarse_trajectories",
    "a1_dense_trajectories",
    "a1_valid_mask",
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _scalar(value):
    arr = np.asarray(value)
    return arr.reshape(-1)[0].item() if arr.size else None


def _read_manifest(dataset_root: Path) -> dict:
    path = dataset_root / "reports" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"A1 dataset manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError(f"A1 dataset is not complete: {path}")
    if manifest.get("schema_version") != "trace_expert_road_geometry_a1_k2_v2":
        raise RuntimeError(
            "Expected a completed TRACE A1/K2 dataset "
            f"(schema={manifest.get('schema_version')!r})"
        )
    return manifest


def _find_episode_shard(dataset_root: Path, episode: int) -> Path:
    direct = dataset_root / "shards" / f"shard_{episode:06d}.npz"
    if direct.is_file():
        return direct

    shards = sorted((dataset_root / "shards").glob("shard_*.npz"))
    for path in shards:
        with np.load(path, allow_pickle=False) as data:
            if "episode_index" in data.files:
                ids = np.asarray(data["episode_index"]).reshape(-1).astype(np.int64)
                if np.any(ids == episode):
                    return path
    raise FileNotFoundError(
        f"Cannot find A1 shard for episode={episode} under {dataset_root / 'shards'}"
    )


def load_sample(dataset_root: Path, episode: int, frame: int, ego: int):
    manifest = _read_manifest(dataset_root)
    shard = _find_episode_shard(dataset_root, episode)
    with np.load(shard, allow_pickle=False) as data:
        missing = [key for key in A1_REQUIRED if key not in data.files]
        if missing:
            raise KeyError(f"{shard.name} is missing required fields: {missing}")

        ep = np.asarray(data["episode_index"]).reshape(-1).astype(np.int64)
        fr = np.asarray(data["frame_index"]).reshape(-1).astype(np.int64)
        eg = np.asarray(data["ego_index"]).reshape(-1).astype(np.int64)
        matches = np.flatnonzero((ep == episode) & (fr == frame) & (eg == ego))
        if matches.size != 1:
            available = sorted(set(zip(fr[ep == episode].tolist(), eg[ep == episode].tolist())))
            if matches.size == 0:
                raise RuntimeError(
                    f"Sample not found: episode={episode}, frame={frame}, ego={ego}. "
                    f"Available (frame,ego) pairs for this episode include {available[:30]}"
                )
            raise RuntimeError(
                f"Sample key is not unique: episode={episode}, frame={frame}, ego={ego}, "
                f"matches={matches.size}"
            )
        idx = int(matches[0])
        row = {key: np.array(data[key][idx], copy=True) for key in A1_REQUIRED}
        # Additional fields are optional and used only for context/plotting.
        for key in ("scenario", "expert_trajectory_xy", "episode_step", "koopman_payload_kg", "koopman_road_mu"):
            if key in data.files:
                row[key] = np.array(data[key][idx], copy=True)
    return manifest, shard, idx, row


def load_a0_bank(path: Path) -> tuple[np.ndarray, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"A0 control bank not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        for key in ("anchors_controls", "metadata_json"):
            if key not in data.files:
                raise KeyError(f"A0 bank is missing {key!r}: {path}")
        controls = np.asarray(data["anchors_controls"], dtype=np.float32)
        metadata_text = str(np.asarray(data["metadata_json"]).item())
    if controls.ndim != 3 or controls.shape[1:] != (40, 2):
        raise ValueError(f"A0 anchors_controls must be [K,40,2], got {controls.shape}")
    metadata = json.loads(metadata_text)
    return controls, metadata


def load_diffusion_dynamics(
    path: Path, fallback_dt: float, fallback_wheelbase: float, fallback_max_steer: float
):
    if not path.is_file():
        return fallback_dt, fallback_wheelbase, fallback_max_steer, "A1 candidate config fallback"
    try:
        import yaml
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        control = cfg.get("control") or {}
        dynamics = control.get("dynamics") or {}
        dt = float(dynamics.get("dt", fallback_dt))
        wheelbase = float(dynamics.get("wheelbase_m", fallback_wheelbase))
        max_steer = float(dynamics.get("max_steer_rad", fallback_max_steer))
        return dt, wheelbase, max_steer, str(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read diffusion dynamics from {path}: {exc}") from exc


def rollout_controls(controls: np.ndarray, initial_speed: float, dt: float, wheelbase: float):
    """Use the project's exact differentiable bicycle rollout implementation."""
    import torch
    from highway_env.planner.diffusion.common.kinematics import rollout_bicycle

    k = int(controls.shape[0])
    u = torch.as_tensor(controls, dtype=torch.float32, device="cpu")
    v0 = torch.full((k,), max(float(initial_speed), 0.0), dtype=torch.float32)
    with torch.no_grad():
        poses, speeds = rollout_bicycle(u, v0, dt=float(dt), wheelbase_m=float(wheelbase))
    return poses.cpu().numpy().astype(np.float32), speeds.cpu().numpy().astype(np.float32)


def _lane_polylines(row):
    geometry = np.asarray(row["road_lane_geometry"], dtype=np.float32)
    point_mask = np.asarray(row["road_lane_point_mask"], dtype=bool)
    lane_mask = np.asarray(row["road_lane_mask"], dtype=bool)
    lane_ids = np.asarray(row["road_lane_index"])
    lanes = []
    for lane_idx in np.flatnonzero(lane_mask):
        keep = point_mask[lane_idx]
        g = geometry[lane_idx, keep]
        if len(g) < 2:
            continue
        lanes.append({
            "index": int(lane_idx),
            "id": "/".join(str(x) for x in np.asarray(lane_ids[lane_idx]).reshape(-1) if str(x)),
            "geometry": g,
            "center": g[:, 0:2],
            "left": g[:, 2:4],
            "right": g[:, 4:6],
            "width": g[:, 7],
        })
    if not lanes:
        raise RuntimeError("No valid road lane polylines found in road_lane_geometry")
    return lanes


def project_to_lane(point: np.ndarray, lane: dict):
    """Closest segment projection; signed cross-track is positive to the left."""
    line = np.asarray(lane["center"], dtype=np.float64)
    if len(line) < 2:
        return float("inf"), float("nan"), float("nan")
    p0 = line[:-1]
    vec = line[1:] - p0
    norm2 = np.sum(vec * vec, axis=1)
    valid = norm2 > 1e-10
    if not np.any(valid):
        return float("inf"), float("nan"), float("nan")
    p0 = p0[valid]
    vec = vec[valid]
    norm2 = norm2[valid]
    alpha = np.clip(np.sum((point[None, :] - p0) * vec, axis=1) / norm2, 0.0, 1.0)
    proj = p0 + alpha[:, None] * vec
    residual = point[None, :] - proj
    euclidean = np.linalg.norm(residual, axis=1)
    j = int(np.argmin(euclidean))
    length = math.sqrt(float(norm2[j]))
    signed_cte = float((vec[j, 0] * residual[j, 1] - vec[j, 1] * residual[j, 0]) / max(length, 1e-9))

    # Width is linearly interpolated on the original segment. Recover the
    # corresponding original segment index after dropping degenerate segments.
    original_valid = np.flatnonzero(np.sum(np.diff(line, axis=0) ** 2, axis=1) > 1e-10)
    original_j = int(original_valid[j])
    widths = np.asarray(lane["width"], dtype=np.float64)
    width = float((1.0 - alpha[j]) * widths[original_j] + alpha[j] * widths[original_j + 1])
    return float(euclidean[j]), signed_cte, width


def evaluate_road(trajectories: np.ndarray, lanes: list[dict]):
    """Evaluate center-point containment in the union of captured lane strips."""
    k, steps = trajectories.shape[:2]
    nearest_lane = np.full((k, steps), -1, dtype=np.int32)
    nearest_cte = np.full((k, steps), np.nan, dtype=np.float32)
    nearest_width = np.full((k, steps), np.nan, dtype=np.float32)
    strip_margin = np.full((k, steps), -np.inf, dtype=np.float32)
    inside_any = np.zeros((k, steps), dtype=bool)

    for m in range(k):
        for t in range(steps):
            point = trajectories[m, t]
            best_euclidean = float("inf")
            best_cte = float("nan")
            best_width = float("nan")
            best_lane = -1
            best_margin = -float("inf")
            for lane in lanes:
                euclidean, cte, width = project_to_lane(point, lane)
                if not np.isfinite(cte) or not np.isfinite(width) or width <= 0:
                    continue
                margin = 0.5 * width - abs(cte)
                if euclidean < best_euclidean:
                    best_euclidean = euclidean
                    best_cte = cte
                    best_width = width
                    best_lane = lane["index"]
                best_margin = max(best_margin, margin)
            nearest_lane[m, t] = best_lane
            nearest_cte[m, t] = best_cte
            nearest_width[m, t] = best_width
            strip_margin[m, t] = best_margin
            inside_any[m, t] = best_margin >= -0.05

    return {
        "nearest_lane_index": nearest_lane,
        "nearest_lane_cte_m": nearest_cte,
        "nearest_lane_width_m": nearest_width,
        "lane_strip_margin_m": strip_margin,
        "inside_any_lane_strip": inside_any,
    }


def steering_metrics(method: str, controls: np.ndarray, poses: np.ndarray,
                     speeds: np.ndarray, road_eval: dict, dt: float,
                     wheelbase: float, max_steer: float):
    rows = []
    per_step = []
    k = controls.shape[0]
    for m in range(k):
        accel = controls[m, :, 0].astype(np.float64)
        steer = controls[m, :, 1].astype(np.float64)
        steer_rate = np.diff(steer) / float(dt)
        speed_mid = 0.5 * (speeds[m, :-1] + speeds[m, 1:])
        yaw_rate = speed_mid * np.tan(steer) / float(wheelbase)
        lateral_accel = speed_mid ** 2 * np.tan(steer) / float(wheelbase)
        active_sign = np.sign(np.where(np.abs(steer) >= 5e-4, steer, 0.0))
        nonzero_sign = active_sign[active_sign != 0]
        sign_changes = int(np.count_nonzero(nonzero_sign[1:] != nonzero_sign[:-1]))
        saturation = np.abs(steer) >= float(max_steer) * 0.995
        xy = poses[m, :, :2]
        step_distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cte = road_eval["nearest_lane_cte_m"][m]
        margin = road_eval["lane_strip_margin_m"][m]
        inside = road_eval["inside_any_lane_strip"][m]

        rows.append({
            "method": method,
            "anchor_index": m,
            "max_abs_steer_rad": float(np.max(np.abs(steer))),
            "mean_abs_steer_rad": float(np.mean(np.abs(steer))),
            "rms_steer_rad": float(np.sqrt(np.mean(steer ** 2))),
            "steering_saturation_steps": int(np.count_nonzero(saturation)),
            "steering_saturation_ratio": float(np.mean(saturation)),
            "max_abs_steer_rate_rad_s": float(np.max(np.abs(steer_rate), initial=0.0)),
            "steering_total_variation_rad": float(np.sum(np.abs(np.diff(steer)))),
            "steering_sign_changes": sign_changes,
            "max_abs_yaw_rate_rad_s": float(np.max(np.abs(yaw_rate), initial=0.0)),
            "max_abs_lateral_accel_mps2": float(np.max(np.abs(lateral_accel), initial=0.0)),
            "max_speed_mps": float(np.max(speeds[m])),
            "final_yaw_deg": float(np.rad2deg(poses[m, -1, 2])),
            "path_length_m": float(np.sum(step_distance)),
            "max_abs_nearest_lane_cte_m": float(np.nanmax(np.abs(cte))),
            "mean_abs_nearest_lane_cte_m": float(np.nanmean(np.abs(cte))),
            "min_lane_strip_margin_m": float(np.nanmin(margin)),
            "center_offroad_steps": int(np.count_nonzero(~inside)),
            "center_offroad_ratio": float(np.mean(~inside)),
        })

        for t in range(controls.shape[1]):
            per_step.append({
                "method": method,
                "anchor_index": m,
                "step": t + 1,
                "time_s": (t + 1) * float(dt),
                "acceleration_mps2": float(accel[t]),
                "steering_rad": float(steer[t]),
                "steering_rate_rad_s": float(steer_rate[t - 1]) if t > 0 else 0.0,
                "speed_mps": float(speeds[m, t + 1]),
                "yaw_rad": float(poses[m, t + 1, 2]),
                "x_m": float(xy[t + 1, 0]),
                "y_m": float(xy[t + 1, 1]),
                "nearest_lane_index": int(road_eval["nearest_lane_index"][m, t + 1]),
                "nearest_lane_cte_m": float(cte[t + 1]),
                "lane_strip_margin_m": float(margin[t + 1]),
                "inside_any_lane_strip": bool(inside[t + 1]),
            })
    return rows, per_step


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_outputs(output_dir: Path, lanes: list[dict], row: dict,
                  trajectories: dict, controls: dict, dt_by_method: dict,
                  max_steer_by_method: dict, consistency: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    current_slot = int(_scalar(row["road_current_lane_slot"]))
    target_slot = int(_scalar(row["road_target_lane_slot"]))

    fig, ax = plt.subplots(figsize=(12, 7))
    for lane in lanes:
        g = lane["geometry"]
        idx = lane["index"]
        label = f"lane {idx} center" if idx == current_slot else None
        ax.plot(g[:, 0], g[:, 1], color="0.65", linewidth=0.8, linestyle=":",
                label=label)
        ax.plot(g[:, 2], g[:, 3], color="0.25", linewidth=0.9, linestyle="--",
                label="road/lane edges" if idx == lanes[0]["index"] else None)
        ax.plot(g[:, 4], g[:, 5], color="0.25", linewidth=0.9, linestyle="--")
        if idx == target_slot and target_slot != current_slot:
            ax.plot(g[:, 0], g[:, 1], color="green", linewidth=1.4,
                    label="recorded target lane")
    ax.scatter([0.0], [0.0], marker="o", color="black", s=35, zorder=10, label="ego origin")

    styles = {
        "A0_FIXED": ("tab:blue", "--", 0.95),
        "A1_OFFLINE": ("tab:orange", ":", 0.85),
        "A1_ONLINE": ("tab:red", "-", 1.0),
    }
    for method, (color, linestyle, alpha) in styles.items():
        if method not in trajectories:
            continue
        tr = trajectories[method]
        for m in range(tr.shape[0]):
            label = method.replace("_", " ") if m == 0 else None
            ax.plot(tr[m, :, 0], tr[m, :, 1], color=color, linestyle=linestyle,
                    linewidth=1.2, alpha=alpha, label=label)
            ax.scatter([tr[m, -1, 0]], [tr[m, -1, 1]], color=color, s=10, alpha=alpha)
    if "expert_trajectory_xy" in row:
        gt = np.asarray(row["expert_trajectory_xy"], dtype=np.float32)
        if gt.ndim == 2 and gt.shape[-1] == 2:
            gt = np.concatenate([np.zeros((1, 2), dtype=np.float32), gt], axis=0)
            ax.plot(gt[:, 0], gt[:, 1], color="black", linewidth=2.0,
                    linestyle="-.", label="expert trajectory")
    ax.set_title("A0/A1 Anchor Rollouts over Captured Road Geometry")
    ax.set_xlabel("Longitudinal position (m)")
    ax.set_ylabel("Lateral position (m)")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    road_png = output_dir / "anchor_road_overlay.png"
    fig.savefig(road_png, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, method in zip(axes, ("A0_FIXED", "A1_ONLINE")):
        if method not in controls:
            ax.set_visible(False)
            continue
        u = controls[method]
        dt = dt_by_method[method]
        t = (np.arange(u.shape[1]) + 1) * dt
        for m in range(u.shape[0]):
            ax.plot(t, u[m, :, 1], linewidth=0.9, alpha=0.8, label=f"M{m:02d}")
        limit = max_steer_by_method[method]
        ax.axhline(limit, color="black", linestyle="--", linewidth=0.9)
        ax.axhline(-limit, color="black", linestyle="--", linewidth=0.9)
        ax.set_title(method.replace("_", " "))
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=2)
    axes[0].set_ylabel("Front-wheel steering (rad)")
    fig.suptitle("Steering Profiles and Saturation Limits")
    fig.tight_layout()
    steer_png = output_dir / "steering_profiles.png"
    fig.savefig(steer_png, dpi=180)
    plt.close(fig)

    # Show stored-vs-replayed A1 trajectory mismatch. If they overlap, the
    # remaining road deviation is in candidate generation/tracking, not a
    # discrepancy between the saved trajectory and the control rollout.
    fig, ax = plt.subplots(figsize=(8, 5))
    if "A1_OFFLINE" in trajectories and "A1_ONLINE" in trajectories:
        stored = trajectories["A1_OFFLINE"]
        replay = trajectories["A1_ONLINE"]
        for m in range(min(stored.shape[0], replay.shape[0])):
            ax.plot(stored[m, :, 0], stored[m, :, 1], color="tab:orange",
                    linestyle=":", linewidth=1.0, alpha=0.75)
            ax.plot(replay[m, :, 0], replay[m, :, 1], color="tab:red",
                    linestyle="-", linewidth=0.8, alpha=0.75)
    ax.set_title(
        "A1 Stored vs Rebuilt Rollout\n"
        f"max |stored - replay| = {consistency.get('stored_dense_vs_online_replay_max_abs_m', float('nan')):.3e} m"
    )
    ax.set_xlabel("Longitudinal position (m)")
    ax.set_ylabel("Lateral position (m)")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    fig.tight_layout()
    replay_png = output_dir / "a1_replay_consistency.png"
    fig.savefig(replay_png, dpi=180)
    plt.close(fig)

    pdf_path = output_dir / "anchor_steering_diagnostic.pdf"
    with PdfPages(pdf_path) as pdf:
        for image_path in (road_png, steer_png, replay_png):
            image = plt.imread(image_path)
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return road_png, steer_png, replay_png, pdf_path


def run(args):
    dataset_root = resolve_path(args.dataset_root)
    a0_path = resolve_path(args.a0_bank)
    diffusion_config_path = resolve_path(args.diffusion_config)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, shard, row_index, row = load_sample(
        dataset_root, args.episode, args.frame, args.ego
    )
    a1_cfg = dict(manifest.get("a1_anchor_parameters") or {})
    if not a1_cfg:
        raise RuntimeError("A1 manifest does not contain a1_anchor_parameters")

    from highway_env.planner.a1_anchor_builder import A1AnchorBuilder
    builder = A1AnchorBuilder(a1_cfg)
    online_a1 = builder.build(row)

    stored_controls = np.asarray(row["a1_coarse_controls"], dtype=np.float32)
    stored_xy = np.asarray(row["a1_coarse_trajectories"], dtype=np.float32)
    stored_dense = np.asarray(row["a1_dense_trajectories"], dtype=np.float32)
    stored_valid = np.asarray(row["a1_valid_mask"], dtype=bool)
    online_controls = np.asarray(online_a1["a1_coarse_controls"], dtype=np.float32)
    online_xy = np.asarray(online_a1["a1_coarse_trajectories"], dtype=np.float32)
    online_dense_numpy = np.asarray(online_a1["a1_dense_trajectories"], dtype=np.float32)
    online_valid = np.asarray(online_a1["a1_valid_mask"], dtype=bool)

    if stored_controls.shape != online_controls.shape:
        raise RuntimeError(
            f"A1 stored/online control shape mismatch: {stored_controls.shape} vs {online_controls.shape}"
        )
    if stored_dense.shape != online_dense_numpy.shape:
        raise RuntimeError(
            f"A1 stored/online dense trajectory shape mismatch: {stored_dense.shape} vs {online_dense_numpy.shape}"
        )

    candidate_base = builder.candidate_builder.base
    a1_dt = float(candidate_base.control_dt)
    a1_wheelbase = float(candidate_base.wheelbase)
    a1_max_steer = float(candidate_base.max_steer)
    a0_controls, a0_metadata = load_a0_bank(a0_path)
    if a0_controls.shape[0] != stored_controls.shape[0]:
        print(
            f"[WARN] A0 K={a0_controls.shape[0]} differs from A1 K={stored_controls.shape[0]}; "
            "both banks will be analyzed independently."
        )
    a0_dt, a0_wheelbase, a0_max_steer, dynamics_source = load_diffusion_dynamics(
        diffusion_config_path, a1_dt, a1_wheelbase, a1_max_steer
    )

    initial_speed = max(float(np.asarray(row["ego_state"]).reshape(-1)[0]), 0.0)
    a0_poses, a0_speeds = rollout_controls(
        a0_controls, initial_speed, a0_dt, a0_wheelbase
    )
    a1_online_poses, a1_online_speeds = rollout_controls(
        online_controls, initial_speed, a1_dt, a1_wheelbase
    )
    a1_stored_poses, a1_stored_speeds = rollout_controls(
        stored_controls, initial_speed, a1_dt, a1_wheelbase
    )

    # Reconcile the common 10-Hz control rollout with A1's stored 10-Hz path.
    replay_dense_error = float(np.max(np.abs(a1_online_poses[:, :, :2] - stored_dense)))
    stored_builder_error = float(np.max(np.abs(online_dense_numpy - stored_dense)))
    stored_control_error = float(np.max(np.abs(online_controls - stored_controls)))
    online_coarse_error = float(np.max(np.abs(online_xy - stored_xy)))
    coarse_from_dense = a1_online_poses[:, 5::5, :2]
    coarse_error = float(np.max(np.abs(coarse_from_dense - stored_xy)))
    mask_mismatches = int(np.count_nonzero(online_valid != stored_valid))

    lanes = _lane_polylines(row)
    # The offline artifact stores the builder's dense NumPy rollout. The online
    # manager returns the same builder output; the Torch replay is retained as
    # an independent implementation-consistency check.
    a1_stored_metric_poses = a1_stored_poses.copy()
    a1_stored_metric_poses[:, :, :2] = stored_dense
    a1_online_metric_poses = a1_online_poses.copy()
    a1_online_metric_poses[:, :, :2] = online_dense_numpy
    road_eval = {
        "A0_FIXED": evaluate_road(a0_poses[:, :, :2], lanes),
        "A1_OFFLINE": evaluate_road(a1_stored_metric_poses[:, :, :2], lanes),
        "A1_ONLINE": evaluate_road(a1_online_metric_poses[:, :, :2], lanes),
    }
    controls = {
        "A0_FIXED": a0_controls,
        "A1_OFFLINE": stored_controls,
        "A1_ONLINE": online_controls,
    }
    poses = {
        "A0_FIXED": a0_poses,
        "A1_OFFLINE": a1_stored_metric_poses,
        "A1_ONLINE": a1_online_metric_poses,
    }
    speeds = {
        "A0_FIXED": a0_speeds,
        "A1_OFFLINE": a1_stored_speeds,
        "A1_ONLINE": a1_online_speeds,
    }
    dt_by_method = {
        "A0_FIXED": a0_dt,
        "A1_OFFLINE": a1_dt,
        "A1_ONLINE": a1_dt,
    }
    max_steer_by_method = {
        "A0_FIXED": a0_max_steer,
        "A1_OFFLINE": a1_max_steer,
        "A1_ONLINE": a1_max_steer,
    }

    metric_rows = []
    step_rows = []
    for method in controls:
        rows_m, rows_t = steering_metrics(
            method, controls[method], poses[method], speeds[method],
            road_eval[method], dt_by_method[method],
            a0_wheelbase if method == "A0_FIXED" else a1_wheelbase,
            max_steer_by_method[method],
        )
        metric_rows.extend(rows_m)
        step_rows.extend(rows_t)

    consistency = {
        "a1_stored_vs_online_control_max_abs": stored_control_error,
        "a1_stored_vs_online_builder_dense_max_abs_m": stored_builder_error,
        "a1_stored_vs_online_torch_replay_max_abs_m": replay_dense_error,
        "a1_stored_vs_online_builder_coarse_max_abs_m": online_coarse_error,
        "a1_stored_coarse_vs_online_torch_replay_max_abs_m": coarse_error,
        "a1_valid_mask_mismatches": mask_mismatches,
        "a1_replay_tolerance_m": 1e-4,
        "a1_replay_consistent": bool(
            stored_control_error <= 1e-6
            and stored_builder_error <= 1e-4
            and replay_dense_error <= 1e-4
            and online_coarse_error <= 1e-4
            and coarse_error <= 1e-4
            and mask_mismatches == 0
        ),
    }

    def aggregate(method):
        selected = [x for x in metric_rows if x["method"] == method]
        return {
            "anchors": len(selected),
            "mean_steering_saturation_ratio": float(np.mean([x["steering_saturation_ratio"] for x in selected])),
            "max_steering_saturation_ratio": float(np.max([x["steering_saturation_ratio"] for x in selected])),
            "max_abs_steer_rate_rad_s": float(np.max([x["max_abs_steer_rate_rad_s"] for x in selected])),
            "max_abs_lateral_accel_mps2": float(np.max([x["max_abs_lateral_accel_mps2"] for x in selected])),
            "max_abs_nearest_lane_cte_m": float(np.max([x["max_abs_nearest_lane_cte_m"] for x in selected])),
            "anchors_with_center_offroad_steps": int(sum(x["center_offroad_steps"] > 0 for x in selected)),
            "total_center_offroad_steps": int(sum(x["center_offroad_steps"] for x in selected)),
            "worst_lane_strip_margin_m": float(np.min([x["min_lane_strip_margin_m"] for x in selected])),
        }

    scene = {
        "episode_index": args.episode,
        "frame_index": args.frame,
        "ego_index": args.ego,
        "sample_row_in_shard": row_index,
        "shard": str(shard),
        "scenario": str(_scalar(row["scenario"])) if "scenario" in row else None,
        "episode_step": int(_scalar(row["episode_step"])) if "episode_step" in row else None,
        "initial_speed_mps": initial_speed,
        "payload_kg": float(_scalar(row["koopman_payload_kg"])) if "koopman_payload_kg" in row else None,
        "road_mu": float(_scalar(row["koopman_road_mu"])) if "koopman_road_mu" in row else None,
        "current_lane_slot": int(_scalar(row["road_current_lane_slot"])),
        "target_lane_slot": int(_scalar(row["road_target_lane_slot"])),
    }
    summary = {
        "scene": scene,
        "dataset_root": str(dataset_root),
        "a0_bank": str(a0_path),
        "a0_metadata": a0_metadata,
        "a1_manifest_schema": manifest.get("schema_version"),
        "a1_num_selected": int(stored_controls.shape[0]),
        "a0_num_modes": int(a0_controls.shape[0]),
        "a1_config_from_manifest": manifest.get("a1_anchor_config"),
        "a0_dynamics_source": dynamics_source,
        "dynamics": {
            "a0_dt_s": a0_dt, "a0_wheelbase_m": a0_wheelbase,
            "a1_dt_s": a1_dt, "a1_wheelbase_m": a1_wheelbase,
            "a0_steer_limit_rad": a0_max_steer,
            "a1_steer_limit_rad": a1_max_steer,
        },
        "a1_replay_consistency": consistency,
        "method_summary": {
            method: aggregate(method) for method in controls
        },
        "interpretation": [],
    }

    a1_online_summary = summary["method_summary"]["A1_ONLINE"]
    if not consistency["a1_replay_consistent"]:
        summary["interpretation"].append(
            "A1 offline artifacts do not reproduce the online builder/kinematic rollout within tolerance. "
            "Check A1 configuration, candidate-pool configuration, or dataset/code version mismatch."
        )
    else:
        summary["interpretation"].append(
            "A1 stored controls, online rebuilt controls, and the independent 10-Hz bicycle replay are consistent. "
            "A road deviation in this sample is therefore not explained by a stored-vs-replayed trajectory mismatch."
        )
    if a1_online_summary["mean_steering_saturation_ratio"] > 0.05:
        summary["interpretation"].append(
            "A1 has nontrivial steering saturation. Steering-limit clipping is a plausible contributor to lane-tracking error."
        )
    if a1_online_summary["anchors_with_center_offroad_steps"] > 0:
        summary["interpretation"].append(
            "At least one A1 anchor center point leaves the union of captured lane strips. "
            "This is a center-point diagnostic, not a full truck-footprint collision/road-clearance test."
        )
    if summary["method_summary"]["A0_FIXED"]["anchors_with_center_offroad_steps"] > 0:
        summary["interpretation"].append(
            "A0 has no scene-conditioned road feedback or road-boundary filtering; fixed control centroids can therefore leave the captured road corridor."
        )

    _write_csv(output_dir / "anchor_steering_metrics.csv", metric_rows)
    _write_csv(output_dir / "anchor_steering_timeseries.csv", step_rows)

    trajectory_dict = {
        "A0_FIXED": a0_poses[:, :, :2],
        "A1_OFFLINE": stored_dense,
        "A1_ONLINE": a1_online_poses[:, :, :2],
    }
    _plot_outputs(
        output_dir, lanes, row, trajectory_dict, controls, dt_by_method,
        max_steer_by_method, consistency
    )
    (output_dir / "steering_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("[STEERING DIAGNOSTIC]", json.dumps(scene, ensure_ascii=False), flush=True)
    print("[A1 REPLAY]", json.dumps(consistency, ensure_ascii=False), flush=True)
    for method, stats in summary["method_summary"].items():
        print(f"[{method}]", json.dumps(stats, ensure_ascii=False), flush=True)
    for item in summary["interpretation"]:
        print("[INTERPRETATION]", item, flush=True)
    print(f"[OUTPUT] {output_dir}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Diagnose A0/A1 steering, bicycle rollouts, and road-relative deviation for one real A1 dataset sample."
    )
    parser.add_argument(
        "--dataset-root",
        default="outputs/anchor_ablation/ckpt/A1_anchors",
        help="Completed A1/K2 dataset root containing shards/ and reports/manifest.json.",
    )
    parser.add_argument(
        "--a0-bank",
        default="outputs/anchor_ablation/ckpt/A0_anchors/A0_control_anchors.npz",
        help="A0 fixed control-centroid bank.",
    )
    parser.add_argument("--diffusion-config", default="configs/train/diffusion.yaml")
    parser.add_argument("--episode", type=int, default=469)
    parser.add_argument("--frame", type=int, default=9)
    parser.add_argument("--ego", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="outputs/anchor_ablation/diagnostics/steering_ep469_f9_ego0",
    )
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
