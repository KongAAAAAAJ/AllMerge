#!/usr/bin/env python3
"""Run real control-Diffusion + frozen K2 + JTP on one-ego expert samples.

This is an offline, single-controlled-vehicle validation harness. Other actors
from the same expert sample are retained as constant-velocity collision
obstacles. The script does not step the simulator or publish vehicle controls.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Resolve the repository root robustly for direct script execution.
# The script is located at <repo>/scripts/evaluation/, so parents[2] is the
# repository root. Marker-based discovery avoids future path-depth mistakes.
if __package__ in (None, ""):
    _script_path = Path(__file__).resolve()
    _repo_root = next(
        (
            _parent
            for _parent in _script_path.parents
            if (_parent / "data_pipeline" / "expert_dataset.py").is_file()
            and (_parent / "highway_env").is_dir()
        ),
        None,
    )
    if _repo_root is None:
        raise RuntimeError(
            "Cannot locate AllMerge repository root from "
            f"{_script_path}; expected data_pipeline/expert_dataset.py "
            "and highway_env/ at the repository root."
        )
    _repo_root_str = str(_repo_root)
    if _repo_root_str not in sys.path:
        sys.path.insert(0, _repo_root_str)

from data_pipeline.expert_dataset import (  # noqa: E402
    AllMergeExpertDatasetConfig,
    AllMergeExpertShardDataset,
)
from highway_env.planner.diffusion.common.execution import K2_CONTEXT  # noqa: E402
from highway_env.planner.diffusion.config.experiment import resolve_repo_file  # noqa: E402
from highway_env.planner.diffusion.inference.control import ControlDiffusionRuntime  # noqa: E402
from highway_env.planner.diffusion.projection import JointTrajectoryProjector  # noqa: E402
from highway_env.planner.diffusion.projection.constraints import stability_limits  # noqa: E402
from highway_env.planner.diffusion.projection.types import (  # noqa: E402
    ExternalTraffic,
    RoadGeometry,
    constant_velocity_poses,
)


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _world_from_local(points_xy: np.ndarray, pose0: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    c, s = math.cos(float(pose0[2])), math.sin(float(pose0[2]))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return points @ rot.T + np.asarray(pose0[:2], dtype=np.float64)


def _local_from_world(points_xy: np.ndarray, pose0: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    c, s = math.cos(float(pose0[2])), math.sin(float(pose0[2]))
    rot_t = np.asarray([[c, s], [-s, c]], dtype=np.float64)
    return (points - np.asarray(pose0[:2], dtype=np.float64)) @ rot_t.T


def build_road_geometry_from_features(
    features: dict[str, Any],
    pose0: np.ndarray,
) -> tuple[RoadGeometry, dict[str, Any]]:
    """Build the current carriageway corridor from same-road lane features.

    Feature channels follow AllMergeFeatureBuilder.MAP_FEATURE_NAMES:
    x_rel, y_rel, sin_heading_rel, cos_heading_rel, lane_width, speed_limit,
    is_current_lane, is_left_lane, is_right_lane, is_target_lane.
    """
    polylines = _as_numpy(features["map_polylines"]).astype(np.float64)
    valid = _as_numpy(features["map_valid_mask"]).astype(bool).reshape(-1)
    if polylines.ndim != 3 or polylines.shape[-1] < 10:
        raise ValueError(f"map_polylines must be [M,P,10], got {polylines.shape}")
    if valid.shape[0] != polylines.shape[0]:
        raise ValueError("map_valid_mask and map_polylines dimensions differ")

    indices = [int(i) for i in np.flatnonzero(valid)
               if np.isfinite(polylines[i]).all()
               and np.all(polylines[i, :, 4] > 0.0)]
    if not indices:
        raise ValueError("No valid lane polylines are available for JTP road geometry")

    current = [i for i in indices if float(polylines[i, 0, 6]) > 0.5]
    if not current:
        raise ValueError("No is_current_lane polyline is present in the expert sample")
    current_idx = current[0]

    # The feature builder marks left/right lanes only when they share the
    # current road segment. This avoids merging unrelated parallel roads.
    same_road = [i for i in indices if i == current_idx
                 or float(polylines[i, 0, 7]) > 0.5
                 or float(polylines[i, 0, 8]) > 0.5]
    if current_idx not in same_road:
        same_road.insert(0, current_idx)

    ref = polylines[current_idx]
    ref_xy = ref[:, :2]
    ref_heading = np.arctan2(ref[:, 2], ref[:, 3])
    ref_normal = np.stack((-np.sin(ref_heading), np.cos(ref_heading)), axis=-1)

    lateral_offsets = {}
    for idx in same_road:
        delta = polylines[idx, :, :2] - ref_xy
        lateral_offsets[idx] = float(np.median(np.sum(delta * ref_normal, axis=-1)))
    right_idx = min(same_road, key=lambda i: lateral_offsets[i])
    left_idx = max(same_road, key=lambda i: lateral_offsets[i])
    target_candidates = [i for i in indices if float(polylines[i, 0, 9]) > 0.5]
    target_idx = target_candidates[0] if target_candidates else current_idx
    target_same_road = target_idx in same_road

    def boundary(polyline: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
        xy = polyline[:, :2]
        heading = np.arctan2(polyline[:, 2], polyline[:, 3])
        normal = np.stack((-np.sin(heading), np.cos(heading)), axis=-1)
        half_lane = 0.5 * polyline[:, 4]
        if side == "left":
            return xy + half_lane[:, None] * normal, half_lane
        return xy - half_lane[:, None] * normal, half_lane

    left_boundary, _ = boundary(polylines[left_idx], "left")
    right_boundary, _ = boundary(polylines[right_idx], "right")
    if left_boundary.shape != right_boundary.shape:
        raise ValueError("Left/right lane polylines do not share a common sample grid")

    center_local = 0.5 * (left_boundary + right_boundary)
    half_width = 0.5 * np.linalg.norm(left_boundary - right_boundary, axis=-1)
    keep = [0]
    for i in range(1, len(center_local)):
        if np.linalg.norm(center_local[i] - center_local[keep[-1]]) > 1.0e-5:
            keep.append(i)
    center_local = center_local[keep]
    half_width = half_width[keep]
    if len(center_local) < 2 or not np.isfinite(center_local).all():
        raise ValueError("Constructed JTP road corridor is degenerate or non-finite")
    if not np.isfinite(half_width).all() or np.any(half_width <= 0.0):
        raise ValueError("Constructed JTP road corridor has invalid half-widths")

    center_world = _world_from_local(center_local, pose0).astype(np.float32)
    half_width = half_width.astype(np.float32)
    road = RoadGeometry(
        centerlines_xy=torch.from_numpy(center_world[None]),
        left_width_m=torch.from_numpy(half_width[None]),
        right_width_m=torch.from_numpy(half_width[None]),
    )
    meta = {
        "lane_count": int(len(same_road)),
        "left_lane_feature_index": int(left_idx),
        "right_lane_feature_index": int(right_idx),
        "current_lane_feature_index": int(current_idx),
        "target_lane_feature_index": int(target_idx),
        "target_lane_same_road": bool(target_same_road),
        "min_half_width_m": float(half_width.min()),
        "max_half_width_m": float(half_width.max()),
        "centerline_local_xy": center_local.astype(np.float32),
        "half_width_m": half_width,
    }
    return road, meta


def build_external_traffic_from_features(
    features: dict[str, Any],
    context: dict[str, Any],
    *,
    horizon_steps: int,
    dt_s: float,
    max_actors: int = 8,
    max_distance_m: float = 120.0,
) -> tuple[ExternalTraffic | None, dict[str, Any]]:
    """Convert ego-relative actor features to world-frame CV obstacle poses.

    For single-ego validation, any other controlled actor in the sample is
    treated as an external obstacle because it is not part of this one-ego JTP
    decision variable.
    """
    states = _as_numpy(features["agent_states"]).astype(np.float64)
    valid = _as_numpy(features["agent_valid_mask"]).astype(bool).reshape(-1)
    pose0 = _as_numpy(context["koopman_pose0_si"]).reshape(1, 3)[0].astype(np.float64)
    ego_state_hist = _as_numpy(context["koopman_state_hist_si"]).reshape(5, 5)
    ego_v_body = ego_state_hist[-1, :2]
    c, s = math.cos(float(pose0[2])), math.sin(float(pose0[2]))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    ego_v_world = rot @ ego_v_body

    actors = []
    for idx in np.flatnonzero(valid):
        row = states[int(idx)]
        if row.shape[0] < 10 or not np.isfinite(row[:10]).all():
            continue
        rel_xy = row[:2]
        if np.linalg.norm(rel_xy) > float(max_distance_m):
            continue
        world_xy = pose0[:2] + rot @ rel_xy
        world_velocity = ego_v_world + rot @ row[2:4]
        yaw_rel = math.atan2(float(row[6]), float(row[7]))
        yaw_world = float(pose0[2] + yaw_rel)
        length = float(row[8]) if float(row[8]) > 0.5 else 5.0
        width = float(row[9]) if float(row[9]) > 0.5 else 2.0
        actors.append({
            "feature_index": int(idx),
            "state": [float(world_xy[0]), float(world_xy[1]),
                      float(world_velocity[0]), float(world_velocity[1]), yaw_world],
            "length_m": length,
            "width_m": width,
            "is_controlled": bool(row[10] > 0.5) if row.shape[0] > 10 else False,
            "distance_m": float(np.linalg.norm(rel_xy)),
        })
    actors.sort(key=lambda item: item["distance_m"])
    actors = actors[:max(0, int(max_actors))]
    if not actors:
        return None, {"actor_count": 0, "actor_feature_indices": []}

    initial = torch.tensor([item["state"] for item in actors], dtype=torch.float32)
    poses = constant_velocity_poses(initial, horizon_steps, dt_s)
    circle_count = 7
    offsets, radii = [], []
    for item in actors:
        radius = max(0.5 * item["width_m"], 0.5)
        center_span = max(0.5 * item["length_m"] - radius, 0.0)
        offsets.append(np.linspace(-center_span, center_span, circle_count, dtype=np.float32))
        radii.append(np.full((circle_count,), radius, dtype=np.float32))
    traffic = ExternalTraffic(
        poses_world=poses,
        circle_offsets_m=torch.from_numpy(np.stack(offsets)),
        circle_radius_m=torch.from_numpy(np.stack(radii)),
    )
    return traffic, {
        "actor_count": len(actors),
        "actor_feature_indices": [item["feature_index"] for item in actors],
        "actor_distances_m": [item["distance_m"] for item in actors],
        "actor_is_controlled": [item["is_controlled"] for item in actors],
    }


def _load_raw_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Diffusion config must be a YAML mapping: {path}")
    return data


def _context_from_targets(targets: dict, device: torch.device) -> dict[str, torch.Tensor]:
    missing = [key for key in K2_CONTEXT if key not in targets]
    if missing:
        raise KeyError(
            "Selected expert shard does not contain synchronized K2 fields: "
            f"{missing}. Use the TruckSim K2 paired-anchor dataset."
        )
    context = {}
    for key in K2_CONTEXT:
        value = targets[key]
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        tensor = tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
        context[key] = tensor
    expected = {
        "koopman_state_hist_si": (1, 5, 5),
        "koopman_control_hist_si": (1, 4, 2),
        "koopman_pose0_si": (1, 3),
        "koopman_payload_kg": (1,),
        "koopman_road_mu": (1,),
    }
    for key, shape in expected.items():
        if tuple(context[key].shape) not in (shape, (1, 1) if shape == (1,) else shape):
            raise ValueError(f"Invalid {key} shape {tuple(context[key].shape)}, expected {shape}")
        if not torch.isfinite(context[key]).all():
            raise ValueError(f"Non-finite K2 context field: {key}")
    if float(context["koopman_payload_kg"].reshape(-1)[0]) < 0:
        raise ValueError("koopman_payload_kg must be nonnegative")
    mu = float(context["koopman_road_mu"].reshape(-1)[0])
    if not 0.0 < mu <= 1.0:
        raise ValueError("koopman_road_mu must be in (0,1]")
    return context


def _features_to_numpy(features: dict) -> dict[str, np.ndarray]:
    result = {}
    for key, value in features.items():
        result[key] = _as_numpy(value).astype(bool if key in ("agent_valid_mask", "map_valid_mask", "mode_valid_mask") else np.float32, copy=False)[None]
    return result


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _metric_pair(diagnostics: dict) -> tuple[dict, dict]:
    history = diagnostics.get("history") or []
    initial = diagnostics.get("initial_metrics")
    final = diagnostics.get("final_metrics")
    if initial is None and history:
        initial = history[0]
    if final is None and history:
        final = history[-1]
    return dict(initial or {}), dict(final or {})


def _world_pose_to_local(world_pose: np.ndarray, pose0: np.ndarray) -> np.ndarray:
    world_pose = np.asarray(world_pose, dtype=np.float64)
    xy_local = _local_from_world(world_pose[..., :2], pose0)
    yaw = world_pose[..., 2] - float(pose0[2])
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    return np.concatenate((xy_local, yaw[..., None]), axis=-1).astype(np.float32)


def _save_plots(out_dir: Path, nominal: dict, projected: dict, road_meta: dict,
                external: ExternalTraffic | None, pose0: np.ndarray,
                context: dict, cfg: dict, dt_s: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is unavailable; NPZ/JSON/CSV outputs are still saved.")
        return

    nominal_pose = np.asarray(nominal["pose"])[0]
    projected_pose = np.asarray(projected["trajectory_local"])
    road_xy = np.asarray(road_meta["centerline_local_xy"])
    half_width = np.asarray(road_meta["half_width_m"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(nominal_pose[:, 0], nominal_pose[:, 1], "--", label="Diffusion + K2 nominal")
    ax.plot(projected_pose[:, 0], projected_pose[:, 1], "-", label="JTP projected")
    ax.plot(road_xy[:, 0], road_xy[:, 1], ":", label="Road corridor centerline")
    tangent = np.gradient(road_xy, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=-1, keepdims=True).clip(min=1.0e-8)
    tangent = tangent / tangent_norm
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=-1)
    left_edge = road_xy + half_width[:, None] * normal
    right_edge = road_xy - half_width[:, None] * normal
    ax.plot(left_edge[:, 0], left_edge[:, 1], color="0.5", linewidth=0.8, alpha=0.7)
    ax.plot(right_edge[:, 0], right_edge[:, 1], color="0.5", linewidth=0.8, alpha=0.7)
    ax.fill(np.concatenate((left_edge[:, 0], right_edge[::-1, 0])),
            np.concatenate((left_edge[:, 1], right_edge[::-1, 1])),
            alpha=0.10, label="Approx. road corridor")
    ax.scatter([0.0], [0.0], marker="o", label="Ego start")
    if external is not None:
        ext_poses, _, _ = external.tensors(device="cpu", dtype=torch.float32,
                                            expected_steps=projected_pose.shape[0])
        for i in range(ext_poses.shape[0]):
            ext_local = _world_pose_to_local(ext_poses[i].numpy(), pose0)
            ax.plot(ext_local[:, 0], ext_local[:, 1], "-.", alpha=0.45,
                    label="External actor CV" if i == 0 else None)
    ax.set_xlabel("Longitudinal position in ego frame (m)")
    ax.set_ylabel("Lateral position in ego frame (m)")
    ax.set_title("Single-vehicle JTP trajectory projection")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_comparison.png", dpi=180)
    plt.close(fig)

    t = np.arange(np.asarray(nominal["control"]).shape[1]) * dt_s
    u0 = np.asarray(nominal["control"])[0]
    u1 = np.asarray(projected["control"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, u0[:, 0], "--", label="Nominal")
    axes[0].plot(t, u1[:, 0], label="JTP")
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)
    axes[1].plot(t, u0[:, 1], "--", label="Nominal")
    axes[1].plot(t, u1[:, 1], label="JTP")
    axes[1].set_ylabel("Front-wheel angle (rad)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(); axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "control_comparison.png", dpi=180)
    plt.close(fig)

    nominal_states = np.asarray(nominal["states"])[0]
    projected_states = np.asarray(projected["states"])
    nominal_ltr = np.asarray(nominal["ltr"])[0]
    projected_ltr = np.asarray(projected["ltr"])
    payload = context["koopman_payload_kg"].reshape(-1)[0]
    mu = context["koopman_road_mu"].reshape(-1)[0]
    with torch.no_grad():
        vx = torch.as_tensor(projected_states[1:, 0], dtype=torch.float32)
        bounds = stability_limits(vx, torch.as_tensor(payload), torch.as_tensor(mu), cfg).cpu().numpy()
    r_lim, beta_lim, roll_lim, ltr_lim = bounds.T
    beta0 = np.arctan2(nominal_states[1:, 1], nominal_states[1:, 0])
    beta1 = np.arctan2(projected_states[1:, 1], projected_states[1:, 0])
    series = [
        (np.abs(nominal_states[1:, 2]), np.abs(projected_states[1:, 2]), r_lim, "|Yaw rate| (rad/s)"),
        (np.abs(beta0), np.abs(beta1), beta_lim, "|Sideslip| (rad)"),
        (np.abs(nominal_states[1:, 3]), np.abs(projected_states[1:, 3]), roll_lim, "|Roll angle| (rad)"),
        (np.abs(nominal_ltr[1:]), np.abs(projected_ltr[1:]), ltr_lim, "|LTR|"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    t_state = np.arange(1, len(projected_states)) * dt_s
    for ax, (v0, v1, limit, label) in zip(axes, series):
        ax.plot(t_state, v0, "--", label="Nominal")
        ax.plot(t_state, v1, label="JTP")
        ax.plot(t_state, limit, ":", label="Limit")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "stability_comparison.png", dpi=180)
    plt.close(fig)


def _prepare_sample(dataset, index: int, device: torch.device):
    features, targets, metadata = dataset[index]
    features_np = _features_to_numpy(features)
    context = _context_from_targets(targets, device)
    for key in K2_CONTEXT:
        features_np[key] = context[key].detach().cpu().numpy()
    return features, targets, metadata, features_np, context


def _find_auto_sample(dataset, *, scan_limit: int, device: torch.device,
                      jtp_cfg: dict, requires_coarse_controls: bool):
    required_half_width = (float(jtp_cfg["geometry"]["footprint"]["circle_radius_m"])
                           + float(jtp_cfg["geometry"]["road_clearance_m"]) + 0.05)
    failures = []
    for index in range(min(len(dataset), int(scan_limit))):
        try:
            features, targets, metadata, features_np, context = _prepare_sample(dataset, index, device)
            pose0 = _as_numpy(context["koopman_pose0_si"]).reshape(3)
            road, road_meta = build_road_geometry_from_features(features, pose0)
            if road_meta["min_half_width_m"] < required_half_width:
                failures.append((index, "road corridor too narrow"))
                continue
            if not road_meta["target_lane_same_road"]:
                failures.append((index, "target lane belongs to a different road group"))
                continue
            if requires_coarse_controls and "coarse_controls" not in features_np:
                failures.append((index, "coarse_controls missing for dynamic-anchor checkpoint"))
                continue
            return index, features, targets, metadata, features_np, context, road, road_meta
        except Exception as exc:
            failures.append((index, f"{type(exc).__name__}: {exc}"))
    detail = "; ".join(f"{i}:{reason}" for i, reason in failures[:12])
    raise RuntimeError(
        f"No valid single-vehicle sample found in first {min(len(dataset), scan_limit)} "
        f"validation items. Required road half-width >= {required_half_width:.3f} m. "
        f"First failures: {detail}"
    )


def _run_one(runtime, projector, dataset, index: int, args, jtp_cfg: dict,
             output_root: Path) -> dict:
    features, targets, metadata, features_np, context = _prepare_sample(
        dataset, index, runtime.device)
    if "coarse_controls" not in features_np and runtime.model.anchor_manager is None:
        raise KeyError(
            "Selected expert sample lacks coarse_controls and this checkpoint uses "
            "dynamic anchors. Use the paired-anchor TruckSim dataset."
        )

    pose0 = _as_numpy(context["koopman_pose0_si"]).reshape(3)
    road, road_meta = build_road_geometry_from_features(features, pose0)
    required_half_width = (float(jtp_cfg["geometry"]["footprint"]["circle_radius_m"])
                           + float(jtp_cfg["geometry"]["road_clearance_m"]))
    if road_meta["min_half_width_m"] < required_half_width:
        print(f"[WARN] sample {index}: road half-width {road_meta['min_half_width_m']:.3f} m "
              f"is below footprint+clearance {required_half_width:.3f} m; expect infeasibility.")
    if not road_meta["target_lane_same_road"]:
        print(f"[WARN] sample {index}: target lane is on another road group. "
              "This first offline validator only constructs the current-road corridor; "
              "interpret road-boundary violations cautiously.")

    external, external_meta = build_external_traffic_from_features(
        features, context, horizon_steps=int(jtp_cfg["execution"]["horizon_steps"]),
        dt_s=float(jtp_cfg["execution"]["dt_s"]),
        max_actors=args.max_external_actors, max_distance_m=args.max_external_distance,
    )

    nominal = runtime.infer(features_np)
    required_outputs = ("control", "pose", "states", "ltr")
    missing = [key for key in required_outputs if key not in nominal]
    if missing:
        raise RuntimeError(f"Real control-Diffusion/K2 inference missing outputs: {missing}")
    nominal_control_np = np.asarray(nominal["control"], dtype=np.float32)
    if nominal_control_np.shape != (1, int(jtp_cfg["execution"]["control_steps"]), 2):
        raise ValueError(f"Nominal control shape mismatch: {nominal_control_np.shape}")

    controls = torch.as_tensor(nominal_control_np, device=runtime.device, dtype=torch.float32)
    start = time.perf_counter()
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    result = projector.project(
        nominal_controls=controls,
        context=context,
        road_geometry=road,
        external_traffic=external,
    )
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    jtp_latency_ms = (time.perf_counter() - start) * 1000.0

    nominal_pose_local = np.asarray(nominal["pose"], dtype=np.float32)[0]
    projected_pose_local = _world_pose_to_local(
        result.trajectories.detach().cpu().numpy()[0], pose0)
    nominal_states = np.asarray(nominal["states"], dtype=np.float32)[0]
    nominal_ltr = np.asarray(nominal["ltr"], dtype=np.float32)[0]
    projected_states = result.states.detach().cpu().numpy()[0]
    projected_ltr = result.ltr.detach().cpu().numpy()[0]
    projected_controls = result.controls.detach().cpu().numpy()[0]

    # Confirm that the JTP adapter and the model's nominal selected-mode rollout
    # use the same frozen K2 weights, context, units, and world/local transform.
    with torch.no_grad():
        adapter_nominal = projector.rollout_fn(controls, context)
    adapter_nominal_world = adapter_nominal["trajectory"].detach().cpu().numpy()[0]
    model_nominal_world = _world_from_local(nominal_pose_local[:, :2], pose0)
    yaw_world = nominal_pose_local[:, 2] + pose0[2]
    model_nominal_world_pose = np.column_stack((model_nominal_world, yaw_world))
    consistency_max_abs = float(np.max(np.abs(adapter_nominal_world - model_nominal_world_pose)))

    initial_metrics, final_metrics = _metric_pair(result.diagnostics or {})
    delta_control = projected_controls - nominal_control_np[0]
    trajectory_delta = projected_pose_local[:, :2] - nominal_pose_local[:, :2]
    output_dir = output_root / f"sample_{index:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload_kg = float(context["koopman_payload_kg"].reshape(-1)[0].detach().cpu())
    road_mu = float(context["koopman_road_mu"].reshape(-1)[0].detach().cpu())
    metrics = {
        "sample_index": int(index),
        "sample_metadata": _to_jsonable(metadata),
        "checkpoint": str(args.checkpoint_resolved),
        "dataset_root": str(args.dataset_root_resolved),
        "split": args.split,
        "device": str(runtime.device),
        "model_execution_kind": runtime.model.vehicle_model.kind,
        "model_anchor_strategy": runtime.model.anchors_config.get("strategy", "dynamic"),
        "selected_mode": int(np.asarray(nominal.get("trajectory_mode_idx", [0])).reshape(-1)[0]),
        "payload_kg": payload_kg,
        "road_mu": road_mu,
        "road_geometry": {k: v for k, v in road_meta.items() if k not in ("centerline_local_xy", "half_width_m")},
        "external_traffic": external_meta,
        "nominal_koopman_consistency_max_abs": consistency_max_abs,
        "nominal_koopman_consistency_tolerance": float(args.koopman_consistency_tolerance),
        "nominal_koopman_consistency_pass": bool(consistency_max_abs <= args.koopman_consistency_tolerance),
        "jtp_success": bool(result.success),
        "jtp_status": str(result.status),
        "jtp_iterations": int(result.iterations),
        "jtp_latency_ms": float(jtp_latency_ms),
        "diffusion_latency_ms": float(np.asarray(nominal.get("latency_ms", np.nan))),
        "control_delta_l2": float(np.linalg.norm(delta_control)),
        "control_delta_max_abs": float(np.max(np.abs(delta_control))),
        "trajectory_deviation_rmse_m": float(np.sqrt(np.mean(trajectory_delta ** 2))),
        "trajectory_deviation_max_m": float(np.max(np.linalg.norm(trajectory_delta, axis=-1))),
        "initial_metrics": _to_jsonable(initial_metrics),
        "final_metrics": _to_jsonable(final_metrics),
        "diagnostics": _to_jsonable(result.diagnostics or {}),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "arrays.npz",
        control_nominal=nominal_control_np[0],
        control_projected=projected_controls,
        trajectory_nominal_local=nominal_pose_local,
        trajectory_projected_local=projected_pose_local,
        trajectory_nominal_world=model_nominal_world_pose,
        trajectory_projected_world=result.trajectories.detach().cpu().numpy()[0],
        states_nominal=nominal_states,
        states_projected=projected_states,
        ltr_nominal=nominal_ltr,
        ltr_projected=projected_ltr,
        road_centerline_local=road_meta["centerline_local_xy"],
        road_half_width=road_meta["half_width_m"],
    )
    if not args.no_plots:
        _save_plots(
            output_dir,
            nominal,
            {"control": projected_controls, "trajectory_local": projected_pose_local,
             "states": projected_states, "ltr": projected_ltr},
            road_meta, external, pose0, context, jtp_cfg,
            dt_s=float(jtp_cfg["execution"]["dt_s"]),
        )

    initial_max = initial_metrics.get("max_violation", float("nan"))
    final_max = final_metrics.get("max_violation", float("nan"))
    print(
        f"[JTP SINGLE] sample={index} status={result.status} success={result.success} "
        f"initial_violation={initial_max} final_violation={final_max} "
        f"delta_u={metrics['control_delta_l2']:.6f} "
        f"latency_ms={jtp_latency_ms:.2f}"
    )
    print(f"[K2 CONSISTENCY] max_abs={consistency_max_abs:.3e} "
          f"tolerance={args.koopman_consistency_tolerance:.3e} "
          f"pass={metrics['nominal_koopman_consistency_pass']}")
    if initial_metrics.get("max_violation_by_group") is not None:
        print("[CONSTRAINTS BEFORE] " + json.dumps(
            initial_metrics["max_violation_by_group"], ensure_ascii=False))
    if final_metrics.get("max_violation_by_group") is not None:
        print("[CONSTRAINTS AFTER ] " + json.dumps(
            final_metrics["max_violation_by_group"], ensure_ascii=False))
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate real Control Diffusion + embedded K2 + JTP on one-ego TruckSim samples.")
    parser.add_argument("--config", default="configs/train/diffusion.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Real control-Diffusion checkpoint. Defaults to runtime.checkpoint.control in --config.")
    parser.add_argument("--dataset-root", default=None,
                        help="TruckSim K2 paired-anchor dataset. Defaults to control.training.dataset_root.")
    parser.add_argument("--split", choices=("val", "test", "train", "all"), default="val")
    parser.add_argument("--sample-index", default="auto",
                        help="Dataset index or 'auto' to scan for a sample with a sufficiently wide same-road corridor.")
    parser.add_argument("--scan-limit", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--jtp-config", default="configs/projection/jtp.yaml")
    parser.add_argument("--output-dir", default="outputs/jtp_validation/single_vehicle")
    parser.add_argument("--max-external-actors", type=int, default=8)
    parser.add_argument("--max-external-distance", type=float, default=120.0)
    parser.add_argument("--koopman-consistency-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    repo_config = resolve_repo_file(args.config)
    raw_cfg = _load_raw_config(repo_config)
    control_cfg = raw_cfg.get("control", {})
    training_cfg = control_cfg.get("training", {})
    runtime_cfg = raw_cfg.get("runtime", {})
    checkpoint_cfg = runtime_cfg.get("checkpoint", {})
    checkpoint_value = args.checkpoint or (checkpoint_cfg.get("control") if isinstance(checkpoint_cfg, dict) else None)
    dataset_value = args.dataset_root or training_cfg.get("dataset_root")
    if not checkpoint_value:
        raise ValueError("Specify --checkpoint or runtime.checkpoint.control in the YAML config")
    if not dataset_value:
        raise ValueError("Specify --dataset-root or control.training.dataset_root in the YAML config")
    args.checkpoint_resolved = resolve_repo_file(checkpoint_value)
    args.dataset_root_resolved = resolve_repo_file(dataset_value)
    if not args.checkpoint_resolved.is_file():
        raise FileNotFoundError(f"Control-Diffusion checkpoint not found: {args.checkpoint_resolved}")
    if not args.dataset_root_resolved.is_dir():
        raise FileNotFoundError(f"Expert dataset root not found: {args.dataset_root_resolved}")

    jtp_config_path = resolve_repo_file(args.jtp_config)
    jtp_cfg = yaml.safe_load(jtp_config_path.read_text(encoding="utf-8"))
    if not isinstance(jtp_cfg, dict):
        raise ValueError("JTP YAML must be a mapping")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    print(f"[LOAD] Control-Diffusion checkpoint: {args.checkpoint_resolved}")
    runtime = ControlDiffusionRuntime(
        args.checkpoint_resolved,
        device=str(device),
        deterministic_seed=args.seed,
        inference_steps=args.inference_steps,
        jtp_enabled=False,
    )
    if runtime.model.vehicle_model.kind != "koopman_k2":
        raise RuntimeError(
            "The selected checkpoint does not contain a K2 execution model. "
            f"execution.model={runtime.model.vehicle_model.kind!r}; JTP requires koopman_k2."
        )
    if not runtime.model.vehicle_model.koopman_metadata:
        raise RuntimeError("Loaded control checkpoint has no reconstructed K2 metadata")
    k2_state_keys = [key for key in runtime.model.state_dict()
                     if key.startswith("vehicle_model.koopman.")]
    if not k2_state_keys:
        raise RuntimeError("Loaded control checkpoint contains no embedded K2 model parameters")
    frozen_k2 = runtime.model.vehicle_model.koopman
    if any(parameter.requires_grad for parameter in frozen_k2.parameters()):
        raise RuntimeError("K2 parameters must remain frozen during JTP validation")
    print(f"[MODEL] execution=koopman_k2 | K2_state_tensors={len(k2_state_keys)} "
          f"| K2_parameters={sum(p.numel() for p in frozen_k2.parameters())}")
    model_dt = float(runtime.model.ctrl_cfg.dt)
    model_horizon = int(runtime.model.ctrl_cfg.horizon_steps)
    if not np.isclose(model_dt, float(jtp_cfg["execution"]["dt_s"]), atol=1.0e-8):
        raise ValueError(f"Diffusion dt={model_dt} differs from JTP dt={jtp_cfg['execution']['dt_s']}")
    if model_horizon != int(jtp_cfg["execution"]["horizon_steps"]):
        raise ValueError(f"Diffusion horizon={model_horizon} differs from JTP horizon={jtp_cfg['execution']['horizon_steps']}")

    projector = JointTrajectoryProjector(runtime.model.vehicle_model, config=jtp_config_path)
    dataset = AllMergeExpertShardDataset(AllMergeExpertDatasetConfig(
        dataset_root=args.dataset_root_resolved, split=args.split, include_metadata=True))
    if len(dataset) == 0:
        raise RuntimeError(f"No expert samples in split={args.split!r}")

    if args.sample_index.lower() == "auto":
        (sample_index, _, _, _, _, _, _, _) = _find_auto_sample(
            dataset, scan_limit=args.scan_limit, device=device,
            jtp_cfg=jtp_cfg, requires_coarse_controls=(runtime.model.anchor_manager is None))
        indices = [sample_index]
        print(f"[SELECT] auto-selected validation sample index={sample_index}")
    else:
        indices = [int(x.strip()) for x in args.sample_index.split(",") if x.strip()]
    if not indices:
        raise ValueError("No sample indices were selected")
    for index in indices:
        if index < 0 or index >= len(dataset):
            raise IndexError(f"sample index {index} outside split length {len(dataset)}")

    output_root = resolve_repo_file(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict["checkpoint_resolved"] = str(args.checkpoint_resolved)
    args_dict["dataset_root_resolved"] = str(args.dataset_root_resolved)
    args_dict["device_resolved"] = str(device)
    (output_root / "run_config.json").write_text(
        json.dumps(_to_jsonable(args_dict), ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for index in indices:
        rows.append(_run_one(runtime, projector, dataset, index, args, jtp_cfg, output_root))
    csv_path = output_root / "summary.csv"
    columns = [
        "sample_index", "jtp_success", "jtp_status", "jtp_iterations", "jtp_latency_ms",
        "diffusion_latency_ms", "payload_kg", "road_mu", "external_traffic",
        "nominal_koopman_consistency_max_abs", "nominal_koopman_consistency_pass", "control_delta_l2",
        "trajectory_deviation_rmse_m", "trajectory_deviation_max_m",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            compact = dict(row)
            compact["external_traffic"] = row["external_traffic"].get("actor_count", 0)
            writer.writerow(compact)

    successes = sum(bool(row["jtp_success"]) for row in rows)
    print(f"[DONE] samples={len(rows)} feasible={successes}/{len(rows)}")
    print(f"[OUTPUT] {output_root}")
    if args.no_plots:
        print("[INFO] --no-plots selected; arrays and metrics were saved without figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
