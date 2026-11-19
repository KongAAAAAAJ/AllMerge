from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch

from highway_env.planner.diffusion.projection import RoadGeometry


def _resample_polyline(points: np.ndarray, widths: np.ndarray, count: int = 32):
    points = np.asarray(points, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("Each valid map lane must contain at least two XY points")
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5
    points, widths = points[keep], widths[keep]
    if len(points) < 2:
        raise ValueError("Map lane degenerates after duplicate-point removal")
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if arc[-1] <= 1e-5:
        raise ValueError("Map lane has zero arc length")
    target = np.linspace(0.0, arc[-1], int(count))
    xy = np.stack([np.interp(target, arc, points[:, d]) for d in range(2)], axis=-1)
    width = np.interp(target, arc, widths)
    if not np.isfinite(xy).all() or not np.isfinite(width).all() or np.any(width <= 0):
        raise ValueError("Invalid lane coordinates or widths in JTP road geometry")
    return xy, width


def road_geometry_from_features(features: Mapping, pose0: np.ndarray, *, point_count: int = 32) -> RoadGeometry:
    """Convert ego-local lane features to the world-frame JTP road corridor."""
    maps = np.asarray(features["map_polylines"], dtype=np.float64)
    valid = np.asarray(features["map_valid_mask"], dtype=bool)
    pose0 = np.asarray(pose0, dtype=np.float64).reshape(3)
    if maps.ndim != 3 or maps.shape[-1] < 5 or valid.shape != (maps.shape[0],):
        raise ValueError("Expected map_polylines [L,P,D>=5] and map_valid_mask [L]")
    lanes, widths = [], []
    for lane_id in np.flatnonzero(valid):
        local_xy = maps[lane_id, :, :2]
        lane_width = np.maximum(maps[lane_id, :, 4], 0.1)
        xy, width = _resample_polyline(local_xy, lane_width, point_count)
        c, s = math.cos(float(pose0[2])), math.sin(float(pose0[2]))
        world = np.empty_like(xy)
        world[:, 0] = pose0[0] + c * xy[:, 0] - s * xy[:, 1]
        world[:, 1] = pose0[1] + s * xy[:, 0] + c * xy[:, 1]
        lanes.append(world)
        widths.append(width)
    if not lanes:
        raise ValueError("JTP feasibility requires at least one valid road polyline")
    centerlines = torch.as_tensor(np.stack(lanes), dtype=torch.float32)
    half_widths = torch.as_tensor(np.stack(widths) * 0.5, dtype=torch.float32)
    return RoadGeometry(centerlines_xy=centerlines,
                        left_width_m=half_widths,
                        right_width_m=half_widths)


def _circle_offsets(length: float, width: float, count: int = 3):
    radius = max(float(width) * 0.5, 1e-3)
    start, end = -float(length) * 0.5 + radius, float(length) * 0.5 - radius
    if end <= start:
        return np.asarray([0.0]), radius
    required = int(math.ceil((end - start) / max(2.0 * radius, 1e-6))) + 1
    return np.linspace(start, end, max(int(count), required)), radius


def _circle_centers(pose: np.ndarray, offsets: np.ndarray):
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    return np.asarray(pose[:2])[None] + offsets[:, None] * np.asarray([c, s])


def external_collision_free(local_poses: np.ndarray, features: Mapping, jtp_cfg: Mapping) -> bool:
    """Constant-velocity collision check for uncontrolled traffic actors."""
    poses = np.asarray(local_poses, dtype=np.float64)
    ego = np.asarray(features["ego_state"], dtype=np.float64)
    agents = np.asarray(features["agent_states"], dtype=np.float64)
    valid = np.asarray(features["agent_valid_mask"], dtype=bool)
    footprint = jtp_cfg["geometry"]["footprint"]
    ego_offsets = np.asarray(footprint["circle_offsets_m"], dtype=np.float64)
    ego_radius = float(footprint["circle_radius_m"])
    margin = float(jtp_cfg.get("evaluation", {}).get("collision_margin_m", 0.25))
    dt = float(jtp_cfg["execution"]["dt_s"])
    times = np.arange(len(poses), dtype=np.float64) * dt
    for idx in np.flatnonzero(valid):
        actor = agents[idx]
        if actor.shape[0] > 10 and actor[10] > 0.5:
            continue
        actor_velocity = actor[2:4] + ego[:2]
        actor_heading = math.atan2(float(actor[6]), float(actor[7]))
        actor_length = max(float(actor[8]), 0.5)
        actor_width = max(float(actor[9]), 0.3)
        actor_offsets, actor_radius = _circle_offsets(actor_length, actor_width, 3)
        direction = np.asarray([math.cos(actor_heading), math.sin(actor_heading)])
        for step, t in enumerate(times):
            ego_centers = _circle_centers(poses[step], ego_offsets)
            actor_centers = actor[:2][None] + actor_velocity[None] * t + actor_offsets[:, None] * direction[None]
            d2 = np.sum((ego_centers[:, None] - actor_centers[None]) ** 2, axis=-1)
            if np.any(d2 < (ego_radius + actor_radius + margin) ** 2):
                return False
    return True


def intertruck_group_collision_free(records: Sequence[dict], jtp_cfg: Mapping) -> bool:
    """Synchronized pairwise collision check on raw world-frame K2 predictions."""
    if len(records) <= 1:
        return True
    footprint = jtp_cfg["geometry"]["footprint"]
    offsets = np.asarray(footprint["circle_offsets_m"], dtype=np.float64)
    radius = float(footprint["circle_radius_m"])
    margin = float(jtp_cfg.get("evaluation", {}).get("collision_margin_m", 0.25))
    for i in range(len(records)):
        pi = np.asarray(records[i]["raw_world_poses"], dtype=np.float64)
        for j in range(i + 1, len(records)):
            pj = np.asarray(records[j]["raw_world_poses"], dtype=np.float64)
            if pi.shape != pj.shape:
                raise ValueError("Synchronized platoon predictions must share the same time grid")
            for t in range(len(pi)):
                ci = _circle_centers(pi[t], offsets)
                cj = _circle_centers(pj[t], offsets)
                d2 = np.sum((ci[:, None] - cj[None]) ** 2, axis=-1)
                if np.any(d2 < (2.0 * radius + margin) ** 2):
                    return False
    return True


def combine_preprojection_feasibility(jtp_metrics: Mapping, external_ok: bool,
                                      intertruck_ok: bool) -> dict:
    """Combine exact JTP internal constraints with independently evaluated collisions."""
    jtp_ok = bool(jtp_metrics["projector_constraints_satisfied"])
    return {
        "jtp_constraints_feasible": jtp_ok,
        "external_collision_feasible": bool(external_ok),
        "intertruck_collision_feasible": bool(intertruck_ok),
        "preprojection_feasible": bool(jtp_ok and external_ok and intertruck_ok),
        "jtp_max_violation": float(jtp_metrics.get("max_violation", float("nan"))),
        "jtp_violation_count_by_group": dict(jtp_metrics.get("violation_count_by_group", {})),
    }
