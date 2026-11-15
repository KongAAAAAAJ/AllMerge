from __future__ import annotations

from typing import Mapping, Sequence, Tuple
import math

import numpy as np


_REQUIRED = (
    ("stability.sideslip_limit_rad",),
    ("stability.roll_limit_rad",),
    ("stability.ltr_limit",),
    ("geometry.road_margin_m",),
    ("geometry.curvature_limit_per_m",),
    ("geometry.ego_length_m",),
    ("geometry.ego_width_m",),
    ("geometry.collision_margin_m",),
    ("control.acceleration_rate_max_mps3",),
    ("control.steering_rate_max_rad_s",),
)


def _get(cfg: Mapping, dotted: str, default=None):
    value = cfg
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def validate_feasibility_config(cfg: Mapping) -> None:
    missing = []
    for (key,) in _REQUIRED:
        value = _get(cfg, key)
        if value is None:
            missing.append(key)
    for key in ("stability.sideslip_limit_rad", "stability.roll_limit_rad",
                "stability.ltr_limit", "geometry.curvature_limit_per_m",
                "geometry.ego_length_m", "geometry.ego_width_m"):
        value = _get(cfg, key)
        if value is not None and float(value) <= 0:
            raise ValueError(f"Feasibility parameter {key} must be positive")
    for key in ("geometry.road_margin_m", "geometry.collision_margin_m",
                "control.acceleration_rate_max_mps3", "control.steering_rate_max_rad_s"):
        value = _get(cfg, key)
        if value is not None and float(value) < 0:
            raise ValueError(f"Feasibility parameter {key} must be nonnegative")
    if missing:
        raise ValueError(
            "Full TRACE pre-MATP feasibility requires explicit experimental thresholds. "
            "Fill these values in trace_feasibility.yaml: " + ", ".join(missing)
        )


def _circle_offsets(length: float, width: float, count: int) -> Tuple[np.ndarray, float]:
    radius = max(float(width) / 2.0, 1e-3)
    start, end = -float(length) / 2.0 + radius, float(length) / 2.0 - radius
    if end < start:
        return np.asarray([0.0], dtype=np.float64), radius
    required = int(math.ceil((end - start) / max(2.0 * radius, 1e-6))) + 1
    n = max(int(count), required, 1)
    return np.linspace(start, end, n, dtype=np.float64), radius


def _circle_centers(pose: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    return np.asarray(pose[:2], dtype=np.float64)[None] + offsets[:, None] * np.asarray([c, s])


def _polyline_project(point: np.ndarray, line: np.ndarray):
    seg = line[1:] - line[:-1]
    denom = np.sum(seg * seg, axis=-1).clip(1e-9)
    rel = point[None] - line[:-1]
    t = np.clip(np.sum(rel * seg, axis=-1) / denom, 0.0, 1.0)
    proj = line[:-1] + t[:, None] * seg
    d2 = np.sum((point[None] - proj) ** 2, axis=-1)
    idx = int(np.argmin(d2))
    return proj[idx], idx, float(t[idx]), float(d2[idx])


def _road_boundary_arrays(map_polylines: np.ndarray, map_valid_mask: np.ndarray):
    maps = np.asarray(map_polylines, dtype=np.float64)
    valid = np.asarray(map_valid_mask, dtype=bool)
    ids = np.flatnonzero(valid)
    if len(ids) == 0:
        raise ValueError("Road feasibility requires valid map polylines")
    current = [i for i in ids if np.max(maps[i, :, 6]) > 0.5]
    target = [i for i in ids if np.max(maps[i, :, 9]) > 0.5]
    ref_idx = current[0] if current else (target[0] if target else int(ids[0]))
    ref = maps[ref_idx, :, :2]
    delta = np.diff(ref, axis=0)
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    # Segment normals; point projection returns the local segment index.
    normals = np.stack((-np.sin(heading), np.cos(heading)), axis=-1)
    lower = np.full((len(ref),), np.inf, dtype=np.float64)
    upper = np.full((len(ref),), -np.inf, dtype=np.float64)
    for lane_idx in ids:
        lane = maps[lane_idx, :, :2]
        width = float(np.median(maps[lane_idx, :, 4]))
        lateral = np.sum((lane - ref) * np.vstack((normals, normals[-1])), axis=-1)
        lower = np.minimum(lower, lateral - width / 2.0)
        upper = np.maximum(upper, lateral + width / 2.0)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Could not reconstruct road boundaries from map lane polylines")
    return ref, lower, upper


def _road_feasible(poses: np.ndarray, features: Mapping, cfg: Mapping) -> bool:
    geo = cfg["geometry"]
    length, width = float(geo["ego_length_m"]), float(geo["ego_width_m"])
    margin = float(geo["road_margin_m"])
    offsets, radius = _circle_offsets(length, width, int(geo.get("footprint_circle_count", 7)))
    ref, lower, upper = _road_boundary_arrays(features["map_polylines"], features["map_valid_mask"])
    for pose in poses:
        centers = _circle_centers(pose, offsets)
        for center in centers:
            proj, idx, t, _ = _polyline_project(center, ref)
            seg = ref[idx + 1] - ref[idx]
            norm = np.linalg.norm(seg).clip(1e-9)
            normal = np.asarray([-seg[1], seg[0]]) / norm
            d = float(np.dot(center - proj, normal))
            lo = (1.0 - t) * lower[idx] + t * lower[idx + 1]
            hi = (1.0 - t) * upper[idx] + t * upper[idx + 1]
            if d - radius < lo + margin - 1e-6 or d + radius > hi - margin + 1e-6:
                return False
    return True


def _curvature_feasible(poses: np.ndarray, dt: float, limit: float, eps: float = 1e-3) -> Tuple[bool, float]:
    xy = np.asarray(poses[:, :2], dtype=np.float64)
    if len(xy) < 3:
        return False, float("inf")
    velocity = (xy[2:] - xy[:-2]) / (2.0 * dt)
    acceleration = (xy[2:] - 2.0 * xy[1:-1] + xy[:-2]) / (dt * dt)
    numerator = velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0]
    denominator = (np.sum(velocity * velocity, axis=-1) + eps * eps) ** 1.5
    curvature = numerator / np.maximum(denominator, 1e-9)
    peak = float(np.max(np.abs(curvature))) if len(curvature) else 0.0
    return peak <= limit + 1e-8, peak


def _stability_feasible(states: np.ndarray, ltr: np.ndarray, mu: float,
                        payload_kg: float, cfg: Mapping) -> Dict[str, object]:
    st = cfg["stability"]
    vx, vy, yaw_rate, roll = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    beta = np.arctan2(vy, vx)
    g = float(st.get("gravity_mps2", 9.81))
    v_eps = float(st.get("yaw_rate_speed_epsilon_mps", 1.0))
    yaw_limit = float(mu) * g / np.maximum(np.abs(vx), v_eps)
    beta_limit = float(st["sideslip_limit_rad"]) + float(st.get("sideslip_mu_slope_rad", 0.0)) * (1.0 - float(mu))
    roll_limit = (float(st["roll_limit_rad"])
                  + float(st.get("roll_payload_slope_rad_per_1000kg", 0.0))
                  * (float(payload_kg) - float(st.get("roll_payload_reference_kg", 0.0))) / 1000.0
                  + float(st.get("roll_mu_slope_rad", 0.0)) * (1.0 - float(mu)))
    ltr_limit = float(st["ltr_limit"])
    if min(beta_limit, roll_limit, ltr_limit) <= 0:
        raise ValueError("Condition-dependent stability limit became nonpositive")
    values = {
        "yaw_rate_feasible": bool(np.all(np.abs(yaw_rate[1:]) <= yaw_limit[1:] + 1e-8)),
        "sideslip_feasible": bool(np.all(np.abs(beta[1:]) <= beta_limit + 1e-8)),
        "roll_feasible": bool(np.all(np.abs(roll[1:]) <= roll_limit + 1e-8)),
        "ltr_feasible": bool(np.all(np.abs(ltr[1:]) <= ltr_limit + 1e-8)),
        "max_abs_yaw_rate_rad_s": float(np.max(np.abs(yaw_rate[1:]))),
        "max_abs_sideslip_rad": float(np.max(np.abs(beta[1:]))),
        "max_abs_roll_rad": float(np.max(np.abs(roll[1:]))),
        "max_abs_ltr": float(np.max(np.abs(ltr[1:]))),
    }
    values["stability_feasible"] = all(values[k] for k in (
        "yaw_rate_feasible", "sideslip_feasible", "roll_feasible", "ltr_feasible"))
    return values


def _actor_circles(position: np.ndarray, heading: float, length: float, width: float,
                   count: int) -> Tuple[np.ndarray, float]:
    offsets, radius = _circle_offsets(length, width, count)
    direction = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
    return position[None] + offsets[:, None] * direction[None], radius


def _external_collision_feasible(poses: np.ndarray, features: Mapping, dt: float,
                                 cfg: Mapping) -> bool:
    geo = cfg["geometry"]
    ego_offsets, ego_radius = _circle_offsets(
        float(geo["ego_length_m"]), float(geo["ego_width_m"]),
        int(geo.get("footprint_circle_count", 7)))
    margin = float(geo["collision_margin_m"])
    ego = np.asarray(features["ego_state"], dtype=np.float64)
    agents = np.asarray(features["agent_states"], dtype=np.float64)
    valid = np.asarray(features["agent_valid_mask"], dtype=bool)
    times = np.arange(0, len(poses), dtype=np.float64) * dt
    for agent_idx in np.flatnonzero(valid):
        actor = agents[agent_idx]
        # Other controlled trucks are evaluated jointly after synchronized grouping.
        if actor.shape[0] > 10 and actor[10] > 0.5:
            continue
        actor_v = actor[2:4] + ego[:2]
        actor_heading = math.atan2(float(actor[6]), float(actor[7]))
        actor_length = max(float(actor[8]), 0.5)
        actor_width = max(float(actor[9]), 0.3)
        actor_offsets, actor_radius = _circle_offsets(actor_length, actor_width, 3)
        for step, t in enumerate(times):
            ego_centers = _circle_centers(poses[step], ego_offsets)
            actor_position = actor[:2] + actor_v * t
            actor_centers = actor_position[None] + actor_offsets[:, None] * np.asarray(
                [math.cos(actor_heading), math.sin(actor_heading)])
            d2 = np.sum((ego_centers[:, None] - actor_centers[None]) ** 2, axis=-1)
            if np.any(d2 < (ego_radius + actor_radius + margin) ** 2):
                return False
    return True


def _control_feasible(controls: np.ndarray, dt: float, cfg: Mapping) -> Dict[str, object]:
    c = cfg["control"]
    accel, steer = controls[:, 0], controls[:, 1]
    da = np.diff(accel) / dt
    ds = np.diff(steer) / dt
    result = {
        "acceleration_feasible": bool(np.all(accel >= float(c.get("acceleration_min_mps2", -np.inf)) - 1e-8)
                                      and np.all(accel <= float(c.get("acceleration_max_mps2", np.inf)) + 1e-8)),
        "steering_feasible": bool(np.all(steer >= float(c.get("steering_min_rad", -np.inf)) - 1e-8)
                                   and np.all(steer <= float(c.get("steering_max_rad", np.inf)) + 1e-8)),
        "acceleration_rate_feasible": bool(np.all(np.abs(da) <= float(c["acceleration_rate_max_mps3"]) + 1e-8)),
        "steering_rate_feasible": bool(np.all(np.abs(ds) <= float(c["steering_rate_max_rad_s"]) + 1e-8)),
        "max_abs_acceleration_mps2": float(np.max(np.abs(accel))),
        "max_abs_steering_rad": float(np.max(np.abs(steer))),
        "max_abs_acceleration_rate_mps3": float(np.max(np.abs(da))) if len(da) else 0.0,
        "max_abs_steering_rate_rad_s": float(np.max(np.abs(ds))) if len(ds) else 0.0,
    }
    result["control_feasible"] = all(result[k] for k in (
        "acceleration_feasible", "steering_feasible",
        "acceleration_rate_feasible", "steering_rate_feasible"))
    return result


def evaluate_single_preprojection(plan: Mapping, cfg: Mapping, *, dt: float = 0.1) -> Dict[str, object]:
    """Evaluate the selected diffusion execution trajectory before MATP.

    Implements the original nonlinear checks described in TRACE Section V.
    It does not run the QP/SCP projector and does not alter controls or poses.
    """
    poses = np.asarray(plan["poses"], dtype=np.float64)
    controls = np.asarray(plan["controls"], dtype=np.float64)
    states = np.asarray(plan["states"], dtype=np.float64)
    ltr = np.asarray(plan["ltr"], dtype=np.float64).reshape(-1)
    features = plan["features"]
    mu = float(plan["road_mu"])
    payload = float(plan["payload_kg"])
    if poses.shape != (41, 3) or controls.shape != (40, 2) or states.shape != (41, 5) or ltr.shape != (41,):
        raise ValueError("Raw feasibility expects poses [41,3], controls [40,2], states [41,5], LTR [41]")
    stability = _stability_feasible(states, ltr, mu, payload, cfg)
    road_ok = _road_feasible(poses, features, cfg)
    curvature_ok, curvature_peak = _curvature_feasible(
        poses, dt, float(cfg["geometry"]["curvature_limit_per_m"]))
    collision_ok = _external_collision_feasible(poses, features, dt, cfg)
    control = _control_feasible(controls, dt, cfg)
    result = {
        **stability,
        **control,
        "road_feasible": bool(road_ok),
        "curvature_feasible": bool(curvature_ok),
        "external_collision_feasible": bool(collision_ok),
        "intertruck_collision_feasible": True,  # Filled by synchronized group check.
        "max_abs_curvature_per_m": float(curvature_peak),
        "raw_feasibility_scope": "selected_diffusion_plan_before_joint_projection",
    }
    result["feasible"] = bool(
        result["stability_feasible"] and result["road_feasible"]
        and result["curvature_feasible"] and result["external_collision_feasible"]
        and result["control_feasible"]
    )
    return result


def _world_poses(local_poses: np.ndarray, pose0: np.ndarray) -> np.ndarray:
    x0, y0, yaw0 = map(float, pose0)
    c, s = math.cos(yaw0), math.sin(yaw0)
    x = x0 + c * local_poses[:, 0] - s * local_poses[:, 1]
    y = y0 + s * local_poses[:, 0] + c * local_poses[:, 1]
    yaw = yaw0 + local_poses[:, 2]
    return np.stack((x, y, yaw), axis=-1)


def apply_intertruck_group_constraints(plans: Sequence[dict], cfg: Mapping) -> None:
    """Apply synchronized pairwise truck collision constraints in world frame.

    The collector stores one ego sample per controlled truck and frame. The
    evaluator groups these samples by episode/frame, transforms each K2-local
    prediction to world coordinates, and checks every covering-circle pair.
    """
    if len(plans) <= 1:
        for plan in plans:
            plan["feasibility"]["intertruck_collision_feasible"] = True
        return
    geo = cfg["geometry"]
    offsets, radius = _circle_offsets(float(geo["ego_length_m"]), float(geo["ego_width_m"]),
                                      int(geo.get("footprint_circle_count", 7)))
    margin = float(geo["collision_margin_m"])
    world = [_world_poses(np.asarray(p["poses"], dtype=np.float64), np.asarray(p["pose0"], dtype=np.float64))
             for p in plans]
    group_ok = True
    for i in range(len(plans)):
        for j in range(i + 1, len(plans)):
            for step in range(0, 41):
                ci = _circle_centers(world[i][step], offsets)
                cj = _circle_centers(world[j][step], offsets)
                d2 = np.sum((ci[:, None] - cj[None]) ** 2, axis=-1)
                if np.any(d2 < (2.0 * radius + margin) ** 2):
                    group_ok = False
                    break
            if not group_ok:
                break
        if not group_ok:
            break
    for plan in plans:
        plan["feasibility"]["intertruck_collision_feasible"] = bool(group_ok)
        f = plan["feasibility"]
        f["feasible"] = bool(f["feasible"] and group_ok)
