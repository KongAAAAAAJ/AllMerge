"""Configuration loading and validation for Joint Trajectory Projection."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "projection" / "jtp.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_jtp_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    filename = Path(path) if path is not None else default_config_path()
    if not filename.is_file():
        raise FileNotFoundError(f"JTP YAML configuration not found: {filename}")
    data = yaml.safe_load(filename.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JTP configuration must be a YAML mapping")
    if overrides:
        data = _deep_merge(data, overrides)
    validate_jtp_config(data)
    data["_config_path"] = str(filename.resolve())
    return data


def validate_jtp_config(cfg: dict) -> None:
    required = ("execution", "stability", "geometry", "collision", "control", "optimization", "qp")
    missing = [key for key in required if not isinstance(cfg.get(key), dict)]
    if missing:
        raise ValueError(f"JTP configuration missing mapping sections: {missing}")

    execution = cfg["execution"]
    if float(execution.get("dt_s", 0.0)) <= 0:
        raise ValueError("execution.dt_s must be positive")
    if int(execution.get("horizon_steps", 0)) <= 0 or int(execution.get("control_steps", 0)) <= 0:
        raise ValueError("execution horizon_steps/control_steps must be positive")
    if float(execution.get("wheelbase_m", 0.0)) <= 0:
        raise ValueError("execution.wheelbase_m must be positive")

    stability = cfg["stability"]
    if not 0.0 < float(stability.get("ltr_limit", 0.0)) < 1.0:
        raise ValueError("stability.ltr_limit must be in (0,1)")
    if not 0.0 < float(stability.get("minimum_mu_for_limits", 0.0)) <= 1.0:
        raise ValueError("stability.minimum_mu_for_limits must be in (0,1]")
    if float(stability.get("gravity_mps2", 0.0)) <= 0 or float(stability.get("minimum_speed_for_yaw_limit_mps", 0.0)) <= 0:
        raise ValueError("Stability gravity and minimum speed must be positive")
    if float(stability.get("yaw_rate_safety_factor", 0.0)) <= 0:
        raise ValueError("stability.yaw_rate_safety_factor must be positive")
    for section in ("sideslip", "roll"):
        values = stability.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"stability.{section} must be a mapping")
        if float(values.get("base_limit_deg", 0.0)) <= 0:
            raise ValueError(f"stability.{section}.base_limit_deg must be positive")
        if float(values.get("reference_mu", 0.0)) <= 0:
            raise ValueError(f"stability.{section}.reference_mu must be positive")
        if float(values.get("scale_min", 0.0)) <= 0 or float(values.get("scale_max", 0.0)) < float(values.get("scale_min", 0.0)):
            raise ValueError(f"Invalid stability.{section} friction scale bounds")
    if float(stability["roll"].get("payload_reference_kg", 0.0)) <= 0:
        raise ValueError("stability.roll.payload_reference_kg must be positive")

    geometry = cfg["geometry"]
    if float(geometry.get("curvature_limit_per_m", 0.0)) <= 0:
        raise ValueError("geometry.curvature_limit_per_m must be positive")
    if float(geometry.get("road_clearance_m", -1.0)) < 0:
        raise ValueError("geometry.road_clearance_m must be nonnegative")
    footprint = geometry.get("footprint", {})
    if not footprint.get("circle_offsets_m") or float(footprint.get("circle_radius_m", 0.0)) <= 0:
        raise ValueError("geometry.footprint requires circle_offsets_m and positive circle_radius_m")

    collision = cfg["collision"]
    if collision.get("qp_compact_footprints", False):
        if not collision.get("qp_ego_circle_offsets_m") or float(collision.get("qp_ego_circle_radius_m", 0.0)) <= 0:
            raise ValueError("Compact collision footprint requires ego offsets and positive radius")
        if not collision.get("qp_external_circle_offsets_m") or float(collision.get("qp_external_circle_radius_m", 0.0)) <= 0:
            raise ValueError("Compact collision footprint requires external offsets and positive radius")
    if float(collision.get("clearance_m", -1.0)) < 0:
        raise ValueError("collision.clearance_m must be nonnegative")
    external = collision.get("external_footprint", {})
    if not external.get("circle_offsets_m") or float(external.get("circle_radius_m", 0.0)) <= 0:
        raise ValueError("collision.external_footprint requires circle_offsets_m and positive circle_radius_m")

    control = cfg["control"]
    if float(control["acceleration_min_mps2"]) >= float(control["acceleration_max_mps2"]):
        raise ValueError("Invalid acceleration bounds")
    if float(control["steering_min_rad"]) >= float(control["steering_max_rad"]):
        raise ValueError("Invalid steering bounds")
    if min(float(control["acceleration_rate_max_mps3"]),
           float(control["steering_rate_max_radps"])) <= 0:
        raise ValueError("Control rate limits must be positive")
    if len(control.get("trust_scale", [])) != 2 or min(map(float, control["trust_scale"])) <= 0:
        raise ValueError("control.trust_scale must contain two positive values")

    opt = cfg["optimization"]
    if int(opt.get("max_scp_iterations", 0)) < 1 or int(opt.get("max_qp_retries", 0)) < 0:
        raise ValueError("Invalid SCP iteration/retry count")
    if float(opt.get("lambda_u", -1.0)) < 0:
        raise ValueError("optimization.lambda_u must be nonnegative")
    if len(opt.get("trajectory_weights", [])) != 3 or min(map(float, opt["trajectory_weights"])) < 0:
        raise ValueError("trajectory_weights must contain three nonnegative values")
    if len(opt.get("control_weights", [])) != 2 or min(map(float, opt["control_weights"])) < 0:
        raise ValueError("control_weights must contain two nonnegative values")
    if float(opt.get("minimum_trust_radius", 0.0)) <= 0 or float(opt.get("initial_trust_radius", 0.0)) <= 0:
        raise ValueError("Initial and minimum trust radii must be positive")
    if not 0 < float(opt.get("trust_shrink_factor", 0.0)) < 1 or float(opt.get("trust_grow_factor", 0.0)) <= 1:
        raise ValueError("Invalid trust-region adaptation factors")
    if not 0 < float(opt.get("alpha_shrink_factor", 0.0)) < 1:
        raise ValueError("alpha_shrink_factor must be in (0,1)")
    if float(opt.get("merit_penalty", 0.0)) <= 0:
        raise ValueError("merit_penalty must be positive")
    if float(opt.get("maximum_trust_radius", 0.0)) < float(opt.get("initial_trust_radius", 0.0)):
        raise ValueError("maximum_trust_radius must be >= initial_trust_radius")
    if not 0 < float(opt.get("alpha_minimum", 0.0)) <= float(opt.get("alpha_initial", 0.0)) <= 1:
        raise ValueError("Invalid line-search alpha bounds")
    merit_scales = opt.get("merit_scales", {})
    if len(merit_scales.get("stability", [])) != 4 or min(map(float, merit_scales["stability"])) <= 0:
        raise ValueError("optimization.merit_scales.stability must contain four positive scales")
    for key in ("road_residual_m", "curvature_residual_per_m2", "collision_residual_m2"):
        if float(merit_scales.get(key, 0.0)) <= 0:
            raise ValueError(f"optimization.merit_scales.{key} must be positive")

    profile = cfg.get("constraint_profile", {})
    for name in ("stability", "road_geometry", "collision", "control_box", "control_rate", "trust_region"):
        if name in profile and not isinstance(profile[name], bool):
            raise ValueError(f"constraint_profile.{name} must be boolean")
    if profile.get("road_geometry", True) is False and profile.get("collision", True) is False and profile.get("control_box", True) is False:
        raise ValueError("At least one safety/control constraint group must remain enabled")

    qp = cfg["qp"]
    if int(qp.get("max_iter", 0)) < 1 or float(qp.get("eps_abs", 0.0)) <= 0 or float(qp.get("eps_rel", 0.0)) <= 0:
        raise ValueError("Invalid QP solver settings")
    if float(qp.get("diagonal_regularization", -1.0)) < 0:
        raise ValueError("qp.diagonal_regularization must be nonnegative")
