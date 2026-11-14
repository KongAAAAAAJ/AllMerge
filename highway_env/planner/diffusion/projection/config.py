"""Configuration loading and validation for TRACE Joint Trajectory Projection."""
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
    required = ("execution", "stability", "geometry", "control", "optimization", "projection")
    missing = [key for key in required if not isinstance(cfg.get(key), dict)]
    if missing:
        raise ValueError(f"JTP configuration missing mapping sections: {missing}")
    execution = cfg["execution"]
    if float(execution.get("dt_s", 0)) <= 0 or float(execution.get("wheelbase_m", 0)) <= 0:
        raise ValueError("execution.dt_s and wheelbase_m must be positive")
    if int(execution.get("horizon_steps", 0)) < 1 or int(execution.get("control_steps", 0)) < 1:
        raise ValueError("execution horizon_steps/control_steps must be positive")

    stability = cfg["stability"]
    if not 0 < float(stability.get("ltr_limit", 0)) < 1:
        raise ValueError("stability.ltr_limit must be in (0,1)")
    if not 0 < float(stability.get("minimum_mu_for_limits", 0)) <= 1:
        raise ValueError("stability.minimum_mu_for_limits must be in (0,1]")
    if float(stability.get("gravity_mps2", 0)) <= 0 or float(stability.get("minimum_speed_for_yaw_limit_mps", 0)) <= 0:
        raise ValueError("Stability gravity and minimum speed must be positive")
    if float(stability.get("yaw_rate_safety_factor", 0)) <= 0:
        raise ValueError("stability.yaw_rate_safety_factor must be positive")
    for name in ("sideslip", "roll"):
        item = stability.get(name)
        if not isinstance(item, dict) or float(item.get("base_limit_deg", 0)) <= 0:
            raise ValueError(f"stability.{name}.base_limit_deg must be positive")
        if float(item.get("reference_mu", 0)) <= 0 or float(item.get("scale_min", 0)) <= 0:
            raise ValueError(f"Invalid stability.{name} friction parameters")
        if float(item.get("scale_max", 0)) < float(item.get("scale_min", 0)):
            raise ValueError(f"Invalid stability.{name} scale range")
    if float(stability["roll"].get("payload_reference_kg", 0)) <= 0:
        raise ValueError("stability.roll.payload_reference_kg must be positive")

    geometry = cfg["geometry"]
    if float(geometry.get("curvature_limit_per_m", 0)) <= 0 or float(geometry.get("road_clearance_m", -1)) < 0:
        raise ValueError("Invalid road geometry limits")
    footprint = geometry.get("footprint", {})
    if not footprint.get("circle_offsets_m") or float(footprint.get("circle_radius_m", 0)) <= 0:
        raise ValueError("geometry.footprint requires circle_offsets_m and positive circle_radius_m")

    control = cfg["control"]
    if float(control.get("acceleration_min_mps2", 0)) >= float(control.get("acceleration_max_mps2", 0)):
        raise ValueError("Invalid acceleration bounds")
    if float(control.get("steering_min_rad", 0)) >= float(control.get("steering_max_rad", 0)):
        raise ValueError("Invalid steering bounds")
    if len(control.get("trust_scale", [])) != 2 or min(map(float, control["trust_scale"])) <= 0:
        raise ValueError("control.trust_scale must contain two positive values")

    opt = cfg["optimization"]
    if float(opt.get("lambda_u", -1)) < 0:
        raise ValueError("optimization.lambda_u must be nonnegative")
    if len(opt.get("trajectory_weights", [])) != 3 or min(map(float, opt["trajectory_weights"])) < 0:
        raise ValueError("trajectory_weights must contain three nonnegative values")
    if len(opt.get("control_weights", [])) != 2 or min(map(float, opt["control_weights"])) < 0:
        raise ValueError("control_weights must contain two nonnegative values")

    proj = cfg["projection"]
    if int(proj.get("max_iterations", 0)) < 1 or int(proj.get("max_stalled_iterations", 0)) < 1:
        raise ValueError("Invalid projection iteration limits")
    active = proj.get("max_active_per_group", {})
    for key in ("stability", "road", "curvature", "control"):
        if int(active.get(key, 0)) < 1:
            raise ValueError(f"projection.max_active_per_group.{key} must be positive")
    scales = proj.get("residual_scales", {})
    if len(scales.get("stability", [])) != 4 or min(map(float, scales["stability"])) <= 0:
        raise ValueError("projection.residual_scales.stability must contain four positive values")
    for key in ("road_m", "curvature_per_m2"):
        if float(scales.get(key, 0)) <= 0:
            raise ValueError(f"projection.residual_scales.{key} must be positive")
    for key in ("damping", "metric_regularization", "merit_penalty", "max_step_norm"):
        if float(proj.get(key, 0)) <= 0:
            raise ValueError(f"projection.{key} must be positive")
    if int(proj.get("damping_retries", 0)) < 1:
        raise ValueError("projection.damping_retries must be positive")
    if not 0 < float(proj.get("trust_shrink_factor", 0)) < 1 or float(proj.get("trust_grow_factor", 0)) <= 1:
        raise ValueError("Invalid adaptive trust factors")
    if not 0 < float(proj.get("alpha_shrink_factor", 0)) < 1:
        raise ValueError("alpha_shrink_factor must be in (0,1)")
    if not 0 < float(proj.get("alpha_minimum", 0)) <= float(proj.get("alpha_initial", 0)) <= 1:
        raise ValueError("Invalid projection line-search bounds")
    if not 0 < float(proj.get("minimum_trust_radius", 0)) <= float(proj.get("initial_trust_radius", 0)) <= float(proj.get("maximum_trust_radius", 0)):
        raise ValueError("Invalid projection trust-radius bounds")
