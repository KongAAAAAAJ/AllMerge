"""Configuration loading and validation for TRACE Joint Trajectory Projection."""
from __future__ import annotations
from copy import deepcopy
import math
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
    required = ("execution", "stability", "geometry", "control", "optimization", "projection", "runtime")
    missing = [key for key in required if not isinstance(cfg.get(key), dict)]
    if missing:
        raise ValueError(f"JTP configuration missing mapping sections: {missing}")
    execution = cfg["execution"]
    if float(execution.get("dt_s", 0)) <= 0 or float(execution.get("wheelbase_m", 0)) <= 0:
        raise ValueError("execution.dt_s and wheelbase_m must be positive")
    if int(execution.get("horizon_steps", 0)) < 1 or int(execution.get("control_steps", 0)) < 1:
        raise ValueError("execution horizon_steps/control_steps must be positive")
    if int(execution["control_steps"]) != int(execution["horizon_steps"]):
        raise ValueError("JTP requires identity control interpolation (control_steps=horizon_steps)")
    context = cfg["runtime"].get("context")
    if not isinstance(context, dict):
        raise ValueError("runtime must define context mapping")
    if int(context.get("history_state_steps", 0)) != 5:
        raise ValueError("Current K2 requires runtime.context.history_state_steps=5")
    if abs(float(context.get("sample_dt_s", 0)) - float(execution["dt_s"])) > 1e-8:
        raise ValueError("runtime.context.sample_dt_s must match execution.dt_s")
    if float(context.get("max_gap_tolerance_s", -1)) < 0:
        raise ValueError("runtime.context.max_gap_tolerance_s must be nonnegative")

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
    if len(opt.get("control_weights", [])) != 2 or min(map(float, opt["control_weights"])) < 0:
        raise ValueError("control_weights must contain two nonnegative values")

    proj = cfg['projection']
    if min(int(proj['max_iterations']), int(proj['max_stalled_iterations']), int(proj['constraint_sampling_stride'])) < 1:
        raise ValueError('Projection iteration limits and sampling stride must be positive')
    if len(proj['residual_scales']['stability']) != 4 or min(proj['residual_scales']['stability']) <= 0:
        raise ValueError('Four positive stability scales are required')
    if not 0 < proj['alpha_minimum'] <= proj['alpha_initial'] <= 1 or not 0 < proj['alpha_shrink_factor'] < 1:
        raise ValueError('Invalid line-search settings')
    for key in ('constraint_tolerance', 'nondegradation_tolerance', 'minimum_violation_improvement', 'change_tolerance'):
        if not math.isfinite(float(proj[key])) or float(proj[key]) < 0:
            raise ValueError(f'Invalid projection.{key}')
    qp = proj['qp']
    for key in ('eps_abs', 'eps_rel', 'time_limit_ms', 'control_correction_weight'):
        if not math.isfinite(float(qp[key])) or float(qp[key]) <= 0:
            raise ValueError(f'projection.qp.{key} must be positive and finite')
    if qp['max_iter'] < 1 or min(qp['state_slack'].values()) <= 0:
        raise ValueError('Invalid QP iteration limit or slack penalties')
