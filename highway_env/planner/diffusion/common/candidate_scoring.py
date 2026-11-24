"""Safety-first, target-point-independent scoring of TRACE diffusion candidates.

The same deterministic selector is called by training, inference and paper
/evaluation entry points.  It uses only current scene observations, candidate
K2 execution predictions, road geometry and calibrated vehicle parameters.
No expert future trajectory or navigation target-point distance enters the
candidate score.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch

from .safety_losses import (
    background_constant_velocity_poses,
    pairwise_vehicle_safety,
    road_boundary_safety_loss,
)
from .fuel import FuelParameters, vt_cpfm_from_states
from .progress import centerline_progress


ROAD_KEYS = (
    "road_lane_geometry", "road_lane_point_mask", "road_lane_mask",
    "road_lane_connections", "road_current_lane_slot", "road_target_lane_slot",
)


def _repeat_batch(value: torch.Tensor, modes: int) -> torch.Tensor:
    return value[:, None].expand(value.shape[0], modes, *value.shape[1:]).reshape(
        value.shape[0] * modes, *value.shape[1:]
    )


def _masked_minmax(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-scene min-max normalization; equal-valued candidates receive 0.5."""
    inf = torch.full_like(values, float("inf"))
    ninf = torch.full_like(values, -float("inf"))
    lo = torch.where(valid, values, inf).amin(dim=1, keepdim=True)
    hi = torch.where(valid, values, ninf).amax(dim=1, keepdim=True)
    span = hi - lo
    normalized = (values - lo) / span.clamp_min(1.0e-8)
    normalized = torch.where(span > 1.0e-8, normalized, torch.full_like(values, 0.5))
    return torch.where(valid, normalized.clamp(0.0, 1.0), torch.zeros_like(values))


def _stability_metrics(states: torch.Tensor, ltr: torch.Tensor,
                       payload: torch.Tensor, road_mu: torch.Tensor,
                       cfg: Mapping, residual_scales, tolerance: float):
    """Vectorized counterpart of projection.constraints.stability_residual.

    Args use flattened candidate batches: states [N,H+1,5], ltr [N,H+1],
    payload/mu [N].  Limits and formulas exactly follow the JTP config.
    """
    section = cfg
    sideslip = section["sideslip"]
    roll = section["roll"]
    future = states[:, 1:]
    vx = future[..., 0]
    vy = future[..., 1]
    yaw_rate = future[..., 2]
    roll_angle = future[..., 3]
    beta = torch.atan2(vy, vx)

    mu = road_mu.reshape(-1, 1).to(vx).clamp_min(float(section["minimum_mu_for_limits"]))
    mass = payload.reshape(-1, 1).to(vx).clamp_min(0.0)
    v_safe = vx.abs().clamp_min(float(section["minimum_speed_for_yaw_limit_mps"]))
    yaw_limit = float(section["yaw_rate_safety_factor"]) * mu * float(section["gravity_mps2"]) / v_safe

    beta_scale = (mu / float(sideslip["reference_mu"])).pow(float(sideslip["friction_exponent"]))
    beta_scale = beta_scale.clamp(float(sideslip["scale_min"]), float(sideslip["scale_max"]))
    beta_limit = math.radians(float(sideslip["base_limit_deg"])) * beta_scale

    roll_mu_scale = (mu / float(roll["reference_mu"])).pow(float(roll["friction_exponent"]))
    roll_mu_scale = roll_mu_scale.clamp(float(roll["scale_min"]), float(roll["scale_max"]))
    payload_scale = 1.0 / (1.0 + float(roll["payload_sensitivity"]) *
                           mass / float(roll["payload_reference_kg"]))
    roll_limit = math.radians(float(roll["base_limit_deg"])) * roll_mu_scale * payload_scale
    ltr_limit = vx.new_full(vx.shape, float(section["ltr_limit"]))

    response = torch.stack((yaw_rate, beta, roll_angle, ltr[:, 1:]), dim=-1)
    bounds = torch.stack((yaw_limit, beta_limit.expand_as(vx), roll_limit.expand_as(vx),
                          ltr_limit), dim=-1)
    raw_violation = response.abs() - bounds
    scales = raw_violation.new_tensor(tuple(float(x) for x in residual_scales)).view(1, 1, 4)
    normalized = raw_violation / scales
    positive = torch.relu(normalized)
    violated = normalized > float(tolerance)
    violation_rate = violated.float().mean(dim=(1, 2))
    severity = positive.square().mean(dim=(1, 2))
    max_violation = positive.amax(dim=(1, 2))
    return {
        "violation": violated.any(dim=(1, 2)),
        "violation_rate": violation_rate,
        "severity": severity,
        "max_violation": max_violation,
    }


@torch.no_grad()
def score_candidate_modes(
    controls_all: torch.Tensor,
    prediction: Mapping[str, torch.Tensor],
    features: Mapping[str, torch.Tensor],
    context: Mapping[str, torch.Tensor] | None,
    *,
    dt: float,
    safety_config: Mapping,
    scoring_config: Mapping,
    fuel_parameters: FuelParameters | None,
    execution_kind: str,
):
    """Select one candidate by safety feasibility, violation severity, efficiency.

    Lexicographic policy:
      1. If any candidate is feasible, discard every infeasible candidate.
      2. Otherwise minimize the number of violated safety categories, then the
         summed normalized violation severity.
      3. Within the remaining pool, maximize normalized fuel/progress utility.

    The target_point feature is deliberately never read here.
    """
    if controls_all.ndim != 4 or controls_all.shape[-1] != 2:
        raise ValueError("controls_all must be [B,M,H,2]")
    poses = prediction.get("poses")
    speeds = prediction.get("speeds")
    if poses is None or poses.ndim != 4 or poses.shape[:2] != controls_all.shape[:2]:
        raise ValueError("candidate prediction must contain poses [B,M,H+1,3]")
    batch, modes, steps = poses.shape[:3]
    valid = features.get("mode_valid_mask")
    if valid is None or valid.shape != (batch, modes):
        raise ValueError("mode_valid_mask must be [B,M]")
    valid = valid.bool()
    if not valid.any(dim=1).all():
        raise ValueError("Every scene must contain at least one valid candidate")

    flat_poses = poses.reshape(batch * modes, steps, 3)
    safety = safety_config
    dimensions = flat_poses.new_tensor((float(safety["vehicle_length_m"]),
                                        float(safety["vehicle_width_m"]))).expand(batch * modes, 2)
    zero = flat_poses.new_zeros(batch * modes)

    # All currently observed surrounding vehicles, including controlled
    # platoon members, are propagated with the same constant-velocity rule in
    # training and online inference.  Expert future platoon references are not
    # used by this selector.
    if "agent_states" not in features or "agent_valid_mask" not in features:
        raise KeyError("Candidate scoring requires observed agent_states and agent_valid_mask")
    other_poses, other_dims, other_valid = background_constant_velocity_poses(
        features["agent_states"], features["agent_valid_mask"], features["ego_state"],
        horizon_steps=steps - 1, dt=dt, include_controlled=bool(
            scoring_config.get("include_controlled_agents", True)
        ),
    )
    if other_poses.shape[1] == 0:
        collision_loss = zero
        ttc_loss = zero
        collision_margin_violation = torch.zeros_like(zero, dtype=torch.bool)
        ttc_violation = torch.zeros_like(zero, dtype=torch.bool)
        min_gap = torch.full_like(zero, float("inf"))
    else:
        other_poses = _repeat_batch(other_poses, modes)
        other_dims = _repeat_batch(other_dims, modes)
        other_valid = _repeat_batch(other_valid, modes)
        pair = pairwise_vehicle_safety(
            flat_poses, dimensions, other_poses, other_dims, other_valid,
            dt=dt,
            collision_margin_m=float(safety["collision_margin_m"]),
            ttc_threshold_s=float(safety["ttc_threshold_s"]),
            softplus_temperature_m=float(safety["softplus_temperature_m"]),
            collision_distance_scale_m=float(safety["collision_distance_scale_m"]),
            ttc_distance_scale_m=float(safety["ttc_distance_scale_m"]),
        )
        collision_loss = pair["collision_loss_per_sample"]
        ttc_loss = pair["ttc_loss_per_sample"]
        collision_margin_violation = pair["margin_violation_per_sample"]
        ttc_violation = pair["ttc_violation_per_sample"]
        min_gap = pair["min_gap_per_sample"]

    missing_road = [key for key in ROAD_KEYS if key not in features]
    if missing_road:
        if execution_kind == "koopman_k2":
            raise KeyError(f"K2 Candidate Scoring requires live road geometry: {missing_road}")
        road_loss = zero
        road_violation = torch.zeros_like(zero, dtype=torch.bool)
        road_violation_rate = zero
        road_max_violation = zero
    else:
        road = road_boundary_safety_loss(
            flat_poses, dimensions,
            _repeat_batch(features["road_lane_geometry"], modes),
            _repeat_batch(features["road_lane_point_mask"], modes),
            _repeat_batch(features["road_lane_mask"], modes),
            _repeat_batch(features["road_lane_connections"], modes),
            _repeat_batch(features["road_current_lane_slot"], modes),
            _repeat_batch(features["road_target_lane_slot"], modes),
            softplus_temperature_m=float(safety["softplus_temperature_m"]),
            distance_scale_m=float(safety["road_distance_scale_m"]),
        )
        road_loss = road["loss_per_sample"]
        road_violation_rate = road["violation_rate_per_sample"]
        road_max_violation = road["max_violation_per_sample"]
        road_violation = road_max_violation > float(scoring_config.get("feasibility_tolerance", 1.0e-4))

    # Full chassis stability is evaluated with the same payload/friction
    # conditioned limits as JTP.  It is disabled only for kinematic smoke
    # models, which do not predict the required K2 chassis states.
    stability_violation = torch.zeros_like(zero, dtype=torch.bool)
    stability_violation_rate = zero
    stability_severity = zero
    stability_max_violation = zero
    if execution_kind == "koopman_k2":
        states = prediction.get("states")
        ltr = prediction.get("ltr")
        context = context or {}
        required = ("koopman_payload_kg", "koopman_road_mu")
        missing = [key for key in required if key not in context]
        if states is None or ltr is None or missing:
            raise KeyError(f"K2 candidate scoring requires states/LTR and context fields {missing}")
        stability_cfg = scoring_config["stability"]
        residual_scales = scoring_config["stability_residual_scales"]
        stab = _stability_metrics(
            states.reshape(batch * modes, steps, 5),
            ltr.reshape(batch * modes, steps),
            _repeat_batch(context["koopman_payload_kg"].reshape(batch, -1), modes).reshape(-1),
            _repeat_batch(context["koopman_road_mu"].reshape(batch, -1), modes).reshape(-1),
            stability_cfg, residual_scales,
            float(scoring_config.get("stability_tolerance", 1.0e-4)),
        )
        stability_violation = stab["violation"]
        stability_violation_rate = stab["violation_rate"]
        stability_severity = stab["severity"]
        stability_max_violation = stab["max_violation"]

    # Efficiency terms are candidate-relative within each scene.  This avoids
    # mixing the physical units of fuel and distance and makes the selector
    # invariant to a common offset shared by all candidates.
    if speeds is None:
        raise ValueError("Candidate scoring requires predicted speeds")
    fuel_total = zero
    efficiency_cfg = scoring_config.get("efficiency", {})
    fuel_weight = float(efficiency_cfg.get("fuel_weight", 1.0))
    progress_weight = float(efficiency_cfg.get("progress_weight", 1.0))
    if min(fuel_weight, progress_weight) < 0 or fuel_weight + progress_weight <= 0:
        raise ValueError("Candidate efficiency weights must be nonnegative and not both zero")
    if execution_kind == "koopman_k2" and fuel_weight > 0:
        if fuel_parameters is None:
            raise ValueError("K2 candidate scoring with fuel_weight>0 requires calibrated fuel parameters")
        payload = (context or {})["koopman_payload_kg"].reshape(batch)
        fuel_total = vt_cpfm_from_states(
            speeds.reshape(batch * modes, steps),
            _repeat_batch(payload[:, None], modes).reshape(-1),
            fuel_parameters, dt=dt,
        )["fuel_total"]
    else:
        fuel_weight = 0.0

    lane = features.get("target_lane_polyline")
    if lane is None:
        progress = flat_poses[:, -1, 0]
    else:
        lane_flat = _repeat_batch(lane, modes)
        segment = lane_flat[:, 1:, :2] - lane_flat[:, :-1, :2]
        nondegenerate = segment.square().sum(dim=-1).sum(dim=-1) > 1.0e-5
        progress = flat_poses[:, -1, 0].clone()
        if nondegenerate.any():
            progress[nondegenerate] = centerline_progress(
                flat_poses[nondegenerate, -1], lane_flat[nondegenerate]
            )

    fuel_2d = fuel_total.reshape(batch, modes)
    progress_2d = progress.reshape(batch, modes)
    valid_fuel = _masked_minmax(fuel_2d, valid)
    valid_progress = _masked_minmax(progress_2d, valid)
    utility_fuel = 1.0 - valid_fuel
    utility_progress = valid_progress
    total_efficiency_weight = fuel_weight + progress_weight
    efficiency = (fuel_weight * utility_fuel + progress_weight * utility_progress) / max(
        total_efficiency_weight, 1.0e-12
    )

    collision_2d = collision_loss.reshape(batch, modes)
    ttc_2d = ttc_loss.reshape(batch, modes)
    road_2d = road_loss.reshape(batch, modes)
    stability_2d = stability_severity.reshape(batch, modes)
    collision_flag = collision_margin_violation.reshape(batch, modes)
    ttc_flag = ttc_violation.reshape(batch, modes)
    road_flag = road_violation.reshape(batch, modes)
    stability_flag = stability_violation.reshape(batch, modes)
    flags = torch.stack((collision_flag, ttc_flag, road_flag, stability_flag), dim=-1)
    violation_count = flags.sum(dim=-1)
    severity_cfg = scoring_config.get("severity_weights", {})
    severity = (
        float(severity_cfg.get("collision", 1.0)) * collision_2d
        + float(severity_cfg.get("ttc", 1.0)) * ttc_2d
        + float(severity_cfg.get("road", 1.0)) * road_2d
        + float(severity_cfg.get("stability", 1.0)) * stability_2d
    )

    safe = valid & (violation_count == 0)
    has_safe = safe.any(dim=1)
    selected = torch.zeros(batch, dtype=torch.long, device=poses.device)
    priority = torch.full((batch, modes), -1.0e6, dtype=poses.dtype, device=poses.device)
    severity_tolerance = float(scoring_config.get("severity_tie_tolerance", 1.0e-6))
    for row in range(batch):
        row_valid = valid[row]
        if bool(has_safe[row]):
            pool = safe[row]
            priority[row] = torch.where(
                pool, efficiency[row], torch.full_like(priority[row], -1.0e3)
            )
        else:
            min_count = violation_count[row].masked_fill(~row_valid, modes + 1).min()
            count_pool = row_valid & (violation_count[row] == min_count)
            min_severity = severity[row].masked_fill(~count_pool, float("inf")).min()
            pool = count_pool & (severity[row] <= min_severity + severity_tolerance)
            sev_min = severity[row, row_valid].min()
            sev_max = severity[row, row_valid].max()
            sev_span = (sev_max - sev_min).clamp_min(1.0e-8)
            sev_norm = ((severity[row] - sev_min) / sev_span).clamp(0.0, 1.0)
            fallback_priority = (
                -100.0 * violation_count[row].to(poses.dtype) - sev_norm
            )
            priority[row] = torch.where(
                pool, efficiency[row],
                torch.where(row_valid, fallback_priority,
                            torch.full_like(priority[row], -1.0e6)),
            )
        selected[row] = priority[row].argmax()

    def shaped(value):
        return value.reshape(batch, modes)

    return {
        "selected_mode": selected,
        "candidate_scores": priority,
        "candidate_safe_mask": safe,
        "candidate_violation_count": violation_count,
        "candidate_violation_severity": severity,
        "candidate_efficiency_score": efficiency,
        "candidate_collision_margin_violation": collision_flag,
        "candidate_ttc_violation": ttc_flag,
        "candidate_road_violation": road_flag,
        "candidate_stability_violation": stability_flag,
        "candidate_collision_loss": collision_2d,
        "candidate_ttc_loss": ttc_2d,
        "candidate_road_loss": road_2d,
        "candidate_stability_loss": stability_2d,
        "candidate_road_violation_rate": shaped(road_violation_rate),
        "candidate_stability_violation_rate": shaped(stability_violation_rate),
        "candidate_stability_max_violation": shaped(stability_max_violation),
        "candidate_fuel_total": fuel_2d,
        "candidate_progress_m": progress_2d,
        "candidate_min_gap_m": shaped(min_gap),
    }
