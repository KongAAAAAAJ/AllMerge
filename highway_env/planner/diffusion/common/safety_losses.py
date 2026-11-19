"""Differentiable execution-space safety losses for TRACE control diffusion."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _vehicle_axes(poses: torch.Tensor):
    yaw = poses[..., 2]
    forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    lateral = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
    return forward, lateral


def _smooth_hinge(value: torch.Tensor, temperature: float):
    return (F.softplus(value / temperature) - math.log(2.0)).clamp_min(0.0) * temperature


def background_constant_velocity_poses(
    agent_states: torch.Tensor,
    agent_valid_mask: torch.Tensor,
    ego_state: torch.Tensor,
    horizon_steps: int,
    dt: float,
):
    """Predict uncontrolled background vehicles in the fixed ego-local frame."""
    controlled = agent_states[..., 10] > 0.5
    valid = (
        agent_valid_mask.bool()
        & (~controlled)
        & (agent_states[..., 8] > 0)
        & (agent_states[..., 9] > 0)
    )
    relative_position = agent_states[..., 0:2]
    absolute_velocity_local = agent_states[..., 2:4] + ego_state[:, None, 0:2]
    times = torch.arange(
        horizon_steps + 1, device=agent_states.device, dtype=agent_states.dtype
    ) * float(dt)
    xy = relative_position[:, :, None, :] + absolute_velocity_local[:, :, None, :] * times[None, None, :, None]
    heading = torch.atan2(agent_states[..., 6], agent_states[..., 7])
    yaw = heading[:, :, None].expand(-1, -1, horizon_steps + 1)
    poses = torch.cat((xy, yaw.unsqueeze(-1)), dim=-1)
    return poses, agent_states[..., 8:10], valid


def _finite_difference_velocity(positions: torch.Tensor, dt: float):
    delta = (positions[:, 1:] - positions[:, :-1]) / float(dt)
    return torch.cat((delta[:, :1], delta), dim=1)


def _obb_pairwise_geometry(
    ego_poses: torch.Tensor,
    ego_dimensions: torch.Tensor,
    other_poses: torch.Tensor,
    other_dimensions: torch.Tensor,
    dt: float,
):
    """Return exact 2-D OBB SAT gaps and axis-wise closing speeds.

    The four separating axes are the longitudinal and lateral axes of both
    rectangles. max(axis_gap) is positive iff the rectangles are disjoint and
    negative iff their oriented rectangles overlap.
    """
    batch, agents, steps = other_poses.shape[:3]
    ego_forward, ego_lateral = _vehicle_axes(ego_poses)
    other_forward, other_lateral = _vehicle_axes(other_poses)

    ego_forward = ego_forward[:, None].expand(-1, agents, -1, -1)
    ego_lateral = ego_lateral[:, None].expand(-1, agents, -1, -1)
    axes = torch.stack(
        (ego_forward, ego_lateral, other_forward, other_lateral), dim=-2
    )

    relative_position = other_poses[..., :2] - ego_poses[:, None, :, :2]
    signed_projection = (relative_position.unsqueeze(-2) * axes).sum(dim=-1)
    projected_distance = signed_projection.abs()

    ego_half_length = (ego_dimensions[:, 0] * 0.5)[:, None, None, None]
    ego_half_width = (ego_dimensions[:, 1] * 0.5)[:, None, None, None]
    other_half_length = (other_dimensions[..., 0] * 0.5)[:, :, None, None]
    other_half_width = (other_dimensions[..., 1] * 0.5)[:, :, None, None]

    ego_forward_projection = (
        ego_forward.unsqueeze(-2) * axes
    ).sum(dim=-1).abs()
    ego_lateral_projection = (
        ego_lateral.unsqueeze(-2) * axes
    ).sum(dim=-1).abs()
    other_forward_projection = (
        other_forward.unsqueeze(-2) * axes
    ).sum(dim=-1).abs()
    other_lateral_projection = (
        other_lateral.unsqueeze(-2) * axes
    ).sum(dim=-1).abs()

    ego_support = (
        ego_half_length * ego_forward_projection
        + ego_half_width * ego_lateral_projection
    )
    other_support = (
        other_half_length * other_forward_projection
        + other_half_width * other_lateral_projection
    )
    axis_gaps = projected_distance - ego_support - other_support
    signed_gap = axis_gaps.amax(dim=-1)

    ego_velocity = _finite_difference_velocity(ego_poses[..., :2], dt)
    other_velocity = _finite_difference_velocity(
        other_poses[..., :2].reshape(batch * agents, steps, 2), dt
    ).reshape(batch, agents, steps, 2)
    relative_velocity = other_velocity - ego_velocity[:, None]
    relative_axis_velocity = (
        relative_velocity.unsqueeze(-2) * axes
    ).sum(dim=-1)
    closing_speed = (
        -torch.sign(signed_projection) * relative_axis_velocity
    ).clamp_min(0.0)
    ttc_axis_margin = closing_speed, axis_gaps
    return signed_gap, ttc_axis_margin


def pairwise_vehicle_safety(
    ego_poses: torch.Tensor,
    ego_dimensions: torch.Tensor,
    other_poses: torch.Tensor,
    other_dimensions: torch.Tensor,
    other_valid_mask: torch.Tensor,
    *,
    dt: float,
    collision_margin_m: float,
    ttc_threshold_s: float,
    softplus_temperature_m: float,
    collision_distance_scale_m: float,
    ttc_distance_scale_m: float,
):
    """OBB collision and TTC losses for one ego against reference vehicles."""
    batch, agents, steps = other_poses.shape[:3]
    signed_gap, (closing_speed, axis_gaps) = _obb_pairwise_geometry(
        ego_poses, ego_dimensions, other_poses, other_dimensions, dt
    )

    valid = other_valid_mask.bool()
    has_other = valid.any(dim=1)
    valid_pair_time = valid[:, :, None]

    collision_violation = float(collision_margin_m) - signed_gap
    collision_violation = collision_violation.masked_fill(~valid_pair_time, -torch.inf)
    worst_collision_per_time = collision_violation.amax(dim=1)
    worst_collision_per_time = torch.where(
        has_other[:, None], worst_collision_per_time,
        torch.zeros_like(worst_collision_per_time),
    )
    collision_hinge = _smooth_hinge(worst_collision_per_time, softplus_temperature_m)
    collision_loss = (collision_hinge / collision_distance_scale_m).square().mean()

    # A collision can occur within T only if every SAT axis can close within T.
    # The minimum axis margin is therefore the differentiable bottleneck.
    ttc_axis_violation = (
        float(ttc_threshold_s) * closing_speed
        - (axis_gaps - float(collision_margin_m))
    )
    pair_ttc_violation = ttc_axis_violation.amin(dim=-1)
    pair_ttc_violation = pair_ttc_violation.masked_fill(~valid_pair_time, -torch.inf)
    worst_ttc_per_time = pair_ttc_violation.amax(dim=1)
    worst_ttc_per_time = torch.where(
        has_other[:, None], worst_ttc_per_time, torch.zeros_like(worst_ttc_per_time)
    )
    ttc_hinge = _smooth_hinge(worst_ttc_per_time, softplus_temperature_m)
    ttc_loss = (ttc_hinge / ttc_distance_scale_m).square().mean()

    sample_collision = ((signed_gap < 0.0) & valid_pair_time).any(dim=(1, 2))
    sample_margin_violation = ((signed_gap < float(collision_margin_m)) & valid_pair_time).any(dim=(1, 2))
    sample_ttc_violation = ((pair_ttc_violation > 0.0) & valid_pair_time).any(dim=(1, 2))
    valid_gaps = signed_gap.masked_fill(~valid_pair_time, torch.inf)
    sample_min_gap = valid_gaps.amin(dim=(1, 2))
    min_gap = sample_min_gap[has_other].mean() if has_other.any() else ego_poses.new_zeros(())

    return {
        'collision_loss': collision_loss,
        'ttc_loss': ttc_loss,
        'collision_rate': sample_collision.float().mean(),
        'collision_margin_violation_rate': sample_margin_violation.float().mean(),
        'ttc_violation_rate': sample_ttc_violation.float().mean(),
        'min_gap': min_gap,
    }


def _connected_drivable_lanes(
    road_lane_mask: torch.Tensor,
    road_lane_connections: torch.Tensor,
    road_current_lane_slot: torch.Tensor,
    road_target_lane_slot: torch.Tensor,
):
    """Select the road-graph component containing the current and target lanes."""
    batch, lanes = road_lane_mask.shape
    valid = road_lane_mask.bool()
    adjacency = road_lane_connections.bool() | road_lane_connections.bool().transpose(1, 2)
    adjacency = adjacency & valid[:, :, None] & valid[:, None, :]
    seeds = F.one_hot(road_current_lane_slot.long(), num_classes=lanes).bool()
    seeds = seeds | F.one_hot(road_target_lane_slot.long(), num_classes=lanes).bool()
    reachable = seeds & valid
    for _ in range(lanes):
        neighbours = torch.bmm(
            reachable.float().unsqueeze(1), adjacency.float()
        ).squeeze(1).bool()
        reachable = reachable | neighbours
    return reachable & valid


def _vehicle_footprint_samples(poses: torch.Tensor, dimensions: torch.Tensor):
    """Twelve perimeter samples on the exact oriented rectangular footprint."""
    batch = poses.shape[0]
    half_length = dimensions[:, 0] * 0.5
    half_width = dimensions[:, 1] * 0.5
    corners = torch.stack(
        (
            torch.stack((half_length, half_width), dim=-1),
            torch.stack((half_length, -half_width), dim=-1),
            torch.stack((-half_length, -half_width), dim=-1),
            torch.stack((-half_length, half_width), dim=-1),
        ),
        dim=1,
    )
    next_corners = corners.roll(shifts=-1, dims=1)
    fractions = poses.new_tensor((0.0, 1.0 / 3.0, 2.0 / 3.0))
    perimeter = (
        corners[:, :, None, :] * (1.0 - fractions[None, None, :, None])
        + next_corners[:, :, None, :] * fractions[None, None, :, None]
    ).reshape(batch, 12, 2)
    points_local = torch.cat((perimeter, poses.new_zeros((batch, 1, 2))), dim=1)

    forward, lateral = _vehicle_axes(poses)
    local_x = points_local[:, None, :, 0]
    local_y = points_local[:, None, :, 1]
    return (
        poses[..., None, :2]
        + local_x[..., None] * forward[..., None, :]
        + local_y[..., None] * lateral[..., None, :]
    )


def road_boundary_safety_loss(
    ego_poses: torch.Tensor,
    ego_dimensions: torch.Tensor,
    road_lane_geometry: torch.Tensor,
    road_lane_point_mask: torch.Tensor,
    road_lane_mask: torch.Tensor,
    road_lane_connections: torch.Tensor,
    road_current_lane_slot: torch.Tensor,
    road_target_lane_slot: torch.Tensor,
    *,
    softplus_temperature_m: float,
    distance_scale_m: float,
):
    """Penalize rectangular footprint points outside the connected drivable-road union.

    Internal lane boundaries are excluded: each footprint point may lie in any
    lane corridor connected to the current or target lane.
    """
    batch, steps = ego_poses.shape[:2]
    lane_valid = _connected_drivable_lanes(
        road_lane_mask, road_lane_connections,
        road_current_lane_slot, road_target_lane_slot,
    )
    footprint = _vehicle_footprint_samples(ego_poses, ego_dimensions)

    centerline = road_lane_geometry[..., 0:2]
    widths = road_lane_geometry[..., 7]
    p0, p1 = centerline[:, :, :-1], centerline[:, :, 1:]
    segment = p1 - p0
    segment_sq = segment.square().sum(dim=-1).clamp_min(1e-8)
    valid_segment = (
        lane_valid[:, :, None]
        & road_lane_point_mask.bool()[:, :, :-1]
        & road_lane_point_mask.bool()[:, :, 1:]
        & (segment_sq > 1e-7)
    )

    # Find each lane's segment nearest to the predicted vehicle centre without
    # retaining a large map-distance tensor in the autograd graph. Seven local
    # segments then cover the complete 8-m truck footprint at the exported map
    # spacing, while keeping the differentiable footprint calculation compact.
    with torch.no_grad():
        centre = ego_poses[..., :2].detach()
        relative = centre[:, :, None, None, :] - p0[:, None, :, :, :]
        projection = (
            (relative * segment[:, None, :, :, :]).sum(dim=-1)
            / segment_sq[:, None, :, :]
        ).clamp(0.0, 1.0)
        closest = p0[:, None, :, :, :] + projection[..., None] * segment[:, None, :, :, :]
        centre_distance_sq = (centre[:, :, None, None, :] - closest).square().sum(dim=-1)
        centre_distance_sq = centre_distance_sq.masked_fill(
            ~valid_segment[:, None, :, :], torch.inf
        )
        nearest_segment = centre_distance_sq.argmin(dim=-1)
        offsets = torch.arange(-3, 4, device=ego_poses.device)
        selected_segments = (nearest_segment[..., None] + offsets).clamp(0, segment.shape[2] - 1)

    segment_count = selected_segments.shape[-1]
    p0_selected = torch.gather(
        p0[:, None].expand(-1, steps, -1, -1, -1),
        dim=3,
        index=selected_segments[..., None].expand(-1, -1, -1, -1, 2),
    )
    segment_selected = torch.gather(
        segment[:, None].expand(-1, steps, -1, -1, -1),
        dim=3,
        index=selected_segments[..., None].expand(-1, -1, -1, -1, 2),
    )
    width0_selected = torch.gather(
        widths[:, :, :-1][:, None].expand(-1, steps, -1, -1),
        dim=3, index=selected_segments,
    )
    width1_selected = torch.gather(
        widths[:, :, 1:][:, None].expand(-1, steps, -1, -1),
        dim=3, index=selected_segments,
    )
    valid_selected = torch.gather(
        valid_segment[:, None].expand(-1, steps, -1, -1),
        dim=3, index=selected_segments,
    )

    selected_segment_sq = segment_selected.square().sum(dim=-1).clamp_min(1e-8)
    relative = footprint[:, :, :, None, None, :] - p0_selected[:, :, None, :, :, :]
    projection = (
        (relative * segment_selected[:, :, None, :, :, :]).sum(dim=-1)
        / selected_segment_sq[:, :, None, :, :]
    ).clamp(0.0, 1.0)
    closest = p0_selected[:, :, None, :, :, :] + projection[..., None] * segment_selected[:, :, None, :, :, :]
    lateral_distance = torch.linalg.vector_norm(
        footprint[:, :, :, None, None, :] - closest, dim=-1
    )
    width = (
        width0_selected[:, :, None, :, :]
        + projection * (width1_selected - width0_selected)[:, :, None, :, :]
    )
    violation = lateral_distance - width * 0.5
    violation = violation.masked_fill(~valid_selected[:, :, None, :, :], torch.inf)
    footprint_violation = violation.amin(dim=(-1, -2)).amax(dim=-1)

    road_hinge = _smooth_hinge(footprint_violation, softplus_temperature_m)
    loss = (road_hinge / distance_scale_m).square().mean()
    return {
        'loss': loss,
        'violation_rate': (footprint_violation > 0.0).float().mean(),
        'max_violation': footprint_violation.clamp_min(0.0).amax(),
        'violation': footprint_violation,
    }


def expert_xy_to_poses(expert_xy: torch.Tensor):
    """Convert [B,H,2] ego-local expert positions to [B,H+1,3] poses."""
    batch, horizon, _ = expert_xy.shape
    origin = expert_xy.new_zeros((batch, 1, 2))
    xy = torch.cat((origin, expert_xy), dim=1)
    delta = xy[:, 1:] - xy[:, :-1]
    heading = torch.atan2(delta[..., 1], delta[..., 0])
    heading = torch.cat((heading.new_zeros((batch, 1)), heading), dim=1)
    return torch.cat((xy, heading.unsqueeze(-1)), dim=-1)
