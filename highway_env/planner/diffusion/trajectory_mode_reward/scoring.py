"""Core single-target trajectory-mode counterfactual scoring."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .collision_geometry import (
    obb_overlap_series_batch,
    shared_corridor_gap_series_batch,
)
from .constants import NUM_VEHICLES, TRAJECTORY_SHAPE
from .geometry import (
    _dense_local_trajectories_batch,
    local_to_world,
    road_margin_series_batch,
    tracking_aware_dimensions,
)
from .risk import (
    aggregate_temporal_risk_batch,
    closing_ttc_from_gap_series_batch,
    continuous_outside_road_penalty_batch,
    soft_threshold_risk,
)


class CounterfactualScoringMixin:
    # GRPO_SPEED_V2B_BATCH_GEOMETRY_20261007
    def _score_target_group(
        self,
        *,
        target_role: int,
        target_centerline: np.ndarray | None = None,
        initial_velocity_by_role=None,
        target_trajectories: np.ndarray,
        frozen_argmax_joint_trajectories: np.ndarray,
        poses: list[np.ndarray],
        road: object,
        background_by_actor: Mapping[
            str,
            list[tuple[str, np.ndarray, tuple[float, float]]],
        ],
    ) -> dict[str, object]:
        group_size = int(target_trajectories.shape[0])
        joint = np.broadcast_to(
            frozen_argmax_joint_trajectories,
            (group_size, NUM_VEHICLES, *TRAJECTORY_SHAPE),
        ).copy()
        joint[:, target_role] = target_trajectories

        if initial_velocity_by_role is None:
            raise RuntimeError('V6 score requires planning-time ego_state velocity')
        dense_local, _ = _dense_local_trajectories_batch(joint, self.config,
            initial_velocity_xy=np.stack(initial_velocity_by_role)[None])
        dense_world = np.empty_like(dense_local, dtype=np.float64)
        for role in range(NUM_VEHICLES):
            dense_world[:, role] = local_to_world(dense_local[:, role], poses[role])

        local = dense_local[:, target_role]
        world = dense_world[:, target_role]
        # V6 progress is calculated by the SAME shared tensor implementation
        # used in Guidance; no separate clipping/normalization here.
        progress_score = None

        # The signed road-union reward is also used by the differentiable Guide.
        # The old W4 road penalty/constraints are preserved as distinct metrics.
        from .road_union_reward import production_road_reward
        # Curvature V5 shares the exact MATP-compatible 10Hz cubic geometry.
        from .curvature_reward import curvature_reward_components
        from .spline_dense import evaluate_dense_spline
        import torch
        from .centerline_reward import centerline_reward_components
        with torch.no_grad():
            control_points = torch.as_tensor(target_trajectories, dtype=torch.float64)
            v0 = torch.as_tensor(initial_velocity_by_role[target_role],dtype=torch.float64)
            curvature_dense, _, _ = evaluate_dense_spline(control_points, self.config,initial_velocity_xy=v0)
            from .task_objective import longitudinal_terms
            kin=longitudinal_terms(curvature_dense,self.config,initial_velocity_xy=v0)
            # Reuse existing heading-based lateral comfort only; longitudinal
            # acceleration is penalized by the V6 longitudinal term.
            hd=torch.atan2(torch.diff(curvature_dense[...,1],dim=-1),torch.diff(curvature_dense[...,0],dim=-1))
            dyaw=torch.atan2(torch.sin(torch.diff(hd,dim=-1)),torch.cos(torch.diff(hd,dim=-1)))
            lateral_cost=dyaw.abs().mean(dim=-1)/float(self.config.interpolation_dt_s)
            curvature_terms = curvature_reward_components(curvature_dense, self.config)
            centerline_terms = centerline_reward_components(curvature_dense, target_centerline, self.config)
        curvature_numpy = {k: v.detach().cpu().numpy().astype(np.float32)
                           for k, v in curvature_terms.items()}
        kinematic_numpy={k:v.detach().cpu().numpy().astype(np.float32)
                           for k,v in kin.items() if k not in ('accel_series_mps2','jerk_series_mps3')}
        progress_score=kin['progress_score'].detach().cpu().numpy().astype(np.float64)
        lateral_comfort_penalty=lateral_cost.detach().cpu().numpy().astype(np.float32)
        centerline_numpy = {k: v.detach().cpu().numpy().astype(np.float32)
                            for k, v in centerline_terms.items()}
        road_boundary_reward = production_road_reward(
            target_trajectories, poses[target_role], road, self.config,
            initial_velocity_xy=initial_velocity_by_role[target_role]
        )
        road_margin = road_margin_series_batch(
            world, road, self.config, tracking_aware=True
        )
        minimum_road_margin = np.min(road_margin, axis=1)
        out_of_drivable = np.any(road_margin[:, 1:] < 0.0, axis=1)
        road_penalty = continuous_outside_road_penalty_batch(
            road_margin[:, 1:],
            outside_scale_m=self.config.road_outside_scale_m,
            max_weight=self.config.temporal_max_weight,
            mean_weight=self.config.temporal_mean_weight,
        )

        motion_local = local[:, 1:]
        xy = np.concatenate(
            (
                np.zeros((group_size, 1, 2), dtype=np.float64),
                motion_local[..., :2],
            ),
            axis=1,
        )
        speed = np.linalg.norm(np.diff(xy, axis=1), axis=2) / self.config.interpolation_dt_s
        acceleration = np.diff(speed, axis=1, prepend=speed[:, :1]) / self.config.interpolation_dt_s
        unwrapped_heading = np.unwrap(motion_local[..., 2], axis=1)
        yaw_rate = np.diff(
            unwrapped_heading,
            axis=1,
            prepend=unwrapped_heading[:, :1],
        ) / self.config.interpolation_dt_s
        comfort_penalty = np.clip(
            0.5 * np.mean(np.abs(acceleration), axis=1) / 8.0
            + 0.5 * np.mean(np.abs(yaw_rate), axis=1),
            0.0,
            1.0,
        )

        collision = np.zeros(group_size, dtype=np.bool_)
        minimum_background_gap = np.full(group_size, self.config.no_risk_gap_m, dtype=np.float64)
        minimum_teammate_gap = np.full(group_size, self.config.no_risk_gap_m, dtype=np.float64)
        minimum_ttc = np.full(group_size, self.config.no_risk_ttc_s, dtype=np.float64)
        tracking_dimensions = tracking_aware_dimensions(self.config)
        physical_dimensions = (self.config.vehicle_length_m, self.config.vehicle_width_m)

        interaction_gap_risks: list[np.ndarray] = []
        interaction_ttc_risks: list[np.ndarray] = []

        for actor, predictions in background_by_actor.items():
            branch_gap_risks: list[np.ndarray] = []
            branch_ttc_risks: list[np.ndarray] = []
            for name, predicted, other_dimensions in predictions:
                gap_series = shared_corridor_gap_series_batch(
                    world,
                    tracking_dimensions,
                    predicted,
                    other_dimensions,
                    no_risk_gap_m=self.config.no_risk_gap_m,
                )
                branch_gap_risks.append(
                    soft_threshold_risk(
                        gap_series,
                        warning_threshold=self.config.background_safe_gap_m,
                        softness=self.config.gap_softness_m,
                    )
                )
                ttc = closing_ttc_from_gap_series_batch(
                    gap_series,
                    dt_s=self.config.interpolation_dt_s,
                    closing_speed_epsilon_mps=self.config.closing_speed_epsilon_mps,
                    no_risk_gap_m=self.config.no_risk_gap_m,
                    no_risk_ttc_s=self.config.no_risk_ttc_s,
                )
                minimum_ttc = np.minimum(minimum_ttc, np.min(ttc, axis=1))
                branch_ttc_risks.append(
                    soft_threshold_risk(
                        ttc,
                        warning_threshold=self.config.ttc_warning_s,
                        softness=self.config.ttc_softness_s,
                    )
                )
                if name == actor:
                    minimum_background_gap = np.minimum(
                        minimum_background_gap,
                        np.min(gap_series, axis=1),
                    )
                    collision |= obb_overlap_series_batch(
                        world[:, 1:],
                        physical_dimensions,
                        predicted[1:],
                        other_dimensions,
                        0.0,
                    )
            if branch_gap_risks:
                interaction_gap_risks.append(np.max(np.stack(branch_gap_risks, axis=0), axis=0))
                interaction_ttc_risks.append(np.max(np.stack(branch_ttc_risks, axis=0), axis=0))

        for other_role in range(NUM_VEHICLES):
            if other_role == target_role:
                continue
            teammate = dense_world[:, other_role]
            collision |= obb_overlap_series_batch(
                world[:, 1:],
                physical_dimensions,
                teammate[:, 1:],
                physical_dimensions,
                0.0,
            )
            pair_gap = shared_corridor_gap_series_batch(
                world,
                tracking_dimensions,
                teammate,
                tracking_dimensions,
                no_risk_gap_m=self.config.no_risk_gap_m,
            )
            minimum_teammate_gap = np.minimum(
                minimum_teammate_gap,
                np.min(pair_gap, axis=1),
            )
            interaction_gap_risks.append(
                soft_threshold_risk(
                    pair_gap,
                    warning_threshold=self.config.platoon_safe_gap_m,
                    softness=self.config.gap_softness_m,
                )
            )
            pair_ttc = closing_ttc_from_gap_series_batch(
                pair_gap,
                dt_s=self.config.interpolation_dt_s,
                closing_speed_epsilon_mps=self.config.closing_speed_epsilon_mps,
                no_risk_gap_m=self.config.no_risk_gap_m,
                no_risk_ttc_s=self.config.no_risk_ttc_s,
            )
            minimum_ttc = np.minimum(minimum_ttc, np.min(pair_ttc, axis=1))
            interaction_ttc_risks.append(
                soft_threshold_risk(
                    pair_ttc,
                    warning_threshold=self.config.ttc_warning_s,
                    softness=self.config.ttc_softness_s,
                )
            )

        gap_risk = np.max(np.stack(interaction_gap_risks, axis=0), axis=0)
        ttc_risk = np.max(np.stack(interaction_ttc_risks, axis=0), axis=0)
        gap_penalty = aggregate_temporal_risk_batch(
            gap_risk,
            max_weight=self.config.temporal_max_weight,
            mean_weight=self.config.temporal_mean_weight,
        )
        ttc_penalty = aggregate_temporal_risk_batch(
            ttc_risk,
            max_weight=self.config.temporal_max_weight,
            mean_weight=self.config.temporal_mean_weight,
        )
        clearance_violation = (
            (minimum_background_gap < self.config.background_safe_gap_m)
            | (minimum_teammate_gap < self.config.platoon_safe_gap_m)
        )

        # Official V6 total. No W4 total or duplicate reward algebra.
        # Cast component arrays to float32, matching GRPO/GUIDANCE tensor scoring.
        from .task_objective import combine_reward
        v6_components = {
            name: torch.as_tensor(value, dtype=torch.float32)
            for name, value in {
                'progress_score': progress_score,
                'accel_penalty': kinematic_numpy['accel_penalty'],
                'jerk_penalty': kinematic_numpy['jerk_penalty'],
                'road_boundary_reward': road_boundary_reward,
                'curvature_penalty': curvature_numpy['curvature_penalty'],
                'centerline_penalty': centerline_numpy['centerline_penalty'],
                'lateral_comfort_penalty': lateral_comfort_penalty,
            }.items()
        }
        with torch.no_grad():
            task_rewards = combine_reward(v6_components, self.config).cpu().numpy().astype(np.float32)
        if not np.isfinite(task_rewards).all():
            raise RuntimeError('Nonfinite V6 production task reward')
        unsafe = collision | out_of_drivable | clearance_violation
        return {
            "task_rewards": task_rewards,
            "unsafe": unsafe,
            "collision": collision,
            "out_of_drivable": out_of_drivable,
            "clearance_violation": clearance_violation,
            "components": {
                "progress_score": progress_score.astype(np.float32),
                "lateral_comfort_penalty": lateral_comfort_penalty,
                **kinematic_numpy,
                "gap_penalty": gap_penalty.astype(np.float32),
                "ttc_penalty": ttc_penalty.astype(np.float32),
                "road_penalty": road_penalty.astype(np.float32),
                "comfort_penalty": comfort_penalty.astype(np.float32),
                "road_boundary_reward": road_boundary_reward.astype(np.float32),
                **curvature_numpy,
                **centerline_numpy,
                "minimum_background_gap_m": minimum_background_gap.astype(np.float32),
                "minimum_teammate_gap_m": minimum_teammate_gap.astype(np.float32),
                "minimum_road_margin_m": minimum_road_margin.astype(np.float32),
                "minimum_ttc_s": minimum_ttc.astype(np.float32),
            },
        }
