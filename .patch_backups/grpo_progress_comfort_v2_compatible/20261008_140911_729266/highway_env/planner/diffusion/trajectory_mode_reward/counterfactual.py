"""Public trajectory-mode counterfactual reward scorer."""

from __future__ import annotations

import numpy as np

from .config import (
    TrajectoryModeRewardConfig,
    TrajectoryModeRewardError,
)
from .constants import NUM_MODES, NUM_VEHICLES
from .geometry import _dense_local_trajectories
from .input_validation import CounterfactualInputMixin
from .results import (
    RewardGeometryContext,
    TrajectoryModePretrainRewardResult,
    TrajectoryModeRewardResult,
)
from .scoring import CounterfactualScoringMixin
from .state_adapter import AllMergeRewardStateAdapter


class TrajectoryModeCounterfactualReward(
    CounterfactualInputMixin,
    CounterfactualScoringMixin,
):
    """Score each target trajectory mode with teammates frozen to Stage-1."""

    COMPONENT_NAMES = (
        "progress_score",
        "gap_penalty",
        "ttc_penalty",
        "road_penalty",
        "comfort_penalty",
        "minimum_background_gap_m",
        "minimum_teammate_gap_m",
        "minimum_road_margin_m",
        "minimum_ttc_s",
    )

    def __init__(
        self,
        config: TrajectoryModeRewardConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else TrajectoryModeRewardConfig()
        )
        self.state_adapter = AllMergeRewardStateAdapter(
            self.config
        )

    def build_geometry_context(
        self,
        env: object,
        frozen_argmax_joint_trajectories: object,
    ) -> RewardGeometryContext:
        frozen_argmax = self._frozen_argmax_values(
            frozen_argmax_joint_trajectories
        )
        _, times = _dense_local_trajectories(
            frozen_argmax[None],
            self.config,
        )
        return self.state_adapter.build(
            env,
            times,
        )

    def score_pretrain(
        self,
        env: object,
        frozen_all_mode_trajectories: object,
        frozen_argmax_joint_trajectories: object,
        valid_mode_mask: object,
        *,
        geometry_context: RewardGeometryContext | None = None,
    ) -> TrajectoryModePretrainRewardResult:
        frozen_all = self._frozen_all_values(
            frozen_all_mode_trajectories
        )
        frozen_argmax = self._frozen_argmax_values(
            frozen_argmax_joint_trajectories
        )
        valid = self._valid_values(
            valid_mode_mask
        )

        shape = (
            NUM_VEHICLES,
            NUM_MODES,
        )
        rewards = np.zeros(
            shape,
            dtype=np.float32,
        )
        unsafe = np.zeros(
            shape,
            dtype=np.bool_,
        )
        collision = np.zeros(
            shape,
            dtype=np.bool_,
        )
        out_of_drivable = np.zeros(
            shape,
            dtype=np.bool_,
        )
        clearance = np.zeros(
            shape,
            dtype=np.bool_,
        )
        components = {
            name: np.zeros(
                shape,
                dtype=np.float32,
            )
            for name in self.COMPONENT_NAMES
        }

        context = (
            geometry_context
            if geometry_context is not None
            else self.build_geometry_context(
                env,
                frozen_argmax,
            )
        )

        for role in range(NUM_VEHICLES):
            for mode in range(NUM_MODES):
                if not valid[role, mode]:
                    continue

                scored = self._score_target_group(
                    target_role=role,
                    target_trajectories=(
                        frozen_all[
                            role,
                            mode,
                        ][None]
                    ),
                    frozen_argmax_joint_trajectories=(
                        frozen_argmax
                    ),
                    poses=list(context.poses),
                    road=context.road,
                    background_by_actor=(
                        context.backgrounds[role]
                    ),
                )
                rewards[role, mode] = scored[
                    "rewards"
                ][0]
                unsafe[role, mode] = scored[
                    "unsafe"
                ][0]
                collision[role, mode] = scored[
                    "collision"
                ][0]
                out_of_drivable[role, mode] = scored[
                    "out_of_drivable"
                ][0]
                clearance[role, mode] = scored[
                    "clearance_violation"
                ][0]
                for name in self.COMPONENT_NAMES:
                    components[name][
                        role,
                        mode,
                    ] = scored["components"][name][0]

        return TrajectoryModePretrainRewardResult(
            rewards=rewards,
            valid_mode_mask=valid,
            unsafe=unsafe,
            collision=collision,
            out_of_drivable=out_of_drivable,
            clearance_violation=clearance,
            components=components,
        )

    def score_candidates(
        self,
        env: object,
        candidates: object,
        frozen_argmax_joint_trajectories: object,
        valid_mode_mask: object,
        pretrain: TrajectoryModePretrainRewardResult,
        *,
        geometry_context: RewardGeometryContext | None = None,
    ) -> TrajectoryModeRewardResult:
        candidate_values = self._candidate_values(
            candidates
        )
        frozen_argmax = self._frozen_argmax_values(
            frozen_argmax_joint_trajectories
        )
        valid = self._valid_values(
            valid_mode_mask
        )

        if not isinstance(
            pretrain,
            TrajectoryModePretrainRewardResult,
        ):
            raise TrajectoryModeRewardError(
                "pretrain must be TrajectoryModePretrainRewardResult"
            )
        if not np.array_equal(
            pretrain.valid_mode_mask,
            valid,
        ):
            raise TrajectoryModeRewardError(
                "pretrain and candidate valid_mode_mask differ"
            )

        shape = (
            NUM_VEHICLES,
            NUM_MODES,
            candidate_values.shape[2],
        )
        rewards = np.zeros(
            shape,
            dtype=np.float32,
        )
        unsafe = np.zeros(
            shape,
            dtype=np.bool_,
        )
        collision = np.zeros(
            shape,
            dtype=np.bool_,
        )
        out_of_drivable = np.zeros(
            shape,
            dtype=np.bool_,
        )
        clearance = np.zeros(
            shape,
            dtype=np.bool_,
        )
        components = {
            name: np.zeros(
                shape,
                dtype=np.float32,
            )
            for name in self.COMPONENT_NAMES
        }

        context = (
            geometry_context
            if geometry_context is not None
            else self.build_geometry_context(
                env,
                frozen_argmax,
            )
        )

        for role in range(NUM_VEHICLES):
            for mode in range(NUM_MODES):
                if not valid[role, mode]:
                    continue

                scored = self._score_target_group(
                    target_role=role,
                    target_trajectories=(
                        candidate_values[
                            role,
                            mode,
                        ]
                    ),
                    frozen_argmax_joint_trajectories=(
                        frozen_argmax
                    ),
                    poses=list(context.poses),
                    road=context.road,
                    background_by_actor=(
                        context.backgrounds[role]
                    ),
                )
                rewards[role, mode] = scored[
                    "rewards"
                ]
                unsafe[role, mode] = scored[
                    "unsafe"
                ]
                collision[role, mode] = scored[
                    "collision"
                ]
                out_of_drivable[role, mode] = scored[
                    "out_of_drivable"
                ]
                clearance[role, mode] = scored[
                    "clearance_violation"
                ]
                for name in self.COMPONENT_NAMES:
                    components[name][
                        role,
                        mode,
                    ] = scored["components"][name]

        return TrajectoryModeRewardResult(
            rewards=rewards,
            pretrain_rewards=pretrain.rewards,
            valid_mode_mask=valid,
            unsafe=unsafe,
            collision=collision,
            out_of_drivable=out_of_drivable,
            clearance_violation=clearance,
            pretrain_unsafe=pretrain.unsafe,
            pretrain_collision=pretrain.collision,
            pretrain_out_of_drivable=(
                pretrain.out_of_drivable
            ),
            pretrain_clearance_violation=(
                pretrain.clearance_violation
            ),
            components=components,
            pretrain_components=pretrain.components,
        )

    def score_all_mode_trajectories(
        self,
        env: object,
        all_mode_trajectories: object,
        frozen_argmax_joint_trajectories: object,
        valid_mode_mask: object,
        pretrain: TrajectoryModePretrainRewardResult,
        *,
        geometry_context: RewardGeometryContext | None = None,
    ) -> TrajectoryModeRewardResult:
        values = self._frozen_all_values(
            all_mode_trajectories
        )
        return self.score_candidates(
            env,
            values[:, :, None],
            frozen_argmax_joint_trajectories,
            valid_mode_mask,
            pretrain,
            geometry_context=geometry_context,
        )

    def score_counterfactuals(
        self,
        env: object,
        candidates: object,
        frozen_all_mode_trajectories: object,
        frozen_argmax_joint_trajectories: object,
        valid_mode_mask: object,
    ) -> TrajectoryModeRewardResult:
        context = self.build_geometry_context(
            env,
            frozen_argmax_joint_trajectories,
        )
        pretrain = self.score_pretrain(
            env,
            frozen_all_mode_trajectories,
            frozen_argmax_joint_trajectories,
            valid_mode_mask,
            geometry_context=context,
        )
        return self.score_candidates(
            env,
            candidates,
            frozen_argmax_joint_trajectories,
            valid_mode_mask,
            pretrain,
            geometry_context=context,
        )
