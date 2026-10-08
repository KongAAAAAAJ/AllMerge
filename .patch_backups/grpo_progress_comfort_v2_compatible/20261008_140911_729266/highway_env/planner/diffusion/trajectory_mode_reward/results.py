"""Typed outputs and reusable geometry cache for reward evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import TrajectoryModeRewardError
from .constants import NUM_MODES, NUM_VEHICLES


_COMPONENT_NAMES = (
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


def _check_component_map(
    components: Mapping[str, np.ndarray],
    expected_shape: tuple[int, ...],
    name: str,
) -> None:
    if set(components) != set(_COMPONENT_NAMES):
        raise TrajectoryModeRewardError(
            f"{name} component keys do not match the reward contract"
        )
    for key, value in components.items():
        array = np.asarray(value)
        if (
            array.shape != expected_shape
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise TrajectoryModeRewardError(
                f"{name}.{key} must be finite float32 {expected_shape}"
            )


@dataclass(frozen=True)
class RewardGeometryContext:
    """Process-local cache valid for one unchanged simulator state."""

    poses: tuple[np.ndarray, ...]
    backgrounds: tuple[Mapping, ...]
    road: object


@dataclass(frozen=True)
class TrajectoryModePretrainRewardResult:
    """Frozen same-mode rewards cached once for one live state."""

    rewards: np.ndarray
    valid_mode_mask: np.ndarray
    unsafe: np.ndarray
    collision: np.ndarray
    out_of_drivable: np.ndarray
    clearance_violation: np.ndarray
    components: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        expected = (NUM_VEHICLES, NUM_MODES)
        rewards = np.asarray(self.rewards)
        if (
            rewards.shape != expected
            or rewards.dtype != np.float32
            or not np.isfinite(rewards).all()
        ):
            raise TrajectoryModeRewardError(
                "pretrain rewards must be finite float32 [3,10]"
            )
        valid = np.asarray(self.valid_mode_mask)
        if valid.shape != expected or valid.dtype != np.bool_:
            raise TrajectoryModeRewardError(
                "valid_mode_mask must be bool [3,10]"
            )
        for name in (
            "unsafe",
            "collision",
            "out_of_drivable",
            "clearance_violation",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != expected or value.dtype != np.bool_:
                raise TrajectoryModeRewardError(
                    f"{name} must be bool [3,10]"
                )
        _check_component_map(
            self.components,
            expected,
            "pretrain_components",
        )


@dataclass(frozen=True)
class TrajectoryModeRewardResult:
    """Current and frozen rewards under identical teammate context."""

    rewards: np.ndarray
    pretrain_rewards: np.ndarray
    valid_mode_mask: np.ndarray

    unsafe: np.ndarray
    collision: np.ndarray
    out_of_drivable: np.ndarray
    clearance_violation: np.ndarray

    pretrain_unsafe: np.ndarray
    pretrain_collision: np.ndarray
    pretrain_out_of_drivable: np.ndarray
    pretrain_clearance_violation: np.ndarray

    components: Mapping[str, np.ndarray]
    pretrain_components: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        rewards = np.asarray(self.rewards)
        if (
            rewards.ndim != 3
            or rewards.shape[:2] != (NUM_VEHICLES, NUM_MODES)
            or rewards.shape[2] <= 0
            or rewards.dtype != np.float32
            or not np.isfinite(rewards).all()
        ):
            raise TrajectoryModeRewardError(
                "rewards must be finite float32 [3,10,N]"
            )
        sample_shape = rewards.shape
        pretrain_shape = (NUM_VEHICLES, NUM_MODES)

        pretrain = np.asarray(self.pretrain_rewards)
        if (
            pretrain.shape != pretrain_shape
            or pretrain.dtype != np.float32
            or not np.isfinite(pretrain).all()
        ):
            raise TrajectoryModeRewardError(
                "pretrain_rewards must be finite float32 [3,10]"
            )

        valid = np.asarray(self.valid_mode_mask)
        if valid.shape != pretrain_shape or valid.dtype != np.bool_:
            raise TrajectoryModeRewardError(
                "valid_mode_mask must be bool [3,10]"
            )

        for name in (
            "unsafe",
            "collision",
            "out_of_drivable",
            "clearance_violation",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != sample_shape or value.dtype != np.bool_:
                raise TrajectoryModeRewardError(
                    f"{name} must be bool [3,10,N]"
                )

        for name in (
            "pretrain_unsafe",
            "pretrain_collision",
            "pretrain_out_of_drivable",
            "pretrain_clearance_violation",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != pretrain_shape or value.dtype != np.bool_:
                raise TrajectoryModeRewardError(
                    f"{name} must be bool [3,10]"
                )

        _check_component_map(
            self.components,
            sample_shape,
            "components",
        )
        _check_component_map(
            self.pretrain_components,
            pretrain_shape,
            "pretrain_components",
        )
