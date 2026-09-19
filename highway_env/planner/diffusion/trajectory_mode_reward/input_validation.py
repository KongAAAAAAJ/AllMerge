"""Input-shape validation for the trajectory-mode reward scorer."""

from __future__ import annotations

import numpy as np

from .config import TrajectoryModeRewardError
from .constants import (
    NUM_MODES,
    NUM_VEHICLES,
    TRAJECTORY_SHAPE,
)


class CounterfactualInputMixin:
    @staticmethod
    def _array_without_optional_batch(
        value: object,
        *,
        unbatched_ndim: int,
        name: str,
    ) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.ndim == unbatched_ndim + 1:
            if array.shape[0] != 1:
                raise TrajectoryModeRewardError(
                    f"{name} only supports an optional B=1 axis"
                )
            array = array[0]
        if array.ndim != unbatched_ndim:
            raise TrajectoryModeRewardError(
                f"{name} has the wrong rank"
            )
        return array

    def _candidate_values(
        self,
        candidates: object,
        *,
        trajectories_per_mode: int | None = None,
    ) -> np.ndarray:
        values = self._array_without_optional_batch(
            candidates,
            unbatched_ndim=5,
            name="candidates",
        )
        sample_count = (
            int(trajectories_per_mode)
            if trajectories_per_mode is not None
            else int(values.shape[2])
        )
        expected = (
            NUM_VEHICLES,
            NUM_MODES,
            sample_count,
            *TRAJECTORY_SHAPE,
        )
        if (
            values.shape != expected
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
        ):
            raise TrajectoryModeRewardError(
                "candidates must be finite floating-point [3,10,N,8,2]"
            )
        return np.ascontiguousarray(
            values,
            dtype=np.float32,
        )

    def _frozen_all_values(
        self,
        trajectories: object,
    ) -> np.ndarray:
        values = self._array_without_optional_batch(
            trajectories,
            unbatched_ndim=4,
            name="frozen_all_mode_trajectories",
        )
        expected = (
            NUM_VEHICLES,
            NUM_MODES,
            *TRAJECTORY_SHAPE,
        )
        if (
            values.shape != expected
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
        ):
            raise TrajectoryModeRewardError(
                "frozen_all_mode_trajectories must be finite [3,10,8,2]"
            )
        return np.ascontiguousarray(
            values,
            dtype=np.float32,
        )

    def _frozen_argmax_values(
        self,
        trajectories: object,
    ) -> np.ndarray:
        values = self._array_without_optional_batch(
            trajectories,
            unbatched_ndim=3,
            name="frozen_argmax_joint_trajectories",
        )
        expected = (
            NUM_VEHICLES,
            *TRAJECTORY_SHAPE,
        )
        if (
            values.shape != expected
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
        ):
            raise TrajectoryModeRewardError(
                "frozen_argmax_joint_trajectories must be finite [3,8,2]"
            )
        return np.ascontiguousarray(
            values,
            dtype=np.float32,
        )

    def _valid_values(
        self,
        mask: object,
    ) -> np.ndarray:
        values = self._array_without_optional_batch(
            mask,
            unbatched_ndim=2,
            name="valid_mode_mask",
        )
        if (
            values.shape != (NUM_VEHICLES, NUM_MODES)
            or values.dtype != np.bool_
        ):
            values = np.asarray(
                values,
                dtype=np.bool_,
            )
        if values.shape != (NUM_VEHICLES, NUM_MODES):
            raise TrajectoryModeRewardError(
                "valid_mode_mask must be [3,10]"
            )
        return np.ascontiguousarray(
            values,
            dtype=np.bool_,
        )
