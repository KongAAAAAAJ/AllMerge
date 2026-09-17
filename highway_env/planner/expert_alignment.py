from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from highway_env.planner.mode_definitions import MODE_NAMES, MODE_SLOTS


STRATEGY_NAMES = (
    "valid_any",
    "geometry_any",
    "valid_semantic",
    "geometry_semantic",
)


def _mode_semantic_group(mode_idx: int) -> str:
    try:
        return str(
            MODE_SLOTS[int(mode_idx)].semantic_group
        )
    except Exception:
        mode_idx = int(mode_idx)

        if 0 <= mode_idx <= 2:
            return "KEEP"
        if 3 <= mode_idx <= 5:
            return "LEFT_LC"
        if 6 <= mode_idx <= 8:
            return "RIGHT_LC"
        if mode_idx == 9:
            return "STOP"

        return "UNKNOWN"


class ExpertTrajectoryAligner:
    """
    F.1 diagnostic aligner.

    Four strategies:
      A valid_any
      B geometry_any
      C valid_semantic
      D geometry_semantic
    """

    def align_batch(
        self,
        *,
        features: Dict[str, np.ndarray],
        expert_xy: np.ndarray,
        trajectory_time_s: np.ndarray,
        current_lane_indices: Sequence,
        target_lane_indices: Sequence,
    ) -> Dict[str, object]:
        anchors = np.asarray(
            features["coarse_trajectories"],
            dtype=np.float32,
        )

        mode_valid_mask = np.asarray(
            features["mode_valid_mask"],
            dtype=bool,
        )

        expert_xy = np.asarray(
            expert_xy,
            dtype=np.float32,
        )

        trajectory_time_s = np.asarray(
            trajectory_time_s,
            dtype=np.float32,
        )

        if anchors.ndim != 4:
            raise ValueError(
                "coarse_trajectories must be [B,M,T,2]"
            )

        batch, modes, steps, dims = anchors.shape

        if dims != 2:
            raise ValueError(
                "trajectory coordinate dimension must be 2"
            )

        if expert_xy.shape != (batch, steps, 2):
            raise ValueError(
                f"expert_xy shape={expert_xy.shape}, "
                f"expected={(batch, steps, 2)}"
            )

        if trajectory_time_s.shape != (steps,):
            raise ValueError(
                "trajectory_time_s shape mismatch"
            )

        if mode_valid_mask.shape != (batch, modes):
            raise ValueError(
                "mode_valid_mask shape mismatch"
            )

        geometry_mask = np.any(
            np.abs(anchors) > 1e-6,
            axis=(2, 3),
        )

        residual_xy_all = (
            expert_xy[:, None, :, :]
            - anchors
        )

        point_error_all = np.linalg.norm(
            residual_xy_all,
            axis=-1,
        )

        mode_ade = point_error_all.mean(
            axis=-1
        ).astype(np.float32)

        mode_fde = point_error_all[:, :, -1].astype(
            np.float32
        )

        mode_max_abs_dx = np.abs(
            residual_xy_all[:, :, :, 0]
        ).max(axis=-1).astype(np.float32)

        mode_max_abs_dy = np.abs(
            residual_xy_all[:, :, :, 1]
        ).max(axis=-1).astype(np.float32)

        mode_mse = (
            (
                anchors
                - expert_xy[:, None, :, :]
            )
            ** 2
        ).mean(
            axis=(2, 3)
        ).astype(np.float32)

        expected_semantic_group = [
            self._expected_semantic_group(
                current_lane,
                target_lane,
            )
            for current_lane, target_lane
            in zip(
                current_lane_indices,
                target_lane_indices,
            )
        ]

        mode_semantic_group = tuple(
            _mode_semantic_group(mode_idx)
            for mode_idx in range(modes)
        )

        semantic_mask = np.zeros(
            (batch, modes),
            dtype=bool,
        )

        for ego_idx in range(batch):
            expected = expected_semantic_group[ego_idx]

            for mode_idx in range(modes):
                semantic_mask[
                    ego_idx,
                    mode_idx,
                ] = (
                    mode_semantic_group[mode_idx]
                    == expected
                )

        candidate_masks = {
            "valid_any":
                mode_valid_mask.copy(),

            "geometry_any":
                geometry_mask.copy(),

            "valid_semantic":
                (
                    mode_valid_mask
                    & semantic_mask
                ),

            "geometry_semantic":
                (
                    geometry_mask
                    & semantic_mask
                ),
        }

        strategies = {}

        for strategy_name in STRATEGY_NAMES:
            strategies[strategy_name] = (
                self._select_strategy(
                    candidate_mask=(
                        candidate_masks[
                            strategy_name
                        ]
                    ),
                    anchors=anchors,
                    mode_mse=mode_mse,
                    mode_ade=mode_ade,
                    mode_fde=mode_fde,
                    mode_max_abs_dx=(
                        mode_max_abs_dx
                    ),
                    mode_max_abs_dy=(
                        mode_max_abs_dy
                    ),
                    residual_xy_all=(
                        residual_xy_all
                    ),
                    expected_semantic_group=(
                        expected_semantic_group
                    ),
                    mode_semantic_group=(
                        mode_semantic_group
                    ),
                )
            )

        current = strategies["valid_any"]

        return {
            "expert_trajectory_xy":
                expert_xy.copy(),

            "trajectory_time_s":
                trajectory_time_s.copy(),

            "geometry_mask":
                geometry_mask.copy(),

            "mode_valid_mask":
                mode_valid_mask.copy(),

            "semantic_mask":
                semantic_mask.copy(),

            "mode_semantic_group":
                mode_semantic_group,

            "expected_semantic_group":
                tuple(expected_semantic_group),

            "mode_mse":
                mode_mse,

            "mode_ade":
                mode_ade,

            "mode_fde":
                mode_fde,

            "mode_max_abs_dx":
                mode_max_abs_dx,

            "mode_max_abs_dy":
                mode_max_abs_dy,

            "residual_xy_all":
                residual_xy_all.astype(
                    np.float32
                ),

            "candidate_masks":
                candidate_masks,

            "strategies":
                strategies,

            # compatibility aliases for old debug path
            "nearest_anchor_xy":
                current[
                    "selected_anchor_xy"
                ].copy(),

            "nearest_mode_idx":
                current[
                    "mode_idx"
                ].copy(),

            "nearest_mode_name":
                current[
                    "mode_name"
                ],

            "nearest_semantic_group":
                current[
                    "semantic_group"
                ],

            "semantic_match":
                current[
                    "semantic_match"
                ].copy(),

            "nearest_anchor_ade":
                current[
                    "ade"
                ].copy(),

            "nearest_anchor_fde":
                current[
                    "fde"
                ].copy(),

            "max_abs_dx":
                current[
                    "max_abs_dx"
                ].copy(),

            "max_abs_dy":
                current[
                    "max_abs_dy"
                ].copy(),

            "residual_xy":
                current[
                    "residual_xy"
                ].copy(),
        }

    def _select_strategy(
        self,
        *,
        candidate_mask,
        anchors,
        mode_mse,
        mode_ade,
        mode_fde,
        mode_max_abs_dx,
        mode_max_abs_dy,
        residual_xy_all,
        expected_semantic_group,
        mode_semantic_group,
    ):
        batch = candidate_mask.shape[0]

        selected_idx = np.full(
            (batch,),
            -1,
            dtype=np.int64,
        )

        selected_anchor_xy = np.full(
            (
                batch,
                anchors.shape[2],
                2,
            ),
            np.nan,
            dtype=np.float32,
        )

        selected_residual_xy = np.full(
            (
                batch,
                anchors.shape[2],
                2,
            ),
            np.nan,
            dtype=np.float32,
        )

        selected_mse = np.full(
            (batch,),
            np.nan,
            dtype=np.float32,
        )
        selected_ade = np.full(
            (batch,),
            np.nan,
            dtype=np.float32,
        )
        selected_fde = np.full(
            (batch,),
            np.nan,
            dtype=np.float32,
        )
        selected_max_abs_dx = np.full(
            (batch,),
            np.nan,
            dtype=np.float32,
        )
        selected_max_abs_dy = np.full(
            (batch,),
            np.nan,
            dtype=np.float32,
        )

        candidate_count = candidate_mask.sum(
            axis=1
        ).astype(np.int64)

        mode_name = []
        semantic_group = []
        semantic_match = np.zeros(
            (batch,),
            dtype=bool,
        )

        for ego_idx in range(batch):
            available = np.flatnonzero(
                candidate_mask[ego_idx]
            )

            if available.size == 0:
                mode_name.append("NONE")
                semantic_group.append("NONE")
                continue

            best_local = int(
                np.argmin(
                    mode_mse[
                        ego_idx,
                        available,
                    ]
                )
            )

            mode_idx = int(
                available[
                    best_local
                ]
            )

            selected_idx[ego_idx] = mode_idx

            selected_anchor_xy[
                ego_idx
            ] = anchors[
                ego_idx,
                mode_idx,
            ]

            selected_residual_xy[
                ego_idx
            ] = residual_xy_all[
                ego_idx,
                mode_idx,
            ]

            selected_mse[
                ego_idx
            ] = mode_mse[
                ego_idx,
                mode_idx,
            ]

            selected_ade[
                ego_idx
            ] = mode_ade[
                ego_idx,
                mode_idx,
            ]

            selected_fde[
                ego_idx
            ] = mode_fde[
                ego_idx,
                mode_idx,
            ]

            selected_max_abs_dx[
                ego_idx
            ] = mode_max_abs_dx[
                ego_idx,
                mode_idx,
            ]

            selected_max_abs_dy[
                ego_idx
            ] = mode_max_abs_dy[
                ego_idx,
                mode_idx,
            ]

            current_semantic = (
                mode_semantic_group[
                    mode_idx
                ]
            )

            mode_name.append(
                MODE_NAMES[
                    mode_idx
                ]
            )

            semantic_group.append(
                current_semantic
            )

            semantic_match[
                ego_idx
            ] = (
                current_semantic
                == expected_semantic_group[
                    ego_idx
                ]
            )

        return {
            "candidate_mask":
                candidate_mask.copy(),

            "candidate_count":
                candidate_count,

            "mode_idx":
                selected_idx,

            "mode_name":
                tuple(mode_name),

            "semantic_group":
                tuple(semantic_group),

            "semantic_match":
                semantic_match,

            "mse":
                selected_mse,

            "ade":
                selected_ade,

            "fde":
                selected_fde,

            "max_abs_dx":
                selected_max_abs_dx,

            "max_abs_dy":
                selected_max_abs_dy,

            "selected_anchor_xy":
                selected_anchor_xy,

            "residual_xy":
                selected_residual_xy,
        }

    @staticmethod
    def _lane_id(lane_index) -> int:
        return int(
            lane_index[2]
        )

    def _expected_semantic_group(
        self,
        current_lane_index,
        target_lane_index,
    ) -> str:
        current_id = self._lane_id(
            current_lane_index
        )
        target_id = self._lane_id(
            target_lane_index
        )

        if target_id == current_id:
            return "KEEP"

        if target_id < current_id:
            return "LEFT_LC"

        return "RIGHT_LC"
