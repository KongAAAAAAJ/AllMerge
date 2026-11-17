"""Offline A1 lattice anchors from pre-anchor expert samples and exact lane geometry."""
from __future__ import annotations

import numpy as np
import torch

from highway_env.planner.adaptive_anchors import (
    AdaptiveCandidateBuilder,
    select_diverse_anchors,
)

DEFAULT_NUM_MODES = 10


def mode_names(num_modes: int):
    return tuple(f"DYNAMIC_DIVERSE_{i:02d}" for i in range(int(num_modes)))


class A1AnchorBuilder:
    """Generate a shared candidate pool and retain K diversity-only anchors.

    The raw road-geometry export is converted to the same compact map-feature
    contract consumed by the online A1/A2 candidate generator. This keeps the
    offline A1 artifact builder synchronized with the active diffusion path.
    """

    def __init__(self, config: dict):
        self.cfg = dict(config or {})
        self.candidate_builder = AdaptiveCandidateBuilder(
            self.cfg.get(
                "candidate_generation",
                {"config_path": "configs/anchors/control_candidate_pool.yaml"},
            )
        )
        self.num_selected = int(self.cfg.get("num_selected", DEFAULT_NUM_MODES))
        if self.num_selected <= 0:
            raise ValueError("num_selected must be positive")
        if self.num_selected > self.candidate_builder.num_candidates:
            raise ValueError(
                f"num_selected={self.num_selected} exceeds candidate pool "
                f"{self.candidate_builder.num_candidates}"
            )
        self.mode_names = mode_names(self.num_selected)
        self.trajectory_weight = float(self.cfg.get("trajectory_weight", 1.0))
        self.control_weight = float(self.cfg.get("control_weight", 1.0))
        self.length = float(self.candidate_builder.base.config.get("ego_length_m", 8.0))
        self.width = float(self.candidate_builder.base.config.get("ego_width_m", 2.6))

    def _compact_map_features(self, sample):
        geometry = np.asarray(sample["road_lane_geometry"], dtype=np.float32)
        point_mask = np.asarray(sample["road_lane_point_mask"], dtype=bool)
        lane_mask = np.asarray(sample["road_lane_mask"], dtype=bool)
        lane_meta = np.asarray(sample["road_lane_meta"], dtype=np.float32)
        connections = np.asarray(sample["road_lane_connections"], dtype=bool)
        current_i = int(sample["road_current_lane_slot"])
        target_i = int(sample["road_target_lane_slot"])
        lane_count, point_count = geometry.shape[:2]
        maps = np.zeros((lane_count, point_count, 10), dtype=np.float32)
        valid = lane_mask.copy()

        for lane_id in np.flatnonzero(lane_mask):
            keep = point_mask[lane_id]
            lane = geometry[lane_id, keep]
            if len(lane) < 2:
                valid[lane_id] = False
                continue
            compact = maps[lane_id]
            compact[:len(lane), 0:2] = lane[:, 0:2]
            compact[:len(lane), 2] = np.sin(lane[:, 6])
            compact[:len(lane), 3] = np.cos(lane[:, 6])
            compact[:len(lane), 4] = lane[:, 7]
            compact[:len(lane), 5] = lane_meta[lane_id, 0]
            compact[:len(lane), 6] = float(lane_id == current_i)
            compact[:len(lane), 9] = float(lane_id == target_i)
            # Compact points and repeat the final valid point in padding slots;
            # otherwise zero padding would create a false segment back to origin.
            compact[len(lane):] = compact[len(lane) - 1]

        current = geometry[current_i, point_mask[current_i], :2]
        if len(current) < 2:
            raise ValueError("A1 raw road geometry has fewer than two current-lane points")
        current_origin_idx = int(np.argmin(np.sum(current * current, axis=-1)))
        current_center = current[current_origin_idx]
        adjacent = {"left": None, "right": None}
        for lane_id in np.flatnonzero(lane_mask):
            if lane_id == current_i or not connections[current_i, lane_id]:
                continue
            lane = geometry[lane_id, point_mask[lane_id], :2]
            if len(lane) < 2:
                continue
            nearest = int(np.argmin(np.sum((lane - current_center) ** 2, axis=-1)))
            lateral = float(lane[nearest, 1] - current_center[1])
            side = "left" if lateral > 0.5 else "right" if lateral < -0.5 else None
            if side and (
                adjacent[side] is None or abs(lateral) < adjacent[side][0]
                or lane_id == target_i
            ):
                adjacent[side] = (abs(lateral), lane_id)

        for side, slot in (("left", 7), ("right", 8)):
            if adjacent[side] is not None:
                maps[adjacent[side][1], :, slot] = 1.0
        return maps, valid

    def build(self, sample: dict) -> dict:
        ego_state = np.asarray(sample["ego_state"], dtype=np.float32)
        maps, map_valid = self._compact_map_features(sample)
        pool = self.candidate_builder.build_single(
            ego_state=ego_state,
            map_polylines=maps,
            map_valid_mask=map_valid,
            target_acceleration=float(ego_state[2]) if ego_state.size > 2 else 0.0,
        )
        controls = torch.from_numpy(pool["coarse_controls"][None])
        trajectories = torch.from_numpy(pool["coarse_trajectories"][None])
        valid = torch.from_numpy(pool["mode_valid_mask"][None])
        selected = select_diverse_anchors(
            controls, trajectories, valid,
            num_selected=self.num_selected,
            trajectory_weight=self.trajectory_weight,
            control_weight=self.control_weight,
        )[0].numpy()
        selected_controls = pool["coarse_controls"][selected]
        selected_xy = pool["coarse_trajectories"][selected]
        _, dense_future = self.candidate_builder._rollout_many(
            selected_controls, max(float(ego_state[0]), 0.0), return_dense=True
        )
        dense = np.concatenate(
            [np.zeros((self.num_selected, 1, 2), dtype=np.float32), dense_future],
            axis=1,
        )
        return {
            "a1_coarse_controls": selected_controls.astype(np.float32),
            "a1_coarse_trajectories": selected_xy.astype(np.float32),
            "a1_dense_trajectories": dense.astype(np.float32),
            "a1_valid_mask": np.ones((self.num_selected,), dtype=bool),
            "a1_invalid_reasons": np.full(
                (self.num_selected,), "valid", dtype="U40"
            ),
        }
