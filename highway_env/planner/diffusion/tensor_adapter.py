from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .config import StructuredDiffusionConfig


FLOAT_KEYS = (
    "ego_state",
    "agent_states",
    "map_polylines",
    "target_point",
    "target_lane_polyline",
    "coarse_trajectories",
)

BOOL_KEYS = (
    "agent_valid_mask",
    "map_valid_mask",
    "mode_valid_mask",
)


class PlannerTensorAdapter:
    """
    NumPy -> torch conversion + deterministic physical-unit scaling.

    Environment feature construction remains NumPy-only.
    Torch/model concerns stay here.
    """

    def __init__(
        self,
        config: StructuredDiffusionConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device

        self.ego_scale = torch.tensor(
            config.feature_scales.ego,
            dtype=torch.float32,
            device=device,
        )
        self.agent_scale = torch.tensor(
            config.feature_scales.agent,
            dtype=torch.float32,
            device=device,
        )
        self.map_scale = torch.tensor(
            config.feature_scales.map_point,
            dtype=torch.float32,
            device=device,
        )
        self.target_point_scale = torch.tensor(
            config.feature_scales.target_point,
            dtype=torch.float32,
            device=device,
        )
        self.trajectory_scale = torch.tensor(
            config.feature_scales.trajectory_xy,
            dtype=torch.float32,
            device=device,
        )

    def to_torch(
        self,
        features: Dict[str, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        missing = [
            key
            for key in (*FLOAT_KEYS, *BOOL_KEYS)
            if key not in features
        ]
        if missing:
            raise KeyError(
                f"Planner features missing keys: {missing}"
            )

        result: Dict[str, torch.Tensor] = {}

        for key in FLOAT_KEYS:
            result[key] = torch.as_tensor(
                features[key],
                dtype=torch.float32,
                device=self.device,
            )

        for key in BOOL_KEYS:
            result[key] = torch.as_tensor(
                features[key],
                dtype=torch.bool,
                device=self.device,
            )

        self._validate_shapes(result)
        return result

    def normalize_ego(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x / self.ego_scale

    def normalize_agents(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x / self.agent_scale

    def normalize_map(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x / self.map_scale

    def normalize_target_point(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x / self.target_point_scale

    def normalize_trajectory(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x / self.trajectory_scale

    def denormalize_trajectory(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x * self.trajectory_scale

    def _validate_shapes(
        self,
        features: Dict[str, torch.Tensor],
    ) -> None:
        cfg = self.config
        batch = features["ego_state"].shape[0]

        expected = {
            "ego_state": (
                batch,
                cfg.ego_dim,
            ),
            "agent_states": (
                batch,
                cfg.max_agents,
                cfg.agent_dim,
            ),
            "agent_valid_mask": (
                batch,
                cfg.max_agents,
            ),
            "map_polylines": (
                batch,
                cfg.max_map_polylines,
                cfg.map_points,
                cfg.map_dim,
            ),
            "map_valid_mask": (
                batch,
                cfg.max_map_polylines,
            ),
            "target_point": (
                batch,
                2,
            ),
            "target_lane_polyline": (
                batch,
                cfg.map_points,
                cfg.map_dim,
            ),
            "coarse_trajectories": (
                batch,
                cfg.num_modes,
                cfg.horizon_steps,
                2,
            ),
            "mode_valid_mask": (
                batch,
                cfg.num_modes,
            ),
        }

        for key, shape in expected.items():
            actual = tuple(
                features[key].shape
            )
            if actual != shape:
                raise ValueError(
                    f"{key}: shape={actual}, expected={shape}"
                )
