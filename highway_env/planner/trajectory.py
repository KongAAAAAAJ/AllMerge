from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from highway_env.planner.geometry import world_to_ego_point


def build_planner_time_grid(
    horizon_steps: int = 8,
    trajectory_dt: float = 0.5,
) -> np.ndarray:
    horizon_steps = int(horizon_steps)
    trajectory_dt = float(trajectory_dt)

    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if trajectory_dt <= 0.0:
        raise ValueError("trajectory_dt must be positive")

    return (
        np.arange(1, horizon_steps + 1, dtype=np.float32)
        * trajectory_dt
    )


@dataclass
class PlannerTrajectory:
    """Common external planner trajectory: ego-centric xy + explicit time."""

    xy: np.ndarray
    time_s: np.ndarray
    source: str = ""
    ego_position_world: Optional[np.ndarray] = None
    ego_heading_world: Optional[float] = None

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=np.float32)
        self.time_s = np.asarray(self.time_s, dtype=np.float32)

        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError("PlannerTrajectory.xy must be [T,2]")
        if self.time_s.ndim != 1:
            raise ValueError("PlannerTrajectory.time_s must be [T]")
        if self.xy.shape[0] != self.time_s.shape[0]:
            raise ValueError("xy/time_s length mismatch")
        if not np.all(np.isfinite(self.xy)):
            raise ValueError("PlannerTrajectory.xy contains NaN/Inf")
        if not np.all(np.isfinite(self.time_s)):
            raise ValueError("PlannerTrajectory.time_s contains NaN/Inf")
        if np.any(np.diff(self.time_s) <= 0.0):
            raise ValueError("PlannerTrajectory.time_s must be strictly increasing")

        if self.ego_position_world is not None:
            self.ego_position_world = np.asarray(
                self.ego_position_world,
                dtype=np.float32,
            ).copy()

        if self.ego_heading_world is not None:
            self.ego_heading_world = float(self.ego_heading_world)

    @classmethod
    def from_world_path(
        cls,
        *,
        source_time_s,
        world_xy,
        ego_position_world,
        ego_heading_world: float,
        horizon_steps: int = 8,
        trajectory_dt: float = 0.5,
        source: str = "Polynomial",
    ) -> "PlannerTrajectory":
        source_time_s = np.asarray(source_time_s, dtype=np.float32)
        world_xy = np.asarray(world_xy, dtype=np.float32)

        if source_time_s.ndim != 1:
            raise ValueError("source_time_s must be 1-D")
        if world_xy.ndim != 2 or world_xy.shape[1] != 2:
            raise ValueError("world_xy must be [N,2]")
        if source_time_s.shape[0] != world_xy.shape[0]:
            raise ValueError("source_time_s/world_xy length mismatch")
        if source_time_s.shape[0] < 2:
            raise ValueError("source path is too short")
        if np.any(np.diff(source_time_s) <= 0.0):
            raise ValueError("source_time_s must be strictly increasing")

        target_time_s = build_planner_time_grid(
            horizon_steps=horizon_steps,
            trajectory_dt=trajectory_dt,
        )

        tolerance = 1e-4

        if source_time_s[0] > target_time_s[0] + tolerance:
            raise ValueError("source path starts after first planner timestamp")

        if source_time_s[-1] < target_time_s[-1] - tolerance:
            raise ValueError(
                "source path does not cover requested "
                f"{target_time_s[-1]:.3f}s horizon; "
                f"source ends at {source_time_s[-1]:.3f}s"
            )

        sampled_world_xy = np.stack(
            [
                np.interp(target_time_s, source_time_s, world_xy[:, 0]),
                np.interp(target_time_s, source_time_s, world_xy[:, 1]),
            ],
            axis=-1,
        ).astype(np.float32)

        local_xy = world_to_ego_point(
            sampled_world_xy,
            ego_position_world,
            float(ego_heading_world),
        )

        return cls(
            xy=local_xy,
            time_s=target_time_s,
            source=source,
            ego_position_world=np.asarray(
                ego_position_world,
                dtype=np.float32,
            ),
            ego_heading_world=float(ego_heading_world),
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "xy": self.xy.copy(),
            "time_s": self.time_s.copy(),
            "source": self.source,
            "ego_position_world": (
                None
                if self.ego_position_world is None
                else self.ego_position_world.copy()
            ),
            "ego_heading_world": self.ego_heading_world,
        }
