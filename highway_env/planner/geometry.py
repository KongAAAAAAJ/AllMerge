"""Coordinate transforms shared by AllMerge planner feature construction.

Planner convention
------------------
The local frame is ego-centric:
  * +x: ego forward
  * +y: lateral axis after rotating the world frame by -ego heading
  * angles: radians

Lane-left / lane-right semantics must not be inferred from the sign of local y.
Use lane topology / lane index roles for maneuver semantics.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(angle):
    """Wrap angle(s) to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def _world_to_ego_rotation(ego_heading: float) -> np.ndarray:
    c = float(np.cos(ego_heading))
    s = float(np.sin(ego_heading))
    return np.asarray([[c, s], [-s, c]], dtype=np.float32)


def _ego_to_world_rotation(ego_heading: float) -> np.ndarray:
    c = float(np.cos(ego_heading))
    s = float(np.sin(ego_heading))
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def world_to_ego_point(point, ego_position, ego_heading: float) -> np.ndarray:
    """Transform world point(s) to the ego frame."""
    point = np.asarray(point, dtype=np.float32)
    ego_position = np.asarray(ego_position, dtype=np.float32)
    delta = point - ego_position
    rotation = _world_to_ego_rotation(ego_heading)
    return np.asarray(delta @ rotation.T, dtype=np.float32)


def world_to_ego_vector(vector, ego_heading: float) -> np.ndarray:
    """Rotate world vector(s) to the ego frame, without translation."""
    vector = np.asarray(vector, dtype=np.float32)
    rotation = _world_to_ego_rotation(ego_heading)
    return np.asarray(vector @ rotation.T, dtype=np.float32)


def ego_to_world_point(point, ego_position, ego_heading: float) -> np.ndarray:
    """Transform ego-frame point(s) to world coordinates."""
    point = np.asarray(point, dtype=np.float32)
    ego_position = np.asarray(ego_position, dtype=np.float32)
    rotation = _ego_to_world_rotation(ego_heading)
    return np.asarray(point @ rotation.T + ego_position, dtype=np.float32)


def ego_to_world_vector(vector, ego_heading: float) -> np.ndarray:
    """Rotate ego-frame vector(s) to the world frame, without translation."""
    vector = np.asarray(vector, dtype=np.float32)
    rotation = _ego_to_world_rotation(ego_heading)
    return np.asarray(vector @ rotation.T, dtype=np.float32)
