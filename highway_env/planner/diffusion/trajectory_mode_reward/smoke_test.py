"""Executable smoke test for the migrated trajectory-mode reward."""

from __future__ import annotations

import numpy as np

from .evaluator import evaluate_candidates


class _StraightLane:
    length = 500.0
    width = 4.0

    def local_coordinates(self, position):
        return float(position[0]), float(position[1])

    def position(self, longitudinal, lateral):
        return np.asarray(
            [float(longitudinal), float(lateral)],
            dtype=np.float64,
        )

    def heading_at(self, longitudinal):
        return 0.0

    def width_at(self, longitudinal):
        return self.width


class _Network:
    def __init__(self):
        self.lane = _StraightLane()
        self.graph = {
            "a": {
                "b": [self.lane],
            }
        }

    def get_lane(self, lane_index):
        return self.lane


class _Road:
    def __init__(self):
        self.network = _Network()
        self.vehicles = []


class _Vehicle:
    LENGTH = 5.74
    WIDTH = 2.3

    def __init__(self, road, x, speed=10.0):
        self.position = np.asarray(
            [float(x), 0.0],
            dtype=np.float64,
        )
        self.heading = 0.0
        self.speed = float(speed)
        self.lane_index = ("a", "b", 0)
        road.vehicles.append(self)


class _Env:
    def __init__(self):
        self.road = _Road()
        self.controlled_vehicles = [
            _Vehicle(self.road, 100.0),
            _Vehicle(self.road, 85.0),
            _Vehicle(self.road, 70.0),
        ]
        self.background_vehicles = [
            _Vehicle(self.road, 180.0),
        ]


def main() -> None:
    env = _Env()

    step_x = (
        np.arange(1, 9, dtype=np.float32)
        * 5.0
    )
    base = np.zeros(
        (3, 10, 8, 2),
        dtype=np.float32,
    )
    base[..., 0] = step_x

    frozen_argmax = base[:, 0].copy()
    valid = np.ones(
        (3, 10),
        dtype=np.bool_,
    )

    candidates = np.repeat(
        base[:, :, None],
        2,
        axis=2,
    )
    candidates[:, :, 1, :, 1] = np.linspace(
        0.0,
        10.0,
        8,
        dtype=np.float32,
    )

    result = evaluate_candidates(
        env,
        candidates,
        base,
        frozen_argmax,
        valid,
    )

    assert result.rewards.shape == (3, 10, 2)
    assert bool(
        result.out_of_drivable[0, 0, 1]
    )
    assert (
        float(result.rewards[0, 0, 0])
        > float(result.rewards[0, 0, 1])
    )

    print(
        "[OK] trajectory_mode_reward smoke test: "
        f"safe={result.rewards[0,0,0]:.4f}, "
        f"offroad={result.rewards[0,0,1]:.4f}"
    )


if __name__ == "__main__":
    main()
