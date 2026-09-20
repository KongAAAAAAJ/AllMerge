#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from highway_env.planner.diffusion.grpo.reward_adapter import (
    _reward_decomposition_metrics,
)


def main() -> int:
    v, m, n = 3, 4, 3
    valid = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=np.bool_,
    )

    pretrain = np.zeros((v, m), dtype=np.float32)
    rewards = np.zeros((v, m, n), dtype=np.float32)
    rewards[0] = 0.20
    rewards[1] = -0.40
    rewards[2] = 0.10

    def component(a: float, b: float, c: float) -> np.ndarray:
        arr = np.zeros((v, m, n), dtype=np.float32)
        arr[0] = a
        arr[1] = b
        arr[2] = c
        return arr

    components = {
        "progress_score": component(0.8, 0.5, 0.7),
        "gap_penalty": component(0.1, 0.7, 0.2),
        "ttc_penalty": component(0.1, 0.6, 0.2),
        "road_penalty": component(0.0, 0.0, 0.0),
        "comfort_penalty": component(0.1, 0.1, 0.1),
        "minimum_background_gap_m": component(12.0, 12.0, 12.0),
        "minimum_teammate_gap_m": component(9.0, 5.0, 8.0),
        "minimum_road_margin_m": component(2.0, 2.0, 2.0),
        "minimum_ttc_s": component(8.0, 2.5, 6.0),
    }
    pretrain_components = {
        name: values[:, :, 0].copy()
        for name, values in components.items()
    }

    zeros = np.zeros((v, m, n), dtype=np.bool_)
    zeros_pre = np.zeros((v, m), dtype=np.bool_)

    result = SimpleNamespace(
        rewards=rewards,
        pretrain_rewards=pretrain,
        valid_mode_mask=valid,
        components=components,
        pretrain_components=pretrain_components,
        collision=zeros,
        pretrain_collision=zeros_pre,
        out_of_drivable=zeros,
        pretrain_out_of_drivable=zeros_pre,
        clearance_violation=zeros,
        pretrain_clearance_violation=zeros_pre,
        unsafe=zeros,
        pretrain_unsafe=zeros_pre,
    )

    metrics = _reward_decomposition_metrics(result)

    p0 = "diagnostics/reward_decomp/vehicle_0"
    p1 = "diagnostics/reward_decomp/vehicle_1"
    p2 = "diagnostics/reward_decomp/vehicle_2"

    assert metrics[f"{p1}_gap_penalty_mean"] > metrics[
        f"{p0}_gap_penalty_mean"
    ]
    assert metrics[f"{p1}_gap_penalty_mean"] > metrics[
        f"{p2}_gap_penalty_mean"
    ]
    assert metrics[f"{p1}_ttc_penalty_mean"] > metrics[
        f"{p0}_ttc_penalty_mean"
    ]
    assert metrics[f"{p1}_minimum_teammate_gap_m_mean"] < metrics[
        f"{p0}_minimum_teammate_gap_m_mean"
    ]
    assert metrics[f"{p1}_reward_mean"] < metrics[f"{p0}_reward_mean"]
    assert all(np.isfinite(x) for x in metrics.values())

    print("PASS: reward decomposition diagnostics smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
