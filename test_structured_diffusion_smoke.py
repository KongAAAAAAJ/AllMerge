import numpy as np
import torch

from highway_env.planner.diffusion.config import (
    build_structured_diffusion_config,
)
from highway_env.planner.diffusion.runtime import (
    DiffusionPlannerRuntime,
)


def fake_features(
    batch=3,
):
    rng = np.random.default_rng(0)

    features = {
        "ego_state": rng.normal(
            size=(batch, 14)
        ).astype(np.float32),

        "agent_states": rng.normal(
            size=(batch, 16, 11)
        ).astype(np.float32),

        "agent_valid_mask": np.ones(
            (batch, 16),
            dtype=bool,
        ),

        "map_polylines": rng.normal(
            size=(batch, 8, 32, 10)
        ).astype(np.float32),

        "map_valid_mask": np.zeros(
            (batch, 8),
            dtype=bool,
        ),

        "target_point": rng.normal(
            size=(batch, 2)
        ).astype(np.float32),

        "target_lane_polyline": rng.normal(
            size=(batch, 32, 10)
        ).astype(np.float32),

        "coarse_trajectories": np.zeros(
            (batch, 10, 8, 2),
            dtype=np.float32,
        ),

        "mode_valid_mask": np.zeros(
            (batch, 10),
            dtype=bool,
        ),
    }

    features[
        "map_valid_mask"
    ][:, :3] = True

    # Simple forward anchors.
    t = np.arange(
        1,
        9,
        dtype=np.float32,
    ) * 10.0

    for b in range(batch):
        for m in range(10):
            features[
                "coarse_trajectories"
            ][b, m, :, 0] = t

            features[
                "coarse_trajectories"
            ][b, m, :, 1] = (
                (m % 3 - 1)
                * 4.0
            )

    features[
        "mode_valid_mask"
    ][:, [0, 2, 3, 6, 9]] = True

    return features


def main():
    runtime = DiffusionPlannerRuntime(
        {
            "device": (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            ),
            "allow_random_weights": True,
            "deterministic_seed": 0,
        }
    )

    features = fake_features()

    output = runtime.infer(
        features
    )

    print(
        "trajectory:",
        output[
            "trajectory"
        ].shape,
    )
    print(
        "candidates:",
        output[
            "trajectory_candidates"
        ].shape,
    )
    print(
        "logits:",
        output[
            "trajectory_mode_logits"
        ].shape,
    )
    print(
        "mode_idx:",
        output[
            "trajectory_mode_idx"
        ],
    )
    print(
        "latency_ms:",
        output[
            "latency_ms"
        ],
    )

    assert (
        output[
            "trajectory"
        ].shape
        == (3, 8, 2)
    )

    assert (
        output[
            "trajectory_candidates"
        ].shape
        == (3, 10, 8, 2)
    )

    assert (
        output[
            "trajectory_mode_logits"
        ].shape
        == (3, 10)
    )

    selected = output[
        "trajectory_mode_idx"
    ]

    for b, mode in enumerate(
        selected
    ):
        assert features[
            "mode_valid_mask"
        ][b, mode]

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
