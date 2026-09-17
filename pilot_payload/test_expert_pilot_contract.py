from __future__ import annotations

import sys

import numpy as np
import torch

from highway_env.planner.diffusion.mode_assignment import (
    SEMANTIC_KEEP,
    SEMANTIC_LEFT_LC,
    assign_expert_mode,
)


def make_map(
    semantics,
):
    batch = len(
        semantics
    )

    maps = torch.zeros(
        (
            batch,
            3,
            4,
            10,
        ),
        dtype=torch.float32,
    )

    valid = torch.ones(
        (
            batch,
            3,
        ),
        dtype=torch.bool,
    )

    maps[
        :,
        0,
        :,
        6,
    ] = 1.0

    maps[
        :,
        1,
        :,
        7,
    ] = 1.0

    maps[
        :,
        2,
        :,
        8,
    ] = 1.0

    for batch_idx, semantic in enumerate(
        semantics
    ):
        target_lane = (
            0
            if semantic
            == SEMANTIC_KEEP
            else 1
        )

        maps[
            batch_idx,
            target_lane,
            :,
            9,
        ] = 1.0

    return (
        maps,
        valid,
    )


def main():
    batch = 2
    steps = 8

    t = (
        torch.arange(
            1,
            steps + 1,
            dtype=torch.float32,
        )
        * 0.5
    )

    expert_keep = torch.stack(
        [
            20.0
            * t,

            torch.zeros_like(
                t
            ),
        ],
        dim=-1,
    )

    expert_left = torch.stack(
        [
            15.0
            * t,

            -2.0
            * torch.ones_like(
                t
            ),
        ],
        dim=-1,
    )

    expert = torch.stack(
        [
            expert_keep,
            expert_left,
        ],
        dim=0,
    )

    anchors = torch.zeros(
        (
            batch,
            10,
            steps,
            2,
        ),
        dtype=torch.float32,
    )

    anchors[
        0,
        0,
    ] = expert_keep

    anchors[
        0,
        6,
    ] = (
        expert_keep
        + torch.tensor(
            [
                0.2,
                3.0,
            ]
        )
    )

    anchors[
        1,
        5,
    ] = (
        expert_left
        + torch.tensor(
            [
                0.5,
                0.3,
            ]
        )
    )

    anchors[
        1,
        9,
    ] = (
        expert_left
        + torch.tensor(
            [
                0.1,
                0.1,
            ]
        )
    )

    maps, map_valid = make_map(
        [
            SEMANTIC_KEEP,
            SEMANTIC_LEFT_LC,
        ]
    )

    traffic = torch.ones(
        (
            batch,
            10,
        ),
        dtype=torch.bool,
    )

    traffic[
        0,
        0,
    ] = False

    traffic[
        1,
        5,
    ] = False

    features = {
        "map_polylines":
            maps,

        "map_valid_mask":
            map_valid,

        "coarse_trajectories":
            anchors,

        "mode_valid_mask":
            traffic,
    }

    assignment = assign_expert_mode(
        features=features,
        target_trajectory=expert,
    )

    assert assignment.target_mode.tolist() == [
        0,
        5,
    ]

    selected_valid = torch.gather(
        traffic,
        1,
        assignment.target_mode[
            :,
            None,
        ],
    ).squeeze(1)

    assert selected_valid.tolist() == [
        False,
        False,
    ]

    print(
        "Expert pilot F.2 contract "
        "smoke test passed."
    )


if __name__ == "__main__":
    main()
