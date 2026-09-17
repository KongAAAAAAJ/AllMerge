from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODE_NAMES = (
    "KEEP_HIGH",
    "KEEP_MEDIUM",
    "KEEP_LOW",
    "LEFT_LC_HIGH",
    "LEFT_LC_MEDIUM",
    "LEFT_LC_LOW",
    "RIGHT_LC_HIGH",
    "RIGHT_LC_MEDIUM",
    "RIGHT_LC_LOW",
    "STOP",
)

SEMANTIC_NAMES = (
    "KEEP",
    "LEFT_LC",
    "RIGHT_LC",
    "STOP",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        nargs="?",
        default=(
            "outputs/expert_pilot/"
            "expert_pilot.npz"
        ),
    )

    return parser.parse_args()


def percentile_dict(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:
        return {
            "count":
                0,
        }

    p = np.percentile(
        values,
        [
            50,
            90,
            95,
            99,
        ],
    )

    return {
        "count":
            int(
                values.size
            ),

        "mean":
            float(
                values.mean()
            ),

        "max":
            float(
                values.max()
            ),

        "p50":
            float(
                p[0]
            ),

        "p90":
            float(
                p[1]
            ),

        "p95":
            float(
                p[2]
            ),

        "p99":
            float(
                p[3]
            ),
    }


def distribution(
    values,
    names,
):
    values = np.asarray(
        values,
        dtype=np.int64,
    ).reshape(-1)

    total = values.size

    result = {}

    for idx, name in enumerate(
        names
    ):
        count = int(
            (
                values
                == idx
            ).sum()
        )

        result[
            name
        ] = {
            "count":
                count,

            "ratio":
                (
                    float(
                        count
                        / total
                    )
                    if total
                    else 0.0
                ),
        }

    return result


def main():
    args = parse_args()

    dataset = Path(
        args.dataset
    ).resolve()

    with np.load(
        dataset,
        allow_pickle=False,
    ) as data:
        target_mode = data[
            "target_mode"
        ]

        target_semantic = data[
            "target_semantic"
        ]

        traffic_valid = data[
            "target_mode_traffic_valid"
        ].astype(
            bool
        )

        residual = data[
            "geometry_semantic_residual_xy"
        ]

        max_dx = data[
            "geometry_semantic_max_abs_dx"
        ]

        max_dy = data[
            "geometry_semantic_max_abs_dy"
        ]

        ade = data[
            "geometry_semantic_ade"
        ]

        fde = data[
            "geometry_semantic_fde"
        ]

        summary = {
            "frame_count":
                int(
                    target_mode.shape[
                        0
                    ]
                ),

            "ego_sample_count":
                int(
                    target_mode.size
                ),

            "target_mode_distribution":
                distribution(
                    target_mode,
                    MODE_NAMES,
                ),

            "semantic_distribution":
                distribution(
                    target_semantic,
                    SEMANTIC_NAMES,
                ),

            "target_mode_traffic_valid": {
                "valid_count":
                    int(
                        traffic_valid.sum()
                    ),

                "invalid_count":
                    int(
                        (
                            ~traffic_valid
                        ).sum()
                    ),

                "valid_ratio":
                    float(
                        traffic_valid.mean()
                    ),

                "invalid_ratio":
                    float(
                        (
                            ~traffic_valid
                        ).mean()
                    ),
            },

            "geometry_semantic_residual": {
                "pointwise_abs_dx_m":
                    percentile_dict(
                        np.abs(
                            residual[
                                ...,
                                0,
                            ]
                        )
                    ),

                "pointwise_abs_dy_m":
                    percentile_dict(
                        np.abs(
                            residual[
                                ...,
                                1,
                            ]
                        )
                    ),

                "per_sample_max_abs_dx_m":
                    percentile_dict(
                        max_dx
                    ),

                "per_sample_max_abs_dy_m":
                    percentile_dict(
                        max_dy
                    ),

                "per_sample_ADE_m":
                    percentile_dict(
                        ade
                    ),

                "per_sample_FDE_m":
                    percentile_dict(
                        fde
                    ),
            },
        }

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
