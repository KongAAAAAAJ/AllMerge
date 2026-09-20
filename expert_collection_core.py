from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Repository / working-directory handling
# ---------------------------------------------------------------------------
#
# AllMerge still contains a few historical file opens such as:
#     all_merge/highway_env/vehicle/parameters/truck.json
#
# Therefore the simulation must run with cwd=<parent of all_merge>.
# This script enforces that automatically while keeping repo_root on sys.path.
REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

os.chdir(
    WORKSPACE_ROOT
)


from stable_baselines3.common.utils import set_random_seed  # noqa: E402

from decide.decide import decide  # noqa: E402
from env_reset import env_reset  # noqa: E402
from mode_selector.mode_selector import set_platoon_mode  # noqa: E402

from highway_env.planner.diffusion.mode_assignment import (  # noqa: E402
    SEMANTIC_KEEP,
    SEMANTIC_LEFT_LC,
    SEMANTIC_RIGHT_LC,
    SEMANTIC_STOP,
    SEMANTIC_NAMES,
    assign_expert_mode,
)


SOURCE_COMMIT = "0940278"

FEATURE_KEYS = (
    "ego_state",
    "agent_states",
    "agent_valid_mask",
    "map_polylines",
    "map_valid_mask",
    "target_point",
    "target_lane_polyline",
    "coarse_trajectories",
    "mode_valid_mask",
)

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

SEMANTIC_NAME_TO_ID = {
    "KEEP":
        SEMANTIC_KEEP,

    "LEFT_LC":
        SEMANTIC_LEFT_LC,

    "RIGHT_LC":
        SEMANTIC_RIGHT_LC,

    "STOP":
        SEMANTIC_STOP,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect a 200-500 planning-frame Polynomial expert pilot "
            "for structured diffusion pretraining diagnostics."
        )
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help=(
            "Planning frames to collect. "
            "One frame contains all controlled egos. "
            "Default: 300."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="First episode seed.",
    )

    parser.add_argument(
        "--env-name",
        type=str,
        default="highway-platoon-v0",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            REPO_ROOT
            / "outputs/expert_pilot"
        ),
    )

    parser.add_argument(
        "--allow-contract-mismatch",
        action="store_true",
        help=(
            "Record contract mismatches instead of failing immediately. "
            "Not recommended for the first pilot."
        ),
    )


    parser.add_argument(
        "--save-episode-videos",
        action="store_true",
        help=(
            "Record every expert-collection episode as an MP4. "
            "The terminal collision transition is included in the video "
            "even though done transitions are still excluded from expert labels."
        ),
    )

    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help=(
            "Directory for per-episode MP4 files. "
            "Default: <output-dir>/videos."
        ),
    )

    args = parser.parse_args()

    if not (
        200
        <= args.frames
        <= 500
    ):
        parser.error(
            "--frames must be between 200 and 500 "
            "for this pilot stage."
        )

    return args


def get_vec_attr(
    env,
    name,
    env_idx=0,
    required=True,
):
    try:
        values = env.get_attr(
            name
        )

    except AttributeError:
        if required:
            raise

        return None

    if (
        not values
        or env_idx
        >= len(
            values
        )
    ):
        if required:
            raise RuntimeError(
                f"Cannot get env attribute {name!r}"
            )

        return None

    value = values[
        env_idx
    ]

    if (
        required
        and value is None
    ):
        raise RuntimeError(
            f"env attribute {name!r} is None"
        )

    return value


def to_torch_features(
    features: Dict[str, np.ndarray],
) -> Dict[str, torch.Tensor]:
    output = {}

    for key in FEATURE_KEYS:
        if key not in features:
            raise KeyError(
                f"Missing planner feature: {key}"
            )

        value = np.asarray(
            features[
                key
            ]
        )

        tensor = torch.as_tensor(
            value
        )

        if key in (
            "agent_valid_mask",
            "map_valid_mask",
            "mode_valid_mask",
        ):
            tensor = tensor.bool()
        else:
            tensor = tensor.float()

        output[
            key
        ] = tensor

    return output


def derive_f2_assignment(
    features,
    alignment,
):
    tensor_features = (
        to_torch_features(
            features
        )
    )

    expert_xy = torch.as_tensor(
        np.asarray(
            alignment[
                "expert_trajectory_xy"
            ],
            dtype=np.float32,
        )
    )

    with torch.no_grad():
        assignment = assign_expert_mode(
            features=tensor_features,
            target_trajectory=expert_xy,
        )

    target_mode = (
        assignment.target_mode
        .cpu()
        .numpy()
        .astype(
            np.int64
        )
    )

    target_semantic = (
        assignment.target_semantic
        .cpu()
        .numpy()
        .astype(
            np.int64
        )
    )

    geometry_mask = (
        assignment.geometry_mask
        .cpu()
        .numpy()
        .astype(
            bool
        )
    )

    anchor_distance = (
        assignment.anchor_distance
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    return (
        target_mode,
        target_semantic,
        geometry_mask,
        anchor_distance,
    )


def check_f1_f2_contract(
    *,
    alignment,
    target_mode,
    target_semantic,
) -> Tuple[
    bool,
    List[str],
]:
    """
    Pilot label-contract check after F.2.

    F.1 and F.2 intentionally use different fine-grained
    nearest-anchor metrics:

        F.1 diagnostic:
            xy-MSE

        F.2 training label:
            mean pointwise Euclidean distance (anchor ADE)

    Therefore F.1 geometry_semantic and F.2 target_mode may
    select different H/M/L modes inside the same semantic group.
    This is not a label-contract error.

    Strict checks retained:
      1. F.1/RuleMaker semantic == F.2 map-derived semantic.
      2. F.1 geometry_semantic diagnostics exist.

    F.2 target geometry validity is checked separately in
    build_frame_record().
    """
    errors = []

    if (
        "strategies" not in alignment
        or "geometry_semantic" not in alignment["strategies"]
    ):
        errors.append(
            "F.1 geometry_semantic diagnostics missing"
        )
        return False, errors

    f1_semantic_names = tuple(
        alignment["expected_semantic_group"]
    )

    f1_semantic = np.asarray(
        [
            SEMANTIC_NAME_TO_ID[str(name)]
            for name in f1_semantic_names
        ],
        dtype=np.int64,
    )

    if not np.array_equal(
        f1_semantic,
        target_semantic,
    ):
        errors.append(
            "F.1 expected semantic "
            f"{f1_semantic.tolist()} != "
            "F.2 map-derived semantic "
            f"{target_semantic.tolist()}"
        )

    # Do NOT require:
    #   F.1 geometry_semantic mode == F.2 target_mode
    #
    # F.1 uses xy-MSE while F.2 uses mean pointwise Euclidean
    # distance (anchor ADE). Different H/M/L choices inside the
    # same semantic group are legitimate.

    return len(errors) == 0, errors


def build_frame_record(
    *,
    frame_index,
    episode_index,
    episode_step,
    seed,
    features,
    alignment,
    allow_contract_mismatch,
):
    (
        target_mode,
        target_semantic,
        geometry_mask,
        anchor_distance,
    ) = derive_f2_assignment(
        features,
        alignment,
    )

    contract_ok, contract_errors = (
        check_f1_f2_contract(
            alignment=alignment,
            target_mode=target_mode,
            target_semantic=target_semantic,
        )
    )

    if (
        not contract_ok
        and not allow_contract_mismatch
    ):
        raise RuntimeError(
            "F.1/F.2 label contract mismatch:\n  "
            + "\n  ".join(
                contract_errors
            )
        )

    expert_xy = np.asarray(
        alignment[
            "expert_trajectory_xy"
        ],
        dtype=np.float32,
    )

    time_s = np.asarray(
        alignment[
            "trajectory_time_s"
        ],
        dtype=np.float32,
    )

    # STAGE3_DENSE_EXPERT_V2
    future_trajectory_dense = np.asarray(
        alignment["future_trajectory_dense"],
        dtype=np.float32,
    )
    dense_dt = np.float32(alignment["dense_dt"])
    trajectory_horizon_s = np.float32(
        alignment["trajectory_horizon_s"]
    )

    if future_trajectory_dense.shape != (expert_xy.shape[0], 40, 2):
        raise ValueError(
            "future_trajectory_dense must be [B,40,2], got "
            f"{future_trajectory_dense.shape}"
        )
    expected_sparse = future_trajectory_dense[:, [4, 9, 14, 19, 24, 29, 34, 39]]
    if not np.allclose(expert_xy, expected_sparse, atol=1e-5, rtol=0.0):
        max_error = float(np.max(np.abs(expert_xy - expected_sparse)))
        raise ValueError(
            "expert_trajectory_xy is not an exact 0.5 s downsample of the "
            f"native dense target; max_abs_error={max_error:.6g}"
        )

    anchors = np.asarray(
        features[
            "coarse_trajectories"
        ],
        dtype=np.float32,
    )

    traffic_valid = np.asarray(
        features[
            "mode_valid_mask"
        ],
        dtype=bool,
    )

    batch = expert_xy.shape[
        0
    ]

    batch_index = np.arange(
        batch,
        dtype=np.int64,
    )

    selected_anchor_xy = anchors[
        batch_index,
        target_mode,
    ]

    residual_xy = (
        expert_xy
        - selected_anchor_xy
    ).astype(
        np.float32
    )

    point_error = np.linalg.norm(
        residual_xy,
        axis=-1,
    )

    ade = point_error.mean(
        axis=-1
    ).astype(
        np.float32
    )

    fde = point_error[
        :,
        -1,
    ].astype(
        np.float32
    )

    max_abs_dx = np.abs(
        residual_xy[
            :,
            :,
            0,
        ]
    ).max(
        axis=-1
    ).astype(
        np.float32
    )

    max_abs_dy = np.abs(
        residual_xy[
            :,
            :,
            1,
        ]
    ).max(
        axis=-1
    ).astype(
        np.float32
    )

    target_mode_traffic_valid = traffic_valid[
        batch_index,
        target_mode,
    ]

    target_mode_geometry_valid = geometry_mask[
        batch_index,
        target_mode,
    ]

    selected_assignment_distance = anchor_distance[
        batch_index,
        target_mode,
    ]

    if not bool(
        target_mode_geometry_valid.all()
    ):
        raise RuntimeError(
            "F.2 selected a geometry-invalid mode"
        )

    feature_copy = {
        key:
            np.asarray(
                features[
                    key
                ]
            ).copy()

        for key
        in FEATURE_KEYS
    }

    return {
        "frame_index":
            int(
                frame_index
            ),

        "episode_index":
            int(
                episode_index
            ),

        "episode_step":
            int(
                episode_step
            ),

        "seed":
            int(
                seed
            ),

        "features":
            feature_copy,

        "expert_trajectory_xy":
            expert_xy.copy(),

        "trajectory_time_s":
            time_s.copy(),

        "future_trajectory_dense":
            future_trajectory_dense.copy(),

        "dense_dt":
            dense_dt,

        "trajectory_horizon_s":
            trajectory_horizon_s,

        "target_mode":
            target_mode,

        "target_semantic":
            target_semantic,

        "target_mode_traffic_valid":
            target_mode_traffic_valid.astype(
                bool
            ),

        "target_mode_geometry_valid":
            target_mode_geometry_valid.astype(
                bool
            ),

        "target_mode_assignment_distance":
            selected_assignment_distance.astype(
                np.float32
            ),

        "selected_anchor_xy":
            selected_anchor_xy.astype(
                np.float32
            ),

        "geometry_semantic_residual_xy":
            residual_xy,

        "geometry_semantic_ade":
            ade,

        "geometry_semantic_fde":
            fde,

        "geometry_semantic_max_abs_dx":
            max_abs_dx,

        "geometry_semantic_max_abs_dy":
            max_abs_dy,

        "contract_ok":
            bool(
                contract_ok
            ),

        "contract_errors":
            tuple(
                contract_errors
            ),
    }


def stack_records(
    records,
):
    if not records:
        raise RuntimeError(
            "No pilot frames were collected"
        )

    data = {}

    # Metadata.
    for key in (
        "frame_index",
        "episode_index",
        "episode_step",
        "seed",
    ):
        data[
            key
        ] = np.asarray(
            [
                record[
                    key
                ]
                for record
                in records
            ],
            dtype=np.int64,
        )

    # Features.
    for key in FEATURE_KEYS:
        data[
            key
        ] = np.stack(
            [
                record[
                    "features"
                ][
                    key
                ]
                for record
                in records
            ],
            axis=0,
        )

    # Frame-level trajectory time grid is identical; save once.
    first_time = records[
        0
    ][
        "trajectory_time_s"
    ]

    for record in records[
        1:
    ]:
        if not np.allclose(
            record[
                "trajectory_time_s"
            ],
            first_time,
            atol=1e-6,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Trajectory time grid changed during pilot"
            )

    data[
        "trajectory_time_s"
    ] = first_time.copy()

    for key in (
        "expert_trajectory_xy",
        "target_mode",
        "target_semantic",
        "target_mode_traffic_valid",
        "target_mode_geometry_valid",
        "target_mode_assignment_distance",
        "selected_anchor_xy",
        "geometry_semantic_residual_xy",
        "geometry_semantic_ade",
        "geometry_semantic_fde",
        "geometry_semantic_max_abs_dx",
        "geometry_semantic_max_abs_dy",
    ):
        data[
            key
        ] = np.stack(
            [
                record[
                    key
                ]
                for record
                in records
            ],
            axis=0,
        )

    data[
        "contract_ok"
    ] = np.asarray(
        [
            record[
                "contract_ok"
            ]
            for record
            in records
        ],
        dtype=bool,
    )

    return data


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


def distribution_dict(
    values,
    *,
    names,
):
    values = np.asarray(
        values,
        dtype=np.int64,
    ).reshape(-1)

    total = int(
        values.size
    )

    output = {}

    for idx, name in enumerate(
        names
    ):
        count = int(
            (
                values
                == idx
            ).sum()
        )

        output[
            str(
                name
            )
        ] = {
            "index":
                int(
                    idx
                ),

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

    return output


def traffic_valid_breakdown(
    semantic,
    mode,
    valid,
):
    semantic = np.asarray(
        semantic,
        dtype=np.int64,
    ).reshape(-1)

    mode = np.asarray(
        mode,
        dtype=np.int64,
    ).reshape(-1)

    valid = np.asarray(
        valid,
        dtype=bool,
    ).reshape(-1)

    output = {
        "overall": {
            "count":
                int(
                    valid.size
                ),

            "valid_count":
                int(
                    valid.sum()
                ),

            "valid_ratio":
                (
                    float(
                        valid.mean()
                    )
                    if valid.size
                    else 0.0
                ),

            "invalid_ratio":
                (
                    float(
                        (
                            ~valid
                        ).mean()
                    )
                    if valid.size
                    else 0.0
                ),
        },
        "by_semantic": {},
        "by_mode": {},
    }

    for idx, name in enumerate(
        SEMANTIC_NAMES
    ):
        mask = semantic == idx

        if not bool(
            mask.any()
        ):
            continue

        values = valid[
            mask
        ]

        output[
            "by_semantic"
        ][
            name
        ] = {
            "count":
                int(
                    values.size
                ),

            "valid_ratio":
                float(
                    values.mean()
                ),

            "invalid_ratio":
                float(
                    (
                        ~values
                    ).mean()
                ),
        }

    for idx, name in enumerate(
        MODE_NAMES
    ):
        mask = mode == idx

        if not bool(
            mask.any()
        ):
            continue

        values = valid[
            mask
        ]

        output[
            "by_mode"
        ][
            name
        ] = {
            "count":
                int(
                    values.size
                ),

            "valid_ratio":
                float(
                    values.mean()
                ),

            "invalid_ratio":
                float(
                    (
                        ~values
                    ).mean()
                ),
        }

    return output


def build_summary(
    data,
):
    residual = np.asarray(
        data[
            "geometry_semantic_residual_xy"
        ],
        dtype=np.float32,
    )

    summary = {
        "source_commit":
            SOURCE_COMMIT,

        "frame_count":
            int(
                data[
                    "target_mode"
                ].shape[
                    0
                ]
            ),

        "ego_count_per_frame":
            int(
                data[
                    "target_mode"
                ].shape[
                    1
                ]
            ),

        "ego_sample_count":
            int(
                data[
                    "target_mode"
                ].size
            ),

        "trajectory_time_s":
            data[
                "trajectory_time_s"
            ].astype(
                float
            ).tolist(),

        "target_mode_distribution":
            distribution_dict(
                data[
                    "target_mode"
                ],
                names=MODE_NAMES,
            ),

        "semantic_distribution":
            distribution_dict(
                data[
                    "target_semantic"
                ],
                names=SEMANTIC_NAMES,
            ),

        "target_mode_traffic_valid":
            traffic_valid_breakdown(
                data[
                    "target_semantic"
                ],
                data[
                    "target_mode"
                ],
                data[
                    "target_mode_traffic_valid"
                ],
            ),

        # Primary residual-bound calibration statistics.
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
                    data[
                        "geometry_semantic_max_abs_dx"
                    ]
                ),

            "per_sample_max_abs_dy_m":
                percentile_dict(
                    data[
                        "geometry_semantic_max_abs_dy"
                    ]
                ),

            "per_sample_ADE_m":
                percentile_dict(
                    data[
                        "geometry_semantic_ade"
                    ]
                ),

            "per_sample_FDE_m":
                percentile_dict(
                    data[
                        "geometry_semantic_fde"
                    ]
                ),

            "assignment_ADE_m":
                percentile_dict(
                    data[
                        "target_mode_assignment_distance"
                    ]
                ),
        },

        "contract": {
            "all_frames_ok":
                bool(
                    data[
                        "contract_ok"
                    ].all()
                ),

            "failed_frame_count":
                int(
                    (
                        ~data[
                            "contract_ok"
                        ]
                    ).sum()
                ),
        },
    }

    return summary


def save_sample_csv(
    data,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_count, batch = (
        data[
            "target_mode"
        ].shape
    )

    fieldnames = (
        "frame_index",
        "episode_index",
        "episode_step",
        "seed",
        "ego_index",
        "semantic_id",
        "semantic_name",
        "target_mode",
        "target_mode_name",
        "target_mode_traffic_valid",
        "target_mode_geometry_valid",
        "assignment_ade_m",
        "residual_ade_m",
        "residual_fde_m",
        "max_abs_dx_m",
        "max_abs_dy_m",
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for frame_idx in range(
            frame_count
        ):
            for ego_idx in range(
                batch
            ):
                semantic_id = int(
                    data[
                        "target_semantic"
                    ][
                        frame_idx,
                        ego_idx,
                    ]
                )

                mode_idx = int(
                    data[
                        "target_mode"
                    ][
                        frame_idx,
                        ego_idx,
                    ]
                )

                writer.writerow({
                    "frame_index":
                        int(
                            data[
                                "frame_index"
                            ][
                                frame_idx
                            ]
                        ),

                    "episode_index":
                        int(
                            data[
                                "episode_index"
                            ][
                                frame_idx
                            ]
                        ),

                    "episode_step":
                        int(
                            data[
                                "episode_step"
                            ][
                                frame_idx
                            ]
                        ),

                    "seed":
                        int(
                            data[
                                "seed"
                            ][
                                frame_idx
                            ]
                        ),

                    "ego_index":
                        ego_idx,

                    "semantic_id":
                        semantic_id,

                    "semantic_name":
                        SEMANTIC_NAMES[
                            semantic_id
                        ],

                    "target_mode":
                        mode_idx,

                    "target_mode_name":
                        MODE_NAMES[
                            mode_idx
                        ],

                    "target_mode_traffic_valid":
                        bool(
                            data[
                                "target_mode_traffic_valid"
                            ][
                                frame_idx,
                                ego_idx,
                            ]
                        ),

                    "target_mode_geometry_valid":
                        bool(
                            data[
                                "target_mode_geometry_valid"
                            ][
                                frame_idx,
                                ego_idx,
                            ]
                        ),

                    "assignment_ade_m":
                        float(
                            data[
                                "target_mode_assignment_distance"
                            ][
                                frame_idx,
                                ego_idx,
                            ]
                        ),

                    "residual_ade_m":
                        float(
                            data[
                                "geometry_semantic_ade"
                            ][
                                frame_idx,
                                ego_idx,
                            ]
                        ),

                    "residual_fde_m":
                        float(
                            data[
                                "geometry_semantic_fde"
                            ][
                                frame_idx,
                                ego_idx,
                            ]
                        ),

                    "max_abs_dx_m":
                        float(
                            data[
                                "geometry_semantic_max_abs_dx"
                            ][
                                frame_idx,
                                ego_idx,
                            ]
                        ),

                    "max_abs_dy_m":
                        float(
                            data[
                                "geometry_semantic_max_abs_dy"
                            ][
                                frame_idx,
                                ego_idx,
                            ]
                        ),
                })


def save_pilot(
    records,
    output_dir,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = stack_records(
        records
    )

    dataset_path = (
        output_dir
        / "expert_pilot.npz"
    )

    np.savez_compressed(
        dataset_path,
        **data,
    )

    summary = build_summary(
        data
    )

    summary_path = (
        output_dir
        / "expert_pilot_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    csv_path = (
        output_dir
        / "expert_pilot_samples.csv"
    )

    save_sample_csv(
        data,
        csv_path,
    )

    metadata = {
        "source_commit":
            SOURCE_COMMIT,

        "feature_keys":
            list(
                FEATURE_KEYS
            ),

        "mode_names":
            list(
                MODE_NAMES
            ),

        "semantic_names":
            list(
                SEMANTIC_NAMES
            ),

        "frame_definition":
            (
                "One 10-Hz planning instant containing "
                "all controlled vehicles."
            ),

        "training_label_contract":
            (
                "F.2 semantic + geometry + nearest-anchor ADE"
            ),

        "traffic_validity_role":
            (
                "diagnostic/inference safety mask; "
                "not expert-label assignment"
            ),
    }

    metadata_path = (
        output_dir
        / "expert_pilot_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return (
        dataset_path,
        summary_path,
        csv_path,
        metadata_path,
        summary,
    )


def print_summary(
    summary,
):
    print(
        "\n"
        + "=" * 108
    )

    print(
        "EXPERT PILOT SUMMARY"
    )

    print(
        "=" * 108
    )

    print(
        "frames:",
        summary[
            "frame_count"
        ],
    )

    print(
        "ego samples:",
        summary[
            "ego_sample_count"
        ],
    )

    print(
        "\nSemantic distribution:"
    )

    for name, item in summary[
        "semantic_distribution"
    ].items():
        if item[
            "count"
        ] == 0:
            continue

        print(
            f"  {name:<10s} "
            f"{item['count']:>5d} "
            f"({100.0 * item['ratio']:6.2f}%)"
        )

    print(
        "\nTarget mode distribution:"
    )

    for name, item in summary[
        "target_mode_distribution"
    ].items():
        if item[
            "count"
        ] == 0:
            continue

        print(
            f"  {name:<20s} "
            f"{item['count']:>5d} "
            f"({100.0 * item['ratio']:6.2f}%)"
        )

    traffic = summary[
        "target_mode_traffic_valid"
    ][
        "overall"
    ]

    print(
        "\nTarget-mode traffic validity:"
    )

    print(
        f"  valid ratio   = "
        f"{100.0 * traffic['valid_ratio']:.2f}%"
    )

    print(
        f"  invalid ratio = "
        f"{100.0 * traffic['invalid_ratio']:.2f}%"
    )

    residual = summary[
        "geometry_semantic_residual"
    ]

    print(
        "\nGeometry-semantic residual percentiles:"
    )

    for key in (
        "per_sample_max_abs_dx_m",
        "per_sample_max_abs_dy_m",
        "per_sample_ADE_m",
        "per_sample_FDE_m",
    ):
        values = residual[
            key
        ]

        print(
            f"  {key:<28s} "
            f"P50={values['p50']:.3f} "
            f"P90={values['p90']:.3f} "
            f"P95={values['p95']:.3f} "
            f"P99={values['p99']:.3f} "
            f"max={values['max']:.3f}"
        )

    print(
        "\nContract all frames OK:",
        summary[
            "contract"
        ][
            "all_frames_ok"
        ],
    )

    print(
        "=" * 108
    )



# ---------------------------------------------------------------------------
# Optional per-episode video recording
# ---------------------------------------------------------------------------

def _configure_episode_video(
    video_env,
    video_path: Path,
) -> Path:
    """
    Configure per-episode recording for both old and new SB3 APIs.
    Returns the actual mp4 path.
    """
    if video_env is None:
        raise RuntimeError(
            "--save-episode-videos requested, but env_reset() returned "
            "video_env=None."
        )

    video_path = Path(video_path).resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Newer SB3 API.
    if hasattr(video_env, "video_path"):
        video_env.video_folder = str(video_path.parent)

        if hasattr(video_env, "video_name"):
            video_env.video_name = video_path.name

        video_env.video_path = str(video_path)
        return video_path

    # Older SB3 API.
    has_old_api = (
        hasattr(video_env, "video_recorder")
        and hasattr(video_env, "start_video_recorder")
        and hasattr(video_env, "close_video_recorder")
    )

    if has_old_api:
        # env_reset() may already have started a recorder with a default name.
        # Close and restart it WITHOUT resetting the environment.
        try:
            video_env.close_video_recorder()
        except Exception:
            pass

        video_env.video_folder = str(video_path.parent)
        video_env.name_prefix = video_path.stem

        try:
            video_env.video_length = max(
                int(video_env.video_length),
                10000,
            )
        except Exception:
            video_env.video_length = 10000

        video_env.start_video_recorder()

        recorder = getattr(
            video_env,
            "video_recorder",
            None,
        )

        recorder_path = getattr(
            recorder,
            "path",
            None,
        )

        if recorder_path:
            return Path(recorder_path).resolve()

        step_id = int(
            getattr(video_env, "step_id", 0)
        )

        video_length = int(
            getattr(video_env, "video_length", 10000)
        )

        actual_name = (
            f"{video_path.stem}"
            f"-step-{step_id}"
            f"-to-step-{step_id + video_length}.mp4"
        )

        return (
            video_path.parent
            / actual_name
        ).resolve()

    attrs = sorted(
        name
        for name in dir(video_env)
        if (
            "video" in name.lower()
            or "record" in name.lower()
        )
    )

    raise RuntimeError(
        "Unsupported VecVideoRecorder API. "
        f"Relevant attributes: {attrs}"
    )


def _episode_terminal_summary(
    info,
) -> dict:
    """
    Keep a compact sidecar record for matching each MP4 to collected samples.
    """
    info = (
        info
        if isinstance(
            info,
            dict,
        )
        else {}
    )

    episode_state = info.get(
        "episode_state",
        {},
    )

    if not isinstance(
        episode_state,
        dict,
    ):
        episode_state = {}

    crashed = episode_state.get(
        "crashed",
        None,
    )

    if crashed is None:
        crashed_value = info.get(
            "crashed",
            False,
        )

        if isinstance(
            crashed_value,
            (list, tuple, np.ndarray),
        ):
            crashed = [
                bool(v)
                for v
                in crashed_value
            ]
        else:
            crashed = [
                bool(
                    crashed_value
                )
            ]

    else:
        crashed = [
            bool(v)
            for v
            in crashed
        ]

    terminated = bool(
        episode_state.get(
            "terminated",
            any(
                crashed
            ),
        )
    )

    truncated = bool(
        episode_state.get(
            "truncated",
            info.get(
                "TimeLimit.truncated",
                False,
            ),
        )
    )

    if any(
        crashed
    ):
        reason = (
            "controlled_vehicle_collision"
        )

    elif truncated:
        reason = (
            "truncated"
        )

    elif terminated:
        reason = (
            "terminated_other"
        )

    else:
        reason = (
            "done_unresolved"
        )

    return {
        "reason":
            reason,

        "terminated":
            terminated,

        "truncated":
            truncated,

        "time":
            episode_state.get(
                "time",
                None,
            ),

        "crashed":
            crashed,

        "lane_index":
            episode_state.get(
                "lane_index",
                None,
            ),

        "on_road":
            episode_state.get(
                "on_road",
                None,
            ),

        "speed":
            info.get(
                "speed",
                None,
            ),

        "x_position":
            info.get(
                "x_position",
                None,
            ),

        "y_position":
            info.get(
                "y_position",
                None,
            ),

        "follow_distance":
            info.get(
                "follow_distance",
                None,
            ),

        "follow_ttc":
            info.get(
                "follow_ttc",
                None,
            ),
    }


def _write_episode_video_manifest(
    records,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                "source_commit":
                    SOURCE_COMMIT,

                "video_count":
                    len(
                        records
                    ),

                "episodes":
                    records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )



def main():
    args = parse_args()

    set_random_seed(
        args.seed,
        using_cuda=(
            torch.cuda.is_available()
        ),
    )

    output_dir = Path(
        args.output_dir
    ).resolve()

    video_output_dir = (
        Path(
            args.video_dir
        ).resolve()
        if args.video_dir
        else (
            output_dir
            / "videos"
        )
    )

    if args.save_episode_videos:
        video_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    episode_video_manifest_path = (
        video_output_dir
        / "episode_video_manifest.json"
    )

    episode_video_records = []

    records = []

    episode_index = 0

    skipped_done_steps = 0

    print(
        "=" * 108
    )

    print(
        "ALLMERGE EXPERT PILOT COLLECTION"
    )

    print(
        "=" * 108
    )

    print(
        "source commit:",
        SOURCE_COMMIT,
    )

    print(
        "target frames:",
        args.frames,
    )

    print(
        "working directory:",
        Path.cwd(),
    )

    print(
        "repository root:",
        REPO_ROOT,
    )

    print(
        "output:",
        output_dir,
    )

    print(
        "save episode videos:",
        bool(
            args.save_episode_videos
        ),
    )

    if args.save_episode_videos:
        print(
            "video output:",
            video_output_dir,
        )

    print(
        "=" * 108
    )

    while len(
        records
    ) < args.frames:
        episode_seed = (
            args.seed
            + episode_index
        )

        env = None
        video_env = None

        episode_start_frame = len(
            records
        )

        episode_step = 0
        episode_done = False

        episode_terminal = {
            "reason":
                "collection_target_reached",

            "terminated":
                False,

            "truncated":
                False,
        }

        episode_video_path = None

        try:
            (
                env,
                video_env,
                obs,
            ) = env_reset(
                args.env_name,
                seed=episode_seed,
            )

            if args.save_episode_videos:
                episode_video_path = (
                    video_output_dir
                    / (
                        f"episode_{episode_index:04d}"
                        f"_seed_{episode_seed:06d}.mp4"
                    )
                )

                episode_video_path = (
                    _configure_episode_video(
                        video_env,
                        episode_video_path,
                    )
                )

            # Default path remains the raw VecEnv, exactly as before.
            # When video recording is explicitly enabled, VecVideoRecorder is
            # only a transparent stepping/render wrapper around the same env.
            runner_env = (
                video_env
                if args.save_episode_videos
                else env
            )

            while len(
                records
            ) < args.frames:
                mode = (
                    set_platoon_mode()
                )

                action = decide(
                    obs,
                    mode,
                )

                (
                    obs,
                    reward,
                    done,
                    infos,
                ) = runner_env.step(
                    action
                )

                episode_step += 1

                # DummyVecEnv auto-resets on done. Skip that transition so
                # latest_* cannot accidentally refer to the reset state.
                if bool(
                    done[
                        0
                    ]
                ):
                    skipped_done_steps += 1
                    episode_done = True

                    terminal_info = (
                        infos[
                            0
                        ]
                        if infos
                        else {}
                    )

                    episode_terminal = (
                        _episode_terminal_summary(
                            terminal_info
                        )
                    )

                    break

                features = get_vec_attr(
                    env,
                    "latest_planner_features",
                )

                alignment = get_vec_attr(
                    env,
                    "latest_expert_alignment",
                )

                record = build_frame_record(
                    frame_index=len(
                        records
                    ),
                    episode_index=(
                        episode_index
                    ),
                    episode_step=(
                        episode_step
                    ),
                    seed=(
                        episode_seed
                    ),
                    features=features,
                    alignment=alignment,
                    allow_contract_mismatch=(
                        args.allow_contract_mismatch
                    ),
                )

                records.append(
                    record
                )

                if (
                    len(
                        records
                    )
                    % 25
                    == 0
                    or len(
                        records
                    )
                    == args.frames
                ):
                    traffic_valid = np.concatenate(
                        [
                            r[
                                "target_mode_traffic_valid"
                            ].reshape(-1)
                            for r
                            in records
                        ]
                    )

                    print(
                        f"[pilot] "
                        f"{len(records):>3d}/"
                        f"{args.frames} frames | "
                        f"{len(records) * record['target_mode'].size} "
                        f"ego samples | "
                        f"target traffic-valid="
                        f"{100.0 * traffic_valid.mean():.1f}%"
                    )

        finally:
            # Closing VecVideoRecorder also closes its wrapped VecEnv.
            if video_env is not None:
                try:
                    video_env.close()
                except Exception:
                    pass

            elif env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        if args.save_episode_videos:
            if (
                episode_video_path is None
                or not episode_video_path.exists()
            ):
                raise RuntimeError(
                    "Episode video was requested but was not created. "
                    f"episode={episode_index}, "
                    f"expected={episode_video_path}. "
                    "Check MoviePy/FFmpeg and VecVideoRecorder."
                )

            episode_video_records.append(
                {
                    "episode_index":
                        int(
                            episode_index
                        ),

                    "seed":
                        int(
                            episode_seed
                        ),

                    "video_path":
                        str(
                            episode_video_path
                        ),

                    "environment_steps":
                        int(
                            episode_step
                        ),

                    "done":
                        bool(
                            episode_done
                        ),

                    "collected_frame_start":
                        int(
                            episode_start_frame
                        ),

                    "collected_frame_end_exclusive":
                        int(
                            len(
                                records
                            )
                        ),

                    "collected_frames":
                        int(
                            len(
                                records
                            )
                            - episode_start_frame
                        ),

                    "terminal":
                        episode_terminal,
                }
            )

            _write_episode_video_manifest(
                episode_video_records,
                episode_video_manifest_path,
            )

            print(
                f"[video] episode={episode_index:04d} | "
                f"steps={episode_step} | "
                f"frames={len(records) - episode_start_frame} | "
                f"reason={episode_terminal.get('reason')} | "
                f"{episode_video_path}"
            )

        episode_index += 1

    (
        dataset_path,
        summary_path,
        csv_path,
        metadata_path,
        summary,
    ) = save_pilot(
        records,
        output_dir,
    )

    print_summary(
        summary
    )

    print(
        "\nSaved:"
    )

    print(
        " ",
        dataset_path,
    )

    print(
        " ",
        summary_path,
    )

    print(
        " ",
        csv_path,
    )

    print(
        " ",
        metadata_path,
    )

    print(
        "\nepisodes used:",
        episode_index,
    )

    print(
        "done transitions skipped:",
        skipped_done_steps,
    )
