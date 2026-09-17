from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


# Semantic IDs are deliberately independent from mode indices.
SEMANTIC_KEEP = 0
SEMANTIC_LEFT_LC = 1
SEMANTIC_RIGHT_LC = 2
SEMANTIC_STOP = 3

SEMANTIC_NAMES = (
    "KEEP",
    "LEFT_LC",
    "RIGHT_LC",
    "STOP",
)

# Fixed AllMerge / DiffusionDrive-style mode slots:
#   0..2 KEEP H/M/L
#   3..5 LEFT_LC H/M/L
#   6..8 RIGHT_LC H/M/L
#   9    STOP
MODE_SEMANTIC_IDS = (
    SEMANTIC_KEEP,
    SEMANTIC_KEEP,
    SEMANTIC_KEEP,
    SEMANTIC_LEFT_LC,
    SEMANTIC_LEFT_LC,
    SEMANTIC_LEFT_LC,
    SEMANTIC_RIGHT_LC,
    SEMANTIC_RIGHT_LC,
    SEMANTIC_RIGHT_LC,
    SEMANTIC_STOP,
)


@dataclass
class ExpertModeAssignment:
    """
    Derived supervision for the existing multimodal planner.

    target_mode:
        Fine-grained mode index in [0, 9].

    target_semantic:
        KEEP / LEFT_LC / RIGHT_LC / STOP semantic id.

    geometry_mask:
        True when an anchor geometrically exists.
        This is intentionally independent from traffic mode_valid_mask.

    semantic_mask:
        True for modes that belong to target_semantic.

    candidate_mask:
        geometry_mask & semantic_mask.

    anchor_distance:
        Mean per-timestep Euclidean distance to expert trajectory.
        This follows DiffusionDrive's nearest-plan-anchor matching metric.
    """

    target_mode: torch.Tensor
    target_semantic: torch.Tensor
    geometry_mask: torch.Tensor
    semantic_mask: torch.Tensor
    candidate_mask: torch.Tensor
    anchor_distance: torch.Tensor


def _mode_semantic_tensor(
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(
        MODE_SEMANTIC_IDS,
        dtype=torch.long,
        device=device,
    )


def derive_semantic_from_map_features(
    features: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Derive RuleMaker target-lane semantic from the existing map feature flags.

    map point channels:
        6 is_current_lane
        7 is_left_lane
        8 is_right_lane
        9 is_target_lane

    FeatureBuilder guarantees current/target lanes are retained in the map
    set. For ordinary KEEP / lane-change samples, this yields a deterministic
    semantic label without adding a new model input or dataset field.

    STOP cannot be inferred from lane topology alone. Future emergency-stop
    expert data should pass target_semantic explicitly as SEMANTIC_STOP.
    """
    if "map_polylines" not in features:
        raise KeyError(
            "features must contain map_polylines "
            "to derive expert semantic"
        )

    if "map_valid_mask" not in features:
        raise KeyError(
            "features must contain map_valid_mask "
            "to derive expert semantic"
        )

    map_polylines = features[
        "map_polylines"
    ]

    map_valid_mask = features[
        "map_valid_mask"
    ].bool()

    if map_polylines.ndim != 4:
        raise ValueError(
            "map_polylines must be [B,N_map,P,D]"
        )

    if map_polylines.shape[-1] < 10:
        raise ValueError(
            "map_polylines last dimension must contain "
            "current/left/right/target flags at indices 6..9"
        )

    # Flags are repeated along each lane polyline. `amax` is robust to any
    # future per-point masking or interpolation.
    is_current = (
        map_polylines[
            ...,
            6,
        ].amax(
            dim=2
        )
        > 0.5
    )

    is_left = (
        map_polylines[
            ...,
            7,
        ].amax(
            dim=2
        )
        > 0.5
    )

    is_right = (
        map_polylines[
            ...,
            8,
        ].amax(
            dim=2
        )
        > 0.5
    )

    is_target = (
        map_polylines[
            ...,
            9,
        ].amax(
            dim=2
        )
        > 0.5
    )

    target_lane_mask = (
        map_valid_mask
        & is_target
    )

    has_target = target_lane_mask.any(
        dim=1
    )

    if not bool(
        has_target.all()
    ):
        bad = torch.nonzero(
            ~has_target,
            as_tuple=False,
        ).flatten().tolist()

        raise RuntimeError(
            "Cannot derive target semantic: "
            f"no target-lane polyline for batch indices {bad}. "
            "Pass target_semantic explicitly for unsupported topology."
        )

    target_is_current = (
        target_lane_mask
        & is_current
    ).any(
        dim=1
    )

    target_is_left = (
        target_lane_mask
        & is_left
    ).any(
        dim=1
    )

    target_is_right = (
        target_lane_mask
        & is_right
    ).any(
        dim=1
    )

    semantic_count = (
        target_is_current.to(
            torch.int64
        )
        + target_is_left.to(
            torch.int64
        )
        + target_is_right.to(
            torch.int64
        )
    )

    if not bool(
        (semantic_count == 1).all()
    ):
        bad = torch.nonzero(
            semantic_count != 1,
            as_tuple=False,
        ).flatten().tolist()

        raise RuntimeError(
            "Ambiguous target-lane semantic for batch indices "
            f"{bad}. Pass target_semantic explicitly."
        )

    target_semantic = torch.full(
        (
            map_polylines.shape[0],
        ),
        SEMANTIC_KEEP,
        dtype=torch.long,
        device=map_polylines.device,
    )

    target_semantic[
        target_is_left
    ] = SEMANTIC_LEFT_LC

    target_semantic[
        target_is_right
    ] = SEMANTIC_RIGHT_LC

    return target_semantic


def build_geometry_mask(
    anchors: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Geometry mask from the current AnchorBuilder contract.

    Geometrically existing anchors are always retained as nonzero trajectories,
    even when traffic-invalid. Geometrically unavailable modes remain all-zero.
    """
    if anchors.ndim != 4:
        raise ValueError(
            "anchors must be [B,M,T,2]"
        )

    return anchors.abs().amax(
        dim=(-1, -2)
    ) > float(
        epsilon
    )


def assign_expert_mode(
    *,
    features: Dict[str, torch.Tensor],
    target_trajectory: torch.Tensor,
    target_semantic: Optional[
        torch.Tensor
    ] = None,
) -> ExpertModeAssignment:
    """
    Semantic-constrained nearest-anchor assignment.

    This preserves the DiffusionDrive multimodal structure:
      expert -> nearest plan anchor -> cls target + selected regression branch

    Modification relative to the naive global nearest rule:
      candidate = geometry_exists AND same semantic group

    Traffic mode_valid_mask is intentionally NOT used here.
    It remains an inference-time safety/selectability mask.
    """
    anchors = features[
        "coarse_trajectories"
    ]

    if anchors.ndim != 4:
        raise ValueError(
            "coarse_trajectories must be [B,M,T,2]"
        )

    if anchors.shape[1] != len(
        MODE_SEMANTIC_IDS
    ):
        raise ValueError(
            "F.2 semantic assignment expects exactly "
            f"{len(MODE_SEMANTIC_IDS)} modes, "
            f"got {anchors.shape[1]}"
        )

    expected_target_shape = (
        anchors.shape[0],
        anchors.shape[2],
        2,
    )

    if target_trajectory.shape != (
        expected_target_shape
    ):
        raise ValueError(
            "target_trajectory shape="
            f"{tuple(target_trajectory.shape)}, "
            f"expected={expected_target_shape}"
        )

    if target_semantic is None:
        target_semantic = (
            derive_semantic_from_map_features(
                features
            )
        )
    else:
        target_semantic = (
            torch.as_tensor(
                target_semantic,
                dtype=torch.long,
                device=anchors.device,
            )
            .reshape(-1)
        )

    if target_semantic.shape != (
        anchors.shape[0],
    ):
        raise ValueError(
            "target_semantic must be [B]"
        )

    if bool(
        (
            (
                target_semantic
                < SEMANTIC_KEEP
            )
            | (
                target_semantic
                > SEMANTIC_STOP
            )
        ).any()
    ):
        raise ValueError(
            "target_semantic contains unsupported values"
        )

    geometry_mask = build_geometry_mask(
        anchors
    )

    mode_semantic = (
        _mode_semantic_tensor(
            device=anchors.device
        )
    )

    semantic_mask = (
        mode_semantic[
            None,
            :,
        ]
        == target_semantic[
            :,
            None,
        ]
    )

    candidate_mask = (
        geometry_mask
        & semantic_mask
    )

    has_candidate = candidate_mask.any(
        dim=1
    )

    if not bool(
        has_candidate.all()
    ):
        bad = torch.nonzero(
            ~has_candidate,
            as_tuple=False,
        ).flatten().tolist()

        bad_semantic = [
            SEMANTIC_NAMES[
                int(
                    target_semantic[
                        idx
                    ].item()
                )
            ]
            for idx
            in bad
        ]

        raise RuntimeError(
            "No geometry-compatible anchor exists for "
            f"batch indices {bad}, semantics={bad_semantic}. "
            "Do not silently fall back to another semantic group."
        )

    # Match original DiffusionDrive mode assignment:
    # mean Euclidean distance over trajectory points (anchor ADE).
    anchor_distance = torch.linalg.vector_norm(
        target_trajectory[
            :,
            None,
            :,
            :,
        ]
        - anchors,
        dim=-1,
    ).mean(
        dim=-1
    )

    masked_distance = anchor_distance.masked_fill(
        ~candidate_mask,
        float(
            "inf"
        ),
    )

    target_mode = masked_distance.argmin(
        dim=-1
    )

    return ExpertModeAssignment(
        target_mode=target_mode,
        target_semantic=target_semantic,
        geometry_mask=geometry_mask,
        semantic_mask=semantic_mask,
        candidate_mask=candidate_mask,
        anchor_distance=anchor_distance,
    )
