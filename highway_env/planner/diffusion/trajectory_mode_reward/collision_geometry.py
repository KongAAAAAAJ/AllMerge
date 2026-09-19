"""Oriented-box geometry shared by trajectory-mode reward terms."""

from __future__ import annotations

import numpy as np


def shared_corridor_gap_series(
    first: np.ndarray,
    first_dimensions: tuple[float, float],
    second: np.ndarray,
    second_dimensions: tuple[float, float],
    *,
    no_risk_gap_m: float,
) -> np.ndarray:
    """Return bumper gaps when two OBBs share a lateral corridor."""
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    if (
        first_values.shape != second_values.shape
        or first_values.ndim != 2
        or first_values.shape[1] != 3
        or not np.isfinite(first_values).all()
        or not np.isfinite(second_values).all()
    ):
        raise ValueError(
            "OBB trajectories must be finite matching [N,3] arrays"
        )

    first_size = np.asarray(first_dimensions, dtype=np.float64)
    second_size = np.asarray(second_dimensions, dtype=np.float64)
    if (
        first_size.shape != (2,)
        or second_size.shape != (2,)
        or not np.isfinite(first_size).all()
        or not np.isfinite(second_size).all()
        or np.any(first_size <= 0.0)
        or np.any(second_size <= 0.0)
    ):
        raise ValueError(
            "OBB dimensions must be positive finite (length,width)"
        )

    sentinel = float(no_risk_gap_m)
    if not np.isfinite(sentinel) or sentinel <= 0.0:
        raise ValueError(
            "no_risk_gap_m must be positive and finite"
        )

    forward = np.column_stack(
        (np.cos(first_values[:, 2]), np.sin(first_values[:, 2]))
    )
    lateral_axis = np.column_stack(
        (-forward[:, 1], forward[:, 0])
    )
    delta = second_values[:, :2] - first_values[:, :2]
    longitudinal = np.abs(
        np.einsum("ij,ij->i", delta, forward)
    )
    lateral = np.abs(
        np.einsum("ij,ij->i", delta, lateral_axis)
    )

    relative_heading = second_values[:, 2] - first_values[:, 2]
    cosine = np.abs(np.cos(relative_heading))
    sine = np.abs(np.sin(relative_heading))

    first_longitudinal_support = 0.5 * first_size[0]
    first_lateral_support = 0.5 * first_size[1]
    second_longitudinal_support = (
        0.5 * second_size[0] * cosine
        + 0.5 * second_size[1] * sine
    )
    second_lateral_support = (
        0.5 * second_size[0] * sine
        + 0.5 * second_size[1] * cosine
    )

    same_corridor = (
        lateral
        <= first_lateral_support
        + second_lateral_support
        + 1e-6
    )
    bumper = (
        longitudinal
        - first_longitudinal_support
        - second_longitudinal_support
    )
    return np.ascontiguousarray(
        np.where(same_corridor, bumper, sentinel),
        dtype=np.float64,
    )


def obb_overlap_series(
    first: np.ndarray,
    first_dimensions: tuple[float, float],
    second: np.ndarray,
    second_dimensions: tuple[float, float],
    margin_m,
) -> bool:
    """Return whether two synchronized [N,3] OBB pose series overlap."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (
        first.shape != second.shape
        or first.ndim != 2
        or first.shape[1] != 3
    ):
        raise ValueError(
            "OBB trajectories must have matching [N,3] shapes"
        )

    margins = np.asarray(margin_m, dtype=np.float64)
    if margins.ndim == 0:
        margins = np.full(
            (len(first), 2),
            float(margins),
            dtype=np.float64,
        )
    elif margins.shape == (len(first),):
        margins = np.repeat(
            margins[:, None],
            2,
            axis=1,
        )
    if margins.shape != (len(first), 2) or np.any(margins < 0.0):
        raise ValueError(
            "OBB margin must be non-negative scalar, [N], or [N,2]"
        )

    first_half = (
        np.asarray(first_dimensions, dtype=np.float64)[None, :]
        * 0.5
        + margins
    )
    second_half = (
        np.asarray(second_dimensions, dtype=np.float64)[None, :]
        * 0.5
        + margins
    )

    first_long = np.column_stack(
        [np.cos(first[:, 2]), np.sin(first[:, 2])]
    )
    first_lat = np.column_stack(
        [-first_long[:, 1], first_long[:, 0]]
    )
    second_long = np.column_stack(
        [np.cos(second[:, 2]), np.sin(second[:, 2])]
    )
    second_lat = np.column_stack(
        [-second_long[:, 1], second_long[:, 0]]
    )
    delta = second[:, :2] - first[:, :2]

    separated = np.zeros(len(first), dtype=np.bool_)
    for axis in (
        first_long,
        first_lat,
        second_long,
        second_lat,
    ):
        first_radius = (
            first_half[:, 0]
            * np.abs(np.einsum("ij,ij->i", first_long, axis))
            + first_half[:, 1]
            * np.abs(np.einsum("ij,ij->i", first_lat, axis))
        )
        second_radius = (
            second_half[:, 0]
            * np.abs(np.einsum("ij,ij->i", second_long, axis))
            + second_half[:, 1]
            * np.abs(np.einsum("ij,ij->i", second_lat, axis))
        )
        separated |= (
            np.abs(np.einsum("ij,ij->i", delta, axis))
            > first_radius + second_radius
        )

    return bool(np.any(~separated))
