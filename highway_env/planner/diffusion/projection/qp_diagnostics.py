"""Artifacts for diagnosing OSQP primal-infeasible JTP group problems."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp


_STABILITY_METRICS = ("yaw_rate", "sideslip", "roll_angle", "ltr")
_CONTROL_NAMES = ("acceleration", "steering")


def family_row_labels(family: str, indices, horizon: int, circle_count: int) -> list[str]:
    """Map flattened JTP residual indices to physical constraint labels."""
    labels = []
    for raw_index in indices:
        index = int(raw_index)
        if family == "stability":
            half = horizon * 4
            sign = "upper" if index < half else "lower"
            local = index if index < half else index - half
            time_index = local // 4 + 1
            metric = _STABILITY_METRICS[local % 4]
            labels.append(f"stability.{sign}.{metric}@t={time_index}")
        elif family == "road":
            per_time = circle_count * 2
            time_index = index // per_time + 1
            local = index % per_time
            circle_index = local // 2
            side = "left" if local % 2 == 0 else "right"
            labels.append(
                f"road.{side}.circle={circle_index}@t={time_index}")
        elif family == "curvature":
            labels.append(f"curvature.limit@t={index + 2}")
        elif family == "control_rate":
            per_sign = horizon * 2
            sign = "upper" if index < per_sign else "lower"
            local = index if index < per_sign else index - per_sign
            time_index = local // 2 + 1
            control = _CONTROL_NAMES[local % 2]
            labels.append(f"control_rate.{sign}.{control}@t={time_index}")
        else:
            raise ValueError(f"Unknown JTP constraint family: {family}")
    return labels


def _finite_json_value(value):
    value = float(value)
    return value if np.isfinite(value) else None


def analyze_primal_infeasibility_certificate(
    matrix, lower, upper, certificate, row_blocks, row_families, row_labels
) -> tuple[np.ndarray, dict]:
    """Compute OSQP's certificate separation value and row/family contributions."""
    certificate = np.asarray(certificate, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    row_blocks = list(row_blocks)
    row_families = list(row_families)
    row_labels = list(row_labels)

    contributions = np.zeros_like(certificate)
    positive = certificate > 0.0
    negative = certificate < 0.0
    finite_upper = np.isfinite(upper)
    finite_lower = np.isfinite(lower)
    upper_mask = positive & finite_upper
    lower_mask = negative & finite_lower
    contributions[upper_mask] = certificate[upper_mask] * upper[upper_mask]
    contributions[lower_mask] = certificate[lower_mask] * lower[lower_mask]

    aty = np.asarray(matrix.T @ certificate).reshape(-1)
    block_summary = {}
    family_summary = {}
    for block in dict.fromkeys(row_blocks):
        mask = np.asarray([value == block for value in row_blocks], dtype=bool)
        block_summary[block] = {
            "row_count": int(mask.sum()),
            "certificate_support_count": int((np.abs(certificate[mask]) > 1.0e-10).sum()),
            "bound_contribution": float(contributions[mask].sum()),
        }
    for family in dict.fromkeys(row_families):
        mask = np.asarray([value == family for value in row_families], dtype=bool)
        family_summary[family] = {
            "row_count": int(mask.sum()),
            "certificate_support_count": int((np.abs(certificate[mask]) > 1.0e-10).sum()),
            "bound_contribution": float(contributions[mask].sum()),
        }

    support = np.flatnonzero(np.abs(certificate) > 1.0e-10)
    active_rows = [{
        "row_index": int(index),
        "block": row_blocks[index],
        "family": row_families[index],
        "label": row_labels[index],
        "certificate_multiplier": float(certificate[index]),
        "bound_contribution": float(contributions[index]),
        "lower": _finite_json_value(lower[index]),
        "upper": _finite_json_value(upper[index]),
    } for index in support]
    active_rows.sort(key=lambda item: abs(item["bound_contribution"]), reverse=True)

    separation_value = float(contributions.sum())
    analysis = {
        "certificate_length": int(certificate.size),
        "certificate_support_count": int(support.size),
        "certificate_separation_value": separation_value,
        "certificate_margin": -separation_value,
        "certificate_is_separating": separation_value < 0.0,
        "A_transpose_certificate_inf_norm": (
            float(np.abs(aty).max()) if aty.size else 0.0),
        "row_block_contributions": block_summary,
        "constraint_family_contributions": family_summary,
        "certificate_supported_rows": active_rows,
    }
    return contributions, analysis


def _csc_payload(prefix: str, matrix) -> dict:
    value = sp.csc_matrix(matrix)
    return {
        f"{prefix}_data": value.data,
        f"{prefix}_indices": value.indices,
        f"{prefix}_indptr": value.indptr,
        f"{prefix}_shape": np.asarray(value.shape, dtype=np.int64),
    }


def write_primal_infeasibility_diagnostic(
    problem,
    *,
    metadata: dict,
    arrays: dict,
    solver_status: str,
    solver_status_val: int,
    solver_info: dict,
    primal_certificate,
    dual_certificate,
    solver_x,
    solver_y,
    output_dir: str | Path,
) -> dict:
    """Write the exact sparse OSQP problem, bounds, and infeasibility certificate."""
    P, q, A, lower, upper = problem
    P = sp.csc_matrix(P)
    A = sp.csc_matrix(A)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    primal_certificate = np.asarray(primal_certificate, dtype=np.float64).reshape(-1)
    dual_certificate = (np.asarray(dual_certificate, dtype=np.float64).reshape(-1)
                        if dual_certificate is not None else np.empty(0, dtype=np.float64))
    solver_x = (np.asarray(solver_x, dtype=np.float64).reshape(-1)
                if solver_x is not None else np.empty(0, dtype=np.float64))
    solver_y = (np.asarray(solver_y, dtype=np.float64).reshape(-1)
                if solver_y is not None else np.empty(0, dtype=np.float64))

    row_blocks = metadata["row_blocks"]
    row_families = metadata["row_families"]
    row_labels = metadata["row_labels"]
    contributions, certificate_analysis = analyze_primal_infeasibility_certificate(
        A, lower, upper, primal_certificate,
        row_blocks, row_families, row_labels)

    context = metadata["context"]
    parts = [
        f"seed_{context.get('seed', 'na')}",
        f"episode_{context.get('episode', 'na')}",
        f"frame_{context.get('frame', 'na')}",
        f"vehicle_{context.get('vehicle_index', 'na')}",
        f"iter_{metadata['iteration']}",
        f"group_{metadata['group_index']}_{metadata['group_role']}",
        datetime.now().strftime("%Y%m%dT%H%M%S_%f"),
    ]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(map(str, parts)))
    directory = Path(output_dir)
    if not directory.is_absolute():
        directory = Path.cwd() / directory
    directory.mkdir(parents=True, exist_ok=True)
    matrix_path = directory / f"{stem}.npz"
    metadata_path = directory / f"{stem}.json"

    variable_labels = metadata["variable_labels"]
    payload = {
        **_csc_payload("P", P),
        **_csc_payload("A", A),
        "q": np.asarray(q, dtype=np.float64),
        "lower": lower,
        "upper": upper,
        "primal_infeasibility_certificate": primal_certificate,
        "dual_infeasibility_certificate": dual_certificate,
        "solver_x": solver_x,
        "solver_y": solver_y,
        "certificate_bound_contributions": contributions,
        "row_blocks": np.asarray(row_blocks, dtype=str),
        "row_families": np.asarray(row_families, dtype=str),
        "row_labels": np.asarray(row_labels, dtype=str),
        "variable_labels": np.asarray(variable_labels, dtype=str),
    }
    payload.update({key: np.asarray(value) for key, value in arrays.items()})
    np.savez_compressed(matrix_path, **payload)

    document = {
        "schema_version": 1,
        "artifact_type": "jtp_qp_primal_infeasibility",
        "solver_status": solver_status,
        "solver_status_val": int(solver_status_val),
        "solver_info": solver_info,
        "context": context,
        "iteration": int(metadata["iteration"]),
        "group_index": int(metadata["group_index"]),
        "group_role": metadata["group_role"],
        "group_families": metadata["group_families"],
        "qp_group_order": metadata["qp_group_order"],
        "qp_settings": metadata["qp_settings"],
        "matrix_file": matrix_path.name,
        "matrix_shapes": {
            "P": list(P.shape),
            "A": list(A.shape),
            "q": list(np.asarray(q).shape),
        },
        "variable_blocks": metadata["variable_blocks"],
        "row_blocks": metadata["row_block_ranges"],
        "control_bounds": metadata["control_bounds"],
        "constraint_families": metadata["constraint_families"],
        "certificate_analysis": certificate_analysis,
    }
    metadata_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "matrix_file": str(matrix_path),
        "metadata_file": str(metadata_path),
        "certificate_separation_value": certificate_analysis["certificate_separation_value"],
        "certificate_A_transpose_inf_norm": certificate_analysis["A_transpose_certificate_inf_norm"],
        "certificate_support_count": certificate_analysis["certificate_support_count"],
    }
