"""Full-group priority-constrained QP backend for TRACE JTP."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .qp_diagnostics import family_row_labels


def _family_balanced_weights(labels):
    labels = np.asarray(labels)
    weights = np.empty(labels.size, dtype=np.float64)
    for family in dict.fromkeys(labels.tolist()):
        mask = labels == family
        weights[mask] = 1.0 / int(mask.sum())
    return weights


def _build_qp_problem(control_diagonal, control_lower, control_upper,
                      state_jacobian, state_residual, state_slack_upper,
                      state_weights, protection_jacobian, protection_bound,
                      protection_slack_upper, protection_weights, cfg):
    """Build one full-row priority QP. No active-set truncation or trust rows."""
    nu = int(control_diagonal.size)
    ns = int(state_residual.size)
    np_rows = int(protection_bound.size)
    nvar = nu + ns + np_rows

    state_cfg = cfg["qp"]["state_slack"]
    protection_cfg = cfg["qp"]["protection_slack"]
    p_diagonal = np.concatenate((
        control_diagonal,
        float(state_cfg["quadratic_penalty"]) * state_weights,
        float(protection_cfg["quadratic_penalty"]) * protection_weights,
    ))
    q = np.concatenate((
        np.zeros(nu, dtype=np.float64),
        float(state_cfg["linear_penalty"]) * state_weights,
        float(protection_cfg["linear_penalty"]) * protection_weights,
    ))
    P = sp.diags(p_diagonal, format="csc")

    zero_us = sp.csc_matrix((nu, ns))
    zero_up = sp.csc_matrix((nu, np_rows))
    A_control = sp.hstack((sp.eye(nu, format="csc"), zero_us, zero_up), format="csc")
    lower_parts = [control_lower]
    upper_parts = [control_upper]
    blocks = [A_control]

    if ns:
        zero_sp = sp.csc_matrix((ns, np_rows))
        A_state = sp.hstack((sp.csc_matrix(state_jacobian),
                             -sp.eye(ns, format="csc"), zero_sp), format="csc")
        A_state_slack = sp.hstack((sp.csc_matrix((ns, nu)),
                                   sp.eye(ns, format="csc"), zero_sp), format="csc")
        blocks.extend((A_state, A_state_slack))
        lower_parts.extend((np.full(ns, -np.inf), np.zeros(ns)))
        upper_parts.extend((-state_residual, state_slack_upper))

    if np_rows:
        zero_ps = sp.csc_matrix((np_rows, ns))
        A_protection = sp.hstack((sp.csc_matrix(protection_jacobian), zero_ps,
                                  -sp.eye(np_rows, format="csc")), format="csc")
        A_protection_slack = sp.hstack((sp.csc_matrix((np_rows, nu)), zero_ps,
                                        sp.eye(np_rows, format="csc")), format="csc")
        blocks.extend((A_protection, A_protection_slack))
        lower_parts.extend((np.full(np_rows, -np.inf), np.zeros(np_rows)))
        upper_parts.extend((protection_bound, protection_slack_upper))

    A = sp.vstack(blocks, format="csc")
    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)
    return P, q, A, lower, upper


def _solve_osqp(problem, cfg, *, diagnostic_metadata=None):
    import osqp

    P, q, A, lower, upper = problem
    qp_cfg = cfg["qp"]
    solver = osqp.OSQP()
    solver.setup(
        P=P,
        q=q,
        A=A,
        l=lower,
        u=upper,
        verbose=bool(qp_cfg["verbose"]),
        eps_abs=float(qp_cfg["eps_abs"]),
        eps_rel=float(qp_cfg["eps_rel"]),
        max_iter=int(qp_cfg["max_iter"]),
        time_limit=float(qp_cfg["time_limit_ms"]) / 1000.0,
        polishing=bool(qp_cfg["polishing"]),
        adaptive_rho=bool(qp_cfg["adaptive_rho"]),
    )
    result = solver.solve()
    info = result.info
    diagnostics = {
        "backend": "osqp",
        "solver_status": str(info.status),
        "solver_status_val": int(info.status_val),
        "solver_iterations": int(info.iter),
        "solver_time_limit_ms": float(qp_cfg["time_limit_ms"]),
        "primal_residual": float(info.prim_res),
        "dual_residual": float(info.dual_res),
        "solver_objective": float(info.obj_val),
        "solver_setup_time_s": float(info.setup_time),
        "solver_solve_time_s": float(info.solve_time),
    }
    if (diagnostic_metadata is not None
            and cfg["qp"]["diagnostics"]["save_primal_infeasible"]
            and int(info.status_val) in (3, 4)):
        from .qp_diagnostics import write_primal_infeasibility_diagnostic
        artifact = write_primal_infeasibility_diagnostic(
            problem,
            metadata=diagnostic_metadata["metadata"],
            arrays=diagnostic_metadata["arrays"],
            solver_status=str(info.status),
            solver_status_val=int(info.status_val),
            solver_info=diagnostics,
            primal_certificate=result.prim_inf_cert,
            dual_certificate=result.dual_inf_cert,
            solver_x=result.x,
            solver_y=result.y,
            output_dir=cfg["qp"]["diagnostics"]["output_dir"],
        )
        diagnostics["infeasibility_diagnostic"] = artifact
    if int(info.status_val) != 1:
        return None, diagnostics
    return np.asarray(result.x, dtype=np.float64), diagnostics


def _control_objective_diagonal(controls, control_weights, cfg):
    if control_weights is None:
        raise ValueError("constrained_qp requires optimization.control_weights")
    weights = np.asarray(control_weights, dtype=np.float64)
    return (float(cfg["qp"]["control_correction_weight"])
            * np.tile(weights, controls.numel() // 2))


def _family_labels(names, rows):
    return np.concatenate([
        np.full(int(rows[name].shape[0]), name, dtype=object) for name in names
    ])


def _state_slack_upper(base_residual, nondegradation_tolerance):
    """Active rows may retain their original excess; inactive rows cannot become violated."""
    return np.where(
        base_residual > 0.0,
        base_residual + float(nondegradation_tolerance),
        0.0,
    )


def _protection_bounds(base_residual, cumulative_change, protection_tolerance,
                        nondegradation_tolerance):
    """Protect all prior rows relative to the outer-iteration baseline.

    Previously violated rows may increase only within the Stage-2 tolerance.
    Previously satisfied rows must remain satisfied, preserving the group's
    violation count as well as per-row non-degradation.
    """
    was_violated = base_residual > 0.0
    slack_limit = min(float(protection_tolerance), float(nondegradation_tolerance))
    bound = np.where(was_violated, -cumulative_change,
                     -(base_residual + cumulative_change))
    slack_upper = np.where(was_violated, slack_limit, 0.0)
    return bound, slack_upper


def solve_groups_qp(controls, groups, selected, rows, thresholds, order, cfg,
                    control_lower, control_upper, control_weights,
                    diagnostic_context=None):
    """Solve hard-box restoration and all rows in each of the three priority groups."""
    flat = controls.detach().reshape(-1)
    nu = flat.numel()
    control_diagonal = _control_objective_diagonal(controls, control_weights, cfg)
    control_lo = np.tile(control_lower.detach().reshape(-1).cpu().numpy(), nu // 2)
    control_hi = np.tile(control_upper.detach().reshape(-1).cpu().numpy(), nu // 2)
    bound_scale = max(1.0, float(np.abs(control_lo).max()), float(np.abs(control_hi).max()))
    bound_margin = max(float(torch.finfo(flat.dtype).eps) * 4.0,
                       float(cfg["qp"]["eps_abs"]) * 10.0) * bound_scale
    hard_lower = control_lo + bound_margin - flat.cpu().numpy()
    hard_upper = control_hi - bound_margin - flat.cpu().numpy()

    empty_jacobian = np.empty((0, nu), dtype=np.float64)
    empty_vector = np.empty(0, dtype=np.float64)
    empty_weights = np.empty(0, dtype=np.float64)
    restoration_info = {
        "backend": "osqp",
        "solver_status": "not_required",
        "solver_iterations": 0,
        "solver_solve_time_s": 0.0,
        "primal_residual": 0.0,
        "dual_residual": 0.0,
    }
    if np.all(hard_lower <= 0.0) and np.all(hard_upper >= 0.0):
        p = np.zeros(nu, dtype=np.float64)
    else:
        problem = _build_qp_problem(
            control_diagonal, hard_lower, hard_upper,
            empty_jacobian, empty_vector, empty_vector, empty_weights,
            empty_jacobian, empty_vector, empty_vector, empty_weights, cfg)
        solution, restoration_info = _solve_osqp(problem, cfg)
        if solution is None:
            return None, {
                "reason": "hard_box_restoration_qp_failed",
                "solver_solve_time_total_s": restoration_info["solver_solve_time_s"],
                "hard_box_restoration": restoration_info,
                "solver_method": "constrained_qp",
                "trust_region_enabled": False,
            }
        p = solution[:nu]

    hard_restoration_norm = float(np.linalg.norm(p))
    history_jacobian = empty_jacobian
    history_base_residual = empty_vector
    history_labels = np.empty(0, dtype=object)
    history_row_labels = []
    group_details = []
    total_solver_iterations = int(restoration_info.get("solver_iterations", 0))
    total_solve_time = float(restoration_info["solver_solve_time_s"])
    diagnostics_enabled = cfg["qp"].get("diagnostics", {}).get(
        "save_primal_infeasible", False)

    for group_index, names in enumerate(order, start=1):
        use = [name for name in names if name in rows and rows[name].shape[0]]
        if not use:
            continue

        jacobian_parts = [rows[name].detach().to(device="cpu", dtype=torch.float64).numpy()
                          for name in use]
        jac = np.concatenate(jacobian_parts, axis=0)
        base_residual_parts = [
            (groups[name][selected[name]] - thresholds[name][selected[name]])
            .detach().to(device="cpu", dtype=torch.float64).numpy()
            for name in use
        ]
        base_residual = np.concatenate(base_residual_parts)
        residual = base_residual + jac @ p
        labels = _family_labels(use, rows)
        state_row_labels = []
        if diagnostics_enabled:
            circle_count = int((diagnostic_context or {})["footprint_circle_count"])
            horizon_steps = int((diagnostic_context or {})["horizon_steps"])
            for name in use:
                state_row_labels.extend(family_row_labels(
                    name, selected[name].detach().cpu().tolist(),
                    horizon_steps, circle_count))
        state_weights = _family_balanced_weights(labels)
        state_slack_upper = _state_slack_upper(
            base_residual, cfg["nondegradation_tolerance"])

        if history_jacobian.shape[0]:
            cumulative_history_change = history_jacobian @ p
            protection_bound, protection_slack_upper = _protection_bounds(
                history_base_residual, cumulative_history_change,
                cfg["protection_tolerance"], cfg["nondegradation_tolerance"])
            protection_weights = _family_balanced_weights(history_labels)
        else:
            protection_bound = empty_vector
            protection_slack_upper = empty_vector
            protection_weights = empty_weights

        lower_step = hard_lower - p
        upper_step = hard_upper - p
        problem = _build_qp_problem(
            control_diagonal, lower_step, upper_step,
            jac, residual, state_slack_upper, state_weights,
            history_jacobian, protection_bound, protection_slack_upper,
            protection_weights, cfg)
        diagnostic_metadata = None
        if diagnostics_enabled and ("road" in use or "curvature" in use):
            state_family_labels = [name for name in use for _ in selected[name].tolist()]
            protection_family_labels = history_labels.tolist()
            control_variable_labels = [
                f"delta_u.{('acceleration' if index % 2 == 0 else 'steering')}@t={index // 2 + 1}"
                for index in range(nu)
            ]
            state_variable_labels = [f"state_slack::{label}" for label in state_row_labels]
            protection_variable_labels = [
                f"protection_slack::{label}" for label in history_row_labels]
            state_slack_labels = [
                f"state_slack.nonnegative::{label}" for label in state_row_labels]
            protection_slack_labels = [
                f"protection_slack.nonnegative::{label}" for label in history_row_labels]
            row_labels = (
                control_variable_labels + state_row_labels + state_slack_labels
                + history_row_labels + protection_slack_labels)
            row_blocks = (
                ["control_box"] * nu
                + ["state_inequality"] * len(state_row_labels)
                + ["state_slack_nonnegative"] * len(state_row_labels)
                + ["protection_inequality"] * len(history_row_labels)
                + ["protection_slack_nonnegative"] * len(history_row_labels))
            row_families = (
                ["control_box"] * nu
                + state_family_labels + state_family_labels
                + protection_family_labels + protection_family_labels)
            state_row_start = nu
            state_slack_start = state_row_start + len(state_row_labels)
            protection_start = state_slack_start + len(state_row_labels)
            protection_slack_start = protection_start + len(history_row_labels)
            diagnostic_metadata = {
                "metadata": {
                    "context": dict(diagnostic_context or {}),
                    "iteration": int((diagnostic_context or {}).get("iteration", 0)),
                    "group_index": group_index,
                    "group_role": "road_curvature",
                    "group_families": list(use),
                    "qp_group_order": [list(group) for group in order],
                    "qp_settings": {
                        "eps_abs": float(cfg["qp"]["eps_abs"]),
                        "eps_rel": float(cfg["qp"]["eps_rel"]),
                        "max_iter": int(cfg["qp"]["max_iter"]),
                        "time_limit_ms": float(cfg["qp"]["time_limit_ms"]),
                        "polishing": bool(cfg["qp"]["polishing"]),
                        "adaptive_rho": bool(cfg["qp"]["adaptive_rho"]),
                    },
                    "row_labels": row_labels,
                    "row_blocks": row_blocks,
                    "row_families": row_families,
                    "variable_labels": (
                        control_variable_labels + state_variable_labels
                        + protection_variable_labels),
                    "variable_blocks": {
                        "control_delta": [0, nu],
                        "state_slack": [nu, nu + len(state_row_labels)],
                        "protection_slack": [
                            nu + len(state_row_labels),
                            nu + len(state_row_labels) + len(history_row_labels)],
                    },
                    "row_block_ranges": {
                        "control_box": [0, nu],
                        "state_inequality": [state_row_start, state_slack_start],
                        "state_slack_nonnegative": [state_slack_start, protection_start],
                        "protection_inequality": [protection_start, protection_slack_start],
                        "protection_slack_nonnegative": [
                            protection_slack_start,
                            protection_slack_start + len(history_row_labels)],
                    },
                    "control_bounds": {
                        "absolute_lower": control_lo.tolist(),
                        "absolute_upper": control_hi.tolist(),
                        "delta_lower_at_group_entry": lower_step.tolist(),
                        "delta_upper_at_group_entry": upper_step.tolist(),
                        "cumulative_delta_lower": hard_lower.tolist(),
                        "cumulative_delta_upper": hard_upper.tolist(),
                    },
                    "constraint_families": {
                        "state": list(use),
                        "protection": list(dict.fromkeys(protection_family_labels)),
                    },
                },
                "arrays": {
                    "state_jacobian": jac,
                    "state_residual_at_group_entry": residual,
                    "state_residual_at_outer_start": base_residual,
                    "state_slack_upper": state_slack_upper,
                    "state_slack_weights": state_weights,
                    "protection_jacobian": history_jacobian,
                    "protection_bound": protection_bound,
                    "protection_slack_upper": protection_slack_upper,
                    "protection_slack_weights": protection_weights,
                    "nominal_control": flat.cpu().numpy(),
                    "control_at_group_entry": flat.cpu().numpy() + p,
                    "cumulative_control_correction": p,
                    "control_lower_absolute": control_lo,
                    "control_upper_absolute": control_hi,
                    "control_delta_lower_at_group_entry": lower_step,
                    "control_delta_upper_at_group_entry": upper_step,
                },
            }
        if diagnostic_metadata is None:
            solution, solve_info = _solve_osqp(problem, cfg)
        else:
            solution, solve_info = _solve_osqp(
                problem, cfg, diagnostic_metadata=diagnostic_metadata)
        total_solve_time += float(solve_info["solver_solve_time_s"])
        if solution is None:
            return None, {
                "reason": "priority_group_qp_failed",
                "solver_solve_time_total_s": total_solve_time,
                "failed_group": names,
                "failed_group_solver": solve_info,
                "groups": group_details,
                "hard_box_restoration": restoration_info,
                "solver_method": "constrained_qp",
                "trust_region_enabled": False,
            }

        state_count = int(base_residual.size)
        protection_count = int(history_base_residual.size)
        delta = solution[:nu]
        state_slack = solution[nu:nu + state_count]
        protection_slack = solution[nu + state_count:nu + state_count + protection_count]
        predicted_residual = residual + jac @ delta
        linear_before = np.maximum(residual, 0.0)
        linear_after = np.maximum(predicted_residual, 0.0)
        protection_after = history_base_residual + history_jacobian @ (p + delta)
        group_details.append({
            "families": list(names),
            "constraint_rows": state_count,
            "constraint_rows_by_family": {
                name: int(rows[name].shape[0]) for name in use
            },
            "violating_rows_at_group_entry": int((residual > 0.0).sum()),
            "inactive_rows_at_group_entry": int((residual <= 0.0).sum()),
            **solve_info,
            "state_slack_sum": float(state_slack.sum()),
            "state_slack_max": float(state_slack.max()) if state_slack.size else 0.0,
            "state_slack_l2": float(np.linalg.norm(state_slack)),
            "state_slack_upper_sum": float(state_slack_upper.sum()),
            "protection_rows": protection_count,
            "protection_slack_sum": float(protection_slack.sum()) if protection_slack.size else 0.0,
            "protection_slack_max": float(protection_slack.max()) if protection_slack.size else 0.0,
            "protection_linearized_regression_max": (
                float(np.maximum(protection_after - history_base_residual, 0.0).max())
                if protection_after.size else 0.0),
            "increment_norm": float(np.linalg.norm(delta)),
            "cumulative_correction_norm": float(np.linalg.norm(p + delta)),
            "linear_max_before": float(linear_before.max()) if linear_before.size else 0.0,
            "linear_max_after": float(linear_after.max()) if linear_after.size else 0.0,
            "linear_violation_count_before": int((residual > 0.0).sum()),
            "linear_violation_count_after": int((predicted_residual > 0.0).sum()),
        })
        total_solver_iterations += int(solve_info["solver_iterations"])

        history_jacobian = np.concatenate((history_jacobian, jac), axis=0)
        history_base_residual = np.concatenate((history_base_residual, base_residual))
        history_labels = np.concatenate((history_labels, labels))
        if diagnostics_enabled:
            history_row_labels.extend(state_row_labels)
        p = p + delta

    final_controls = flat.cpu().numpy() + p
    control_violation = np.maximum(control_lo - final_controls, final_controls - control_hi)
    hard_box_tolerance = max(float(torch.finfo(flat.dtype).eps) * 8.0,
                             float(cfg["qp"]["eps_abs"]) * 10.0)
    if float(np.maximum(control_violation, 0.0).max()) > hard_box_tolerance:
        raise RuntimeError("Grouped QP returned controls outside the hard control range")

    return torch.as_tensor(p.reshape(tuple(controls.shape)), device=controls.device,
                           dtype=controls.dtype), {
        "groups": group_details,
        "jacobian_shape": [sum(int(j.shape[0]) for j in rows.values()), nu],
        "metric_factorizations": 0,
        "solver_method": "constrained_qp",
        "backend": "osqp",
        "hard_box_restoration_norm": hard_restoration_norm,
        "hard_box_restoration": restoration_info,
        "hard_box_active_lower": int(np.isclose(p, hard_lower, atol=1.0e-7).sum()),
        "hard_box_active_upper": int(np.isclose(p, hard_upper, atol=1.0e-7).sum()),
        "solver_iterations_total": total_solver_iterations,
        "solver_solve_time_total_s": total_solve_time,
        "trust_region_enabled": False,
        "constraint_scope": "all_rows_per_priority_group",
        "constraint_rows_by_family": {
            name: int(value.shape[0]) for name, value in rows.items()
        },
    }
