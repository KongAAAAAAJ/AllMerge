"""Priority-grouped constrained QP backend for TRACE JTP."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def _family_balanced_weights(labels):
    labels = np.asarray(labels)
    weights = np.empty(labels.size, dtype=np.float64)
    for family in dict.fromkeys(labels.tolist()):
        mask = labels == family
        weights[mask] = 1.0 / int(mask.sum())
    return weights


def _build_qp_problem(control_diagonal, control_lower, control_upper,
                      state_jacobian, state_residual, state_weights,
                      protection_jacobian, protection_weights, cfg):
    """Build OSQP matrices for one priority group; no trust-region rows."""
    nu = int(control_diagonal.size)
    ns = int(state_residual.size)
    np_rows = int(protection_jacobian.shape[0])
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
        upper_parts.extend((-state_residual, state_residual))

    if np_rows:
        zero_ps = sp.csc_matrix((np_rows, ns))
        A_protection = sp.hstack((sp.csc_matrix(protection_jacobian), zero_ps,
                                  -sp.eye(np_rows, format="csc")), format="csc")
        A_protection_slack = sp.hstack((sp.csc_matrix((np_rows, nu)), zero_ps,
                                        sp.eye(np_rows, format="csc")), format="csc")
        blocks.extend((A_protection, A_protection_slack))
        lower_parts.extend((np.full(np_rows, -np.inf), np.zeros(np_rows)))
        upper_parts.extend((np.zeros(np_rows),
                            np.full(np_rows, float(cfg["protection_tolerance"]))))

    A = sp.vstack(blocks, format="csc")
    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)
    return P, q, A, lower, upper


def _solve_osqp(problem, cfg):
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
        "primal_residual": float(info.prim_res),
        "dual_residual": float(info.dual_res),
        "solver_objective": float(info.obj_val),
        "solver_setup_time_s": float(info.setup_time),
        "solver_solve_time_s": float(info.solve_time),
    }
    if int(info.status_val) != 1:
        return None, diagnostics
    return np.asarray(result.x, dtype=np.float64), diagnostics


def _control_objective_diagonal(controls, control_weights, cfg):
    if control_weights is None:
        raise ValueError("constrained_qp requires optimization.control_weights")
    weights = np.asarray(control_weights, dtype=np.float64)
    return (float(cfg["qp"]["control_correction_weight"])
            * np.tile(weights, controls.numel() // 2))


def _family_rows(names, selected, active_mask=None):
    labels = np.concatenate([
        np.full(int(selected[name].numel()), name, dtype=object) for name in names
    ])
    return labels if active_mask is None else labels[active_mask]


def solve_groups_qp(controls, groups, selected, rows, thresholds, order, cfg,
                    control_lower, control_upper, control_weights):
    """Solve the hard-box restoration and three priority groups with OSQP."""
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
        "primal_residual": 0.0,
        "dual_residual": 0.0,
    }
    if np.all(hard_lower <= 0.0) and np.all(hard_upper >= 0.0):
        p = np.zeros(nu, dtype=np.float64)
    else:
        problem = _build_qp_problem(
            control_diagonal, hard_lower, hard_upper,
            empty_jacobian, empty_vector, empty_weights,
            empty_jacobian, empty_weights, cfg)
        solution, restoration_info = _solve_osqp(problem, cfg)
        if solution is None:
            return None, {
                "reason": "hard_box_restoration_qp_failed",
                "hard_box_restoration": restoration_info,
                "solver_method": "constrained_qp",
                "trust_region_enabled": False,
            }
        p = solution[:nu]

    hard_restoration_norm = float(np.linalg.norm(p))
    history_rows = []
    history_weights = []
    group_details = []
    total_solver_iterations = int(restoration_info.get("solver_iterations", 0))
    total_solve_time = float(restoration_info.get("solver_solve_time_s", 0.0))

    for names in order:
        use = [name for name in names if name in rows and selected[name].numel()]
        if not use:
            continue

        jacobian_parts = [rows[name].detach().to(device="cpu", dtype=torch.float64).numpy()
                          for name in use]
        jac = np.concatenate(jacobian_parts, axis=0)
        residual_parts = [
            (groups[name][selected[name]] - thresholds[name][selected[name]])
            .detach().to(device="cpu", dtype=torch.float64).numpy()
            for name in use
        ]
        residual = np.concatenate(residual_parts) + jac @ p
        active = residual > 0.0
        active_count = int(active.sum())

        if active_count:
            state_jac = jac[active]
            state_residual = residual[active]
            active_labels = _family_rows(use, selected, active)
            state_weights = _family_balanced_weights(active_labels)
        else:
            state_jac = empty_jacobian
            state_residual = empty_vector
            state_weights = empty_weights

        if history_rows:
            protection_jac = np.concatenate(history_rows, axis=0)
            protection_weights = np.concatenate(history_weights)
        else:
            protection_jac = empty_jacobian
            protection_weights = empty_weights

        lower_step = hard_lower - p
        upper_step = hard_upper - p
        problem = _build_qp_problem(
            control_diagonal, lower_step, upper_step,
            state_jac, state_residual, state_weights,
            protection_jac, protection_weights, cfg)
        solution, solve_info = _solve_osqp(problem, cfg)
        if solution is None:
            return None, {
                "reason": "priority_group_qp_failed",
                "failed_group": names,
                "failed_group_solver": solve_info,
                "groups": group_details,
                "hard_box_restoration": restoration_info,
                "solver_method": "constrained_qp",
                "trust_region_enabled": False,
            }

        delta = solution[:nu]
        state_slack = solution[nu:nu + active_count]
        protection_slack = solution[nu + active_count:]
        predicted_after = residual[active] + state_jac @ delta if active_count else empty_vector
        before_max = float(state_residual.max()) if active_count else 0.0
        after_max = float(np.maximum(predicted_after, 0.0).max()) if active_count else 0.0
        group_details.append({
            "families": list(names),
            "active_rows": active_count,
            **solve_info,
            "state_slack_sum": float(state_slack.sum()) if active_count else 0.0,
            "state_slack_max": float(state_slack.max()) if active_count else 0.0,
            "protection_rows": int(protection_jac.shape[0]),
            "protection_slack_sum": float(protection_slack.sum()) if protection_slack.size else 0.0,
            "protection_slack_max": float(protection_slack.max()) if protection_slack.size else 0.0,
            "protection_linearized_increase_max": (
                float(np.maximum(protection_jac @ delta, 0.0).max())
                if protection_jac.shape[0] else 0.0),
            "increment_norm": float(np.linalg.norm(delta)),
            "cumulative_correction_norm": float(np.linalg.norm(p + delta)),
            "linear_max_before": before_max,
            "linear_max_after": after_max,
        })
        total_solver_iterations += int(solve_info["solver_iterations"])
        total_solve_time += float(solve_info["solver_solve_time_s"])
        p = p + delta

        history_rows.append(jac)
        history_weights.append(_family_balanced_weights(_family_rows(use, selected)))

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
    }
