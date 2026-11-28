"""One stability QP with hard acceleration and steering bounds per outer iteration."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch


def _build_qp_problem(control_diagonal, lower, upper, jacobian, residual, slack_lower, slack_upper, cfg):
    nu, ns = len(control_diagonal), len(residual)
    penalty = cfg['qp']['state_slack']
    weights = np.full(ns, 1. / ns) if ns else np.empty(0)
    P = sp.diags(np.r_[control_diagonal, penalty['quadratic_penalty'] * weights], format='csc')
    q = np.r_[np.zeros(nu), penalty['linear_penalty'] * weights]
    A = sp.bmat([[sp.eye(nu), sp.csc_matrix((nu, ns))],
                 [sp.csc_matrix(jacobian), -sp.eye(ns)],
                 [sp.csc_matrix((ns, nu)), sp.eye(ns)]], format='csc')
    return P, q, A, np.r_[lower, np.full(ns, -np.inf), slack_lower], np.r_[upper, -residual, slack_upper]


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
    if int(info.status_val) != 1:
        return None, diagnostics
    return np.asarray(result.x, dtype=np.float64), diagnostics



def solve_qp(controls, stability, selected, jacobian, thresholds, cfg, lower, upper, weights):
    flat = controls.detach().cpu().numpy().reshape(-1)
    lo = np.tile(lower.detach().cpu().numpy(), controls.numel() // 2)
    hi = np.tile(upper.detach().cpu().numpy(), controls.numel() // 2)
    margin = max(float(torch.finfo(controls.dtype).eps) * 4.,
                 float(cfg['qp']['eps_abs']) * 10.) * max(1., float(abs(lo).max()), float(abs(hi).max()))
    residual = (stability[selected] - thresholds[selected]).detach().cpu().double().numpy()
    bounds = np.asarray([cfg['qp']['state_slack']['bounds'][key]
                         for key in ('yaw', 'sideslip', 'roll', 'ltr')], dtype=np.float64)
    bounds /= np.asarray(cfg['residual_scales']['stability'])[:, None]
    channels = selected.detach().cpu().numpy() % 4
    slack_lower = np.where(residual > 0, bounds[channels, 0], 0.)
    slack_upper = np.where(residual > 0, bounds[channels, 1], 0.)
    diagonal = float(cfg['qp']['control_correction_weight']) * np.tile(weights, controls.numel() // 2)
    problem = _build_qp_problem(diagonal, lo + margin - flat, hi - margin - flat,
                               jacobian.detach().cpu().double().numpy(), residual, slack_lower, slack_upper, cfg)
    solution, info = _solve_osqp(problem, cfg)
    if solution is None:
        return None, info
    delta = solution[:controls.numel()]
    violation = np.maximum(lo - (flat + delta), flat + delta - hi)
    tolerance = max(float(torch.finfo(controls.dtype).eps) * 8., float(cfg['qp']['eps_abs']) * 10.)
    if np.maximum(violation, 0.).max() > tolerance:
        raise RuntimeError('QP returned controls outside the hard control range')
    slack = solution[controls.numel():]
    info.update(state_slack_sum=float(slack.sum()), state_slack_max=float(slack.max()) if len(slack) else 0.,
                constraint_rows=len(residual), increment_norm=float(np.linalg.norm(delta)))
    return torch.as_tensor(delta.reshape(controls.shape), device=controls.device, dtype=controls.dtype), info
