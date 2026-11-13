"""Sparse joint QP assembly for one SCP linearization of JTP."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class LinearizedQP:
    P: sparse.csc_matrix
    q: np.ndarray
    A: sparse.csc_matrix
    lower: np.ndarray
    upper: np.ndarray
    n_tau: int
    n_control: int
    diagnostics: dict


def _as_np(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _block_diag(items):
    return sparse.block_diag(items, format="csc") if items else sparse.csc_matrix((0, 0))



def _rate_matrix(horizon: int) -> sparse.csc_matrix:
    """Map flattened [H,2] controls to first difference at every node."""
    rows, cols, data = [], [], []
    for k in range(horizon):
        for channel in range(2):
            row = 2 * k + channel
            current = 2 * k + channel
            rows.append(row); cols.append(current); data.append(1.0)
            if k > 0:
                rows.append(row); cols.append(2 * (k - 1) + channel); data.append(-1.0)
    return sparse.coo_matrix((data, (rows, cols)), shape=(2 * horizon, 2 * horizon)).tocsc()


def build_joint_qp(
    *,
    controls,
    controls_nominal,
    tau_current,
    tau_nominal,
    jac_tau_u,
    g_stability,
    jac_stability_u,
    g_geometry,
    jac_geometry_tau,
    g_collision,
    jac_collision_tau,
    previous_controls,
    trust_radius: float,
    cfg: dict,
) -> LinearizedQP:
    """Build the paper's joint QP with decision variables [Delta_tau, Delta_U]."""
    u = _as_np(controls)
    u0 = _as_np(controls_nominal)
    tau = _as_np(tau_current)
    tau0 = _as_np(tau_nominal)
    prev = _as_np(previous_controls)
    if u.ndim != 3 or u.shape[-1] != 2:
        raise ValueError("controls must be [N,K,2]")
    n, k, _ = u.shape
    if tau.ndim != 3 or tau.shape[0] != n or tau.shape[-1] != 3:
        raise ValueError("tau_current must be [N,H+1,3]")
    if u0.shape != u.shape or tau0.shape != tau.shape or prev.shape != (n, 2):
        raise ValueError("Nominal controls/trajectory or previous control shape mismatch")
    t = tau.shape[1]
    d_tau_vehicle = t * 3
    d_u_vehicle = k * 2
    n_tau = n * d_tau_vehicle
    n_u = n * d_u_vehicle

    jt_blocks = [sparse.csc_matrix(_as_np(j)) for j in jac_tau_u]
    jg_blocks = [sparse.csc_matrix(_as_np(j)) for j in jac_stability_u]
    jgeo_blocks = [sparse.csc_matrix(_as_np(j)) for j in jac_geometry_tau]
    if len(jt_blocks) != n or len(jg_blocks) != n or len(jgeo_blocks) != n:
        raise ValueError("Per-vehicle Jacobian list length must equal N")
    j_tau_u = _block_diag(jt_blocks)
    j_stab_u = _block_diag(jg_blocks)
    j_geo_tau = _block_diag(jgeo_blocks)
    j_col_tau = sparse.csc_matrix(_as_np(jac_collision_tau))

    opt = cfg["optimization"]
    w_tau = np.tile(np.asarray(opt["trajectory_weights"], dtype=np.float64), n * t)
    w_u = np.tile(np.asarray(opt["control_weights"], dtype=np.float64), n * k)
    lambda_u = float(opt["lambda_u"])
    reg = float(cfg["qp"].get("diagonal_regularization", 1.0e-8))
    p_diag = np.concatenate((w_tau, lambda_u * w_u)) + reg
    P = sparse.diags(p_diag, format="csc")
    e_tau = (tau - tau0).reshape(-1)
    e_u = (u - u0).reshape(-1)
    q = np.concatenate((w_tau * e_tau, lambda_u * w_u * e_u))

    blocks, lower, upper = [], [], []

    # Linearized dynamics consistency: Delta_tau - J_tau_U Delta_U = 0.
    A_eq = sparse.hstack((sparse.eye(n_tau, format="csc"), -j_tau_u), format="csc")
    blocks.append(A_eq); lower.append(np.zeros(n_tau)); upper.append(np.zeros(n_tau))

    # Full-horizon stability inequalities.
    g_stab = np.concatenate([_as_np(g).reshape(-1) for g in g_stability])
    if g_stab.size:
        A_stab = sparse.hstack((sparse.csc_matrix((g_stab.size, n_tau)), j_stab_u), format="csc")
        blocks.append(A_stab); lower.append(np.full(g_stab.size, -np.inf)); upper.append(-g_stab)

    # Road-boundary and curvature inequalities.
    g_geo = np.concatenate([_as_np(g).reshape(-1) for g in g_geometry])
    if g_geo.size:
        A_geo = sparse.hstack((j_geo_tau, sparse.csc_matrix((g_geo.size, n_u))), format="csc")
        blocks.append(A_geo); lower.append(np.full(g_geo.size, -np.inf)); upper.append(-g_geo)

    # Joint inter-truck and external-traffic collision inequalities.
    g_col = _as_np(g_collision).reshape(-1)
    if g_col.size:
        if j_col_tau.shape != (g_col.size, n_tau):
            raise ValueError(f"Collision Jacobian shape {j_col_tau.shape} != {(g_col.size, n_tau)}")
        A_col = sparse.hstack((j_col_tau, sparse.csc_matrix((g_col.size, n_u))), format="csc")
        blocks.append(A_col); lower.append(np.full(g_col.size, -np.inf)); upper.append(-g_col)

    # Physical acceleration and steering bounds on the control nodes.
    control_cfg = cfg["control"]
    lower_u = np.tile([float(control_cfg["acceleration_min_mps2"]),
                       float(control_cfg["steering_min_rad"])], n * k)
    upper_u = np.tile([float(control_cfg["acceleration_max_mps2"]),
                       float(control_cfg["steering_max_rad"])], n * k)
    A_box = sparse.hstack((sparse.csc_matrix((n_u, n_tau)), sparse.eye(n_u, format="csc")), format="csc")
    blocks.append(A_box)
    lower.append(lower_u - u.reshape(-1)); upper.append(upper_u - u.reshape(-1))

    # Rate constraints include the last measured/applied control at k=-1.
    d_one = _rate_matrix(k)
    d_all = _block_diag([d_one] * n)
    rate_base = u.copy()
    rate_base[:, 0, :] -= prev
    if k > 1:
        rate_base[:, 1:, :] = u[:, 1:, :] - u[:, :-1, :]
    rate_limit = np.tile([float(control_cfg["acceleration_rate_max_mps3"]) *
                          float(cfg["execution"]["dt_s"]),
                          float(control_cfg["steering_rate_max_radps"]) *
                          float(cfg["execution"]["dt_s"])], n * k)
    A_rate = sparse.hstack((sparse.csc_matrix((n_u, n_tau)), d_all), format="csc")
    blocks.append(A_rate)
    lower.append(-rate_limit - rate_base.reshape(-1))
    upper.append(rate_limit - rate_base.reshape(-1))

    # Infinity-norm trust region in normalized physical-control coordinates.
    trust_scale = np.tile(np.asarray(control_cfg["trust_scale"], dtype=np.float64), n * k)
    trust = float(trust_radius) * trust_scale
    A_trust = A_box
    blocks.append(A_trust); lower.append(-trust); upper.append(trust)

    A = sparse.vstack(blocks, format="csc")
    l = np.concatenate(lower)
    ub = np.concatenate(upper)
    return LinearizedQP(
        P=P, q=q, A=A, lower=l, upper=ub,
        n_tau=n_tau, n_control=n_u,
        diagnostics={
            "n_vehicles": n,
            "trajectory_variables": n_tau,
            "control_variables": n_u,
            "qp_variables": n_tau + n_u,
            "equality_rows": n_tau,
            "stability_rows": int(g_stab.size),
            "geometry_rows": int(g_geo.size),
            "collision_rows": int(g_col.size),
            "control_box_rows": n_u,
            "control_rate_rows": n_u,
            "trust_region_rows": n_u,
            "total_constraint_rows": int(A.shape[0]),
        },
    )
