"""Priority-grouped LS/QP dispatch and shared nonlinear acceptance."""
from __future__ import annotations

import torch


def build_priority_groups(groups, cfg=None, thresholds=None):
    """Return the fixed JTP priority order: stability, geometry, control rate."""
    order = [["stability"], ["road", "curvature"], ["control_rate"]]
    scores = {"G1_stability": 1.0, "G2_road_curvature": 2.0, "G3_control_rate": 3.0}
    return order, scores


def group_maxima(metrics, priority_groups):
    return [max((metrics["max_violation_by_group"][name] for name in group), default=0.0)
            for group in priority_groups]


def accept_nonlinear(before_groups, after_groups, thresholds, priority_groups, cfg):
    """Evaluate full-resolution nonlinear acceptance for every state constraint row.

    A candidate is state-safe only when every row's tolerance-excess is
    non-increasing and each priority group has a non-increasing violation count.
    At least one row must improve materially, or a group must lose violations.
    """
    tolerance = float(cfg["nondegradation_tolerance"])
    improvement = float(cfg["minimum_violation_improvement"])
    group_details = {}
    all_non_degraded = True
    all_counts_nonincreasing = True
    state_progress = False

    for group_index, families in enumerate(priority_groups):
        before_parts = []
        after_parts = []
        for name in families:
            before_excess = (before_groups[name] - thresholds[name]).clamp_min(0.0)
            after_excess = (after_groups[name] - thresholds[name]).clamp_min(0.0)
            if before_excess.shape != after_excess.shape:
                raise ValueError(
                    f"Constraint row shape changed during line search for {name}: "
                    f"{tuple(before_excess.shape)} vs {tuple(after_excess.shape)}"
                )
            before_parts.append(before_excess.reshape(-1))
            after_parts.append(after_excess.reshape(-1))

        before = torch.cat(before_parts) if before_parts else torch.empty(0)
        after = torch.cat(after_parts) if after_parts else torch.empty(0)
        if not torch.isfinite(before).all() or not torch.isfinite(after).all():
            group_details[f"group_{group_index + 1}"] = {
                "families": list(families),
                "finite": False,
                "nondegradation_ok": False,
                "violation_count_nonincrease_ok": False,
                "violation_count_before": None,
                "violation_count_after": None,
                "nondegradation_failure_count": None,
                "improved_constraint_count": 0,
            }
            all_non_degraded = False
            all_counts_nonincreasing = False
            continue

        regressions = after > before + tolerance
        count_before = int((before > 0.0).sum().detach().cpu())
        count_after = int((after > 0.0).sum().detach().cpu())
        nondegradation_ok = not bool(regressions.any())
        count_nonincrease_ok = count_after <= count_before
        improved = before - after > improvement
        improved_count = int(improved.sum().detach().cpu())
        count_reduction = count_after < count_before
        state_progress = state_progress or improved_count > 0 or count_reduction

        group_details[f"group_{group_index + 1}"] = {
            "families": list(families),
            "finite": True,
            "constraint_count": int(before.numel()),
            "nondegradation_ok": nondegradation_ok,
            "violation_count_nonincrease_ok": count_nonincrease_ok,
            "violation_count_before": count_before,
            "violation_count_after": count_after,
            "violation_count_delta": count_after - count_before,
            "nondegradation_failure_count": int(regressions.sum().detach().cpu()),
            "improved_constraint_count": improved_count,
            "max_excess_before": float(before.max().detach().cpu()) if before.numel() else 0.0,
            "max_excess_after": float(after.max().detach().cpu()) if after.numel() else 0.0,
            "l2_excess_before": float(torch.linalg.vector_norm(before).detach().cpu()),
            "l2_excess_after": float(torch.linalg.vector_norm(after).detach().cpu()),
        }
        all_non_degraded = all_non_degraded and nondegradation_ok
        all_counts_nonincreasing = all_counts_nonincreasing and count_nonincrease_ok

    return {
        "nondegradation_ok": all_non_degraded,
        "violation_count_nonincrease_ok": all_counts_nonincreasing,
        "state_progress": state_progress,
        "accepted": all_non_degraded and all_counts_nonincreasing and state_progress,
        "groups": group_details,
    }


def _weighted_system(jac, residual, cfg):
    rhs = residual.clamp_min(0.0)
    weights = (1.0 + float(cfg["violation_weight_beta"]) * rhs /
               rhs.max().clamp_min(1.0e-8)).sqrt()
    return weights[:, None] * jac, weights * rhs


def _damping_value(weighted_jac, metric_chol, cfg, damping_multiplier):
    inv_jt = torch.cholesky_solve(weighted_jac.T, metric_chol)
    gram = weighted_jac @ inv_jt
    gram = 0.5 * (gram + gram.T)
    scale = float(gram.diagonal().mean().abs())
    damping = max(float(cfg["damping"]) * max(scale, 1.0e-6) * damping_multiplier, 1.0e-8)
    return inv_jt, gram, damping


def _box_qp_cholesky(hessian, gradient, lower, upper):
    """Solve a strictly convex box QP with a primal active-set Cholesky method."""
    dtype = gradient.dtype
    tolerance = max(float(torch.finfo(dtype).eps) * 20.0, 1.0e-12)
    fixed = (upper - lower).abs() <= tolerance
    x = torch.maximum(torch.minimum(torch.zeros_like(gradient), upper), lower)
    x = torch.where(fixed, lower, x)
    at_lower = (x <= lower + tolerance) & ~fixed
    at_upper = (x >= upper - tolerance) & ~fixed
    n = gradient.numel()

    for iteration in range(4 * n + 20):
        grad = hessian @ x + gradient
        free = ~(at_lower | at_upper | fixed)
        direction = torch.zeros_like(x)
        if bool(free.any()):
            h_ff = hessian[free][:, free]
            chol, info = torch.linalg.cholesky_ex(h_ff)
            if int(info) != 0:
                raise RuntimeError("Bounded LS reduced Hessian is not positive definite")
            direction[free] = torch.cholesky_solve((-grad[free])[:, None], chol).squeeze(-1)

        if float(direction.abs().max()) <= tolerance:
            lower_release = at_lower & (grad < -tolerance)
            upper_release = at_upper & (grad > tolerance)
            release_score = torch.where(lower_release, -grad,
                                        torch.where(upper_release, grad, torch.zeros_like(grad)))
            if float(release_score.max()) <= tolerance:
                return x, {"active_lower": int(at_lower.sum()),
                           "active_upper": int(at_upper.sum()),
                           "active_set_iterations": iteration + 1}
            release = int(release_score.argmax())
            at_lower[release] = False
            at_upper[release] = False
            continue

        alpha = 1.0
        hit_lower = torch.zeros_like(at_lower)
        hit_upper = torch.zeros_like(at_upper)
        positive = free & (direction > tolerance)
        negative = free & (direction < -tolerance)
        if bool(positive.any()):
            ratios = (upper[positive] - x[positive]) / direction[positive]
            alpha_pos = float(ratios.min())
            if alpha_pos < alpha:
                alpha = max(0.0, alpha_pos)
        if bool(negative.any()):
            ratios = (lower[negative] - x[negative]) / direction[negative]
            alpha_neg = float(ratios.min())
            if alpha_neg < alpha:
                alpha = max(0.0, alpha_neg)

        x = x + alpha * direction
        if alpha < 1.0 - tolerance:
            hit_lower = free & ((x - lower).abs() <= tolerance * 8.0)
            hit_upper = free & ((x - upper).abs() <= tolerance * 8.0)
            x = torch.where(hit_lower, lower, torch.where(hit_upper, upper, x))
            at_lower |= hit_lower
            at_upper |= hit_upper
        else:
            at_lower |= free & ((x - lower).abs() <= tolerance * 8.0)
            at_upper |= free & ((x - upper).abs() <= tolerance * 8.0)
            at_lower &= ~at_upper

    raise RuntimeError("Bounded LS active-set solver did not converge")


def bounded_damped_ls(jac, residual, metric, metric_chol, lower, upper, cfg,
                      damping_multiplier=1.0, trust_lower=None, trust_upper=None):
    """Damped LS with hard control bounds and damping-first trust adaptation."""
    weighted_jac, weighted_rhs = _weighted_system(jac, residual, cfg)
    combined_lower = hard_lower = lower
    combined_upper = hard_upper = upper
    if trust_lower is not None:
        combined_lower = torch.maximum(hard_lower, trust_lower)
        combined_upper = torch.minimum(hard_upper, trust_upper)
    current_multiplier = float(damping_multiplier)
    final_damping = None
    damping_attempts = 0
    trust_retries = 0
    unconstrained = None

    for attempt in range(int(cfg["damping_retries"])):
        damping_attempts = attempt + 1
        inv_jt, gram, damping = _damping_value(
            weighted_jac, metric_chol, cfg, current_multiplier)
        eye = torch.eye(len(gram), device=gram.device, dtype=gram.dtype)
        chol, info = torch.linalg.cholesky_ex(gram + damping * eye)
        if int(info) != 0:
            current_multiplier *= 10.0
            continue
        solved = torch.cholesky_solve(weighted_rhs[:, None], chol).squeeze(-1)
        unconstrained = -(inv_jt @ solved)
        final_damping = damping

        hard_feasible = bool((unconstrained >= hard_lower).all() and
                             (unconstrained <= hard_upper).all())
        trust_feasible = (trust_lower is None or
                          bool((unconstrained >= trust_lower).all() and
                               (unconstrained <= trust_upper).all()))
        if hard_feasible and trust_feasible:
            return unconstrained, {"damping": damping, "backend": "cholesky",
                                   "active_lower": 0, "active_upper": 0,
                                   "active_set_iterations": 0,
                                   "damping_trust_retries": trust_retries}
        if not hard_feasible or trust_lower is None:
            break
        if trust_retries >= int(cfg["trust_damping_retries"]):
            break
        trust_retries += 1
        current_multiplier *= 10.0

    if final_damping is None or unconstrained is None:
        raise RuntimeError("Damped Gram Cholesky failed after damping retries")

    hessian = weighted_jac.T @ weighted_jac + final_damping * metric
    hessian = 0.5 * (hessian + hessian.T)
    gradient = weighted_jac.T @ weighted_rhs
    solution, active = _box_qp_cholesky(hessian, gradient, combined_lower, combined_upper)
    return solution, {"damping": final_damping,
                      "backend": "cholesky_box_active_set",
                      "damping_trust_retries": trust_retries,
                      "damping_factorization_attempts": damping_attempts, **active}


def _project_halfspaces_with_box(direction, history, metric_chol, cfg, lower, upper):
    protection_tolerance = float(cfg["protection_tolerance"])
    bound_tolerance = 0.0
    history_inverse = (torch.cholesky_solve(history.T, metric_chol)
                       if history.numel() else None)
    history_denominator = ((history * history_inverse.T).sum(-1).clamp_min(1.0e-20)
                           if history.numel() else None)

    def violations(value):
        history_excess = (history @ value - protection_tolerance
                          if history.numel() else value.new_empty((0,)))
        lower_excess = lower - value if lower is not None else value.new_empty((0,))
        upper_excess = value - upper if upper is not None else value.new_empty((0,))
        peak = max(float(history_excess.clamp_min(0.0).max()) if history_excess.numel() else 0.0,
                   float(lower_excess.clamp_min(0.0).max()) if lower_excess.numel() else 0.0,
                   float(upper_excess.clamp_min(0.0).max()) if upper_excess.numel() else 0.0)
        return history_excess, lower_excess, upper_excess, peak

    _, _, _, initial_peak = violations(direction)
    if initial_peak <= 0.0:
        return direction, 0, 0.0, True

    max_sweeps = int(cfg["protection_max_sweeps"])
    for sweep in range(max_sweeps + 1):
        history_excess, lower_excess, upper_excess, peak = violations(direction)
        if peak <= 0.0:
            return direction, sweep, peak, True
        if sweep == max_sweeps:
            break

        if history.numel():
            for index in torch.nonzero(history_excess > 0.0, as_tuple=False).flatten().tolist():
                violation = (history[index] @ direction - protection_tolerance).clamp_min(0.0)
                direction = direction - violation / history_denominator[index] * history_inverse[:, index]

        if lower is not None:
            for index in torch.nonzero((lower - direction) > bound_tolerance,
                                       as_tuple=False).flatten().tolist():
                row = torch.zeros_like(direction)
                row[index] = -1.0
                inverse_row = torch.cholesky_solve(row[:, None], metric_chol).squeeze(-1)
                denominator = inverse_row[index].abs().clamp_min(1.0e-20)
                violation = (lower[index] - direction[index]).clamp_min(0.0)
                direction = direction - violation / denominator * inverse_row
            for index in torch.nonzero((direction - upper) > bound_tolerance,
                                       as_tuple=False).flatten().tolist():
                row = torch.zeros_like(direction)
                row[index] = 1.0
                inverse_row = torch.cholesky_solve(row[:, None], metric_chol).squeeze(-1)
                denominator = inverse_row[index].abs().clamp_min(1.0e-20)
                violation = (direction[index] - upper[index]).clamp_min(0.0)
                direction = direction - violation / denominator * inverse_row

    return torch.zeros_like(direction), max_sweeps, peak, False

def protect_direction(d, history, chol, cfg, lower=None, upper=None):
    """Metric half-space protection with the current control/trust box retained."""
    return _project_halfspaces_with_box(d, history, chol, cfg, lower, upper)


def solve_groups(controls, groups, selected, rows, thresholds, metric, order, cfg,
                 scale, trust, damping_multiplier, control_lower, control_upper,
                 control_weights=None):
    if cfg["solver_method"] == "constrained_qp":
        from .qp_solver import solve_groups_qp
        return solve_groups_qp(
            controls, groups, selected, rows, thresholds, order, cfg,
            control_lower, control_upper, control_weights)
    return _solve_groups_ls(
        controls, groups, selected, rows, thresholds, metric, order, cfg,
        scale, trust, damping_multiplier, control_lower, control_upper)


def _solve_groups_ls(controls, groups, selected, rows, thresholds, metric, order, cfg,
                     scale, trust, damping_multiplier, control_lower, control_upper):
    if not torch.isfinite(metric).all():
        return None, {"reason": "nonfinite_metric"}
    metric_chol, info = torch.linalg.cholesky_ex(metric)
    if int(info) != 0:
        return None, {"reason": "invalid_metric"}

    flat_controls = controls.flatten()
    control_lower = control_lower.reshape(-1).repeat(controls.numel() // 2)
    control_upper = control_upper.reshape(-1).repeat(controls.numel() // 2)
    scale_flat = scale.repeat(controls.numel() // 2)
    bound_margin = float(torch.finfo(flat_controls.dtype).eps) * 4.0 * max(
        1.0, float(control_lower.abs().max()), float(control_upper.abs().max()))
    hard_lower = control_lower + bound_margin - flat_controls
    hard_upper = control_upper - bound_margin - flat_controls
    required_step = torch.maximum(hard_lower.clamp_min(0.0), (-hard_upper).clamp_min(0.0))
    required_radius = float((required_step / scale_flat).max())
    effective_trust = max(min(float(trust), float(cfg["max_step_norm"])), required_radius)
    trust_bound = scale_flat * effective_trust
    initial_lower = torch.maximum(hard_lower, -trust_bound)
    initial_upper = torch.minimum(hard_upper, trust_bound)
    if bool((initial_lower <= 0.0).all() and (initial_upper >= 0.0).all()):
        p = torch.zeros_like(flat_controls)
        hard_projection = {"active_lower": 0, "active_upper": 0, "active_set_iterations": 0}
    else:
        p, hard_projection = _box_qp_cholesky(
            metric, torch.zeros_like(flat_controls), initial_lower, initial_upper)
    hard_projection_norm = float(torch.linalg.vector_norm(p))

    history = p.new_empty((0, p.numel()))
    details = []
    for names in order:
        use = [name for name in names if name in rows and selected[name].numel()]
        if not use:
            continue
        jac = torch.cat([rows[name] for name in use], dim=0)
        residual_parts = [groups[name][selected[name]] - thresholds[name][selected[name]]
                          for name in use]
        residual = torch.cat(residual_parts).detach() + jac @ p
        active = residual > 0.0
        if not bool(active.any()):
            history = torch.cat((history, jac), dim=0)
            continue

        hard_lower_step = hard_lower - p
        hard_upper_step = hard_upper - p
        trust_lower_step = -trust_bound - p
        trust_upper_step = trust_bound - p
        lower_step = torch.maximum(hard_lower_step, trust_lower_step)
        upper_step = torch.minimum(hard_upper_step, trust_upper_step)
        delta, solve_info = bounded_damped_ls(
            jac[active], residual[active], metric, metric_chol,
            hard_lower_step, hard_upper_step, cfg, damping_multiplier,
            trust_lower=trust_lower_step, trust_upper=trust_upper_step)
        if not torch.isfinite(delta).all():
            return None, {"reason": "nonfinite_group_step"}
        delta, sweeps, peak, protected = protect_direction(
            delta, history, metric_chol, cfg, lower_step, upper_step)

        before = float(residual.clamp_min(0.0).max())
        after = float((residual + jac @ delta).clamp_min(0.0).max())
        if after > before + float(cfg["nondegradation_tolerance"]):
            delta = torch.zeros_like(delta)
            after = before
            protected = False
        protection_peak = (float((history @ delta).clamp_min(0.0).max())
                           if history.numel() else 0.0)
        details.append({
            "families": names,
            "active_rows": int(active.sum()),
            "backend": solve_info["backend"],
            "damping": solve_info["damping"],
            "damping_trust_retries": solve_info["damping_trust_retries"],
            "bounded_active_lower": solve_info["active_lower"],
            "bounded_active_upper": solve_info["active_upper"],
            "bounded_active_set_iterations": solve_info["active_set_iterations"],
            "protection_sweeps": sweeps,
            "protection_trial_peak": peak,
            "protection_peak": protection_peak,
            "protection_ok": protected,
            "linear_max_before": before,
            "linear_max_after": after,
        })
        p = p + delta
        history = torch.cat((history, jac), dim=0)

    final_controls = flat_controls + p
    control_violation = torch.maximum(control_lower - final_controls,
                                      final_controls - control_upper).clamp_min(0.0)
    if float(control_violation.max()) > float(torch.finfo(final_controls.dtype).eps) * 8.0:
        raise RuntimeError("Grouped LS returned controls outside the hard control range")
    trust_violation = (p.abs() - trust_bound).clamp_min(0.0)
    if float(trust_violation.max()) > float(torch.finfo(p.dtype).eps) * 8.0:
        raise RuntimeError("Grouped LS exceeded its cumulative trust box")

    return p.reshape_as(controls), {
        "groups": details,
        "jacobian_shape": [sum(len(j) for j in rows.values()), p.numel()],
        "metric_factorizations": 1,
        "solver_method": cfg["solver_method"],
        "hard_box_restoration_norm": hard_projection_norm,
        "hard_box_active_lower": hard_projection["active_lower"],
        "hard_box_active_upper": hard_projection["active_upper"],
        "hard_box_active_set_iterations": hard_projection["active_set_iterations"],
        "effective_trust_radius": effective_trust,
    }
