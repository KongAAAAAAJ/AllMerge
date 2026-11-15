"""Joint Jacobian projection for TRACE.

The solver applies an active-set, violation-weighted minimum-norm
projection to TRACE control variables and the frozen Koopman execution map.
It does not build or solve a QP. Only selected violated constraints are
linearized at each iteration; the nonlinear Koopman rollout is re-evaluated
for every candidate update.
"""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any

import torch

from highway_env.planner.diffusion.common.execution import K2_CONTEXT, VehicleExecution

from .config import load_jtp_config, validate_jtp_config
from .constraints import stability_residual
from .execution_adapter import KoopmanExecutionAdapter
from .geometry import geometry_residual, nearest_road_association, footprint_circle_centers
from .types import JTPResult, RoadGeometry


@dataclass
class _Evaluation:
    rollout: dict
    groups: dict[str, torch.Tensor]
    associations: list[tuple[torch.Tensor, torch.Tensor]]
    metrics: dict


class JointTrajectoryProjector:
    """Adaptive, violation-weighted joint projection in control space.

    External-actor collision avoidance is intentionally excluded. It is handled
    upstream by Adaptive Anchor scoring and selection.
    """

    def __init__(self, execution_model, config=None, *, control_interpolation_matrix=None):
        self.cfg = (load_jtp_config(config)
                    if isinstance(config, (str, bytes, PathLike)) or config is None
                    else config)
        validate_jtp_config(self.cfg)
        self.execution_model = execution_model
        if isinstance(execution_model, VehicleExecution):
            self.rollout_fn = KoopmanExecutionAdapter(execution_model)
        elif isinstance(execution_model, KoopmanExecutionAdapter):
            self.rollout_fn = execution_model
        elif callable(execution_model):
            self.rollout_fn = execution_model
        else:
            raise TypeError("execution_model must be VehicleExecution, KoopmanExecutionAdapter, or callable")

        self.horizon = int(self.cfg["execution"]["horizon_steps"])
        self.control_steps = int(self.cfg["execution"]["control_steps"])
        self.dt = float(self.cfg["execution"]["dt_s"])
        if control_interpolation_matrix is None:
            if self.horizon != self.control_steps:
                raise ValueError("K != H requires an explicit fixed control_interpolation_matrix [H,K]")
            self.control_interpolation = torch.eye(self.horizon)
        else:
            matrix = torch.as_tensor(control_interpolation_matrix, dtype=torch.float32)
            if matrix.shape != (self.horizon, self.control_steps) or not torch.isfinite(matrix).all():
                raise ValueError(f"control interpolation must be finite [{self.horizon},{self.control_steps}]")
            self.control_interpolation = matrix

    @classmethod
    def from_koopman_checkpoint(cls, checkpoint, config=None, *, device="cpu"):
        cfg = load_jtp_config(config) if isinstance(config, (str, bytes, PathLike)) or config is None else config
        execution = VehicleExecution(
            {"model": "koopman_k2", "checkpoint": str(checkpoint)},
            dt=float(cfg["execution"]["dt_s"]),
            wheelbase_m=float(cfg["execution"]["wheelbase_m"]),
        ).to(device)
        execution.requires_grad_(False)
        execution.eval()
        return cls(execution, cfg)

    @staticmethod
    def _to_context(context: dict, *, n: int, device, dtype) -> dict:
        if not isinstance(context, dict):
            raise TypeError("K2 context must be a dictionary")
        missing = [key for key in K2_CONTEXT if key not in context]
        if missing:
            raise KeyError(f"JTP requires synchronized K2 context fields: {missing}")
        result = {}
        for key in K2_CONTEXT:
            tensor = torch.as_tensor(context[key], device=device, dtype=dtype)
            if tensor.ndim == 0 or tensor.shape[0] != n:
                raise ValueError(f"Context field {key} must have leading dimension N={n}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Context field {key} contains NaN/Inf")
            result[key] = tensor
        if result["koopman_state_hist_si"].shape != (n, 5, 5):
            raise ValueError("koopman_state_hist_si must be [N,5,5]")
        if result["koopman_control_hist_si"].shape != (n, 4, 2):
            raise ValueError("koopman_control_hist_si must be [N,4,2]")
        if result["koopman_pose0_si"].shape != (n, 3):
            raise ValueError("koopman_pose0_si must be [N,3]")
        if result["koopman_payload_kg"].reshape(n).lt(0).any():
            raise ValueError("Payload must be nonnegative")
        mu = result["koopman_road_mu"].reshape(n)
        if (mu <= 0).any() or (mu > 1).any():
            raise ValueError("Road friction mu must be in (0,1]")
        return result

    def _ensure_execution_device(self, device, dtype):
        module = self.execution_model
        if isinstance(module, KoopmanExecutionAdapter):
            module = module.vehicle_execution
        if isinstance(module, torch.nn.Module):
            module.to(device=device, dtype=dtype)
            module.eval()
            module.requires_grad_(False)

    def _execution_controls(self, controls):
        matrix = self.control_interpolation.to(device=controls.device, dtype=controls.dtype)
        return torch.einsum("hk,nkc->nhc", matrix, controls)

    def _rollout(self, controls, context):
        output = self.rollout_fn(self._execution_controls(controls), context)
        required = ("trajectory", "states", "ltr")
        missing = [key for key in required if key not in output]
        if missing:
            raise KeyError(f"JTP execution function missing outputs: {missing}")
        n = controls.shape[0]
        trajectory, states, ltr = output["trajectory"], output["states"], output["ltr"]
        expected = ((n, self.horizon + 1, 3), (n, self.horizon + 1, 5), (n, self.horizon + 1))
        for name, value, shape in zip(required, (trajectory, states, ltr), expected):
            if tuple(value.shape) != shape:
                raise ValueError(f"Koopman {name} must be {shape}, got {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Koopman {name} contains NaN/Inf")
        return {"trajectory": trajectory, "states": states, "ltr": ltr}

    @staticmethod
    def _road_list(road_geometry, n, *, device, dtype):
        roads = ([road_geometry] * n if isinstance(road_geometry, RoadGeometry)
                 else list(road_geometry) if isinstance(road_geometry, (list, tuple)) and len(road_geometry) == n
                 else None)
        if roads is None or any(not isinstance(road, RoadGeometry) for road in roads):
            raise ValueError("road_geometry must be RoadGeometry or a list with one entry per vehicle")
        for road in roads:
            road.tensors(device=device, dtype=dtype)
        return roads

    def _associations(self, rollout, roads, controls):
        offsets = torch.as_tensor(self.cfg["geometry"]["footprint"]["circle_offsets_m"],
                                  device=controls.device, dtype=controls.dtype)
        result = []
        for i, road in enumerate(roads):
            centers = footprint_circle_centers(rollout["trajectory"][i, 1:].detach(), offsets)
            result.append(nearest_road_association(centers, road))
        return result

    def _constraint_groups(self, controls, context, roads, associations=None):
        rollout = self._rollout(controls, context)
        if associations is None:
            associations = self._associations(rollout, roads, controls)
        n, k, _ = controls.shape
        stab_scale = controls.new_tensor(self.cfg["projection"]["residual_scales"]["stability"])
        road_scale = float(self.cfg["projection"]["residual_scales"]["road_m"])
        curve_scale = float(self.cfg["projection"]["residual_scales"]["curvature_per_m2"])
        stability, road_values, curve_values = [], [], []
        geometry_cfg = self.cfg["geometry"]
        footprint = geometry_cfg["footprint"]
        for i in range(n):
            raw_stability = stability_residual(
                rollout["states"][i], rollout["ltr"][i],
                context["koopman_payload_kg"][i], context["koopman_road_mu"][i], self.cfg)
            stability.append(raw_stability.reshape(2, self.horizon, 4) /
                             stab_scale.reshape(1, 1, 4))
            raw_geometry, meta = geometry_residual(
                rollout["trajectory"][i], roads[i], self.cfg, association=associations[i])
            road_values.append(raw_geometry[:meta["road_count"]] / road_scale)
            curve_values.append(raw_geometry[meta["road_count"]:] / curve_scale)

        control_cfg = self.cfg["control"]
        lower = controls.new_tensor([control_cfg["acceleration_min_mps2"], control_cfg["steering_min_rad"]])
        upper = controls.new_tensor([control_cfg["acceleration_max_mps2"], control_cfg["steering_max_rad"]])
        control_scale = controls.new_tensor(self.cfg["control"]["trust_scale"]).clamp_min(1.0e-6)
        control_box = torch.cat(((lower - controls) / control_scale,
                                 (controls - upper) / control_scale), dim=0)
        groups = {
            "stability": torch.cat([x.reshape(-1) for x in stability]),
            "road": torch.cat([x.reshape(-1) for x in road_values]),
            "curvature": torch.cat([x.reshape(-1) for x in curve_values]),
            "control": control_box.reshape(-1),
        }
        return rollout, groups, associations

    def _metrics(self, groups):
        tol = float(self.cfg["projection"]["constraint_tolerance"])
        maxima = {key: (float(torch.relu(value).max().detach().cpu()) if value.numel() else 0.0)
                  for key, value in groups.items()}
        counts = {key: int((value > tol).sum().detach().cpu()) for key, value in groups.items()}
        return {
            "max_violation": max(maxima.values(), default=0.0),
            "max_violation_by_group": maxima,
            "violation_count_by_group": counts,
            "stability_violation_count": counts["stability"],
            "geometry_violation_count": counts["road"] + counts["curvature"],
            "control_box_violation_count": counts["control"],
            # Kept as zero-valued compatibility fields. Collision is handled upstream;
            # control-rate constraints are deliberately outside the current JTP profile.
            "collision_violation_count": 0,
            "control_rate_violation_count": 0,
            "feasible": all(value <= tol for value in maxima.values()),
        }

    def _evaluate(self, controls, context, roads, associations=None):
        rollout, groups, associations = self._constraint_groups(controls, context, roads, associations)
        return _Evaluation(rollout, groups, associations, self._metrics(groups))

    def _objective(self, controls, rollout, controls_nominal, trajectory_nominal):
        opt = self.cfg["optimization"]
        wt = rollout["trajectory"].new_tensor(opt["trajectory_weights"])
        wu = controls.new_tensor(opt["control_weights"])
        tau_error = rollout["trajectory"] - trajectory_nominal
        u_error = controls - controls_nominal
        return (0.5 * (tau_error.square() * wt).sum()
                + 0.5 * float(opt["lambda_u"]) * (u_error.square() * wu).sum())

    def _merit(self, controls, evaluation, controls_nominal, trajectory_nominal):
        objective = self._objective(controls, evaluation.rollout, controls_nominal, trajectory_nominal)
        violation = sum(torch.relu(value).square().sum() for value in evaluation.groups.values())
        penalty = float(self.cfg["projection"]["merit_penalty"]) * violation
        return float((objective + penalty).detach().cpu()), float(objective.detach().cpu()), float(violation.detach().cpu())

    def _active_indices(self, groups):
        cfg = self.cfg["projection"]
        selected = {}
        counts = {}
        for name, values in groups.items():
            ids = torch.nonzero(values.detach() > float(cfg["active_tolerance"]), as_tuple=False).reshape(-1)
            cap = int(cfg["max_active_per_group"][name])
            if ids.numel() > cap:
                local = torch.topk(values.detach()[ids], k=cap, largest=True, sorted=False).indices
                ids = ids[local]
            selected[name] = ids
            counts[name] = int(ids.numel())
        return selected, counts

    def _active_residual(self, flat_controls, shape, context, roads, associations, selected):
        controls = flat_controls.reshape(shape)
        _, groups, _ = self._constraint_groups(controls, context, roads, associations)
        return torch.cat([groups[name][selected[name]] for name in groups if selected[name].numel()])

    @staticmethod
    def _jacobian(function, point):
        try:
            return torch.autograd.functional.jacobian(
                function, point, create_graph=False, strict=False, vectorize=True)
        except (RuntimeError, NotImplementedError):
            return torch.autograd.functional.jacobian(
                function, point, create_graph=False, strict=False, vectorize=False)

    def _objective_metric(self, controls, context, trajectory_weights):
        """Local positive-definite metric approximating the TRACE projection cost."""
        flat = controls.detach().reshape(-1).requires_grad_(True)
        shape = controls.shape
        def trajectory_fn(z):
            return self._rollout(z.reshape(shape), context)["trajectory"].reshape(-1)
        jac_tau = self._jacobian(trajectory_fn, flat)
        weights = trajectory_weights.to(jac_tau).repeat(controls.shape[0] * (self.horizon + 1))
        metric = jac_tau.T @ (weights[:, None] * jac_tau)
        wu = controls.new_tensor(self.cfg["optimization"]["control_weights"]).repeat(
            controls.shape[0] * self.control_steps)
        metric = metric + float(self.cfg["optimization"]["lambda_u"]) * torch.diag(wu)
        scales = controls.new_tensor(self.cfg["control"]["trust_scale"]).repeat(
            controls.shape[0] * self.control_steps)
        reg = float(self.cfg["projection"]["metric_regularization"])
        metric = metric + reg * torch.diag(1.0 / scales.square().clamp_min(1.0e-8))
        return 0.5 * (metric + metric.T)

    def _joint_correction(self, controls, context, roads, evaluation, selected, damping_multiplier=1.0):
        flat = controls.detach().reshape(-1).requires_grad_(True)
        active_fn = lambda z: self._active_residual(
            z, controls.shape, context, roads, evaluation.associations, selected)
        residual = active_fn(flat)
        jac = self._jacobian(active_fn, flat)
        if jac.ndim != 2 or jac.shape[0] == 0:
            return None, {"reason": "empty_active_jacobian", "active_rows": 0}
        if not torch.isfinite(jac).all() or not torch.isfinite(residual).all():
            return None, {"reason": "nonfinite_active_jacobian", "active_rows": int(jac.shape[0])}

        proj_cfg = self.cfg["projection"]
        beta = float(proj_cfg["violation_weight_beta"])
        max_v = residual.detach().max().clamp_min(1.0e-8)
        row_weights = (1.0 + beta * residual.detach().clamp_min(0.0) / max_v).sqrt()
        weighted_jac = row_weights[:, None] * jac
        weighted_rhs = row_weights * residual.detach().clamp_min(0.0)

        wt = controls.new_tensor(self.cfg["optimization"]["trajectory_weights"])
        metric = self._objective_metric(controls, context, wt)
        metric_chol, info = torch.linalg.cholesky_ex(metric)
        if int(info.detach().cpu()) != 0:
            return None, {"reason": "objective_metric_not_positive_definite", "active_rows": int(jac.shape[0])}
        inv_jt = torch.cholesky_solve(weighted_jac.T, metric_chol)
        gram = weighted_jac @ inv_jt
        gram = 0.5 * (gram + gram.T)
        base = float(proj_cfg["damping"])
        scale = float(torch.diagonal(gram).mean().detach().abs().cpu()) if gram.numel() else 1.0
        damping = max(base * max(scale, 1.0e-6) * float(damping_multiplier), 1.0e-8)
        solved = None
        for _ in range(int(proj_cfg["damping_retries"])):
            matrix = gram + damping * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            chol, chol_info = torch.linalg.cholesky_ex(matrix)
            if int(chol_info.detach().cpu()) == 0:
                solved = torch.cholesky_solve(weighted_rhs[:, None], chol).squeeze(-1)
                break
            damping *= 10.0
        if solved is None:
            matrix = gram + damping * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            solved = torch.linalg.lstsq(matrix, weighted_rhs[:, None]).solution.squeeze(-1)
        delta = -(inv_jt @ solved)
        if not torch.isfinite(delta).all():
            return None, {"reason": "nonfinite_projection_step", "active_rows": int(jac.shape[0]),
                          "damping": damping}
        condition = float(torch.linalg.cond(gram + damping * torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype)).detach().cpu())
        return delta.reshape_as(controls), {
            "active_rows": int(jac.shape[0]),
            "active_counts": {name: int(ids.numel()) for name, ids in selected.items()},
            "jacobian_shape": list(jac.shape),
            "damping": damping,
            "gram_condition": condition,
            "max_active_violation": float(residual.detach().max().cpu()),
        }

    def project(self, nominal_controls: torch.Tensor, context: dict,
                road_geometry: RoadGeometry | list[RoadGeometry]) -> JTPResult:
        """Project nominal controls while preserving Koopman execution consistency.

        Collision avoidance is handled upstream by Adaptive Anchor.
        """
        if not torch.is_tensor(nominal_controls) or nominal_controls.ndim != 3 or nominal_controls.shape[-1] != 2:
            raise ValueError("nominal_controls must be a torch tensor [N,K,2]")
        if not nominal_controls.is_floating_point() or not torch.isfinite(nominal_controls).all():
            raise ValueError("nominal_controls must be finite floating-point SI controls")
        n, k, _ = nominal_controls.shape
        if k != self.control_steps:
            raise ValueError(f"Expected K={self.control_steps} control nodes, got {k}")
        self._ensure_execution_device(nominal_controls.device, nominal_controls.dtype)
        context = self._to_context(context, n=n, device=nominal_controls.device, dtype=nominal_controls.dtype)
        roads = self._road_list(road_geometry, n, device=nominal_controls.device, dtype=nominal_controls.dtype)

        nominal = nominal_controls.detach().clone()
        controls = nominal.clone()
        with torch.enable_grad():
            nominal_rollout = self._rollout(nominal, context)
            tau_nominal = nominal_rollout["trajectory"].detach().clone()
            current = self._evaluate(controls, context, roads)
        history = [{"iteration": 0, **current.metrics, "status": "nominal"}]
        if current.metrics["feasible"]:
            return JTPResult(controls, current.rollout["trajectory"].detach(),
                             current.rollout["states"].detach(), current.rollout["ltr"].detach(),
                             True, "nominal_already_feasible", 0,
                             {"history": history, "objective": 0.0, "projection": [],
                              "collision_handled_by": "adaptive_anchor",
                              "config_path": self.cfg.get("_config_path")})

        proj_cfg = self.cfg["projection"]
        trust = float(proj_cfg["initial_trust_radius"])
        damping_scale = 1.0
        accepted_iterations = 0
        projection_history = []
        status = "maximum_iterations_reached"
        stalled = 0
        for iteration in range(1, int(proj_cfg["max_iterations"]) + 1):
            with torch.enable_grad():
                current = self._evaluate(controls, context, roads)
                selected, active_counts = self._active_indices(current.groups)
                active_rows = sum(active_counts.values())
                if current.metrics["feasible"]:
                    status = "converged_feasible"
                    break
                if active_rows == 0:
                    status = "no_active_violations"
                    break
                delta, jac_diag = self._joint_correction(controls, context, roads, current, selected,
                                                       damping_multiplier=damping_scale)
            if delta is None:
                trust *= float(proj_cfg["trust_shrink_factor"])
                damping_scale *= 10.0
                stalled += 1
                projection_history.append({"iteration": iteration, "accepted": False,
                                           "trust_radius": trust, **jac_diag})
                if stalled >= int(proj_cfg["max_stalled_iterations"]):
                    status = f"projection_linear_system_failed:{jac_diag.get('reason', 'unknown')}"
                    break
                continue

            control_scale = controls.new_tensor(self.cfg["control"]["trust_scale"]).view(1, 1, 2)
            normalized_delta = delta / control_scale
            # Per-node trust clipping avoids shrinking every control update by
            # sqrt(K), which is especially harmful for long (40-node) horizons.
            node_norm = torch.linalg.vector_norm(normalized_delta, dim=-1, keepdim=True).clamp_min(1.0e-9)
            max_step = min(trust, float(proj_cfg["max_step_norm"]))
            normalized_delta = normalized_delta * torch.clamp(max_step / node_norm, max=1.0)
            delta = normalized_delta * control_scale

            current_merit, current_objective, current_penalty = self._merit(
                controls, current, nominal, tau_nominal)
            alpha = float(proj_cfg["alpha_initial"])
            accepted = False
            trial_diag = None
            while alpha >= float(proj_cfg["alpha_minimum"]):
                candidate_controls = controls + alpha * delta
                # Exact box projection is a safety guard; the box residual remains
                # part of the joint active set so the other constraints are rechecked.
                lower = candidate_controls.new_tensor([
                    self.cfg["control"]["acceleration_min_mps2"],
                    self.cfg["control"]["steering_min_rad"]])
                upper = candidate_controls.new_tensor([
                    self.cfg["control"]["acceleration_max_mps2"],
                    self.cfg["control"]["steering_max_rad"]])
                candidate_controls = torch.maximum(torch.minimum(candidate_controls, upper), lower)
                with torch.enable_grad():
                    candidate = self._evaluate(candidate_controls, context, roads)
                candidate_merit, candidate_objective, candidate_penalty = self._merit(
                    candidate_controls, candidate, nominal, tau_nominal)
                max_reduced = candidate.metrics["max_violation"] < current.metrics["max_violation"] - float(proj_cfg["minimum_violation_improvement"])
                merit_reduced = candidate_merit < current_merit - float(proj_cfg["merit_tolerance"])
                if merit_reduced or max_reduced or candidate.metrics["feasible"]:
                    controls = candidate_controls.detach()
                    current = candidate
                    accepted = True
                    accepted_iterations += 1
                    stalled = 0
                    trial_diag = {
                        "iteration": iteration, "accepted": True, "alpha": alpha,
                        "objective": candidate_objective, "merit": candidate_merit,
                        "penalty": candidate_penalty,
                        "max_violation": candidate.metrics["max_violation"],
                        "max_violation_by_group": candidate.metrics["max_violation_by_group"],
                        "active_counts": active_counts,
                        "trust_radius": trust,
                        **jac_diag,
                    }
                    if alpha >= 0.75 and merit_reduced:
                        trust = min(float(proj_cfg["maximum_trust_radius"]),
                                    trust * float(proj_cfg["trust_grow_factor"]))
                    elif alpha <= 0.25:
                        trust = max(float(proj_cfg["minimum_trust_radius"]),
                                    trust * float(proj_cfg["trust_shrink_factor"]))
                    damping_scale = max(1.0, damping_scale / 2.0)
                    break
                alpha *= float(proj_cfg["alpha_shrink_factor"])

            if not accepted:
                stalled += 1
                trust *= float(proj_cfg["trust_shrink_factor"])
                damping_scale *= 10.0
                trial_diag = {"iteration": iteration, "accepted": False,
                              "trust_radius": trust, "current_merit": current_merit,
                              "current_objective": current_objective,
                              "current_penalty": current_penalty,
                              "active_counts": active_counts, **jac_diag}
            projection_history.append(trial_diag)
            history.append({"iteration": iteration, **current.metrics,
                            "status": "accepted" if accepted else "rejected"})
            if current.metrics["feasible"]:
                status = "converged_feasible"
                break
            if trust < float(proj_cfg["minimum_trust_radius"]):
                status = "stalled_infeasible:minimum_trust_radius"
                break
            if stalled >= int(proj_cfg["max_stalled_iterations"]):
                status = "stalled_infeasible:no_merit_improvement"
                break

        with torch.enable_grad():
            final = self._evaluate(controls, context, roads)
        objective = float(self._objective(controls, final.rollout, nominal, tau_nominal).detach().cpu())
        success = bool(final.metrics["feasible"])
        if success:
            status = "converged_feasible"
        return JTPResult(
            controls=controls.detach(), trajectories=final.rollout["trajectory"].detach(),
            states=final.rollout["states"].detach(), ltr=final.rollout["ltr"].detach(),
            success=success, status=status, iterations=len(projection_history),
            diagnostics={
                "accepted_iterations": accepted_iterations,
                "history": history,
                "projection": projection_history,
                "objective": objective,
                "initial_metrics": history[0],
                "final_metrics": final.metrics,
                "final_trust_radius": trust,
                "collision_handled_by": "adaptive_anchor",
                "control_rate_constraints_enabled": False,
                "config_path": self.cfg.get("_config_path"),
            },
        )
