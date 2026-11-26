"""Priority-grouped LS/QP joint trajectory projection with shared K2 linearization."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from os import PathLike
from typing import Any

import torch

from highway_env.planner.diffusion.common.execution import K2_CONTEXT, VehicleExecution

from .config import load_jtp_config, validate_jtp_config
from .constraints import stability_residual, control_rate_residual
from .execution_adapter import KoopmanExecutionAdapter
from .geometry import (
    geometry_residual, nearest_road_association, footprint_circle_centers,
    intertruck_collision_residual, external_collision_residual,
)
from .types import JTPResult, RoadGeometry
from .linearization import K2StaticCache, shared_linearization
from .grouped_solver import build_priority_groups, group_maxima, accept_nonlinear, solve_groups


@dataclass
class _Evaluation:
    rollout: dict
    groups: dict[str, torch.Tensor]
    associations: list[tuple[torch.Tensor, torch.Tensor]]
    metrics: dict


class JointTrajectoryProjector:
    """Priority-grouped LS/QP projection with full nonlinear verification."""

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

    def _constraint_groups(self, controls, context, roads, associations=None, external_actors=(), *, rollout=None):
        if rollout is None:
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
        intertruck_collision = intertruck_collision_residual(rollout["trajectory"], self.cfg)
        external_collision = external_collision_residual(
            rollout["trajectory"], external_actors, self.cfg)
        control_rate = control_rate_residual(controls, context, self.cfg)
        groups = {
            "stability": torch.cat([x.reshape(-1) for x in stability]),
            "road": torch.cat([x.reshape(-1) for x in road_values]),
            "curvature": torch.cat([x.reshape(-1) for x in curve_values]),
            # Retained for metrics only; grouped_solver enforces this as a fixed box.
            "control": control_box.reshape(-1),
            "intertruck_collision": intertruck_collision.reshape(-1),
            "external_collision": external_collision.reshape(-1),
            "control_rate": control_rate.reshape(-1),
        }
        return rollout, groups, associations

    def _constraint_thresholds(self, groups):
        """Return the single tolerance tensor used by activity and evaluation."""
        tol = float(self.cfg["projection"]["constraint_tolerance"])
        thresholds = {key: torch.full_like(value, tol) for key, value in groups.items()}
        # Control range is a hard constraint and has no feasibility slack.
        if "control" in thresholds:
            thresholds["control"] = torch.zeros_like(groups["control"])
        # Control-rate limits retain their channel-specific SI tolerance after normalization.
        rate_values = groups.get("control_rate")
        if rate_values is not None and rate_values.numel():
            rate_scale = self.cfg["projection"]["residual_scales"]["control_rate"]
            tol_si = float(self.cfg["control"]["rate_tolerance_si"])
            channel_tol = rate_values.new_tensor([
                tol_si / float(rate_scale[0]), tol_si / float(rate_scale[1])])
            n = rate_values.numel() // (2 * self.horizon * 2)
            thresholds["control_rate"] = channel_tol.view(1, 1, 1, 2).expand(
                2, n, self.horizon, 2).reshape(-1)
        return thresholds

    def _metrics(self, groups):
        thresholds = self._constraint_thresholds(groups)
        positive = {key: torch.relu(value - thresholds[key]) for key, value in groups.items()}
        maxima = {key: (float(value.max().detach().cpu()) if value.numel() else 0.0)
                  for key, value in positive.items()}
        counts = {key: int((value > thresholds[key]).sum().detach().cpu())
                  for key, value in groups.items()}
        l2 = {key: float(torch.linalg.vector_norm(value).detach().cpu())
              for key, value in positive.items()}
        control_feasible = counts["control"] == 0
        return {
            "max_violation": max(maxima.values(), default=0.0),
            "max_violation_by_group": maxima,
            "violation_count_by_group": counts,
            "violation_l2_by_group": l2,
            "stability_violation_count": counts["stability"],
            "geometry_violation_count": counts["road"] + counts["curvature"],
            "control_box_violation_count": counts["control"],
            "control_box_feasible": control_feasible,
            "intertruck_collision_violation_count": counts.get("intertruck_collision", 0),
            "external_collision_violation_count": counts.get("external_collision", 0),
            "control_rate_violation_count": counts.get("control_rate", 0),
            "projector_constraints_satisfied": all(value == 0 for value in counts.values()),
        }

    def _evaluate(self, controls, context, roads, associations=None, external_actors=(), *, rollout=None):
        rollout, groups, associations = self._constraint_groups(
            controls, context, roads, associations, external_actors, rollout=rollout)
        return _Evaluation(rollout, groups, associations, self._metrics(groups))

    def _objective(self, controls, rollout, controls_nominal, trajectory_nominal):
        opt = self.cfg["optimization"]
        wt = rollout["trajectory"].new_tensor(opt["trajectory_weights"])
        wu = controls.new_tensor(opt["control_weights"])
        tau_error = rollout["trajectory"] - trajectory_nominal
        u_error = controls - controls_nominal
        return (0.5 * (tau_error.square() * wt).sum()
                + 0.5 * float(opt["lambda_u"]) * (u_error.square() * wu).sum())

    def _road_active_candidate_indices(self, road_values):
        """Return center-circle road rows at every second future step.

        The full road residual remains [N,H,C,2] flattened in vehicle-major,
        time-major, circle-major, boundary-major order. Only the solver candidate
        pool is reduced; nonlinear metrics and acceptance retain every row.
        """
        offsets = [float(value) for value in self.cfg["geometry"]["footprint"]["circle_offsets_m"]]
        center_circle = offsets.index(0.0)
        circle_count = len(offsets)
        time_indices = torch.arange(0, self.horizon, 2, device=road_values.device)
        boundary_indices = torch.arange(2, device=road_values.device)
        local_indices = (
            (time_indices[:, None] * circle_count + center_circle) * 2
            + boundary_indices[None, :]
        ).reshape(-1)
        rows_per_vehicle = self.horizon * circle_count * 2
        vehicle_count = road_values.numel() // rows_per_vehicle
        vehicle_offsets = torch.arange(
            vehicle_count, device=road_values.device
        ) * rows_per_vehicle
        return (vehicle_offsets[:, None] + local_indices[None, :]).reshape(-1)

    def _active_indices(self, groups):
        """Select grouped-solver rows from state-constraint pools using evaluation tolerances."""
        cfg = self.cfg["projection"]
        thresholds = self._constraint_thresholds(groups)
        selected = {}
        counts = {}
        total_counts = {}
        truncated_counts = {}
        candidate_pool_sizes = {}
        candidate_total_counts = {}
        sampling_omitted_counts = {}
        topk_truncated_counts = {}
        for name, values in groups.items():
            if name == "control":
                continue
            full_active = torch.nonzero(
                values.detach() > thresholds[name], as_tuple=False
            ).reshape(-1)
            if name == "road":
                candidate_indices = self._road_active_candidate_indices(values)
            else:
                candidate_indices = torch.arange(values.numel(), device=values.device)
            candidate_active = candidate_indices[
                values.detach()[candidate_indices] > thresholds[name][candidate_indices]
            ]
            candidate_total = int(candidate_active.numel())
            cap = int(cfg["max_active_per_group"][name])
            ids = candidate_active
            if candidate_total > cap:
                local = torch.topk(
                    values.detach()[candidate_active] - thresholds[name][candidate_active],
                    k=cap, largest=True, sorted=False
                ).indices
                ids = candidate_active[local]
            selected[name] = ids
            counts[name] = int(ids.numel())
            total_counts[name] = int(full_active.numel())
            candidate_pool_sizes[name] = int(candidate_indices.numel())
            candidate_total_counts[name] = candidate_total
            sampling_omitted_counts[name] = int(full_active.numel()) - candidate_total
            topk_truncated_counts[name] = candidate_total - int(ids.numel())
            truncated_counts[name] = int(full_active.numel()) - int(ids.numel())
        diagnostics = {
            "active_counts": counts,
            "active_total_counts": total_counts,
            "active_truncated_counts": truncated_counts,
            "active_candidate_pool_sizes": candidate_pool_sizes,
            "active_candidate_total_counts": candidate_total_counts,
            "active_sampling_omitted_counts": sampling_omitted_counts,
            "active_topk_truncated_counts": topk_truncated_counts,
        }
        return selected, diagnostics

    def evaluate_preprojection(self, nominal_controls: torch.Tensor, context: dict,
                                road_geometry, external_actors=()) -> dict:
        """Evaluate nominal controls with the exact JTP constraints, without projection.

        The method shares the same K2 rollout, road geometry, residuals,
        normalization, and tolerance as :meth:`project`. It never modifies
        the supplied controls and never performs a solver iteration.
        Supplied external-actor predictions and inter-truck collision are
        evaluated using the same residuals as the iterative projection.
        """
        if (not torch.is_tensor(nominal_controls) or nominal_controls.ndim != 3
                or nominal_controls.shape[-1] != 2):
            raise ValueError("nominal_controls must be [N,K,2]")
        if not nominal_controls.is_floating_point() or not torch.isfinite(nominal_controls).all():
            raise ValueError("nominal_controls must be finite floating-point SI controls")
        n, k, _ = nominal_controls.shape
        if k != self.control_steps:
            raise ValueError(f"Expected K={self.control_steps} control nodes, got {k}")
        external_actors = tuple(external_actors or ())
        self._ensure_execution_device(nominal_controls.device, nominal_controls.dtype)
        ctx = self._to_context(
            context, n=n, device=nominal_controls.device, dtype=nominal_controls.dtype
        )
        roads = self._road_list(
            road_geometry, n, device=nominal_controls.device, dtype=nominal_controls.dtype
        )
        with torch.no_grad():
            evaluated = self._evaluate(
                nominal_controls.detach(), ctx, roads, external_actors=external_actors)
        return {
            "controls": nominal_controls.detach(),
            "trajectories": evaluated.rollout["trajectory"].detach(),
            "states": evaluated.rollout["states"].detach(),
            "ltr": evaluated.rollout["ltr"].detach(),
            "projector_constraints_satisfied": bool(
                evaluated.metrics["projector_constraints_satisfied"]
            ),
            "metrics": evaluated.metrics,
            "constraint_groups": {
                key: value.detach() for key, value in evaluated.groups.items()
            },
        }

    def project(self, nominal_controls: torch.Tensor, context: dict,
                road_geometry: RoadGeometry | list[RoadGeometry],
                external_actors=()) -> JTPResult:
        """Project nominal controls while preserving Koopman execution consistency.

        The returned controls are always the best accepted JTP projection iterate.
        ``projector_constraints_satisfied`` reports only the constraint families
        implemented inside this projector; full paper feasibility is evaluated
        independently by the synchronized-frame evaluation module. External
        actors must be supplied as world-frame predicted paths when available.
        """
        if not torch.is_tensor(nominal_controls) or nominal_controls.ndim != 3 or nominal_controls.shape[-1] != 2:
            raise ValueError("nominal_controls must be a torch tensor [N,K,2]")
        if not nominal_controls.is_floating_point() or not torch.isfinite(nominal_controls).all():
            raise ValueError("nominal_controls must be finite floating-point SI controls")
        n, k, _ = nominal_controls.shape
        if k != self.control_steps:
            raise ValueError(f"Expected K={self.control_steps} control nodes, got {k}")
        external_actors = tuple(external_actors or ())
        self._ensure_execution_device(nominal_controls.device, nominal_controls.dtype)
        context = self._to_context(context, n=n, device=nominal_controls.device, dtype=nominal_controls.dtype)
        roads = self._road_list(road_geometry, n, device=nominal_controls.device, dtype=nominal_controls.dtype)

        nominal = nominal_controls.detach().clone()
        controls = nominal.clone()
        proj_cfg = self.cfg["projection"]
        solver_method = proj_cfg["solver_method"]
        use_ls = solver_method == "damped_least_squares"
        with torch.no_grad():
            current = self._evaluate(controls, context, roads, external_actors=external_actors)
        tau_nominal = current.rollout["trajectory"].detach().clone()
        thresholds = self._constraint_thresholds(current.groups)
        order, scores = build_priority_groups(current.groups, proj_cfg, thresholds)
        history = [{"iteration": 0, **current.metrics, "priority_group_maxima": group_maxima(current.metrics, order), "status": "nominal"}]
        initial_metrics = current.metrics
        projection_history = []
        trust = float(proj_cfg["initial_trust_radius"]) if use_ls else None
        damping_scale = 1.0 if use_ls else None
        accepted_iterations = stalled = validations = linearizations = 0
        cache = None
        cache_ms = 0.
        status = "nominal_already_feasible" if current.metrics["projector_constraints_satisfied"] else "maximum_iterations_reached"
        def clock():
            if controls.is_cuda:
                torch.cuda.synchronize(controls.device)
            return perf_counter()
        scale = controls.new_tensor(self.cfg["control"]["trust_scale"]) if use_ls else None
        lower = controls.new_tensor([self.cfg["control"]["acceleration_min_mps2"], self.cfg["control"]["steering_min_rad"]])
        upper = controls.new_tensor([self.cfg["control"]["acceleration_max_mps2"], self.cfg["control"]["steering_max_rad"]])
        if (use_ls and not current.metrics["projector_constraints_satisfied"]
                and isinstance(self.rollout_fn, KoopmanExecutionAdapter)):
            tick = clock()
            cache = K2StaticCache(self.rollout_fn, context, self.control_interpolation)
            cache_ms = (clock()-tick)*1000
        for iteration in range(1, int(proj_cfg["max_iterations"])+1):
            if current.metrics["projector_constraints_satisfied"]:
                break
            selected, active_diagnostics = self._active_indices(current.groups)
            if not sum(active_diagnostics["active_counts"].values()) and current.metrics["control_box_feasible"]:
                status = "no_active_violations"
                break
            tick = clock()
            has_active_state_rows = sum(active_diagnostics["active_counts"].values()) > 0
            if (not use_ls and has_active_state_rows and cache is None
                    and isinstance(self.rollout_fn, KoopmanExecutionAdapter)):
                cache = K2StaticCache(self.rollout_fn, context, self.control_interpolation)
                cache_ms = (clock()-tick)*1000
            with torch.enable_grad():
                rows, metric, method = shared_linearization(
                    self, controls, context, roads, current, selected, external_actors, cache,
                    build_metric=use_ls)
            linearizations += 1
            diag = {"iteration":iteration, **active_diagnostics,
                    "hard_control_violation_count_before":current.metrics["control_box_violation_count"],
                    "linearization_method":method,
                    "linearization_ms":(clock()-tick)*1000,
                    "trust_radius":trust,
                    "trust_region_enabled":use_ls}
            tick = clock()
            if not all(torch.isfinite(j).all() for j in rows.values()):
                delta, info = None, {"reason":"nonfinite_active_jacobian"}
            else:
                with torch.no_grad():
                    delta, info = solve_groups(
                        controls, current.groups, selected, rows, thresholds, metric, order,
                        proj_cfg, scale, trust, damping_scale, lower, upper,
                        control_weights=self.cfg["optimization"]["control_weights"])
            diag.update(info)
            diag["grouped_solver_ms"]=(clock()-tick)*1000
            before=group_maxima(current.metrics,order)
            diag["priority_group_maxima_before"]=before
            accepted = False
            accepted_reason = None
            trials = []
            tick = clock()
            if delta is not None and bool(delta.abs().max() > 0):
                alpha = float(proj_cfg["alpha_initial"])
                hard_feasible_before = bool(current.metrics["control_box_feasible"])
                while alpha >= float(proj_cfg["alpha_minimum"]):
                    candidate_controls = controls + alpha * delta
                    with torch.no_grad():
                        candidate = self._evaluate(
                            candidate_controls, context, roads,
                            associations=current.associations,
                            external_actors=external_actors)
                    validations += 1
                    after = group_maxima(candidate.metrics, order)
                    hard_feasible_after = bool(candidate.metrics["control_box_feasible"])
                    hard_restoration = not hard_feasible_before and hard_feasible_after
                    acceptance = accept_nonlinear(
                        current.groups, candidate.groups, thresholds, order, proj_cfg)
                    state_safe = (acceptance["nondegradation_ok"]
                                  and acceptance["violation_count_nonincrease_ok"])
                    state_improvement = bool(acceptance["state_progress"])
                    ok = hard_feasible_after and state_safe and (hard_restoration or state_improvement)
                    if not hard_feasible_after:
                        acceptance_reason = "hard_control_box_violation"
                    elif not acceptance["nondegradation_ok"]:
                        acceptance_reason = "state_constraint_regression"
                    elif not acceptance["violation_count_nonincrease_ok"]:
                        acceptance_reason = "priority_group_violation_count_increased"
                    elif hard_restoration:
                        acceptance_reason = "hard_control_restoration"
                    elif state_improvement:
                        acceptance_reason = "per_constraint_state_improvement"
                    else:
                        acceptance_reason = "no_state_improvement"
                    trials.append({
                        "alpha": alpha,
                        "priority_group_maxima": after,
                        "hard_control_feasible": hard_feasible_after,
                        "state_non_degradation": acceptance["nondegradation_ok"],
                        "group_violation_counts_nonincreasing": acceptance["violation_count_nonincrease_ok"],
                        "state_progress": state_improvement,
                        "acceptance_diagnostics": acceptance,
                        "accepted": ok,
                        "acceptance_reason": acceptance_reason,
                    })
                    if ok:
                        controls = candidate_controls.detach()
                        # Refresh nearest-road associations only after the frozen-association
                        # line search has accepted the candidate. Reuse its K2 rollout.
                        current = self._evaluate(
                            controls, context, roads, external_actors=external_actors,
                            rollout=candidate.rollout)
                        accepted = True
                        accepted_reason = acceptance_reason
                        accepted_iterations += 1
                        stalled = 0
                        if use_ls:
                            if alpha >= .75:
                                trust = min(float(proj_cfg["maximum_trust_radius"]),
                                            trust * float(proj_cfg["trust_grow_factor"]))
                            elif alpha <= .25:
                                trust = max(float(proj_cfg["minimum_trust_radius"]),
                                            trust * float(proj_cfg["trust_shrink_factor"]))
                            damping_scale = max(1.0, damping_scale / 2.0)
                        break
                    if not hard_feasible_before and not hard_feasible_after:
                        # A smaller fraction of a hard-feasibility restoration
                        # cannot enter the box when the active-set solution ends
                        # on its nearest bound.
                        break
                    alpha *= float(proj_cfg["alpha_shrink_factor"])
            diag.update({"accepted":accepted,"line_search_trials":trials,"nonlinear_validation_ms":(clock()-tick)*1000,
                         "priority_group_maxima_after":group_maxima(current.metrics,order),
                         "max_violation":current.metrics["max_violation"],"max_violation_by_group":current.metrics["max_violation_by_group"],
                         "candidate_violation_count_by_group":current.metrics["violation_count_by_group"],
                         "candidate_violation_l2_by_group":current.metrics["violation_l2_by_group"],
                         "acceptance_reason": accepted_reason if accepted else "rejected_or_zero_step"})
            projection_history.append(diag)
            history.append({"iteration":iteration,**current.metrics,"priority_group_maxima":group_maxima(current.metrics,order),"status":"accepted" if accepted else "rejected"})
            if not use_ls and delta is None:
                status = "solver_failed:" + str(info.get("reason", "unknown"))
                break
            if current.metrics["projector_constraints_satisfied"]:
                status="converged_feasible"; break
            if not accepted:
                stalled += 1
                if use_ls:
                    trust *= float(proj_cfg["trust_shrink_factor"])
                    damping_scale *= 10.0
            if use_ls and trust < float(proj_cfg["minimum_trust_radius"]):
                status="stalled_infeasible:minimum_trust_radius"; break
            if stalled>=int(proj_cfg["max_stalled_iterations"]):
                status="stalled_infeasible:no_group_improvement"; break
        final=current
        objective=float(self._objective(controls,final.rollout,nominal,tau_nominal))
        group_progress={}
        for name in current.groups:
            d={}
            for key,metric_name in (("max_violation","max_violation_by_group"),("violation_count","violation_count_by_group"),("violation_l2","violation_l2_by_group")):
                a=initial_metrics[metric_name][name]; b=final.metrics[metric_name][name]
                d.update({"initial_"+key:a,"final_"+key:b,key+"_reduction":a-b})
            group_progress[name]=d
        return JTPResult(controls=controls.detach(),trajectories=final.rollout["trajectory"].detach(),states=final.rollout["states"].detach(),ltr=final.rollout["ltr"].detach(),
            projector_constraints_satisfied=bool(final.metrics["projector_constraints_satisfied"]),status=status,iterations=len(projection_history),
            diagnostics={"accepted_iterations":accepted_iterations,"history":history,"projection":projection_history,"objective":objective,
            "initial_metrics":initial_metrics,"final_metrics":final.metrics,"group_progress":group_progress,"termination_status":status,"final_trust_radius":trust,"trust_region_enabled":use_ls,
            "priority_groups":order,"priority_scores":scores,"solver_method":proj_cfg["solver_method"],"static_cache_builds":int(cache is not None),"static_cache_ms":cache_ms,
            "linearization_count":linearizations,"nonlinear_validation_count":validations,"collision_handled_by":"joint_projection","external_actor_count":len(external_actors),
            "acceptance_policy":"hard_control_box_plus_per_constraint_non_degradation_and_group_violation_count_nonincrease","intertruck_collision_constraints_enabled":True,"external_collision_constraints_enabled":True,
            "control_rate_constraints_enabled":True,
            "road_active_sampling":{
                "time_stride":2,
                "circle_offset_m":0.0,
                "candidate_rows_per_vehicle":2 * ((self.horizon + 1) // 2),
                "global_top_k":int(proj_cfg["max_active_per_group"]["road"]),
                "nonlinear_validation":"full_footprint_all_steps",
            },
            "config_path":self.cfg.get("_config_path")})
