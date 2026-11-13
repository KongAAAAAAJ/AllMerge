"""Sequential-convex Joint Trajectory Projection (JTP) solver.

The solver implements the finalized TRACE formulation with joint decision
variables Delta_tau and Delta_U. It is independent of the diffusion runtime:
callers provide nominal control sequences, synchronized K2 context, and road
corridors. The frozen Koopman execution model is differentiated only w.r.t.
its control inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from os import PathLike

import numpy as np
import torch

from highway_env.planner.diffusion.common.execution import K2_CONTEXT, VehicleExecution

from .config import load_jtp_config, validate_jtp_config
from .constraints import stability_residual
from .execution_adapter import KoopmanExecutionAdapter
from .geometry import (
    collision_residual,
    footprint_circle_centers,
    geometry_residual,
    nearest_road_association,
)
from .qp_builder import build_joint_qp
from .types import ExternalTraffic, JTPResult, RoadGeometry


@dataclass
class _ConstraintEvaluation:
    rollout: dict
    stability: list[torch.Tensor]
    geometry: list[torch.Tensor]
    geometry_meta: list[dict]
    collision: torch.Tensor
    feasible: bool
    max_violation: float
    metrics: dict


class JointTrajectoryProjector:
    """Independent JTP solver using frozen Koopman execution and sparse OSQP.

    Args:
        execution_model: Existing ``VehicleExecution`` configured for K2, a
            ``KoopmanExecutionAdapter``, or a callable implementing
            ``fn(controls[N,H,2], context) -> {trajectory, states, ltr}``.
        config: Parsed JTP YAML mapping or path to ``configs/projection/jtp.yaml``.
        control_interpolation_matrix: Optional fixed C[H,K] mapping from
            diffusion control nodes to Koopman execution-grid inputs. Current
            TRACE uses K=H=40, so identity interpolation is the default.
    """

    def __init__(self, execution_model, config=None, *, control_interpolation_matrix=None):
        self.cfg = load_jtp_config(config) if isinstance(config, (str, bytes, PathLike)) or config is None else config
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
                raise ValueError(
                    "K != H requires an explicit fixed control_interpolation_matrix [H,K]"
                )
            self.control_interpolation = torch.eye(self.horizon)
        else:
            matrix = torch.as_tensor(control_interpolation_matrix, dtype=torch.float32)
            if matrix.shape != (self.horizon, self.control_steps):
                raise ValueError(
                    f"control interpolation must be [{self.horizon},{self.control_steps}]"
                )
            if not torch.isfinite(matrix).all():
                raise ValueError("Control interpolation matrix contains NaN/Inf")
            self.control_interpolation = matrix

    @classmethod
    def from_koopman_checkpoint(cls, checkpoint, config=None, *, device="cpu"):
        """Construct the existing frozen K2 VehicleExecution from a checkpoint."""
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
        # Ignore unrelated runtime metadata (e.g. model IDs); the execution
        # adapter consumes exactly the five synchronized K2 context tensors.
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

    @staticmethod
    def _slice_context(context: dict, index: int) -> dict:
        return {key: value[index:index + 1] for key, value in context.items()}

    def _ensure_execution_device(self, device, dtype) -> None:
        module = self.execution_model
        if isinstance(module, KoopmanExecutionAdapter):
            module = module.vehicle_execution
        if isinstance(module, torch.nn.Module):
            module.to(device=device, dtype=dtype)
            module.eval()
            module.requires_grad_(False)

    def _execution_controls(self, controls: torch.Tensor) -> torch.Tensor:
        matrix = self.control_interpolation.to(device=controls.device, dtype=controls.dtype)
        return torch.einsum("hk,nkc->nhc", matrix, controls)

    def _rollout(self, controls: torch.Tensor, context: dict) -> dict:
        execution_controls = self._execution_controls(controls)
        output = self.rollout_fn(execution_controls, context)
        required = ("trajectory", "states", "ltr")
        missing = [key for key in required if key not in output]
        if missing:
            raise KeyError(f"JTP execution function missing outputs: {missing}")
        trajectory, states, ltr = output["trajectory"], output["states"], output["ltr"]
        n = controls.shape[0]
        if trajectory.shape != (n, self.horizon + 1, 3):
            raise ValueError(f"trajectory must be [N,{self.horizon + 1},3], got {tuple(trajectory.shape)}")
        if states.shape != (n, self.horizon + 1, 5):
            raise ValueError(f"states must be [N,{self.horizon + 1},5], got {tuple(states.shape)}")
        if ltr.shape != (n, self.horizon + 1):
            raise ValueError(f"ltr must be [N,{self.horizon + 1}], got {tuple(ltr.shape)}")
        if not torch.isfinite(trajectory).all() or not torch.isfinite(states).all() or not torch.isfinite(ltr).all():
            raise ValueError("Koopman execution produced NaN/Inf")
        return {"trajectory": trajectory, "states": states, "ltr": ltr}

    def _road_list(self, road_geometry, n: int, *, device, dtype):
        if isinstance(road_geometry, RoadGeometry):
            roads = [road_geometry] * n
        elif isinstance(road_geometry, (list, tuple)) and len(road_geometry) == n:
            roads = list(road_geometry)
        else:
            raise ValueError("road_geometry must be one RoadGeometry or a list with one per vehicle")
        for road in roads:
            if not isinstance(road, RoadGeometry):
                raise TypeError("Every road entry must be RoadGeometry")
            road.tensors(device=device, dtype=dtype)
        return roads

    def _evaluate_constraints(self, controls, context, roads, external_traffic) -> _ConstraintEvaluation:
        rollout = self._rollout(controls, context)
        n = controls.shape[0]
        stability, geometry, geometry_meta = [], [], []
        footprint = self.cfg["geometry"]["footprint"]
        offsets = torch.as_tensor(footprint["circle_offsets_m"], device=controls.device, dtype=controls.dtype)
        for i in range(n):
            stability.append(stability_residual(
                rollout["states"][i], rollout["ltr"][i],
                context["koopman_payload_kg"][i], context["koopman_road_mu"][i], self.cfg,
            ))
            circle_centers = footprint_circle_centers(rollout["trajectory"][i, 1:].detach(), offsets)
            association = nearest_road_association(circle_centers, roads[i])
            g_geo, meta = geometry_residual(rollout["trajectory"][i], roads[i], self.cfg,
                                            association=association,
                                            include_curvature=bool(self.cfg["geometry"].get("enforce_curvature", True)))
            geometry.append(g_geo)
            geometry_meta.append(meta)
        collision = collision_residual(rollout["trajectory"], self.cfg, external_traffic)
        metrics = self._violation_metrics(controls, context, stability, geometry, collision, geometry_meta)
        feasible = metrics["max_violation"] <= float(self.cfg["optimization"]["constraint_tolerance"])
        return _ConstraintEvaluation(rollout, stability, geometry, geometry_meta,
                                     collision, feasible, metrics["max_violation"], metrics)

    def _control_residuals(self, controls, context):
        cfg = self.cfg["control"]
        lower = controls.new_tensor([float(cfg["acceleration_min_mps2"]), float(cfg["steering_min_rad"])])
        upper = controls.new_tensor([float(cfg["acceleration_max_mps2"]), float(cfg["steering_max_rad"])])
        box = torch.cat(((lower - controls).reshape(-1), (controls - upper).reshape(-1)))
        previous = context["koopman_control_hist_si"][:, -1]
        differences = torch.cat((controls[:, :1] - previous[:, None], controls[:, 1:] - controls[:, :-1]), dim=1)
        rate_limit = controls.new_tensor([
            float(cfg["acceleration_rate_max_mps3"]) * self.dt,
            float(cfg["steering_rate_max_radps"]) * self.dt,
        ])
        rate = differences.abs() - rate_limit
        return box, rate

    def _violation_metrics(self, controls, context, stability, geometry, collision, geometry_meta):
        tol = float(self.cfg["optimization"]["constraint_tolerance"])
        box, rate = self._control_residuals(controls, context)
        groups = {
            "stability": torch.cat(stability) if stability else controls.new_empty(0),
            "geometry": torch.cat(geometry) if geometry else controls.new_empty(0),
            "collision": collision,
            "control_box": box,
            "control_rate": rate.reshape(-1),
        }
        maxima = {key: float(torch.relu(value).max().detach().cpu()) if value.numel() else 0.0
                  for key, value in groups.items()}
        profile = self.cfg.get("constraint_profile", {})
        active = {
            "stability": bool(profile.get("stability", True)),
            "geometry": bool(profile.get("road_geometry", True)),
            "collision": bool(profile.get("collision", True)),
            "control_box": bool(profile.get("control_box", True)),
            "control_rate": bool(profile.get("control_rate", True)),
        }
        active_maxima = [maxima[key] for key, enabled in active.items() if enabled]
        return {
            "max_violation": max(active_maxima, default=0.0),
            "max_violation_by_group": maxima,
            "active_constraint_groups": [key for key, enabled in active.items() if enabled],
            "disabled_constraint_groups": [key for key, enabled in active.items() if not enabled],
            "feasible": all(value <= tol for value in active_maxima),
            "stability_violation_count": int((groups["stability"] > tol).sum().item()),
            "geometry_violation_count": int((groups["geometry"] > tol).sum().item()),
            "collision_violation_count": int((groups["collision"] > tol).sum().item()),
            "control_box_violation_count": int((groups["control_box"] > tol).sum().item()),
            "control_rate_violation_count": int((groups["control_rate"] > tol).sum().item()),
        }

    def _jacobian(self, function: Callable, point: torch.Tensor) -> torch.Tensor:
        use_vectorize = bool(self.cfg["optimization"].get("jacobian_vectorize", True))
        try:
            return torch.autograd.functional.jacobian(
                function, point, create_graph=False, strict=False, vectorize=use_vectorize,
            )
        except (RuntimeError, NotImplementedError):
            if not use_vectorize:
                raise
            return torch.autograd.functional.jacobian(
                function, point, create_graph=False, strict=False, vectorize=False,
            )

    def _linearize(self, controls, context, roads, external_traffic, evaluation):
        n, k, _ = controls.shape
        tau = evaluation.rollout["trajectory"]
        jac_tau_u, jac_stability_u = [], []
        jac_geometry_tau = []
        for i in range(n):
            context_i = self._slice_context(context, i)
            payload_i = context["koopman_payload_kg"][i]
            mu_i = context["koopman_road_mu"][i]
            u0 = controls[i].detach().reshape(-1).requires_grad_(True)

            profile = self.cfg.get("constraint_profile", {})
            stability_enabled = bool(profile.get("stability", True))

            def execution_and_stability(u_flat):
                u = u_flat.reshape(1, k, 2)
                pred = self._rollout(u, context_i)
                tau_i = pred["trajectory"][0]
                if not stability_enabled:
                    return tau_i.reshape(-1)
                g_stab = stability_residual(pred["states"][0], pred["ltr"][0],
                                             payload_i, mu_i, self.cfg)
                return torch.cat((tau_i.reshape(-1), g_stab))

            jac = self._jacobian(execution_and_stability, u0)
            tau_dim = (self.horizon + 1) * 3
            jac_tau_u.append(jac[:tau_dim].detach())
            jac_stability_u.append(jac[tau_dim:].detach() if stability_enabled
                                    else jac.new_empty((0, jac.shape[1])))

            # Road segment/corridor associations are frozen at this SCP iterate.
            offsets = torch.as_tensor(self.cfg["geometry"]["footprint"]["circle_offsets_m"],
                                      device=controls.device, dtype=controls.dtype)
            centers = footprint_circle_centers(tau[i, 1:].detach(), offsets)
            association = nearest_road_association(centers, roads[i])
            tau_point = tau[i].detach().reshape(-1).requires_grad_(True)

            def geometry_fn(tau_flat):
                return geometry_residual(tau_flat.reshape(self.horizon + 1, 3), roads[i],
                                         self.cfg, association=association,
                                         include_curvature=bool(self.cfg["geometry"].get("enforce_curvature", True)))[0]

            jac_geometry_tau.append(self._jacobian(geometry_fn, tau_point).detach())

        tau_point = tau.detach().reshape(-1).requires_grad_(True)
        if evaluation.collision.numel():
            def collision_fn(tau_flat):
                return collision_residual(tau_flat.reshape_as(tau), self.cfg, external_traffic)
            jac_collision_tau = self._jacobian(collision_fn, tau_point).detach()
        else:
            jac_collision_tau = tau.new_empty((0, tau.numel()))
        return jac_tau_u, jac_stability_u, jac_geometry_tau, jac_collision_tau

    def _objective(self, controls, trajectory, controls_nominal, trajectory_nominal) -> torch.Tensor:
        opt = self.cfg["optimization"]
        wt = trajectory.new_tensor(opt["trajectory_weights"])
        wu = controls.new_tensor(opt["control_weights"])
        trajectory_error = trajectory - trajectory_nominal
        control_error = controls - controls_nominal
        return (0.5 * (trajectory_error.square() * wt).sum()
                + 0.5 * float(opt["lambda_u"]) * (control_error.square() * wu).sum())

    def _merit(self, controls, evaluation, controls_nominal, trajectory_nominal, context):
        objective = self._objective(controls, evaluation.rollout["trajectory"],
                                    controls_nominal, trajectory_nominal)
        penalty = 0.0
        # Normalize each physical residual family before applying the merit penalty.
        merit_scales = self.cfg["optimization"]["merit_scales"]
        profile = self.cfg.get("constraint_profile", {})
        if bool(profile.get("stability", True)):
            stab_scale = controls.new_tensor(merit_scales["stability"])
            for value in evaluation.stability:
                shaped = value.reshape(2, self.horizon, 4)
                penalty += float((torch.relu(shaped / stab_scale).square().sum()).detach().cpu())
        for value, meta in zip(evaluation.geometry, evaluation.geometry_meta):
            road_count = meta["road_count"]
            road_scale = float(merit_scales["road_residual_m"])
            curve_scale = float(merit_scales["curvature_residual_per_m2"])
            penalty += float(torch.relu(value[:road_count] / road_scale).square().sum().detach().cpu())
            if meta["curvature_count"]:
                penalty += float(torch.relu(value[road_count:] / curve_scale).square().sum().detach().cpu())
        if bool(profile.get("collision", True)) and evaluation.collision.numel():
            collision_scale = float(merit_scales["collision_residual_m2"])
            penalty += float(torch.relu(evaluation.collision / collision_scale).square().sum().detach().cpu())
        box, rate = self._control_residuals(controls, context)
        if bool(profile.get("control_box", True)):
            control_scale = controls.new_tensor(self.cfg["control"]["trust_scale"]).repeat(controls.shape[0] * controls.shape[1])
            penalty += float(torch.relu(box / control_scale.repeat(2)).square().sum().detach().cpu())
        if bool(profile.get("control_rate", True)):
            rate_scale = controls.new_tensor([
                float(self.cfg["control"]["acceleration_rate_max_mps3"]) * self.dt,
                float(self.cfg["control"]["steering_rate_max_radps"]) * self.dt,
            ]).repeat(controls.shape[0] * controls.shape[1])
            penalty += float(torch.relu(rate.reshape(-1) / rate_scale).square().sum().detach().cpu())
        total = float(objective.detach().cpu()) + float(self.cfg["optimization"]["merit_penalty"]) * penalty
        return total, float(objective.detach().cpu()), penalty

    def _solve_qp(self, qp):
        try:
            import osqp
        except ImportError as exc:
            raise ImportError(
                "JTP SCP requires OSQP. Install manually with: "
                "python -m pip install 'osqp>=1.0' 'scipy>=1.10'"
            ) from exc
        settings = self.cfg["qp"]
        solver = osqp.OSQP()
        kwargs = dict(
            P=qp.P, q=qp.q, A=qp.A, l=qp.lower, u=qp.upper,
            verbose=bool(settings["verbose"]),
            eps_abs=float(settings["eps_abs"]), eps_rel=float(settings["eps_rel"]),
            max_iter=int(settings["max_iter"]), adaptive_rho=bool(settings["adaptive_rho"]),
        )
        try:
            solver.setup(**kwargs, polishing=bool(settings.get("polishing", True)))
        except TypeError:
            solver.setup(**kwargs, polish=bool(settings.get("polishing", True)))
        result = solver.solve()
        status = str(result.info.status).lower()
        if result.x is None or not status.startswith("solved"):
            return None, status, {"status": status, "iterations": int(result.info.iter)}
        return np.asarray(result.x, dtype=np.float64), status, {
            "status": status,
            "iterations": int(result.info.iter),
            "objective": float(result.info.obj_val),
            "primal_residual": float(result.info.prim_res),
            "dual_residual": float(result.info.dual_res),
        }

    def project(self, nominal_controls: torch.Tensor, context: dict,
                road_geometry: RoadGeometry | list[RoadGeometry],
                external_traffic: ExternalTraffic | None = None) -> JTPResult:
        """Project a jointly planned set of nominal K2 control sequences.

        Args:
            nominal_controls: [N,K,2] in SI units, channel order [ax, delta_f].
            context: batched K2 context with leading dimension N.
            road_geometry: shared world-frame road corridor(s), or one per truck.
            external_traffic: optional constant-velocity-predicted [A,H+1,3]
                world poses. Use ``constant_velocity_poses`` to construct it.
        """
        if not torch.is_tensor(nominal_controls) or nominal_controls.ndim != 3 or nominal_controls.shape[-1] != 2:
            raise ValueError("nominal_controls must be a torch tensor [N,K,2]")
        if not nominal_controls.is_floating_point() or not torch.isfinite(nominal_controls).all():
            raise ValueError("nominal_controls must be finite floating-point SI controls")
        n, k, _ = nominal_controls.shape
        if k != self.control_steps:
            raise ValueError(f"Expected K={self.control_steps} control nodes, got {k}")
        if self.control_interpolation.shape != (self.horizon, k):
            raise ValueError("Control interpolation matrix does not match the current horizon")
        self._ensure_execution_device(nominal_controls.device, nominal_controls.dtype)
        context = self._to_context(context, n=n, device=nominal_controls.device, dtype=nominal_controls.dtype)
        roads = self._road_list(road_geometry, n, device=nominal_controls.device, dtype=nominal_controls.dtype)
        if external_traffic is not None:
            external_traffic.tensors(device=nominal_controls.device, dtype=nominal_controls.dtype,
                                     expected_steps=self.horizon + 1)

        controls_nominal = nominal_controls.detach().clone()
        controls = controls_nominal.clone()
        with torch.enable_grad():
            nominal_rollout = self._rollout(controls_nominal, context)
            trajectory_nominal = nominal_rollout["trajectory"].detach().clone()
            current = self._evaluate_constraints(controls, context, roads, external_traffic)
        history = [{"iteration": 0, **current.metrics, "status": "nominal"}]
        if current.feasible:
            return JTPResult(
                controls=controls.detach(), trajectories=current.rollout["trajectory"].detach(),
                states=current.rollout["states"].detach(), ltr=current.rollout["ltr"].detach(),
                success=True, status="nominal_already_feasible", iterations=0,
                diagnostics={"history": history, "objective": 0.0, "qp": [],
                             "config_path": self.cfg.get("_config_path")},
            )

        trust = float(self.cfg["optimization"]["initial_trust_radius"])
        qp_history = []
        status = "maximum_iterations_reached"
        accepted_iterations = 0
        scp_iterations = 0
        for iteration in range(1, int(self.cfg["optimization"]["max_scp_iterations"]) + 1):
            scp_iterations = iteration
            current = self._evaluate_constraints(controls, context, roads, external_traffic)
            if current.feasible:
                status = "converged_feasible"
                break
            jac_tau_u, jac_stab_u, jac_geo_tau, jac_col_tau = self._linearize(
                controls, context, roads, external_traffic, current)
            previous = context["koopman_control_hist_si"][:, -1]
            accepted = False
            last_qp_status = "not_solved"
            for retry in range(int(self.cfg["optimization"]["max_qp_retries"]) + 1):
                qp = build_joint_qp(
                    controls=controls,
                    controls_nominal=controls_nominal,
                    tau_current=current.rollout["trajectory"],
                    tau_nominal=trajectory_nominal,
                    jac_tau_u=jac_tau_u,
                    g_stability=current.stability,
                    jac_stability_u=jac_stab_u,
                    g_geometry=current.geometry,
                    jac_geometry_tau=jac_geo_tau,
                    g_collision=current.collision,
                    jac_collision_tau=jac_col_tau,
                    previous_controls=previous,
                    trust_radius=trust,
                    cfg=self.cfg,
                )
                solution, qp_status, qp_diag = self._solve_qp(qp)
                last_qp_status = qp_status
                qp_history.append({"scp_iteration": iteration, "retry": retry,
                                   **qp.diagnostics, **qp_diag, "trust_radius": trust})
                if solution is None:
                    trust *= float(self.cfg["optimization"]["trust_shrink_factor"])
                    if trust < float(self.cfg["optimization"]["minimum_trust_radius"]):
                        break
                    continue
                delta_u = torch.as_tensor(
                    solution[qp.n_tau:qp.n_tau + qp.n_control].reshape(n, k, 2),
                    device=controls.device, dtype=controls.dtype,
                )
                delta_norm = float(torch.linalg.vector_norm(delta_u).detach().cpu())
                current_merit, current_objective, current_penalty = self._merit(
                    controls, current, controls_nominal, trajectory_nominal, context)
                alpha = float(self.cfg["optimization"]["alpha_initial"])
                while alpha >= float(self.cfg["optimization"]["alpha_minimum"]):
                    candidate_controls = controls + alpha * delta_u
                    with torch.enable_grad():
                        candidate = self._evaluate_constraints(candidate_controls, context, roads, external_traffic)
                    candidate_merit, candidate_objective, candidate_penalty = self._merit(
                        candidate_controls, candidate, controls_nominal, trajectory_nominal, context)
                    improves = candidate_merit < current_merit - float(self.cfg["optimization"]["objective_tolerance"])
                    becomes_feasible = candidate.feasible and not current.feasible
                    if improves or becomes_feasible:
                        controls = candidate_controls.detach()
                        current = candidate
                        accepted = True
                        accepted_iterations += 1
                        history.append({
                            "iteration": iteration,
                            **candidate.metrics,
                            "status": "accepted",
                            "alpha": alpha,
                            "objective": candidate_objective,
                            "merit": candidate_merit,
                            "penalty": candidate_penalty,
                            "delta_control_norm": delta_norm,
                        })
                        if alpha >= 0.999:
                            trust = min(float(self.cfg["optimization"]["maximum_trust_radius"]),
                                        trust * float(self.cfg["optimization"]["trust_grow_factor"]))
                        break
                    alpha *= float(self.cfg["optimization"]["alpha_shrink_factor"])
                if accepted:
                    if current.feasible:
                        status = "converged_feasible"
                    elif delta_norm <= float(self.cfg["optimization"]["convergence_step_norm"]):
                        status = "stalled_infeasible"
                    break
                trust *= float(self.cfg["optimization"]["trust_shrink_factor"])
                if trust < float(self.cfg["optimization"]["minimum_trust_radius"]):
                    break
            if not accepted:
                status = f"qp_or_line_search_stalled:{last_qp_status}"
                break
            if current.feasible:
                status = "converged_feasible"
                break
            if history[-1].get("delta_control_norm", float("inf")) <= float(
                    self.cfg["optimization"]["convergence_step_norm"]):
                status = "stalled_infeasible"
                break

        with torch.enable_grad():
            final = self._evaluate_constraints(controls, context, roads, external_traffic)
        objective = float(self._objective(controls, final.rollout["trajectory"],
                                          controls_nominal, trajectory_nominal).detach().cpu())
        success = bool(final.feasible)
        if success and status != "converged_feasible":
            status = "converged_feasible"
        return JTPResult(
            controls=controls.detach(),
            trajectories=final.rollout["trajectory"].detach(),
            states=final.rollout["states"].detach(),
            ltr=final.rollout["ltr"].detach(),
            success=success,
            status=status,
            iterations=scp_iterations,
            diagnostics={
                "accepted_iterations": accepted_iterations,
                "history": history,
                "qp": qp_history,
                "objective": objective,
                "initial_metrics": history[0],
                "final_metrics": final.metrics,
                "trust_radius_final": trust,
                "config_path": self.cfg.get("_config_path"),
                "control_interpolation": "identity" if self.horizon == self.control_steps else "fixed_matrix",
            },
        )
