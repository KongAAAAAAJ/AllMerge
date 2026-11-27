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
from .geometry import geometry_residual, nearest_road_association, footprint_circle_centers
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

    def _constraint_groups(self, controls, context, roads, associations=None, *, rollout=None):
        if rollout is None:
            rollout = self._rollout(controls, context)
        if associations is None:
            associations = self._associations(rollout, roads, controls)
        n = controls.shape[0]
        stab_scale = controls.new_tensor(self.cfg["projection"]["residual_scales"]["stability"])
        road_scale = float(self.cfg["projection"]["residual_scales"]["road_m"])
        curve_scale = float(self.cfg["projection"]["residual_scales"]["curvature_per_m2"])
        stability, road_values, curve_values = [], [], []
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
        lower = controls.new_tensor([
            control_cfg["acceleration_min_mps2"], control_cfg["steering_min_rad"]])
        upper = controls.new_tensor([
            control_cfg["acceleration_max_mps2"], control_cfg["steering_max_rad"]])
        control_scale = controls.new_tensor(control_cfg["trust_scale"]).clamp_min(1.0e-6)
        control_box = torch.cat(((lower - controls) / control_scale,
                                 (controls - upper) / control_scale), dim=0)
        control_rate = control_rate_residual(controls, context, self.cfg)
        groups = {
            "stability": torch.cat([value.reshape(-1) for value in stability]),
            "road": torch.cat([value.reshape(-1) for value in road_values]),
            "curvature": torch.cat([value.reshape(-1) for value in curve_values]),
            "control": control_box.reshape(-1),
            "control_rate": control_rate.reshape(-1),
        }
        return rollout, groups, associations

    def _constraint_thresholds(self, groups):
        """Return the fixed tolerance tensor used by selection and acceptance."""
        tolerance = float(self.cfg["projection"]["constraint_tolerance"])
        thresholds = {key: torch.full_like(value, tolerance) for key, value in groups.items()}
        thresholds["control"] = torch.zeros_like(groups["control"])
        rate_values = groups["control_rate"]
        if rate_values.numel():
            rate_scale = self.cfg["projection"]["residual_scales"]["control_rate"]
            tolerance_si = float(self.cfg["control"]["rate_tolerance_si"])
            channel_tolerance = rate_values.new_tensor([
                tolerance_si / float(rate_scale[0]),
                tolerance_si / float(rate_scale[1])])
            vehicle_count = rate_values.numel() // (2 * self.horizon * 2)
            thresholds["control_rate"] = channel_tolerance.view(1, 1, 1, 2).expand(
                2, vehicle_count, self.horizon, 2).reshape(-1)
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
            "control_rate_violation_count": counts["control_rate"],
            "projector_constraints_satisfied": all(value == 0 for value in counts.values()),
        }

    def _evaluate(self, controls, context, roads, associations=None, *, rollout=None):
        rollout, groups, associations = self._constraint_groups(
            controls, context, roads, associations, rollout=rollout)
        return _Evaluation(rollout, groups, associations, self._metrics(groups))

    def _objective(self, controls, rollout, controls_nominal, trajectory_nominal):
        opt = self.cfg["optimization"]
        trajectory_weights = rollout["trajectory"].new_tensor(opt["trajectory_weights"])
        control_weights = controls.new_tensor(opt["control_weights"])
        trajectory_error = rollout["trajectory"] - trajectory_nominal
        control_error = controls - controls_nominal
        return (0.5 * (trajectory_error.square() * trajectory_weights).sum()
                + 0.5 * float(opt["lambda_u"])
                * (control_error.square() * control_weights).sum())

    def _initial_sampled_times(self):
        stride = int(self.cfg["projection"]["constraint_sampling_stride"])
        times = set(range(stride, self.horizon + 1, stride))
        times.add(self.horizon)
        return {
            "stability": set(times),
            "road": set(times),
            "curvature": {time for time in times if time >= 2},
            "control_rate": set(range(1, self.horizon + 1)),
        }

    def _sample_constraint_indices(self, groups, sampled_times):
        """Build sparse G1/G2 row indices and retain every G3 control-rate row."""
        device = groups["stability"].device
        horizon = self.horizon
        circle_count = len(self.cfg["geometry"]["footprint"]["circle_offsets_m"])

        stability_rows = []
        for sign in range(2):
            for time_index in sorted(sampled_times["stability"]):
                start = sign * horizon * 4 + (time_index - 1) * 4
                stability_rows.extend(range(start, start + 4))

        road_rows = []
        for time_index in sorted(sampled_times["road"]):
            start = (time_index - 1) * circle_count * 2
            road_rows.extend(range(start, start + circle_count * 2))

        curvature_rows = [time_index - 2
                          for time_index in sorted(sampled_times["curvature"])]
        return {
            "stability": torch.tensor(stability_rows, device=device, dtype=torch.long),
            "road": torch.tensor(road_rows, device=device, dtype=torch.long),
            "curvature": torch.tensor(curvature_rows, device=device, dtype=torch.long),
            "control_rate": torch.arange(
                groups["control_rate"].numel(), device=device, dtype=torch.long),
        }

    def _refine_constraint_indices(self, groups, thresholds, sampled_times, refined_times):
        """Add every full-horizon G1/G2 violation time omitted by the current sample."""
        horizon = self.horizon
        circle_count = len(self.cfg["geometry"]["footprint"]["circle_offsets_m"])
        additions = {"stability": set(), "road": set(), "curvature": set()}
        for name in additions:
            violated = torch.nonzero(
                groups[name].detach() > thresholds[name], as_tuple=False).reshape(-1)
            if not violated.numel():
                continue
            if name == "stability":
                times = (violated.remainder(horizon * 4) // 4 + 1).tolist()
            elif name == "road":
                times = (violated // (circle_count * 2) + 1).tolist()
            else:
                times = (violated + 2).tolist()
            additions[name] = set(map(int, times)) - sampled_times[name]
        for name, times in additions.items():
            sampled_times[name].update(times)
            refined_times[name].update(times)
        return {name: sorted(times) for name, times in additions.items() if times}

    def _active_indices(self, groups, candidate_indices, refined_times):
        """DLS active-set selection over sparse temporal candidates."""
        cfg = self.cfg["projection"]
        thresholds = self._constraint_thresholds(groups)
        guard_indices = self._sample_constraint_indices(
            groups, {name: set(times) for name, times in refined_times.items()} |
            {"control_rate": set()})
        guard_indices["control_rate"] = candidate_indices["control_rate"].new_empty((0,))
        selected = {}
        counts, total_counts, truncated_counts = {}, {}, {}
        candidate_pool_sizes, candidate_total_counts = {}, {}
        sampling_omitted_counts, topk_truncated_counts, guard_counts = {}, {}, {}
        for name, candidate_pool in candidate_indices.items():
            values = groups[name]
            full_active = torch.nonzero(
                values.detach() > thresholds[name], as_tuple=False).reshape(-1)
            candidate_active = candidate_pool[
                values.detach()[candidate_pool] > thresholds[name][candidate_pool]]
            candidate_total = int(candidate_active.numel())
            cap = int(cfg["max_active_per_group"][name])
            active_selected = candidate_active
            if candidate_total > cap:
                local = torch.topk(
                    values.detach()[candidate_active] - thresholds[name][candidate_active],
                    k=cap, largest=True, sorted=False).indices
                active_selected = candidate_active[local]
            guards = guard_indices[name]
            ids = torch.unique(torch.cat((active_selected, guards)), sorted=True)
            selected[name] = ids
            selected_active_count = int((values.detach()[ids] > thresholds[name][ids]).sum())
            counts[name] = selected_active_count
            total_counts[name] = int(full_active.numel())
            candidate_pool_sizes[name] = int(candidate_pool.numel())
            candidate_total_counts[name] = candidate_total
            sampling_omitted_counts[name] = int(full_active.numel()) - candidate_total
            topk_truncated_counts[name] = candidate_total - int(active_selected.numel())
            truncated_counts[name] = max(int(full_active.numel()) - selected_active_count, 0)
            guard_counts[name] = int(guards.numel())
        diagnostics = {
            "active_counts": counts,
            "active_total_counts": total_counts,
            "active_truncated_counts": truncated_counts,
            "active_candidate_pool_sizes": candidate_pool_sizes,
            "active_candidate_total_counts": candidate_total_counts,
            "active_sampling_omitted_counts": sampling_omitted_counts,
            "active_topk_truncated_counts": topk_truncated_counts,
            "active_guard_counts": guard_counts,
        }
        return selected, diagnostics

    def _full_group_indices(self, groups, candidate_indices):
        """Select every row in the sparse temporal pool for Full-Group QP."""
        thresholds = self._constraint_thresholds(groups)
        selected = {name: indices for name, indices in candidate_indices.items()}
        row_counts = {name: int(indices.numel()) for name, indices in selected.items()}
        candidate_active_counts = {
            name: int((groups[name].detach()[indices] > thresholds[name][indices]).sum().cpu())
            for name, indices in selected.items()
        }
        full_active_counts = {
            name: int((groups[name].detach() > thresholds[name]).sum().cpu())
            for name in selected
        }
        diagnostics = {
            "active_counts": candidate_active_counts,
            "active_total_counts": full_active_counts,
            "active_truncated_counts": {
                name: max(full_active_counts[name] - candidate_active_counts[name], 0)
                for name in selected
            },
            "active_candidate_pool_sizes": row_counts,
            "active_candidate_total_counts": candidate_active_counts,
            "active_sampling_omitted_counts": {
                name: max(full_active_counts[name] - candidate_active_counts[name], 0)
                for name in selected
            },
            "active_topk_truncated_counts": {name: 0 for name in selected},
            "solver_constraint_scope": "all_sparse_rows_per_priority_group",
            "solver_constraint_rows_by_family": row_counts,
            "solver_inactive_rows_by_family": {
                name: row_counts[name] - candidate_active_counts[name] for name in selected
            },
        }
        return selected, diagnostics

    def evaluate_preprojection(self, nominal_controls: torch.Tensor, context: dict,
                               road_geometry) -> dict:
        """Evaluate full-horizon JTP constraints without modifying controls."""
        if (not torch.is_tensor(nominal_controls) or nominal_controls.ndim != 3
                or nominal_controls.shape[-1] != 2):
            raise ValueError("nominal_controls must be [N,K,2]")
        if not nominal_controls.is_floating_point() or not torch.isfinite(nominal_controls).all():
            raise ValueError("nominal_controls must be finite floating-point SI controls")
        n, k, _ = nominal_controls.shape
        if k != self.control_steps:
            raise ValueError(f"Expected K={self.control_steps} control nodes, got {k}")
        self._ensure_execution_device(nominal_controls.device, nominal_controls.dtype)
        ctx = self._to_context(
            context, n=n, device=nominal_controls.device, dtype=nominal_controls.dtype)
        roads = self._road_list(
            road_geometry, n, device=nominal_controls.device, dtype=nominal_controls.dtype)
        with torch.no_grad():
            evaluated = self._evaluate(nominal_controls.detach(), ctx, roads)
        return {
            "controls": nominal_controls.detach(),
            "trajectories": evaluated.rollout["trajectory"].detach(),
            "states": evaluated.rollout["states"].detach(),
            "ltr": evaluated.rollout["ltr"].detach(),
            "projector_constraints_satisfied": bool(
                evaluated.metrics["projector_constraints_satisfied"]),
            "metrics": evaluated.metrics,
            "constraint_groups": {
                key: value.detach() for key, value in evaluated.groups.items()},
        }

    def project(self, nominal_controls: torch.Tensor, context: dict,
                road_geometry: RoadGeometry | list[RoadGeometry]) -> JTPResult:
        """Project each controlled vehicle independently in its own 80-D space."""
        if (not torch.is_tensor(nominal_controls) or nominal_controls.ndim != 3
                or nominal_controls.shape[-1] != 2):
            raise ValueError("nominal_controls must be a torch tensor [N,K,2]")
        if not nominal_controls.is_floating_point() or not torch.isfinite(nominal_controls).all():
            raise ValueError("nominal_controls must be finite floating-point SI controls")
        n, k, _ = nominal_controls.shape
        if k != self.control_steps:
            raise ValueError(f"Expected K={self.control_steps} control nodes, got {k}")
        self._ensure_execution_device(nominal_controls.device, nominal_controls.dtype)
        full_context = self._to_context(
            context, n=n, device=nominal_controls.device, dtype=nominal_controls.dtype)
        roads = self._road_list(
            road_geometry, n, device=nominal_controls.device, dtype=nominal_controls.dtype)
        def synchronized_clock():
            if nominal_controls.is_cuda:
                torch.cuda.synchronize(nominal_controls.device)
            return perf_counter()

        results = []
        for vehicle_index in range(n):
            vehicle_context = {
                key: value[vehicle_index:vehicle_index + 1]
                for key, value in full_context.items()
            }
            start = synchronized_clock()
            result = self._project_single(
                nominal_controls[vehicle_index:vehicle_index + 1],
                vehicle_context, [roads[vehicle_index]])
            result.diagnostics["projection_latency_ms"] = (
                synchronized_clock() - start) * 1000.0
            results.append(result)

        controls = torch.cat([result.controls for result in results], dim=0)
        trajectories = torch.cat([result.trajectories for result in results], dim=0)
        states = torch.cat([result.states for result in results], dim=0)
        ltr = torch.cat([result.ltr for result in results], dim=0)
        diagnostics_by_vehicle = [result.diagnostics for result in results]

        def merge_metrics(key):
            metrics = [item[key] for item in diagnostics_by_vehicle]
            names = metrics[0]["max_violation_by_group"]
            maxima = {name: max(item["max_violation_by_group"][name] for item in metrics)
                      for name in names}
            counts = {name: sum(item["violation_count_by_group"][name] for item in metrics)
                      for name in names}
            l2 = {name: sum(item["violation_l2_by_group"][name] ** 2 for item in metrics) ** 0.5
                  for name in names}
            return {
                "max_violation": max(item["max_violation"] for item in metrics),
                "max_violation_by_group": maxima,
                "violation_count_by_group": counts,
                "violation_l2_by_group": l2,
                "stability_violation_count": sum(item["stability_violation_count"] for item in metrics),
                "geometry_violation_count": sum(item["geometry_violation_count"] for item in metrics),
                "control_box_violation_count": sum(item["control_box_violation_count"] for item in metrics),
                "control_box_feasible": all(item["control_box_feasible"] for item in metrics),
                "control_rate_violation_count": sum(item["control_rate_violation_count"] for item in metrics),
                "projector_constraints_satisfied": all(
                    item["projector_constraints_satisfied"] for item in metrics),
            }

        initial_metrics = merge_metrics("initial_metrics")
        final_metrics = merge_metrics("final_metrics")
        group_names = diagnostics_by_vehicle[0]["group_progress"]
        group_progress = {
            name: {
                metric: sum(item["group_progress"][name][metric] for item in diagnostics_by_vehicle) / n
                for metric in diagnostics_by_vehicle[0]["group_progress"][name]
            }
            for name in group_names
        }
        projection_history = []
        history = []
        for vehicle_index, item in enumerate(diagnostics_by_vehicle):
            projection_history.extend(
                {**step, "vehicle_index": vehicle_index} for step in item["projection"])
            history.extend(
                {**step, "vehicle_index": vehicle_index} for step in item["history"])
        statuses = [result.status for result in results]
        solver_cfg = self.cfg["projection"]
        use_ls = solver_cfg["solver_method"] == "damped_least_squares"
        diagnostics = {
            "accepted_iterations": sum(
                item["accepted_iterations"] for item in diagnostics_by_vehicle) / n,
            "total_accepted_iterations": sum(
                item["accepted_iterations"] for item in diagnostics_by_vehicle),
            "mean_accepted_iterations_per_vehicle": sum(
                item["accepted_iterations"] for item in diagnostics_by_vehicle) / n,
            "history": history,
            "projection": projection_history,
            "objective": sum(item["objective"] for item in diagnostics_by_vehicle),
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "group_progress": group_progress,
            "termination_status": statuses[0] if len(set(statuses)) == 1 else "mixed_vehicle_statuses",
            "status_by_vehicle": statuses,
            "iterations_by_vehicle": [result.iterations for result in results],
            "total_iterations": sum(result.iterations for result in results),
            "final_trust_radius_by_vehicle": [
                item["final_trust_radius"] for item in diagnostics_by_vehicle],
            "trust_region_enabled": use_ls,
            "priority_groups": diagnostics_by_vehicle[0]["priority_groups"],
            "priority_scores": diagnostics_by_vehicle[0]["priority_scores"],
            "solver_method": solver_cfg["solver_method"],
            "solver_constraint_scope": "active_set" if use_ls else "all_sparse_rows_per_priority_group",
            "static_cache_builds": sum(item["static_cache_builds"] for item in diagnostics_by_vehicle),
            "static_cache_ms": sum(item["static_cache_ms"] for item in diagnostics_by_vehicle),
            "linearization_count": sum(item["linearization_count"] for item in diagnostics_by_vehicle),
            "nonlinear_validation_count": sum(item["nonlinear_validation_count"] for item in diagnostics_by_vehicle),
            "constraint_sampling_stride": int(solver_cfg["constraint_sampling_stride"]),
            "constraint_sampling_by_vehicle": [item["constraint_sampling"] for item in diagnostics_by_vehicle],
            "projection_latency_ms_by_vehicle": [
                float(item["projection_latency_ms"]) for item in diagnostics_by_vehicle],
            "mean_vehicle_projection_latency_ms": sum(
                float(item["projection_latency_ms"]) for item in diagnostics_by_vehicle) / n,
            "control_rate_constraints_enabled": True,
            "collision_constraints_enabled": False,
            "acceptance_policy": "hard_control_box_plus_full_horizon_per_constraint_non_degradation_and_group_violation_count_nonincrease",
            "config_path": self.cfg.get("_config_path"),
            "per_vehicle": diagnostics_by_vehicle,
        }
        return JTPResult(
            controls=controls.detach(), trajectories=trajectories.detach(),
            states=states.detach(), ltr=ltr.detach(),
            projector_constraints_satisfied=all(
                result.projector_constraints_satisfied for result in results),
            status=diagnostics["termination_status"],
            iterations=max(result.iterations for result in results),
            diagnostics=diagnostics)

    def _project_single(self, nominal_controls, context, roads):
        nominal = nominal_controls.detach().clone()
        controls = nominal.clone()
        proj_cfg = self.cfg["projection"]
        solver_method = proj_cfg["solver_method"]
        use_ls = solver_method == "damped_least_squares"
        with torch.no_grad():
            current = self._evaluate(controls, context, roads)
        tau_nominal = current.rollout["trajectory"].detach().clone()
        thresholds = self._constraint_thresholds(current.groups)
        order, scores = build_priority_groups(current.groups, proj_cfg, thresholds)
        history = [{"iteration": 0, **current.metrics,
                    "priority_group_maxima": group_maxima(current.metrics, order),
                    "status": "nominal"}]
        initial_metrics = current.metrics
        projection_history = []
        trust = float(proj_cfg["initial_trust_radius"]) if use_ls else None
        damping_scale = 1.0 if use_ls else None
        accepted_iterations = stalled = validations = linearizations = 0
        cache = None
        cache_ms = 0.0
        status = ("nominal_already_feasible" if current.metrics["projector_constraints_satisfied"]
                  else "maximum_iterations_reached")

        def clock():
            if controls.is_cuda:
                torch.cuda.synchronize(controls.device)
            return perf_counter()

        scale = controls.new_tensor(self.cfg["control"]["trust_scale"]) if use_ls else None
        lower = controls.new_tensor([
            self.cfg["control"]["acceleration_min_mps2"],
            self.cfg["control"]["steering_min_rad"]])
        upper = controls.new_tensor([
            self.cfg["control"]["acceleration_max_mps2"],
            self.cfg["control"]["steering_max_rad"]])
        sampled_times = self._initial_sampled_times()
        refined_times = {name: set() for name in ("stability", "road", "curvature")}
        sampling_events = []
        candidate_indices = self._sample_constraint_indices(current.groups, sampled_times)
        if (not current.metrics["projector_constraints_satisfied"]
                and isinstance(self.rollout_fn, KoopmanExecutionAdapter)):
            tick = clock()
            cache = K2StaticCache(self.rollout_fn, context, self.control_interpolation)
            cache_ms = (clock() - tick) * 1000.0

        for iteration in range(1, int(proj_cfg["max_iterations"]) + 1):
            if current.metrics["projector_constraints_satisfied"]:
                break
            additions = self._refine_constraint_indices(
                current.groups, thresholds, sampled_times, refined_times)
            if additions:
                sampling_events.append({"iteration": iteration, "source": "current_full_horizon",
                                        "added_times": additions})
                candidate_indices = self._sample_constraint_indices(current.groups, sampled_times)

            if use_ls:
                selected, active_diagnostics = self._active_indices(
                    current.groups, candidate_indices, refined_times)
                if (not sum(active_diagnostics["active_counts"].values())
                        and current.metrics["control_box_feasible"]):
                    status = "no_active_violations"
                    break
            else:
                selected, active_diagnostics = self._full_group_indices(
                    current.groups, candidate_indices)

            tick = clock()
            with torch.enable_grad():
                rows, metric, method = shared_linearization(
                    self, controls, context, roads, current, selected, cache,
                    build_metric=use_ls)
            linearizations += 1
            diag = {
                "iteration": iteration,
                **active_diagnostics,
                "hard_control_violation_count_before": current.metrics["control_box_violation_count"],
                "linearization_method": method,
                "linearization_ms": (clock() - tick) * 1000.0,
                "trust_radius": trust,
                "trust_region_enabled": use_ls,
                "constraint_sampling_times": {
                    name: sorted(times) for name, times in sampled_times.items()},
            }

            tick = clock()
            if not all(torch.isfinite(jac).all() for jac in rows.values()):
                delta, info = None, {"reason": "nonfinite_constraint_jacobian"}
            else:
                with torch.no_grad():
                    delta, info = solve_groups(
                        controls, current.groups, selected, rows, thresholds, metric, order,
                        proj_cfg, scale, trust, damping_scale, lower, upper,
                        control_weights=self.cfg["optimization"]["control_weights"])
            diag.update(info)
            diag["grouped_solver_ms"] = (clock() - tick) * 1000.0
            diag["priority_group_maxima_before"] = group_maxima(current.metrics, order)
            accepted = False
            accepted_reason = None
            sampling_expanded = False
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
                            associations=current.associations)
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
                        current = self._evaluate(controls, context, roads, rollout=candidate.rollout)
                        accepted = True
                        accepted_reason = acceptance_reason
                        accepted_iterations += 1
                        stalled = 0
                        if use_ls:
                            if alpha >= 0.75:
                                trust = min(float(proj_cfg["maximum_trust_radius"]),
                                            trust * float(proj_cfg["trust_grow_factor"]))
                            elif alpha <= 0.25:
                                trust = max(float(proj_cfg["minimum_trust_radius"]),
                                            trust * float(proj_cfg["trust_shrink_factor"]))
                            damping_scale = max(1.0, damping_scale / 2.0)
                        break

                    added_candidate = self._refine_constraint_indices(
                        candidate.groups, thresholds, sampled_times, refined_times)
                    if added_candidate:
                        sampling_events.append({"iteration": iteration,
                                                "source": "rejected_candidate_full_horizon",
                                                "added_times": added_candidate})
                        candidate_indices = self._sample_constraint_indices(
                            current.groups, sampled_times)
                        sampling_expanded = True
                        break
                    if not hard_feasible_before and not hard_feasible_after:
                        break
                    alpha *= float(proj_cfg["alpha_shrink_factor"])

            diag.update({
                "accepted": accepted,
                "line_search_trials": trials,
                "nonlinear_validation_ms": (clock() - tick) * 1000.0,
                "priority_group_maxima_after": group_maxima(current.metrics, order),
                "max_violation": current.metrics["max_violation"],
                "max_violation_by_group": current.metrics["max_violation_by_group"],
                "candidate_violation_count_by_group": current.metrics["violation_count_by_group"],
                "candidate_violation_l2_by_group": current.metrics["violation_l2_by_group"],
                "acceptance_reason": accepted_reason if accepted else "rejected_or_zero_step",
                "sampling_expanded": sampling_expanded,
            })
            projection_history.append(diag)
            history.append({
                "iteration": iteration,
                **current.metrics,
                "priority_group_maxima": group_maxima(current.metrics, order),
                "status": "accepted" if accepted else "rejected",
            })

            if not use_ls and delta is None:
                status = "solver_failed:" + str(info.get("reason", "unknown"))
                break
            if current.metrics["projector_constraints_satisfied"]:
                status = "converged_feasible"
                break
            if sampling_expanded:
                continue
            if not accepted:
                stalled += 1
                if use_ls:
                    trust *= float(proj_cfg["trust_shrink_factor"])
                    damping_scale *= 10.0
            if use_ls and trust < float(proj_cfg["minimum_trust_radius"]):
                status = "stalled_infeasible:minimum_trust_radius"
                break
            if stalled >= int(proj_cfg["max_stalled_iterations"]):
                status = "stalled_infeasible:no_group_improvement"
                break

        final = current
        objective = float(self._objective(controls, final.rollout, nominal, tau_nominal))
        group_progress = {}
        for name in current.groups:
            progress = {}
            for key, metric_name in (
                    ("max_violation", "max_violation_by_group"),
                    ("violation_count", "violation_count_by_group"),
                    ("violation_l2", "violation_l2_by_group")):
                before = initial_metrics[metric_name][name]
                after = final.metrics[metric_name][name]
                progress.update({
                    "initial_" + key: before,
                    "final_" + key: after,
                    key + "_reduction": before - after,
                })
            group_progress[name] = progress

        sampling_diagnostics = {
            "stride": int(proj_cfg["constraint_sampling_stride"]),
            "initial_sampled_times": sorted(self._initial_sampled_times()["stability"]),
            "final_sampled_times": {
                name: sorted(times) for name, times in sampled_times.items()},
            "refined_times": {name: sorted(times) for name, times in refined_times.items()},
            "sampled_rows_by_family": {
                name: int(indices.numel())
                for name, indices in self._sample_constraint_indices(final.groups, sampled_times).items()},
            "refinement_events": sampling_events,
            "nonlinear_validation": "full_horizon_all_constraint_rows",
        }
        return JTPResult(
            controls=controls.detach(),
            trajectories=final.rollout["trajectory"].detach(),
            states=final.rollout["states"].detach(),
            ltr=final.rollout["ltr"].detach(),
            projector_constraints_satisfied=bool(final.metrics["projector_constraints_satisfied"]),
            status=status,
            iterations=len(projection_history),
            diagnostics={
                "accepted_iterations": accepted_iterations,
                "history": history,
                "projection": projection_history,
                "objective": objective,
                "initial_metrics": initial_metrics,
                "final_metrics": final.metrics,
                "group_progress": group_progress,
                "termination_status": status,
                "final_trust_radius": trust,
                "trust_region_enabled": use_ls,
                "priority_groups": order,
                "priority_scores": scores,
                "solver_method": solver_method,
                "solver_constraint_scope": (
                    "active_set" if use_ls else "all_sparse_rows_per_priority_group"),
                "static_cache_builds": int(cache is not None),
                "static_cache_ms": cache_ms,
                "linearization_count": linearizations,
                "nonlinear_validation_count": validations,
                "constraint_sampling": sampling_diagnostics,
                "control_rate_constraints_enabled": True,
                "collision_constraints_enabled": False,
                "acceptance_policy": (
                    "hard_control_box_plus_full_horizon_per_constraint_non_degradation_"
                    "and_group_violation_count_nonincrease"),
                "config_path": self.cfg.get("_config_path"),
            })
