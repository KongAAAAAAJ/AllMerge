"""Single-QP stability projection with nonlinear verification."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from os import PathLike
from typing import Any

import torch

from highway_env.planner.diffusion.common.execution import K2_CONTEXT, VehicleExecution

from .config import load_jtp_config, validate_jtp_config
from .constraints import stability_residual
from .execution_adapter import KoopmanExecutionAdapter
from .types import JTPResult
from .linearization import K2StaticCache, shared_linearization
from .qp_solver import solve_qp


@dataclass
class _Evaluation:
    rollout: dict
    groups: dict[str, torch.Tensor]
    metrics: dict


class JointTrajectoryProjector:
    """Stability-only QP projection with hard control bounds and full-horizon verification."""

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

    def _bounds(self, controls):
        c = self.cfg['control']
        return (controls.new_tensor([c['acceleration_min_mps2'], c['steering_min_rad']]),
                controls.new_tensor([c['acceleration_max_mps2'], c['steering_max_rad']]))

    def _stability(self, rollout, context):
        scale = rollout['states'].new_tensor(self.cfg['projection']['residual_scales']['stability'])
        return torch.cat([stability_residual(rollout['states'][i], rollout['ltr'][i],
                    context['koopman_payload_kg'][i], context['koopman_road_mu'][i], self.cfg)
                    .reshape(2, self.horizon, 4).div(scale).reshape(-1)
                    for i in range(len(rollout['states']))])

    def _evaluate(self, controls, context):
        rollout = self._rollout(controls, context)
        stability = self._stability(rollout, context)
        lower, upper = self._bounds(controls)
        scale = controls.new_tensor(self.cfg['control']['trust_scale'])
        control = torch.cat(((lower - controls) / scale, (controls - upper) / scale)).reshape(-1)
        excess = (stability - self.cfg['projection']['constraint_tolerance']).clamp_min(0.)
        metrics = dict(max_violation=float(excess.max()),
                       stability_violation_count=int((excess > 0).sum()),
                       control_box_feasible=bool((control <= 0).all()),
                       projector_constraints_satisfied=bool((excess <= 0).all() and (control <= 0).all()))
        return _Evaluation(rollout, {'stability': stability, 'control': control}, metrics)

    def project(self, nominal_controls, context):
        controls = nominal_controls.detach()
        self._ensure_execution_device(controls.device, controls.dtype)
        context = self._to_context(context, n=len(controls), device=controls.device, dtype=controls.dtype)
        results = [self._project_single(controls[i:i+1], {k: v[i:i+1] for k,v in context.items()})
                   for i in range(len(controls))]
        diagnostics = [r.diagnostics for r in results]
        statuses = [r.status for r in results]
        return JTPResult(torch.cat([r.controls for r in results]),
                         torch.cat([r.trajectories for r in results]),
                         torch.cat([r.states for r in results]), torch.cat([r.ltr for r in results]),
                         all(r.projector_constraints_satisfied for r in results),
                         statuses[0] if len(set(statuses)) == 1 else 'mixed_vehicle_statuses',
                         max(r.iterations for r in results),
                         dict(per_vehicle=diagnostics, status_by_vehicle=statuses,
                              iterations_by_vehicle=[r.iterations for r in results],
                              optimization_time_ms_by_vehicle=[d['optimization_time_ms'] for d in diagnostics]))

    def _project_single(self, nominal, context):
        cfg = self.cfg['projection']
        controls = nominal.clone()
        with torch.no_grad():
            current = self._evaluate(controls, context)
        initial = dict(current.metrics)
        history, accepted, stalled, iteration = [], 0, 0, 0
        status = 'nominal_already_feasible' if initial['projector_constraints_satisfied'] else 'maximum_iterations_reached'
        cache = None
        def clock():
            if controls.is_cuda:
                torch.cuda.synchronize(controls.device)
            return perf_counter()
        if not initial['projector_constraints_satisfied']:
            if isinstance(self.rollout_fn, KoopmanExecutionAdapter):
                cache = K2StaticCache(self.rollout_fn, context, self.control_interpolation)
            lower, upper = self._bounds(controls)
            for iteration in range(1, int(cfg['max_iterations']) + 1):
                values = current.groups['stability']
                threshold = torch.full_like(values, cfg['constraint_tolerance'])
                mask = torch.zeros_like(values, dtype=torch.bool).reshape(2, self.horizon, 4)
                mask[:, int(cfg['constraint_sampling_stride'])-1::int(cfg['constraint_sampling_stride'])] = True
                mask[:, -1] = True
                selected = torch.nonzero(mask.reshape(-1) | (values > threshold)).reshape(-1)
                tick = clock()
                with torch.enable_grad():
                    rows = shared_linearization(self, controls, context, current, selected, cache)
                linear_ms = (clock() - tick) * 1000
                delta, info = solve_qp(controls, values, selected, rows, threshold, cfg, lower, upper,
                                       self.cfg['optimization']['control_weights'])
                record = dict(iteration=iteration, linearization_ms=linear_ms, **info,
                              accepted=False, line_search_trials=[])
                history.append(record)
                if delta is None:
                    status = 'solver_failed:' + info['solver_status']
                    break
                tick = clock()
                alpha = float(cfg['alpha_initial'])
                before = (values - threshold).clamp_min(0.)
                while alpha >= cfg['alpha_minimum']:
                    candidate_controls = controls + alpha * delta
                    with torch.no_grad():
                        candidate = self._evaluate(candidate_controls, context)
                    after = (candidate.groups['stability'] - threshold).clamp_min(0.)
                    safe = bool((after <= before + cfg['nondegradation_tolerance']).all())
                    counts_ok = int((after > 0).sum()) <= int((before > 0).sum())
                    progress = bool((before - after > cfg['minimum_violation_improvement']).any()) or int((after > 0).sum()) < int((before > 0).sum())
                    restore = not current.metrics['control_box_feasible'] and candidate.metrics['control_box_feasible']
                    ok = candidate.metrics['control_box_feasible'] and safe and counts_ok and (progress or restore)
                    record['line_search_trials'].append(dict(alpha=alpha, accepted=ok,
                        stability_nondegradation=safe, stability_counts_nonincreasing=counts_ok,
                        control_box_feasible=candidate.metrics['control_box_feasible']))
                    if ok:
                        controls, current = candidate_controls.detach(), candidate
                        accepted += 1
                        stalled = 0
                        record['accepted'] = True
                        break
                    alpha *= cfg['alpha_shrink_factor']
                record['nonlinear_validation_ms'] = (clock() - tick) * 1000
                if current.metrics['projector_constraints_satisfied']:
                    status = 'converged_feasible'
                    break
                if not record['accepted']:
                    stalled += 1
                if stalled >= cfg['max_stalled_iterations']:
                    status = 'stalled_infeasible:no_improvement'
                    break
        optimization_ms = 1000 * sum(step['solver_solve_time_s'] for step in history)
        diagnostics = dict(initial_metrics=initial, final_metrics=current.metrics,
                           projection=history, accepted_iterations=accepted,
                           optimization_time_ms=optimization_ms, termination_status=status)
        return JTPResult(controls, current.rollout['trajectory'].detach(),
                         current.rollout['states'].detach(), current.rollout['ltr'].detach(),
                         current.metrics['projector_constraints_satisfied'], status, iteration, diagnostics)
