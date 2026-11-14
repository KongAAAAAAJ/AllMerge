"""Control-space Diffusion inference with optional Joint Trajectory Projection."""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import torch

from highway_env.planner.diffusion.training.control import load_control_checkpoint
from highway_env.planner.diffusion.common.execution import K2_CONTEXT
from highway_env.planner.diffusion.common.tensor_adapter import PlannerTensorAdapter
from highway_env.planner.diffusion.config.experiment import resolve_repo_file


class ControlDiffusionRuntime:
    def __init__(self, checkpoint, device='auto', deterministic_seed=0,
                 inference_steps=None, *, jtp_enabled=False, jtp_config=None):
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available()
                                   else 'cpu' if device == 'auto' else device)
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        self.model, payload = load_control_checkpoint(path, device='cpu')
        self.model.to(self.device).eval()
        self.adapter = PlannerTensorAdapter(self.model.scene_config, self.device)
        self.deterministic_seed = deterministic_seed
        self.inference_steps = inference_steps
        self.jtp_enabled = bool(jtp_enabled)
        self.jtp_projector = None
        self.jtp_config_path = None
        if self.jtp_enabled:
            if self.model.vehicle_model.kind != 'koopman_k2':
                raise ValueError('JTP requires a control-diffusion checkpoint with execution.model=koopman_k2')
            config_path = resolve_repo_file(jtp_config or 'configs/projection/jtp.yaml')
            if not config_path.is_file():
                raise FileNotFoundError(f'JTP configuration not found: {config_path}')
            from highway_env.planner.diffusion.projection import JointTrajectoryProjector
            # Reuse the exact frozen K2 module already loaded with the control
            # diffusion checkpoint. No second Koopman instance is constructed.
            self.jtp_projector = JointTrajectoryProjector(
                self.model.vehicle_model, config=config_path)
            projection_cfg = self.jtp_projector.cfg["execution"]
            model_dt = float(self.model.ctrl_cfg.dt)
            model_horizon = int(self.model.ctrl_cfg.horizon_steps)
            if not np.isclose(float(projection_cfg["dt_s"]), model_dt, atol=1.0e-8):
                raise ValueError(
                    f"JTP execution dt {projection_cfg['dt_s']} does not match "
                    f"control-diffusion dt {model_dt}"
                )
            if (int(projection_cfg["horizon_steps"]) != model_horizon
                    or int(projection_cfg["control_steps"]) != model_horizon):
                raise ValueError(
                    "JTP currently requires identity control interpolation and a "
                    f"matching control horizon; JTP={projection_cfg}, "
                    f"Diffusion horizon={model_horizon}"
                )
            self.jtp_config_path = str(config_path)
        self.last_latency_ms = None
        self.last_diffusion_latency_ms = None
        self.last_jtp_latency_ms = 0.0

    @staticmethod
    def _context_from_features(numpy_features, device):
        absent = [key for key in K2_CONTEXT if key not in numpy_features]
        if absent:
            raise KeyError(f'K2 runtime requires synchronized observed context: {absent}')
        return {key: torch.as_tensor(numpy_features[key], dtype=torch.float32, device=device)
                for key in K2_CONTEXT}

    @staticmethod
    def _wrap_angle(value):
        return torch.atan2(torch.sin(value), torch.cos(value))

    @classmethod
    def _apply_jtp_result(cls, prediction, jtp_result, context, dt_s=0.1):
        """Replace selected nominal output with the verified JTP solution.

        JTP returns world-frame poses; the public Diffusion runtime contract is
        ego-at-planning-time local coordinates. The conversion keeps the
        existing action publisher and visualization API unchanged.
        """
        controls = jtp_result.controls.to(prediction['control'])
        world_pose = jtp_result.trajectories.to(prediction['pose'])
        states = jtp_result.states.to(prediction['pose'])
        ltr = jtp_result.ltr.to(prediction['pose'])
        pose0 = context['koopman_pose0_si'].to(world_pose)
        dx = world_pose[..., 0] - pose0[:, None, 0]
        dy = world_pose[..., 1] - pose0[:, None, 1]
        c0 = torch.cos(pose0[:, 2])[:, None]
        s0 = torch.sin(pose0[:, 2])[:, None]
        local_x = c0 * dx + s0 * dy
        local_y = -s0 * dx + c0 * dy
        local_yaw = cls._wrap_angle(world_pose[..., 2] - pose0[:, None, 2])
        local_pose = torch.stack((local_x, local_y, local_yaw), dim=-1)

        future_pose = local_pose[:, 1:]
        stride = max(1, int(round(0.5 / float(dt_s))))
        prediction['control'] = controls
        prediction['pose'] = local_pose
        prediction['trajectory_dense'] = future_pose[..., :2]
        prediction['trajectory'] = future_pose[:, stride - 1::stride, :2]
        prediction['states'] = states
        prediction['ltr'] = ltr
        prediction['speeds'] = states[..., 0]
        return prediction

    @staticmethod
    def _add_dense_derivatives(prediction, dt_s=0.1):
        dense = prediction['trajectory_dense']
        if dense.ndim != 3 or dense.shape[-1] != 2:
            raise ValueError('Control runtime trajectory_dense must be [B,H,2]')
        states = prediction.get('states')
        pose = prediction.get('pose')
        if states is not None and pose is not None:
            yaw = pose[:, 1:, 2]
            vx_body = states[:, 1:, 0]
            vy_body = states[:, 1:, 1]
            vx_local = vx_body * torch.cos(yaw) - vy_body * torch.sin(yaw)
            vy_local = vx_body * torch.sin(yaw) + vy_body * torch.cos(yaw)
            velocity = torch.stack((vx_local, vy_local), dim=-1)
        else:
            origin = torch.zeros_like(dense[:, :1])
            chain = torch.cat((origin, dense), dim=1)
            velocity = (chain[:, 1:] - chain[:, :-1]) / float(dt_s)
        heading = torch.atan2(velocity[..., 1], velocity[..., 0])
        prediction['trajectory_dense_velocity'] = velocity
        prediction['trajectory_dense_heading'] = heading
        return prediction

    @torch.no_grad()
    def infer(self, numpy_features, *, jtp_inputs=None):
        features = self.adapter.to_torch(numpy_features)
        generator = None
        if self.deterministic_seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(self.deterministic_seed))
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        total_start = time.perf_counter()

        context = {}
        if self.model.vehicle_model.kind == 'koopman_k2':
            context = self._context_from_features(numpy_features, self.device)
        diffusion_start = time.perf_counter()
        prediction = self.model.infer_multimodal(
            features, generator=generator, num_steps=self.inference_steps, context=context)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.last_diffusion_latency_ms = (time.perf_counter() - diffusion_start) * 1000.0

        nominal_control = prediction['control'].detach().clone()
        nominal_dense = prediction['trajectory_dense'].detach().clone()
        jtp_success = False
        jtp_status = 'disabled' if not self.jtp_enabled else 'not_run'
        jtp_error = None
        jtp_diagnostics = {}
        jtp_iterations = 0
        self.last_jtp_latency_ms = 0.0

        if self.jtp_enabled:
            if not isinstance(jtp_inputs, dict) or not jtp_inputs.get('ready', False):
                raise RuntimeError(
                    'JTP is enabled but synchronized live K2/road context is not ready. '
                    'The environment must warm up five 10-Hz state samples before inference.'
                )
            if 'road_geometry' not in jtp_inputs:
                raise KeyError('JTP runtime inputs are missing road_geometry')
            jtp_start = time.perf_counter()
            try:
                projected = self.jtp_projector.project(
                    nominal_controls=nominal_control,
                    context=context,
                    road_geometry=jtp_inputs['road_geometry'],
                    external_traffic=jtp_inputs.get('external_traffic'),
                )
                jtp_status = str(projected.status)
                jtp_success = bool(projected.success)
                jtp_iterations = int(projected.iterations)
                jtp_diagnostics = dict(projected.diagnostics or {})
                if jtp_success:
                    prediction = self._apply_jtp_result(
                        prediction, projected, context,
                        dt_s=float(self.model.ctrl_cfg.dt))
                # On failure, preserve the unmodified nominal Diffusion output.
                # The caller receives an explicit failure status and can use
                # its existing controller rather than treating this path safe.
            except Exception as exc:
                jtp_status = 'projection_exception'
                jtp_error = f'{type(exc).__name__}: {exc}'
                jtp_success = False
            self.last_jtp_latency_ms = (time.perf_counter() - jtp_start) * 1000.0

        self._add_dense_derivatives(prediction, dt_s=float(self.model.ctrl_cfg.dt))
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.last_latency_ms = (time.perf_counter() - total_start) * 1000.0

        result = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                  for k, v in prediction.items()}
        result['control_nominal'] = nominal_control.cpu().numpy()
        result['trajectory_dense_nominal'] = nominal_dense.cpu().numpy()
        result['jtp_enabled'] = np.asarray(self.jtp_enabled, dtype=np.bool_)
        result['jtp_success'] = np.asarray(jtp_success, dtype=np.bool_)
        result['jtp_applied'] = np.asarray(jtp_success, dtype=np.bool_)
        result['jtp_status'] = jtp_status
        result['jtp_error'] = jtp_error
        result['jtp_diagnostics'] = jtp_diagnostics
        result['jtp_iterations'] = np.asarray(jtp_iterations, dtype=np.int32)
        final_metrics = jtp_diagnostics.get('final_metrics', {})
        result['jtp_max_violation'] = np.asarray(
            float(final_metrics.get('max_violation', np.nan)), dtype=np.float32)
        result['jtp_control_delta_norm'] = np.asarray(
            float(torch.linalg.vector_norm(prediction['control'] - nominal_control).cpu()),
            dtype=np.float32)
        result['jtp_latency_ms'] = np.asarray(self.last_jtp_latency_ms, dtype=np.float32)
        result['diffusion_latency_ms'] = np.asarray(self.last_diffusion_latency_ms, dtype=np.float32)
        result['latency_ms'] = np.asarray(self.last_latency_ms, dtype=np.float32)
        result['trajectory_time_s'] = np.arange(1, 9, dtype=np.float32) * .5
        result['trajectory_dense_time_s'] = (
            np.arange(1, int(self.model.ctrl_cfg.horizon_steps) + 1, dtype=np.float32)
            * float(self.model.ctrl_cfg.dt))
        return result
