"""TRACE control-space stochastic diffusion planner with shared mode selection.

Control candidates are refined by DDIM and rolled out by a kinematic bicycle
or frozen Koopman K2 execution model. Train and inference select the same valid
goal-endpoint candidate. Expert XY, LTR, fuel, and progress losses supervise
the chosen execution; no pseudo expert controls are used.

Without clean control labels the objective is trajectory-trained iterative
diffusion refinement, not standard DDPM noise-prediction supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from highway_env.planner.diffusion.config.model import build_structured_diffusion_config
from highway_env.planner.diffusion.models.schedule import TruncatedDDIMSchedule
from highway_env.planner.diffusion.common.encoders import StructuredSceneEncoder
from highway_env.planner.diffusion.common.tensor_adapter import PlannerTensorAdapter
from .blocks import CrossAttentionBlock, SinusoidalTimeEmbedding
from highway_env.planner.diffusion.common.execution import VehicleExecution
from highway_env.planner.diffusion.common.fuel import FuelParameters, vt_cpfm_from_states
from highway_env.planner.diffusion.common.progress import centerline_progress
from highway_env.planner.diffusion.common.anchor_controls import xy_anchors_to_controls


@dataclass
class ControlSpaceConfig:
    horizon_steps: int = 40
    dt: float = 0.1
    wheelbase_m: float = 5.5
    max_accel_mps2: float = 2.5
    max_steer_rad: float = 0.08
    num_train_timesteps: int = 1000
    inference_steps: int = 2           # Match trajectory runtime [8,0] by default.
    inference_start_timestep: int = 8
    inference_noise_scale: float = 1.0
    position_scale_m: float = 10.0
    loss_position: float = 1.0         # Only supervised objective.

    def validate(self):
        if self.horizon_steps != 40 or abs(self.dt - 0.1) > 1e-9:
            raise ValueError('Control-space D1 requires 40 steps at dt=0.1 s')
        if min(self.wheelbase_m, self.max_accel_mps2,
               self.max_steer_rad, self.position_scale_m, self.loss_position) <= 0:
            raise ValueError('wheelbase, limits and trajectory loss scaling must be positive')
        if not 2 <= self.inference_steps <= self.inference_start_timestep + 1:
            raise ValueError('inference_steps must be in [2, start_timestep+1]')
        if not 1 <= self.inference_start_timestep < self.num_train_timesteps:
            raise ValueError('inference_start_timestep must be within (0,num_train_timesteps)')
        if self.inference_noise_scale < 0:
            raise ValueError('inference_noise_scale must be nonnegative')


class DirectControlDenoiser(nn.Module):
    """Conditioned x0 control head for multimodal DDIM-style refinement."""

    def __init__(self, scene_config, steps):
        super().__init__()
        dim = scene_config.d_model
        self.modes = scene_config.num_modes
        self.steps = steps
        self.noisy_encoder = nn.Sequential(
            nn.Linear(steps * 2, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim))
        self.mode_embedding = nn.Embedding(self.modes, dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(dim), nn.Linear(dim, 4 * dim),
            nn.SiLU(), nn.Linear(4 * dim, dim))
        self.layers = nn.ModuleList(CrossAttentionBlock(scene_config)
                                    for _ in range(scene_config.num_denoiser_layers))
        self.control_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, scene_config.d_ffn), nn.GELU(),
            nn.Linear(scene_config.d_ffn, steps * 2))

    def forward(self, noisy, t, scene):
        B, M, H, D = noisy.shape
        if (M, H, D) != (self.modes, self.steps, 2):
            raise ValueError('Expected [B,num_modes,40,2] controls')
        tokens = (self.noisy_encoder(noisy.flatten(2)) +
                  self.mode_embedding(torch.arange(M, device=noisy.device))[None])
        t_emb = self.time_embedding(t)
        for layer in self.layers:
            tokens = layer(tokens, scene, t_emb)
        return torch.tanh(self.control_head(tokens).reshape(B, M, H, 2))


class ControlDiffusionPlanner(nn.Module):
    """Control-space diffusion with goal-endpoint selection in train and inference.

    All candidate controls are rolled out by the chosen execution backend. The
    same valid-mode-filtered goal-endpoint selector chooses the executed mode
    for trajectory supervision, auxiliary objectives, and online inference.
    """

    def __init__(self, model_config=None, control_config=None, *,
                 execution_config=None, loss_config=None, koopman_metadata=None):
        super().__init__()
        self.scene_config = build_structured_diffusion_config(**(model_config or {}))
        self.ctrl_cfg = ControlSpaceConfig(**(control_config or {}))
        self.ctrl_cfg.validate()
        self.adapter = PlannerTensorAdapter(self.scene_config, torch.device('cpu'))
        self.scene_encoder = StructuredSceneEncoder(self.scene_config, self.adapter)
        self.denoiser = DirectControlDenoiser(self.scene_config, self.ctrl_cfg.horizon_steps)
        self.schedule = TruncatedDDIMSchedule(self.ctrl_cfg.num_train_timesteps)
        self.register_buffer('control_limits', torch.tensor(
            [self.ctrl_cfg.max_accel_mps2, self.ctrl_cfg.max_steer_rad], dtype=torch.float32))
        self.execution_config = dict(execution_config or {'model': 'kinematic'})
        self.loss_config = dict(loss_config or {})
        self.loss_config.setdefault('mode_selection', 'goal_endpoint')
        self.vehicle_model = VehicleExecution(
            self.execution_config, dt=self.ctrl_cfg.dt,
            wheelbase_m=self.ctrl_cfg.wheelbase_m,
            koopman_metadata=koopman_metadata,
        )
        self._validate_objectives()

    def _encode_scene(self, features):
        device = features['ego_state'].device
        for attr in ('ego_scale', 'agent_scale', 'map_scale', 'target_point_scale'):
            val = getattr(self.adapter, attr)
            if val.device != device:
                setattr(self.adapter, attr, val.to(device))
        return self.scene_encoder(features)

    def _anchor_controls(self, features):
        """Use the same dynamic XY anchors as trajectory diffusion.

        Current AllMerge anchor_builder saves XY (0.5 s) but no paired
        a/delta history. Reconstruct bounded 10-Hz control anchors once per
        forward pass; no ground-truth future labels are involved.
        """
        anchors = features['coarse_trajectories']
        expected = (features['ego_state'].shape[0], self.scene_config.num_modes,
                    self.scene_config.horizon_steps, 2)
        if tuple(anchors.shape) != expected:
            raise ValueError(f'coarse_trajectories: expected {expected}, got {tuple(anchors.shape)}')
        controls = xy_anchors_to_controls(
            anchors, features['ego_state'][:, 0],
            sparse_dt=self.scene_config.trajectory_dt,
            dense_dt=self.ctrl_cfg.dt,
            wheelbase_m=self.ctrl_cfg.wheelbase_m,
            max_accel_mps2=self.ctrl_cfg.max_accel_mps2,
            max_steer_rad=self.ctrl_cfg.max_steer_rad,
        )
        if controls.shape != (expected[0], expected[1], self.ctrl_cfg.horizon_steps, 2):
            raise ValueError('XY anchor duration must match control horizon')
        # Geometrically absent modes carry zero XY data. Give those inactive
        # slots an actual valid reference rather than an impossible zero-path
        # reconstruction, while retaining the unchanged mode_valid_mask.
        valid = features['mode_valid_mask'].bool()
        if tuple(valid.shape) != expected[:2] or not valid.any(dim=1).all():
            raise ValueError('At least one valid anchor per sample is required')
        first_valid = valid.to(torch.int64).argmax(dim=1)
        fallback = controls[torch.arange(expected[0], device=controls.device), first_valid]
        return torch.where(valid[:, :, None, None], controls, fallback[:, None])

    def _sample_controls(self, features, *, generator=None, num_steps=None):
        """Train/runtime-shared *truncated anchor-noised* control DDIM.

        It now mirrors trajectory-space initialization: x_t = sqrt(alpha_t)
        * anchor + sqrt(1-alpha_t) * Gaussian noise, with t=8 by default.
        Only the variable representation (a,delta vs XY) differs at this stage.
        """
        cfg = self.ctrl_cfg
        B = features['ego_state'].shape[0]
        seed_physical = self._anchor_controls(features)
        seed_norm = seed_physical / self.control_limits.to(seed_physical)
        noise = torch.randn(seed_norm.shape, device=seed_norm.device,
                            dtype=seed_norm.dtype, generator=generator)
        scene = self._encode_scene(features)
        count = int(num_steps if num_steps is not None else cfg.inference_steps)
        start_t = int(cfg.inference_start_timestep)
        if not 2 <= count <= start_t + 1:
            raise ValueError('Invalid truncated DDIM inference step count')
        ts = torch.linspace(start_t, 0, count, device=noise.device).round().long().tolist()
        timesteps = torch.full((B,), start_t, device=noise.device, dtype=torch.long)
        sample = self.schedule.add_noise(
            seed_norm, noise * float(cfg.inference_noise_scale), timesteps)
        for j, t in enumerate(ts):
            x0 = self.denoiser(sample,
                               torch.full((B,), t, device=sample.device, dtype=torch.long),
                               scene)
            next_t = ts[j + 1] if j + 1 < len(ts) else None
            sample = self.schedule.step_predict_x0(sample, x0, t, next_t)
        return sample * self.control_limits

    def _validate_objectives(self):
        cfg = self.loss_config
        if cfg.get('mode_selection') != 'goal_endpoint':
            raise ValueError('Control diffusion training mode_selection must be goal_endpoint')
        trajectory = cfg.get('trajectory', {})
        if trajectory.get('type', 'legacy_smooth_l1') not in ('legacy_smooth_l1', 'paper_mse'):
            raise ValueError('trajectory.type: legacy_smooth_l1 or paper_mse')
        for name in ('trajectory', 'ltr', 'fuel', 'progress'):
            weight = float(cfg.get(name, {}).get('weight',
                self.ctrl_cfg.loss_position if name == 'trajectory' else 0.0))
            if weight < 0:
                raise ValueError(f'{name}.weight must be nonnegative')
        if self._weight('ltr') > 0:
            if self.vehicle_model.kind != 'koopman_k2':
                raise ValueError('LTR penalty requires Koopman K2; kinematic model has no LTR')
            limit = cfg.get('ltr', {}).get('limit')
            if limit is None or not 0 < float(limit) < 1:
                raise ValueError('LTR limit must be explicitly set in (0,1)')
        if self._weight('fuel') > 0:
            self._fuel_parameters()  # Fail fast on missing calibration.
        if self._weight('progress') > 0:
            minimum = cfg.get('progress', {}).get('minimum_progress_m')
            if minimum is None or float(minimum) < 0:
                raise ValueError('progress.minimum_progress_m must be >=0')

    def _weight(self, name):
        default = self.ctrl_cfg.loss_position if name == 'trajectory' else 0.0
        return float(self.loss_config.get(name, {}).get('weight', default))

    def _fuel_parameters(self):
        values = self.loss_config.get('fuel', {}).get('parameters')
        if not isinstance(values, dict):
            raise ValueError('Positive fuel weight requires calibrated fuel.parameters')
        return FuelParameters(**values)

    def _all_predictions(self, controls, features, context=None):
        return self.vehicle_model(controls, features, context)

    def _all_poses(self, controls, features, context=None):
        # Legacy helper retained for external D1 callers.
        result = self._all_predictions(controls, features, context)
        return result['poses'], result['speeds']

    @staticmethod
    def _select_goal_endpoint_mode(poses_all, features):
        """Shared train/inference selector: valid terminal XY nearest target_point.

        The score is unnormalized negative squared physical distance (m^2).
        Ties use torch.argmax's first-index rule in both phases. The discrete
        mode index itself is not differentiable; gradients flow through the
        selected trajectory and its executed vehicle states.
        """
        valid = features['mode_valid_mask'].bool()
        if valid.shape != poses_all.shape[:2]:
            raise ValueError('mode_valid_mask must be [B,num_modes]')
        if not valid.any(dim=1).all():
            raise ValueError('Every episode needs at least one valid mode')
        target = features['target_point'][:, None, :]
        scores = -(poses_all[:, :, -1, :2] - target).square().sum(dim=-1)
        selected = scores.masked_fill(~valid, torch.finfo(scores.dtype).min).argmax(dim=1)
        return selected, scores

    def forward_train(self, features, target_dense_xy, target_mode=None, *,
                      generator=None, context=None):
        """Train on the same goal-endpoint-selected mode used by inference.

        Target XY supplies trajectory supervision after selection only. The
        choice of mode depends on observed target_point, predicted endpoint,
        and mode_valid_mask; it never uses future expert XY as an oracle.
        Auxiliary losses use the same selected mode as trajectory supervision.
        """
        cfg = self.ctrl_cfg
        B = features['ego_state'].shape[0]
        if target_dense_xy.shape != (B, cfg.horizon_steps, 2):
            raise ValueError('Expert trajectory_dense must be [B,40,2]')
        controls_all = self._sample_controls(features, generator=generator)
        pred = self._all_predictions(controls_all, features, context)
        poses_all, speeds_all = pred['poses'], pred['speeds']
        predicted_xy = poses_all[:, :, 1:, :2]
        truth = target_dense_xy[:, None].expand_as(predicted_xy)
        if self.loss_config.get('trajectory', {}).get('type', 'legacy_smooth_l1') == 'paper_mse':
            # Paper: (1/H) sum_j ||p_j - p^GT_j||_2^2 in physical m^2.
            errors = (predicted_xy - truth).square().sum(dim=-1).mean(dim=-1)
        else:
            # Bit-for-bit equivalent to the original D1 SmoothL1 regression.
            errors = F.smooth_l1_loss(predicted_xy / cfg.position_scale_m,
                truth / cfg.position_scale_m, reduction='none').mean(dim=(-2, -1))
        selected, _ = self._select_goal_endpoint_mode(poses_all, features)
        selected_error = errors.gather(1, selected[:, None]).squeeze(1)
        batch_ix = torch.arange(B, device=controls_all.device)
        selected_controls = controls_all[batch_ix, selected]
        selected_poses = poses_all[batch_ix, selected]
        selected_speeds = speeds_all[batch_ix, selected]
        z = selected_error.new_zeros(())
        ltr_loss, fuel_loss, progress_loss = z, z, z
        fuel_total = None
        if self._weight('ltr') > 0:
            selected_ltr = pred['ltr'][batch_ix, selected, 1:]
            limit = float(self.loss_config['ltr']['limit'])
            ltr_loss = torch.relu(selected_ltr.abs() - limit).square().mean()
        if self._weight('fuel') > 0:
            if context is None or 'koopman_payload_kg' not in context:
                raise KeyError('Fuel loss needs physical koopman_payload_kg [B] in expert context')
            payload = context['koopman_payload_kg'].to(selected_speeds).reshape(B)
            if (payload < 0).any() or not torch.isfinite(payload).all():
                raise ValueError('Invalid fuel payload_kg')
            fuel_result = vt_cpfm_from_states(
                selected_speeds, payload, self._fuel_parameters(), dt=cfg.dt)
            fuel_total = fuel_result['fuel_total']
            fuel_loss = fuel_total.mean()
        if self._weight('progress') > 0:
            # Paper: road progress along the reference target lane (not ego x).
            min_progress = float(self.loss_config['progress']['minimum_progress_m'])
            delta_s = centerline_progress(selected_poses[:, -1], features['target_lane_polyline'])
            progress_loss = torch.relu(min_progress - delta_s).square().mean()
        trajectory_loss = selected_error.mean()
        total = (self._weight('trajectory') * trajectory_loss +
                 self._weight('ltr') * ltr_loss +
                 self._weight('fuel') * fuel_loss +
                 self._weight('progress') * progress_loss)
        output = {
            'loss': total, 'position_loss': trajectory_loss,
            'trajectory_loss': trajectory_loss, 'ltr_loss': ltr_loss,
            'fuel_loss': fuel_loss, 'progress_loss': progress_loss,
            'controls': selected_controls, 'poses': selected_poses,
            'speeds': selected_speeds, 'selected_mode': selected,
        }
        if pred['ltr'] is not None:
            output['ltr'] = pred['ltr'][batch_ix, selected]
            output['states'] = pred['states'][batch_ix, selected]
        if fuel_total is not None:
            output['fuel_total'] = fuel_total
        return output

    def infer_multimodal(self, features, generator: Optional[torch.Generator] = None,
                         num_steps: Optional[int] = None, context=None):
        cfg = self.ctrl_cfg
        controls_all = self._sample_controls(features, generator=generator,
                                             num_steps=num_steps)
        pred = self._all_predictions(controls_all, features, context)
        poses_all, speeds_all = pred['poses'], pred['speeds']
        mode, scores = self._select_goal_endpoint_mode(poses_all, features)
        idx = torch.arange(controls_all.shape[0], device=controls_all.device)
        controls = controls_all[idx, mode]
        poses = poses_all[idx, mode]
        speeds = speeds_all[idx, mode]
        future_xy = poses[:, 1:, :2]
        result = {
            'control': controls,
            'control_candidates': controls_all,
            'pose': poses,
            'trajectory_dense': future_xy,
            'trajectory': future_xy[:, 4::5, :],
            'trajectory_candidates': poses_all[:, :, 5::5, :2],
            'trajectory_mode_logits': scores,
            'trajectory_mode_idx': mode,
            'speeds': speeds,
        }
        if pred['ltr'] is not None:
            result['ltr'] = pred['ltr'][idx, mode]
            result['states'] = pred['states'][idx, mode]
        return result
