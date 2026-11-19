"""TRACE control-space stochastic diffusion planner with shared mode selection.

Control candidates are refined by DDIM and rolled out by frozen Koopman K2.
Training selects the same valid goal-endpoint candidate used at inference and
optimizes expert trajectory, collision, TTC, road-boundary, LTR, fuel and
progress objectives in execution space.

Without clean control labels the objective is trajectory-trained iterative
diffusion refinement, not standard DDPM noise-prediction supervision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from highway_env.planner.diffusion.config.model import build_structured_diffusion_config
from highway_env.planner.diffusion.models.schedule import TruncatedDDIMSchedule
from highway_env.planner.diffusion.common.encoders import StructuredSceneEncoder
from highway_env.planner.diffusion.common.tensor_adapter import PlannerTensorAdapter
from .blocks import CrossAttentionBlock, SinusoidalTimeEmbedding
from highway_env.planner.diffusion.common.execution import VehicleExecution
from highway_env.planner.diffusion.common.fuel import FuelParameters, vt_cpfm_from_states
from highway_env.planner.diffusion.common.progress import centerline_progress
from highway_env.planner.diffusion.common.safety_losses import (
    background_constant_velocity_poses,
    pairwise_vehicle_safety,
    road_boundary_safety_loss,
)


@dataclass
class ControlSpaceConfig:
    horizon_steps: int = 40
    dt: float = 0.1
    wheelbase_m: float = 5.5
    max_accel_mps2: float = 2.5
    max_steer_rad: float = 0.08
    num_train_timesteps: int = 1000
    inference_steps: int = 2           # Match truncated trajectory DDIM [8,0].
    inference_start_timestep: int = 8
    inference_noise_scale: float = 1.0
    def validate(self):
        if self.horizon_steps != 40 or abs(self.dt - 0.1) > 1e-9:
            raise ValueError('TRACE control diffusion requires 40 steps at dt=0.1 s')
        if min(self.wheelbase_m, self.max_accel_mps2, self.max_steer_rad) <= 0:
            raise ValueError('wheelbase and control limits must be positive')
        if not 2 <= self.inference_steps <= self.inference_start_timestep + 1:
            raise ValueError('inference_steps must be in [2, inference_start_timestep+1]')
        if not 0 <= self.inference_start_timestep < self.num_train_timesteps:
            raise ValueError('Invalid inference_start_timestep')
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
        tokens = (self.noisy_encoder(noisy.flatten(2)))
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
                 execution_config=None, loss_config=None, koopman_metadata=None, anchors_config=None):
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
        self.anchors_config = dict(anchors_config or {'strategy': 'a1'})
        anchor_strategy = self.anchors_config.get('strategy', 'a1')
        if anchor_strategy not in ('adaptive', 'fixed', 'a0', 'a1', 'a2'):
            raise ValueError('Control anchor strategy must be a0/fixed, a1 or a2/adaptive')
        self.anchors_config["num_selected"] = int(self.scene_config.num_modes)
        self.anchor_manager = None
        if anchor_strategy in ('adaptive', 'a2'):
            from highway_env.planner.anchor_scoring.manager import AdaptiveAnchorManager
            self.anchor_manager = AdaptiveAnchorManager(
                self.anchors_config, num_selected=self.scene_config.num_modes
            )
            self.anchors_config = dict(self.anchor_manager.config)
        elif anchor_strategy in ('fixed', 'a0'):
            from highway_env.planner.fixed_anchor_manager import FixedAnchorManager
            self.anchor_manager = FixedAnchorManager(
                self.anchors_config,
                num_modes=self.scene_config.num_modes,
                dt=self.ctrl_cfg.dt,
                wheelbase_m=self.ctrl_cfg.wheelbase_m,
                max_accel_mps2=self.ctrl_cfg.max_accel_mps2,
                max_steer_rad=self.ctrl_cfg.max_steer_rad,
            )
            # Persist the exact K-means control-bank identity in the planner checkpoint.
            self.anchors_config['bank_sha256'] = self.anchor_manager.bank_sha256
        elif anchor_strategy == 'a1':
            from highway_env.planner.a1_anchor_manager import A1AnchorManager
            self.anchor_manager = A1AnchorManager(
                self.anchors_config, num_selected=self.scene_config.num_modes,
                control_steps=self.ctrl_cfg.horizon_steps,
                dt=self.ctrl_cfg.dt,
                wheelbase_m=self.ctrl_cfg.wheelbase_m,
                max_accel_mps2=self.ctrl_cfg.max_accel_mps2,
                max_steer_rad=self.ctrl_cfg.max_steer_rad,
            )
            self.anchors_config.update(self.anchor_manager.config)
    def _encode_scene(self, features):
        device = features['ego_state'].device
        for attr in ('ego_scale', 'agent_scale', 'map_scale', 'target_point_scale'):
            val = getattr(self.adapter, attr)
            if val.device != device:
                setattr(self.adapter, attr, val.to(device))
        return self.scene_encoder(features)

    def _prepare_anchors(self, features):
        """Generate current-scene A0/A1/A2 anchors without semantic mode identities."""
        return self.anchor_manager.select(features)

    def _sample_controls(self, features, *, generator=None, num_steps=None):
        """Control-space DDIM initialized from current-scene control anchors.

        Current-scene anchors are generated before entering this method.
        No inverse XY decoding or stored anchor library is used.
        """
        cfg = self.ctrl_cfg
        B, M, H = (features['ego_state'].shape[0], self.scene_config.num_modes,
                   cfg.horizon_steps)
        if 'coarse_controls' not in features:
            raise KeyError('Control diffusion requires sampled coarse_controls [B,M,40,2]')
        anchor = features['coarse_controls']
        if anchor.shape != (B, M, H, 2):
            raise ValueError(f'coarse_controls shape {tuple(anchor.shape)} != {(B,M,H,2)}')
        if not torch.isfinite(anchor).all():
            raise ValueError('coarse_controls contains NaN/Inf')
        clean = anchor / self.control_limits
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype,
                            generator=generator) * cfg.inference_noise_scale
        scene = self._encode_scene(features)
        count = int(num_steps or cfg.inference_steps)
        if not 2 <= count <= cfg.inference_start_timestep + 1:
            raise ValueError('Invalid truncated DDIM step count')
        ts = torch.linspace(cfg.inference_start_timestep, 0, count,
                            device=clean.device).round().long().tolist()
        sample = self.schedule.add_noise(
            clean, noise,
            torch.full((B,), ts[0], dtype=torch.long, device=clean.device),
        )
        for j, t in enumerate(ts):
            x0 = self.denoiser(sample,
                               torch.full((B,), t, device=sample.device, dtype=torch.long),
                               scene)
            next_t = ts[j + 1] if j + 1 < len(ts) else -1
            sample = self.schedule.step_predict_x0(sample, x0, t, next_t)
        return sample * self.control_limits

    def _validate_objectives(self):
        cfg = self.loss_config
        if cfg['mode_selection'] != 'goal_endpoint':
            raise ValueError('Control diffusion requires goal_endpoint mode selection')
        required = ('trajectory', 'background_collision', 'intertruck_collision',
                    'ttc', 'road_boundary', 'ltr', 'fuel', 'progress')
        for name in required:
            weight = float(cfg[name]['weight'])
            if weight < 0:
                raise ValueError(f'{name}.weight must be nonnegative')
        for name in ('trajectory', 'background_collision', 'intertruck_collision', 'ttc', 'road_boundary'):
            if float(cfg[name]['weight']) <= 0:
                raise ValueError(f'{name}.weight must be positive')
        safety = cfg['safety']
        for key in ('vehicle_length_m', 'vehicle_width_m', 'collision_margin_m',
                    'softplus_temperature_m', 'collision_distance_scale_m',
                    'ttc_distance_scale_m', 'road_distance_scale_m'):
            if float(safety[key]) <= 0:
                raise ValueError(f'safety.{key} must be positive')
        if float(safety['ttc_threshold_s']) <= 0:
            raise ValueError('safety.ttc_threshold_s must be positive')
        if self._weight('ltr') > 0:
            if self.vehicle_model.kind != 'koopman_k2':
                raise ValueError('LTR penalty requires Koopman K2')
            limit = float(cfg['ltr']['limit'])
            if not 0 < limit < 1:
                raise ValueError('LTR limit must be in (0,1)')
        if self._weight('fuel') > 0:
            self._fuel_parameters()
        if self._weight('progress') > 0 and float(cfg['progress']['minimum_progress_m']) < 0:
            raise ValueError('progress.minimum_progress_m must be nonnegative')

    def _weight(self, name):
        return float(self.loss_config[name]['weight'])

    def _fuel_parameters(self):
        values = self.loss_config.get('fuel', {}).get('parameters')
        if not isinstance(values, dict):
            raise ValueError('Positive fuel weight requires calibrated fuel.parameters')
        return FuelParameters(**values)

    def _all_predictions(self, controls, features, context=None):
        return self.vehicle_model(controls, features, context)

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

    def forward_train(self, features, target_dense_xy, safety_targets, *,
                      generator=None, context=None):
        """Train the selected DDIM control sequence against expert and safety objectives.

        The ego trajectory is predicted by frozen K2. Background vehicles use
        constant-velocity references from agent_states. The other two controlled
        trucks use synchronized expert trajectories supplied by RawExpertDataset.
        """
        cfg = self.ctrl_cfg
        batch = features['ego_state'].shape[0]
        if target_dense_xy.shape != (batch, cfg.horizon_steps, 2):
            raise ValueError('Expert trajectory_dense must be [B,40,2]')
        if safety_targets['platoon_reference_poses'].shape != (batch, 2, cfg.horizon_steps + 1, 3):
            raise ValueError('platoon_reference_poses must be [B,2,41,3]')

        road_safety_features = {
            key: features[key] for key in (
                'road_lane_geometry', 'road_lane_point_mask', 'road_lane_mask',
                'road_lane_connections', 'road_current_lane_slot', 'road_target_lane_slot',
            )
        }
        features = self._prepare_anchors(features)
        controls_all = self._sample_controls(features, generator=generator)
        pred = self._all_predictions(controls_all, features, context)
        poses_all, speeds_all = pred['poses'], pred['speeds']
        predicted_xy = poses_all[:, :, 1:, :2]
        truth = target_dense_xy[:, None].expand_as(predicted_xy)
        errors = (predicted_xy - truth).square().sum(dim=-1).mean(dim=-1)

        selected, _ = self._select_goal_endpoint_mode(poses_all, features)
        batch_ix = torch.arange(batch, device=controls_all.device)
        selected_controls = controls_all[batch_ix, selected]
        selected_poses = poses_all[batch_ix, selected]
        selected_speeds = speeds_all[batch_ix, selected]
        trajectory_loss = errors.gather(1, selected[:, None]).squeeze(1).mean()

        safety = self.loss_config['safety']
        ego_dimensions = selected_poses.new_tensor(
            [float(safety['vehicle_length_m']), float(safety['vehicle_width_m'])]
        ).expand(batch, 2)

        background_poses, background_dimensions, background_valid = background_constant_velocity_poses(
            features['agent_states'], features['agent_valid_mask'], features['ego_state'],
            cfg.horizon_steps, cfg.dt,
        )
        collision_args = {
            'dt': cfg.dt,
            'collision_margin_m': float(safety['collision_margin_m']),
            'ttc_threshold_s': float(safety['ttc_threshold_s']),
            'softplus_temperature_m': float(safety['softplus_temperature_m']),
            'collision_distance_scale_m': float(safety['collision_distance_scale_m']),
            'ttc_distance_scale_m': float(safety['ttc_distance_scale_m']),
        }
        background_stats = pairwise_vehicle_safety(
            selected_poses, ego_dimensions,
            background_poses, background_dimensions, background_valid,
            **collision_args,
        )

        platoon_poses = safety_targets['platoon_reference_poses'].to(selected_poses)
        platoon_dimensions = selected_poses.new_tensor(
            [float(safety['vehicle_length_m']), float(safety['vehicle_width_m'])]
        ).view(1, 1, 2).expand(batch, 2, 2)
        platoon_valid = torch.ones((batch, 2), dtype=torch.bool, device=selected_poses.device)
        platoon_stats = pairwise_vehicle_safety(
            selected_poses, ego_dimensions,
            platoon_poses, platoon_dimensions, platoon_valid,
            **collision_args,
        )

        road_stats = road_boundary_safety_loss(
            selected_poses, ego_dimensions,
            road_safety_features['road_lane_geometry'],
            road_safety_features['road_lane_point_mask'],
            road_safety_features['road_lane_mask'],
            road_safety_features['road_lane_connections'],
            road_safety_features['road_current_lane_slot'],
            road_safety_features['road_target_lane_slot'],
            softplus_temperature_m=float(safety['softplus_temperature_m']),
            distance_scale_m=float(safety['road_distance_scale_m']),
        )
        background_collision_loss = background_stats['collision_loss']
        intertruck_collision_loss = platoon_stats['collision_loss']
        ttc_loss = 0.5 * (background_stats['ttc_loss'] + platoon_stats['ttc_loss'])
        road_boundary_loss = road_stats['loss']

        zero = trajectory_loss.new_zeros(())
        ltr_loss, fuel_loss, progress_loss = zero, zero, zero
        fuel_total = None
        if self._weight('ltr') > 0:
            selected_ltr = pred['ltr'][batch_ix, selected, 1:]
            limit = float(self.loss_config['ltr']['limit'])
            ltr_loss = torch.relu(selected_ltr.abs() - limit).square().mean()
        if self._weight('fuel') > 0:
            payload = context['koopman_payload_kg'].to(selected_speeds).reshape(batch)
            fuel_result = vt_cpfm_from_states(
                selected_speeds, payload, self._fuel_parameters(), dt=cfg.dt)
            fuel_total = fuel_result['fuel_total']
            fuel_loss = fuel_total.mean()
        if self._weight('progress') > 0:
            minimum_progress = float(self.loss_config['progress']['minimum_progress_m'])
            delta_s = centerline_progress(selected_poses[:, -1], features['target_lane_polyline'])
            progress_loss = torch.relu(minimum_progress - delta_s).square().mean()

        total = (
            self._weight('trajectory') * trajectory_loss
            + self._weight('background_collision') * background_collision_loss
            + self._weight('intertruck_collision') * intertruck_collision_loss
            + self._weight('ttc') * ttc_loss
            + self._weight('road_boundary') * road_boundary_loss
            + self._weight('ltr') * ltr_loss
            + self._weight('fuel') * fuel_loss
            + self._weight('progress') * progress_loss
        )
        output = {
            'loss': total,
            'trajectory_loss': trajectory_loss,
            'background_collision_loss': background_collision_loss,
            'intertruck_collision_loss': intertruck_collision_loss,
            'ttc_loss': ttc_loss,
            'ttc_background_loss': background_stats['ttc_loss'],
            'ttc_intertruck_loss': platoon_stats['ttc_loss'],
            'road_boundary_loss': road_boundary_loss,
            'ltr_loss': ltr_loss,
            'fuel_loss': fuel_loss,
            'progress_loss': progress_loss,
            'background_collision_rate': background_stats['collision_rate'],
            'intertruck_collision_rate': platoon_stats['collision_rate'],
            'ttc_background_violation_rate': background_stats['ttc_violation_rate'],
            'ttc_intertruck_violation_rate': platoon_stats['ttc_violation_rate'],
            'road_violation_rate': road_stats['violation_rate'],
            'background_min_gap_m': background_stats['min_gap'],
            'intertruck_min_gap_m': platoon_stats['min_gap'],
            'max_road_violation_m': road_stats['max_violation'],
            'controls': selected_controls,
            'poses': selected_poses,
            'speeds': selected_speeds,
            'selected_mode': selected,
        }
        if pred['ltr'] is not None:
            output['ltr'] = pred['ltr'][batch_ix, selected]
            output['states'] = pred['states'][batch_ix, selected]
        if fuel_total is not None:
            output['fuel_total'] = fuel_total
        return output

    def infer_multimodal(self, features, generator: Optional[torch.Generator] = None,
                         num_steps: Optional[int] = None, context=None, *, anchors_prepared=False):
        cfg = self.ctrl_cfg
        features = features if anchors_prepared else self._prepare_anchors(features)
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
            # Full 40-step K2 trajectories are required for paper minADE_K/minFDE_K.
            'trajectory_dense_candidates': poses_all[:, :, 1:, :2],
            'pose_candidates': poses_all,
            'trajectory_mode_logits': scores,
            'trajectory_mode_idx': mode,
            # Expose the actual post-anchor-selection validity mask so an
            # evaluator cannot count invalid candidate modes as minADE/FDE.
            'trajectory_candidate_valid_mask': features['mode_valid_mask'],
            'speeds': speeds,
        }
        if pred['ltr'] is not None:
            result['ltr'] = pred['ltr'][idx, mode]
            result['states'] = pred['states'][idx, mode]
            result['ltr_candidates'] = pred['ltr']
            result['states_candidates'] = pred['states']
        return result
