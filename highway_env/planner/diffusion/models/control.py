"""TRACE control-space stochastic diffusion planner with switchable objectives.

Control candidates are refined by DDIM and rolled out by frozen Koopman K2.
The baseline objective supervises the candidate selected by shared Candidate
Scoring. The anchor-assigned objective routes expert supervision through
K2-executed Anchor ADE, preserves the remaining Anchor modes, and evaluates
auxiliary objectives over all valid generated modes. Target Point remains a
scene-encoder condition; the same Candidate Scoring rule selects the final
candidate during training, inference, and evaluation.

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
from highway_env.planner.diffusion.common.candidate_scoring import score_candidate_modes
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
    """Control-space diffusion with switchable goal-endpoint and anchor-assigned losses.

    Target Point is encoded by the shared scene encoder in both training and
    inference. The original scheme uses goal-endpoint selection for supervision.
    The anchor-consistent scheme assigns expert supervision to the K2-executed
    anchor nearest to the expert trajectory, preserves the other anchor modes,
    and still uses goal-endpoint selection for inference-compatible diagnostics.
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
        self.loss_config.setdefault('safety', {
            'vehicle_length_m': 8.0, 'vehicle_width_m': 2.6,
            'collision_margin_m': 0.2, 'ttc_threshold_s': 1.0,
            'softplus_temperature_m': 0.2, 'collision_distance_scale_m': 1.0,
            'ttc_distance_scale_m': 5.0, 'road_distance_scale_m': 1.0,
        })
        self.loss_config.setdefault('mode_selection', 'candidate_score')
        # Migrate old checkpoints/configurations away from Target Point endpoint
        # selection. Target Point remains a scene-encoder condition only.
        legacy_selection = str(self.loss_config.get('mode_selection', 'candidate_score'))
        if legacy_selection not in ('goal_endpoint', 'candidate_score'):
            raise ValueError(f'Unsupported mode_selection={legacy_selection!r}')
        if legacy_selection == 'goal_endpoint':
            self.loss_config['legacy_mode_selection_migrated_from'] = 'goal_endpoint'
        self.loss_config['mode_selection'] = 'candidate_score'
        self._initialize_candidate_scoring_config()
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
    def _initialize_candidate_scoring_config(self):
        """Freeze the safety-first candidate ranking policy into checkpoints."""
        from copy import deepcopy
        from highway_env.planner.diffusion.projection.config import load_jtp_config
        from highway_env.planner.diffusion.config.experiment import resolve_repo_file

        cfg = deepcopy(self.loss_config.get('candidate_scoring') or {})
        cfg.setdefault('feasibility_tolerance', 1.0e-4)
        cfg.setdefault('severity_tie_tolerance', 1.0e-6)
        cfg.setdefault('include_controlled_agents', True)
        default_fuel_weight = 1.0 if isinstance((self.loss_config.get('fuel') or {}).get('parameters'), dict) else 0.0
        cfg.setdefault('efficiency', {'fuel_weight': default_fuel_weight, 'progress_weight': 1.0})
        cfg.setdefault('severity_weights', {
            'collision': 1.0, 'ttc': 1.0, 'road': 1.0, 'stability': 1.0,
        })
        if 'stability' not in cfg or 'stability_residual_scales' not in cfg:
            jtp_path = resolve_repo_file(cfg.get('jtp_config', 'configs/projection/jtp.yaml'))
            jtp = load_jtp_config(jtp_path)
            cfg.setdefault('stability', deepcopy(jtp['stability']))
            cfg.setdefault('stability_residual_scales', list(jtp['projection']['residual_scales']['stability']))
            cfg.setdefault('stability_tolerance', float(jtp['projection']['constraint_tolerance']))
            cfg['jtp_config_source'] = str(jtp_path)
        self.loss_config['candidate_scoring'] = cfg

    def _select_candidate_mode(self, controls_all, prediction, features, context=None):
        """Shared target-point-independent selector for train, inference and evaluation."""
        fuel_values = (self.loss_config.get('fuel') or {}).get('parameters')
        fuel_parameters = FuelParameters(**fuel_values) if isinstance(fuel_values, dict) else None
        return score_candidate_modes(
            controls_all, prediction, features, context,
            dt=float(self.ctrl_cfg.dt),
            safety_config=self.loss_config['safety'],
            scoring_config=self.loss_config['candidate_scoring'],
            fuel_parameters=fuel_parameters,
            execution_kind=self.vehicle_model.kind,
        )

    def _encode_scene(self, features):
        device = features['ego_state'].device
        for attr in ('ego_scale', 'agent_scale', 'map_scale', 'target_point_scale'):
            val = getattr(self.adapter, attr)
            if val.device != device:
                setattr(self.adapter, attr, val.to(device))
        return self.scene_encoder(features)

    def _prepare_anchors(self, features):
        """Generate current-scene anchors and retain road geometry for scoring."""
        prepared = self.anchor_manager.select(features)
        for key in (
            'road_lane_geometry', 'road_lane_point_mask', 'road_lane_mask',
            'road_lane_connections', 'road_current_lane_slot', 'road_target_lane_slot',
        ):
            if key in features and key not in prepared:
                prepared[key] = features[key]
        return prepared
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

    def _training_scheme(self):
        """Return the selected objective family without mutating legacy configs."""
        scheme = str(self.loss_config.get('training_scheme', 'goal_endpoint'))
        if scheme not in ('goal_endpoint', 'anchor_consistent'):
            raise ValueError(
                "losses.training_scheme must be 'goal_endpoint' or 'anchor_consistent'"
            )
        return scheme

    def _validate_objectives(self):
        cfg = self.loss_config
        if cfg['mode_selection'] != 'candidate_score':
            raise ValueError('Control diffusion requires candidate_score mode selection')
        scheme = self._training_scheme()
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
        if scheme == 'goal_endpoint':
            if self._weight('fuel') > 0:
                self._fuel_parameters()
        else:
            if self.vehicle_model.kind != 'koopman_k2':
                raise ValueError('anchor_consistent training requires frozen Koopman K2 execution')
            anchor_cfg = cfg.get('anchor_consistency')
            if not isinstance(anchor_cfg, dict):
                raise ValueError('anchor_consistent training requires losses.anchor_consistency')
            anchor_weight = float(anchor_cfg.get('weight', 0.0))
            if anchor_weight <= 0:
                raise ValueError('anchor_consistency.weight must be positive')
            relative_fuel_weight = float(cfg['fuel'].get('relative_weight', 0.0))
            relative_epsilon = float(cfg['fuel'].get('relative_epsilon', 1e-8))
            if relative_fuel_weight <= 0:
                raise ValueError('fuel.relative_weight must be positive for anchor_consistent training')
            if relative_epsilon <= 0:
                raise ValueError('fuel.relative_epsilon must be positive')
            self._fuel_parameters()
        if self._weight('progress') > 0 and float(cfg['progress']['minimum_progress_m']) < 0:
            raise ValueError('progress.minimum_progress_m must be nonnegative')

    def _weight(self, name):
        return float(self.loss_config[name]['weight'])

    def _relative_fuel_weight(self):
        return float(self.loss_config['fuel'].get('relative_weight', 0.0))

    @staticmethod
    def _masked_mode_mean(values, valid_mask, mode_weights=None):
        """Mean over valid [B,M] modes, optionally with per-mode weights."""
        weights = valid_mask.to(dtype=values.dtype)
        if mode_weights is not None:
            weights = weights * mode_weights.to(dtype=values.dtype)
        denominator = weights.sum().clamp_min(torch.finfo(values.dtype).eps)
        return (values * weights).sum() / denominator

    def _relative_fuel_objective(self, generated_speeds, anchor_speeds,
                                 payload_kg, valid_mask):
        """Per-anchor relative VT-CPFM fuel change in frozen-K2 execution space.

        A negative value denotes a fuel reduction relative to the corresponding
        anchor; zero denotes parity. Anchor fuel is detached from autograd.
        """
        batch, modes, steps = generated_speeds.shape
        if anchor_speeds.shape != generated_speeds.shape:
            raise ValueError('Generated and anchor K2 speeds must have identical [B,M,H+1] shapes')
        if valid_mask.shape != (batch, modes):
            raise ValueError('mode_valid_mask must match [B,M] fuel candidates')
        payload = payload_kg.to(generated_speeds).reshape(batch)
        repeated_payload = payload[:, None].expand(batch, modes).reshape(-1)
        parameters = self._fuel_parameters()
        generated = vt_cpfm_from_states(
            generated_speeds.reshape(batch * modes, steps), repeated_payload,
            parameters, dt=self.ctrl_cfg.dt,
        )['fuel_total'].reshape(batch, modes)
        with torch.no_grad():
            anchor = vt_cpfm_from_states(
                anchor_speeds.detach().reshape(batch * modes, steps), repeated_payload,
                parameters, dt=self.ctrl_cfg.dt,
            )['fuel_total'].reshape(batch, modes)
        epsilon = float(self.loss_config['fuel'].get('relative_epsilon', 1e-8))
        ratio = generated / anchor.clamp_min(epsilon)
        relative_change = ratio - 1.0
        loss = self._masked_mode_mean(relative_change, valid_mask)
        mean_ratio = self._masked_mode_mean(ratio, valid_mask)
        improvement_pct = -100.0 * self._masked_mode_mean(relative_change, valid_mask)
        return {
            'loss': loss,
            'ratio': mean_ratio,
            'improvement_pct': improvement_pct,
            'generated_fuel': generated,
            'anchor_fuel': anchor,
        }

    def _fuel_parameters(self):
        values = self.loss_config.get('fuel', {}).get('parameters')
        if not isinstance(values, dict):
            raise ValueError('Fuel objective requires calibrated fuel.parameters')
        return FuelParameters(**values)

    def _all_predictions(self, controls, features, context=None):
        return self.vehicle_model(controls, features, context)


    @staticmethod
    def _select_anchor_expert_mode(anchor_poses, target_dense_xy, valid_mask):
        """Assign expert supervision to the K2-executed anchor with minimum ADE.

        The assignment is training-only and is computed from frozen Anchor K2
        rollouts. Target Point remains an input condition; the same Candidate
        Scoring policy selects the final candidate during training and inference.
        """
        if anchor_poses.ndim != 4 or anchor_poses.shape[-1] != 3:
            raise ValueError('anchor_poses must be [B,M,H+1,3]')
        batch, modes, steps = anchor_poses.shape[:3]
        if target_dense_xy.shape != (batch, steps - 1, 2):
            raise ValueError('target_dense_xy must align with K2 anchor execution horizon')
        if valid_mask.shape != (batch, modes):
            raise ValueError('mode_valid_mask must be [B,M]')
        valid = valid_mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError('Every episode needs at least one valid anchor mode')
        anchor_xy = anchor_poses[:, :, 1:, :2]
        expert = target_dense_xy[:, None].to(anchor_xy)
        per_mode_ade = torch.linalg.vector_norm(anchor_xy - expert, dim=-1).mean(dim=-1)
        per_mode_ade = per_mode_ade.masked_fill(~valid, torch.inf)
        selected = per_mode_ade.argmin(dim=1)
        selected_ade = per_mode_ade.gather(1, selected[:, None]).squeeze(1)
        return selected, selected_ade, per_mode_ade

    def _all_mode_safety_objectives(self, poses_all, features, safety_targets,
                                    road_safety_features, valid_modes):
        """Evaluate collision/TTC/road objectives on every valid generated mode.

        Modes are processed one at a time to bound peak memory for the road
        footprint projection. Each mode contributes in proportion to its number
        of valid batch samples, yielding a mean over all valid (sample, mode) pairs.
        """
        batch, modes, steps = poses_all.shape[:3]
        if valid_modes.shape != (batch, modes) or not valid_modes.any():
            raise ValueError('At least one valid candidate mode is required for safety losses')

        safety = self.loss_config['safety']
        collision_args = {
            'dt': self.ctrl_cfg.dt,
            'collision_margin_m': float(safety['collision_margin_m']),
            'ttc_threshold_s': float(safety['ttc_threshold_s']),
            'softplus_temperature_m': float(safety['softplus_temperature_m']),
            'collision_distance_scale_m': float(safety['collision_distance_scale_m']),
            'ttc_distance_scale_m': float(safety['ttc_distance_scale_m']),
        }
        background_poses, background_dimensions, background_valid = background_constant_velocity_poses(
            features['agent_states'], features['agent_valid_mask'], features['ego_state'],
            self.ctrl_cfg.horizon_steps, self.ctrl_cfg.dt,
        )
        platoon_all = safety_targets['platoon_reference_poses'].to(poses_all)
        total_valid = int(valid_modes.sum().item())
        accum = {
            'background_collision_loss': poses_all.new_zeros(()),
            'intertruck_collision_loss': poses_all.new_zeros(()),
            'ttc_background_loss': poses_all.new_zeros(()),
            'ttc_intertruck_loss': poses_all.new_zeros(()),
            'road_boundary_loss': poses_all.new_zeros(()),
            'background_collision_rate': poses_all.new_zeros(()),
            'intertruck_collision_rate': poses_all.new_zeros(()),
            'ttc_background_violation_rate': poses_all.new_zeros(()),
            'ttc_intertruck_violation_rate': poses_all.new_zeros(()),
            'road_violation_rate': poses_all.new_zeros(()),
            'background_min_gap_m': poses_all.new_zeros(()),
            'intertruck_min_gap_m': poses_all.new_zeros(()),
        }
        max_road_violation = poses_all.new_zeros(())
        background_min_gap_count = 0

        for mode in range(modes):
            batch_indices = torch.nonzero(valid_modes[:, mode].bool(), as_tuple=False).flatten()
            count = int(batch_indices.numel())
            if count == 0:
                continue
            ego_poses = poses_all.index_select(0, batch_indices)[:, mode]
            ego_dimensions = ego_poses.new_tensor(
                [float(safety['vehicle_length_m']), float(safety['vehicle_width_m'])]
            ).expand(count, 2)

            bg_stats = pairwise_vehicle_safety(
                ego_poses, ego_dimensions,
                background_poses.index_select(0, batch_indices),
                background_dimensions.index_select(0, batch_indices),
                background_valid.index_select(0, batch_indices),
                **collision_args,
            )
            platoon_poses = platoon_all.index_select(0, batch_indices)
            platoon_dimensions = ego_poses.new_tensor(
                [float(safety['vehicle_length_m']), float(safety['vehicle_width_m'])]
            ).view(1, 1, 2).expand(count, 2, 2)
            platoon_valid = torch.ones((count, 2), dtype=torch.bool, device=ego_poses.device)
            platoon_stats = pairwise_vehicle_safety(
                ego_poses, ego_dimensions, platoon_poses, platoon_dimensions,
                platoon_valid, **collision_args,
            )
            take = lambda key: road_safety_features[key].index_select(0, batch_indices)
            road_stats = road_boundary_safety_loss(
                ego_poses, ego_dimensions,
                take('road_lane_geometry'), take('road_lane_point_mask'),
                take('road_lane_mask'), take('road_lane_connections'),
                take('road_current_lane_slot'), take('road_target_lane_slot'),
                softplus_temperature_m=float(safety['softplus_temperature_m']),
                distance_scale_m=float(safety['road_distance_scale_m']),
            )

            accum['background_collision_loss'] = accum['background_collision_loss'] + bg_stats['collision_loss'] * count
            accum['intertruck_collision_loss'] = accum['intertruck_collision_loss'] + platoon_stats['collision_loss'] * count
            accum['ttc_background_loss'] = accum['ttc_background_loss'] + bg_stats['ttc_loss'] * count
            accum['ttc_intertruck_loss'] = accum['ttc_intertruck_loss'] + platoon_stats['ttc_loss'] * count
            accum['road_boundary_loss'] = accum['road_boundary_loss'] + road_stats['loss'] * count
            accum['background_collision_rate'] = accum['background_collision_rate'] + bg_stats['collision_rate'] * count
            accum['intertruck_collision_rate'] = accum['intertruck_collision_rate'] + platoon_stats['collision_rate'] * count
            accum['ttc_background_violation_rate'] = accum['ttc_background_violation_rate'] + bg_stats['ttc_violation_rate'] * count
            accum['ttc_intertruck_violation_rate'] = accum['ttc_intertruck_violation_rate'] + platoon_stats['ttc_violation_rate'] * count
            accum['road_violation_rate'] = accum['road_violation_rate'] + road_stats['violation_rate'] * count
            has_background = background_valid.index_select(0, batch_indices).any(dim=1)
            background_count = int(has_background.sum().item())
            if background_count:
                accum['background_min_gap_m'] = (
                    accum['background_min_gap_m'] + bg_stats['min_gap'] * background_count
                )
                background_min_gap_count += background_count
            accum['intertruck_min_gap_m'] = accum['intertruck_min_gap_m'] + platoon_stats['min_gap'] * count
            max_road_violation = torch.maximum(max_road_violation, road_stats['max_violation'])

        for key in accum:
            if key == 'background_min_gap_m':
                denominator = max(background_min_gap_count, 1)
            else:
                denominator = total_valid
            accum[key] = accum[key] / float(denominator)
        accum['ttc_loss'] = 0.5 * (
            accum['ttc_background_loss'] + accum['ttc_intertruck_loss']
        )
        accum['max_road_violation_m'] = max_road_violation
        accum['safety_mode_count'] = valid_modes.sum().to(dtype=poses_all.dtype)
        return accum

    def forward_train(self, features, target_dense_xy, safety_targets, *,
                      generator=None, context=None):
        """Train the selected objective while preserving the original baseline path."""
        scheme = self._training_scheme()
        if scheme == 'anchor_consistent':
            return self._forward_train_anchor_consistent(
                features, target_dense_xy, safety_targets,
                generator=generator, context=context,
            )
        return self._forward_train_goal_endpoint(
            features, target_dense_xy, safety_targets,
            generator=generator, context=context,
        )

    def _forward_train_goal_endpoint(self, features, target_dense_xy, safety_targets, *,
                                     generator=None, context=None):
        """Original single-mode baseline losses; the supervised candidate is selected by Candidate Scoring."""
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

        selection = self._select_candidate_mode(controls_all, pred, features, context)

        selected = selection['selected_mode']

        _ = selection['candidate_scores']
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
            if context is None or 'koopman_payload_kg' not in context:
                raise KeyError('Fuel loss needs physical koopman_payload_kg [B] in expert context')
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
            'anchor_consistency_loss': zero,
            'anchor_execution_gap_m': zero,
            'fuel_relative_loss': zero,
            'fuel_relative_ratio': zero,
            'fuel_improvement_pct': zero,
            'expert_anchor_ade': zero,
            'goal_selected_ade': torch.linalg.vector_norm(
                predicted_xy - target_dense_xy[:, None], dim=-1
            ).mean(dim=-1).gather(1, selected[:, None]).squeeze(1).mean(),
            'expert_goal_mode_disagreement': zero,
            'safety_mode_count': zero + batch,
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
            'goal_selected_mode': selected,
            'expert_mode': selected,
        }
        if pred['ltr'] is not None:
            output['ltr'] = pred['ltr'][batch_ix, selected]
            output['states'] = pred['states'][batch_ix, selected]
        if fuel_total is not None:
            output['fuel_total'] = fuel_total
        return output

    def _forward_train_anchor_consistent(self, features, target_dense_xy, safety_targets, *,
                                         generator=None, context=None):
        """Anchor-assigned expert supervision plus per-mode consistency/objectives.

        The expert mode is selected by ADE between each frozen-K2 anchor rollout
        and the expert trajectory. Every other valid mode is regularized toward
        its own frozen-K2 anchor rollout. Safety, LTR, progress and relative fuel
        objectives are evaluated across all valid generated modes. Target Point
        remains a shared scene condition and the goal-endpoint selector used by
        inference is retained for the selected-output diagnostics.
        """
        cfg = self.ctrl_cfg
        batch = features['ego_state'].shape[0]
        if target_dense_xy.shape != (batch, cfg.horizon_steps, 2):
            raise ValueError('Expert trajectory_dense must be [B,40,2]')
        if safety_targets['platoon_reference_poses'].shape != (batch, 2, cfg.horizon_steps + 1, 3):
            raise ValueError('platoon_reference_poses must be [B,2,41,3]')
        if context is None or 'koopman_payload_kg' not in context:
            raise KeyError('Anchor-consistent training requires physical koopman_payload_kg [B]')

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
        valid_modes = features['mode_valid_mask'].bool()
        batch_ix = torch.arange(batch, device=controls_all.device)
        selection = self._select_candidate_mode(controls_all, pred, features, context)
        goal_selected = selection['selected_mode']
        _ = selection['candidate_scores']
        anchor_controls = features.get('coarse_controls')
        if anchor_controls is None or anchor_controls.shape != controls_all.shape:
            raise ValueError('anchor_consistent training requires paired coarse_controls [B,M,40,2]')
        if not torch.isfinite(anchor_controls).all():
            raise ValueError('Anchor control sequence contains NaN/Inf')
        # Anchor controls are fixed references. Execute them through exactly the
        # same frozen K2 and the same observed context as generated controls.
        with torch.no_grad():
            anchor_pred = self._all_predictions(anchor_controls, features, context)
        anchor_poses = anchor_pred['poses']
        expert_mode, expert_anchor_ade, per_anchor_ade = self._select_anchor_expert_mode(
            anchor_poses, target_dense_xy, valid_modes,
        )

        truth = target_dense_xy[:, None].expand_as(predicted_xy)
        per_mode_expert_mse = (predicted_xy - truth).square().sum(dim=-1).mean(dim=-1)
        trajectory_loss = per_mode_expert_mse.gather(
            1, expert_mode[:, None]
        ).squeeze(1).mean()

        anchor_xy = anchor_poses[:, :, 1:, :2]
        per_mode_anchor_mse = (predicted_xy - anchor_xy).square().sum(dim=-1).mean(dim=-1)
        anchor_mode_mask = valid_modes.clone()
        anchor_mode_mask[batch_ix, expert_mode] = False
        anchor_consistency_loss = self._masked_mode_mean(
            per_mode_anchor_mse, anchor_mode_mask,
        )
        anchor_sparse_xy = anchor_poses[:, :, 5::5, :2]
        coarse_xy = features.get('coarse_trajectories')
        if coarse_xy is None or coarse_xy.shape != anchor_sparse_xy.shape:
            raise ValueError('coarse_trajectories must align with 8 K2 anchor samples')
        anchor_execution_gap = torch.linalg.vector_norm(
            anchor_sparse_xy - coarse_xy, dim=-1
        ).mean(dim=-1)
        anchor_execution_gap_m = self._masked_mode_mean(anchor_execution_gap, valid_modes)

        safety_stats = self._all_mode_safety_objectives(
            poses_all, features, safety_targets, road_safety_features, valid_modes,
        )
        background_collision_loss = safety_stats['background_collision_loss']
        intertruck_collision_loss = safety_stats['intertruck_collision_loss']
        ttc_loss = safety_stats['ttc_loss']
        road_boundary_loss = safety_stats['road_boundary_loss']

        zero = trajectory_loss.new_zeros(())
        ltr_loss = zero
        if self._weight('ltr') > 0:
            if pred['ltr'] is None:
                raise ValueError('LTR loss requires K2 LTR predictions')
            limit = float(self.loss_config['ltr']['limit'])
            per_mode_ltr = torch.relu(pred['ltr'][:, :, 1:].abs() - limit).square().mean(dim=-1)
            ltr_loss = self._masked_mode_mean(per_mode_ltr, valid_modes)

        relative_fuel = self._relative_fuel_objective(
            speeds_all, anchor_pred['speeds'], context['koopman_payload_kg'], valid_modes,
        )
        fuel_loss = relative_fuel['loss']
        fuel_relative_ratio = relative_fuel['ratio']
        fuel_improvement_pct = relative_fuel['improvement_pct']

        progress_loss = zero
        if self._weight('progress') > 0:
            minimum_progress = float(self.loss_config['progress']['minimum_progress_m'])
            modes = poses_all.shape[1]
            terminal = poses_all[:, :, -1, :].reshape(batch * modes, 3)
            lane = features['target_lane_polyline'][:, None].expand(
                batch, modes, *features['target_lane_polyline'].shape[1:]
            ).reshape(batch * modes, *features['target_lane_polyline'].shape[1:])
            progress = centerline_progress(terminal, lane).reshape(batch, modes)
            per_mode_progress = torch.relu(minimum_progress - progress).square()
            progress_loss = self._masked_mode_mean(per_mode_progress, valid_modes)

        total = (
            self._weight('trajectory') * trajectory_loss
            + float(self.loss_config['anchor_consistency']['weight']) * anchor_consistency_loss
            + self._weight('background_collision') * background_collision_loss
            + self._weight('intertruck_collision') * intertruck_collision_loss
            + self._weight('ttc') * ttc_loss
            + self._weight('road_boundary') * road_boundary_loss
            + self._weight('ltr') * ltr_loss
            + self._relative_fuel_weight() * fuel_loss
            + self._weight('progress') * progress_loss
        )

        per_mode_generated_ade = torch.linalg.vector_norm(
            predicted_xy - target_dense_xy[:, None], dim=-1
        ).mean(dim=-1)
        goal_selected_ade = per_mode_generated_ade.gather(
            1, goal_selected[:, None]
        ).squeeze(1).mean()
        expert_goal_disagreement = (expert_mode != goal_selected).float().mean()
        selected_controls = controls_all[batch_ix, goal_selected]
        selected_poses = poses_all[batch_ix, goal_selected]
        selected_speeds = speeds_all[batch_ix, goal_selected]
        output = {
            'loss': total,
            'trajectory_loss': trajectory_loss,
            'anchor_consistency_loss': anchor_consistency_loss,
            'anchor_execution_gap_m': anchor_execution_gap_m,
            'fuel_relative_loss': fuel_loss,
            'fuel_relative_ratio': fuel_relative_ratio,
            'fuel_improvement_pct': fuel_improvement_pct,
            'expert_anchor_ade': expert_anchor_ade.mean(),
            'goal_selected_ade': goal_selected_ade,
            'expert_goal_mode_disagreement': expert_goal_disagreement,
            'safety_mode_count': safety_stats['safety_mode_count'],
            'background_collision_loss': background_collision_loss,
            'intertruck_collision_loss': intertruck_collision_loss,
            'ttc_loss': ttc_loss,
            'ttc_background_loss': safety_stats['ttc_background_loss'],
            'ttc_intertruck_loss': safety_stats['ttc_intertruck_loss'],
            'road_boundary_loss': road_boundary_loss,
            'ltr_loss': ltr_loss,
            'fuel_loss': fuel_loss,
            'progress_loss': progress_loss,
            'background_collision_rate': safety_stats['background_collision_rate'],
            'intertruck_collision_rate': safety_stats['intertruck_collision_rate'],
            'ttc_background_violation_rate': safety_stats['ttc_background_violation_rate'],
            'ttc_intertruck_violation_rate': safety_stats['ttc_intertruck_violation_rate'],
            'road_violation_rate': safety_stats['road_violation_rate'],
            'background_min_gap_m': safety_stats['background_min_gap_m'],
            'intertruck_min_gap_m': safety_stats['intertruck_min_gap_m'],
            'max_road_violation_m': safety_stats['max_road_violation_m'],
            # Keep selected_mode aligned with the inference selector. The
            # expert-supervised mode is exposed separately for auditability.
            'controls': selected_controls,
            'poses': selected_poses,
            'speeds': selected_speeds,
            'selected_mode': goal_selected,
            'goal_selected_mode': goal_selected,
            'expert_mode': expert_mode,
            'anchor_expert_ade_per_mode': per_anchor_ade,
        }
        if pred['ltr'] is not None:
            output['ltr'] = pred['ltr'][batch_ix, goal_selected]
            output['states'] = pred['states'][batch_ix, goal_selected]
        output['fuel_total'] = relative_fuel['generated_fuel'][batch_ix, goal_selected]
        output['anchor_fuel_total'] = relative_fuel['anchor_fuel'][batch_ix, goal_selected]
        return output

    def infer_multimodal(self, features, generator: Optional[torch.Generator] = None,
                         num_steps: Optional[int] = None, context=None, *, anchors_prepared=False):
        cfg = self.ctrl_cfg
        features = features if anchors_prepared else self._prepare_anchors(features)
        controls_all = self._sample_controls(features, generator=generator,
                                             num_steps=num_steps)
        pred = self._all_predictions(controls_all, features, context)
        poses_all, speeds_all = pred['poses'], pred['speeds']
        selection = self._select_candidate_mode(controls_all, pred, features, context)
        mode = selection['selected_mode']
        scores = selection['candidate_scores']
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
