"""D1: trajectory-only supervised, control-space stochastic diffusion planner.

No inferred / pseudo expert controls are used. The model samples noisy control
sequences, performs a differentiable DDIM-style refinement, integrates controls
with a kinematic bicycle model, and is supervised ONLY by expert XY positions.

Important: without clean control labels this is end-to-end, trajectory-trained
iterative diffusion refinement; it is not a standard supervised noise-prediction
DDPM objective. Training and inference run the same refinement schedule.
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
from highway_env.planner.diffusion.common.kinematics import rollout_bicycle


@dataclass
class ControlSpaceConfig:
    horizon_steps: int = 40
    dt: float = 0.1
    wheelbase_m: float = 5.5
    max_accel_mps2: float = 2.5
    max_steer_rad: float = 0.08
    num_train_timesteps: int = 1000
    inference_steps: int = 8           # Same count in training and inference.
    position_scale_m: float = 10.0
    loss_position: float = 1.0         # Only supervised objective.

    def validate(self):
        if self.horizon_steps != 40 or abs(self.dt - 0.1) > 1e-9:
            raise ValueError('Control-space D1 requires 40 steps at dt=0.1 s')
        if min(self.wheelbase_m, self.max_accel_mps2,
               self.max_steer_rad, self.position_scale_m, self.loss_position) <= 0:
            raise ValueError('wheelbase, limits and trajectory loss scaling must be positive')
        if not 2 <= self.inference_steps <= self.num_train_timesteps:
            raise ValueError('inference_steps must be in [2, num_train_timesteps]')


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
    """Diffusion of 40 acceleration/steering inputs, trained by expert XY only.

    forward_train: candidate controls [B,M,40,2] -> all kinematic trajectories;
    selects best valid mode by trajectory error (only trajectory loss).
    infer_multimodal: same DDIM schedule, mode chosen by goal endpoint distance.
    """

    def __init__(self, model_config=None, control_config=None):
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

    def _encode_scene(self, features):
        device = features['ego_state'].device
        for attr in ('ego_scale', 'agent_scale', 'map_scale', 'target_point_scale'):
            val = getattr(self.adapter, attr)
            if val.device != device:
                setattr(self.adapter, attr, val.to(device))
        return self.scene_encoder(features)

    def _sample_controls(self, features, *, generator=None, num_steps=None):
        """Differentiable refinement shared by training and inference."""
        cfg = self.ctrl_cfg
        B, M, H = (features['ego_state'].shape[0], self.scene_config.num_modes,
                   cfg.horizon_steps)
        noise = torch.randn((B, M, H, 2), device=features['ego_state'].device,
                            dtype=features['ego_state'].dtype, generator=generator)
        scene = self._encode_scene(features)
        count = int(num_steps or cfg.inference_steps)
        if not 2 <= count <= cfg.num_train_timesteps:
            raise ValueError('Invalid DDIM inference step count')
        ts = torch.linspace(cfg.num_train_timesteps - 1, 0, count,
                            device=noise.device).round().long().tolist()
        sample = noise
        for j, t in enumerate(ts):
            x0 = self.denoiser(sample,
                               torch.full((B,), t, device=sample.device, dtype=torch.long),
                               scene)
            next_t = ts[j + 1] if j + 1 < len(ts) else -1
            sample = self.schedule.step_predict_x0(sample, x0, t, next_t)
        return sample * self.control_limits

    def _all_poses(self, controls, features):
        B, M, H, _ = controls.shape
        initial_vx = features['ego_state'][:, 0].repeat_interleave(M)
        poses, speeds = rollout_bicycle(controls.reshape(B * M, H, 2),
                                        initial_vx, self.ctrl_cfg.dt,
                                        self.ctrl_cfg.wheelbase_m)
        return poses.reshape(B, M, H + 1, 3), speeds.reshape(B, M, H + 1)

    def forward_train(self, features, target_dense_xy, target_mode=None, *, generator=None):
        """Trajectory-only supervision: NO inverse controls, NO control/CE loss."""
        cfg = self.ctrl_cfg
        B = features['ego_state'].shape[0]
        if target_dense_xy.shape != (B, cfg.horizon_steps, 2):
            raise ValueError('Expert trajectory_dense must be [B,40,2]')
        controls_all = self._sample_controls(features, generator=generator)
        poses_all, speeds_all = self._all_poses(controls_all, features)
        errors = F.smooth_l1_loss(
            poses_all[:, :, 1:, :2] / cfg.position_scale_m,
            target_dense_xy[:, None].expand_as(poses_all[:, :, 1:, :2]) / cfg.position_scale_m,
            reduction='none').mean(dim=(-2, -1))
        valid = features['mode_valid_mask'].bool()
        if not valid.any(dim=1).all():
            raise ValueError('Each episode needs at least one valid mode')
        oracle_err, selected = errors.masked_fill(~valid, float('inf')).min(dim=1)
        loss = cfg.loss_position * oracle_err.mean()
        batch_ix = torch.arange(B, device=controls_all.device)
        return {
            'loss': loss, 'position_loss': oracle_err.mean(),
            'controls': controls_all[batch_ix, selected],
            'poses': poses_all[batch_ix, selected],
            'speeds': speeds_all[batch_ix, selected],
            'selected_mode': selected,
        }

    def infer_multimodal(self, features, generator: Optional[torch.Generator] = None,
                         num_steps: Optional[int] = None):
        cfg = self.ctrl_cfg
        controls_all = self._sample_controls(features, generator=generator,
                                             num_steps=num_steps)
        poses_all, speeds_all = self._all_poses(controls_all, features)
        valid = features['mode_valid_mask'].bool()
        if not valid.any(dim=1).all():
            raise ValueError('Every episode needs at least one valid mode')
        # Geometric endpoint matching is an inference heuristic, not a trained
        # classifier, safety scoring network or imitation-control label.
        target = features['target_point'][:, None, :]
        scores = -(poses_all[:, :, -1, :2] - target).square().sum(dim=-1)
        mode = scores.masked_fill(~valid, torch.finfo(scores.dtype).min).argmax(dim=1)
        idx = torch.arange(controls_all.shape[0], device=controls_all.device)
        controls = controls_all[idx, mode]
        poses = poses_all[idx, mode]
        speeds = speeds_all[idx, mode]
        future_xy = poses[:, 1:, :2]
        return {
            'control': controls,
            'control_candidates': controls_all,
            'pose': poses,
            'trajectory_dense': future_xy,
            'trajectory': future_xy[:, 4::5, :],
            'trajectory_candidates': poses_all[:, :, 5::5, :2],
            'trajectory_mode_logits': scores,  # Geometric, NOT learned logits.
            'trajectory_mode_idx': mode,
            'speeds': speeds,
        }
