"""TRACE control-space differentiable vehicle execution backends.

Inputs are commanded [a_x (m/s2), front-wheel delta (rad)] on the 10-Hz grid.
Koopman history/controls must be supplied from the same instant as the plan.
All output poses are in the *ego-at-planning-time* local frame. No training
labels enter the forward predictor beyond the observed K2 context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from highway_env.planner.diffusion.common.kinematics import rollout_bicycle
from highway_env.vehicle.koopman.models.conditional_model import ConditionalKoopman

K2_CONTEXT = (
    'koopman_state_hist_si', 'koopman_control_hist_si', 'koopman_pose0_si',
    'koopman_payload_kg', 'koopman_road_mu',
)


def _require_context(context, names, batch, *, device, dtype):
    if context is None:
        raise ValueError('K2 execution requires observed TruckSim context; got None')
    missing = [key for key in names if key not in context]
    if missing:
        raise KeyError(f'K2 execution missing context keys: {missing}. '
                       'Use TruckSim expert shards or feed synchronized runtime history.')
    tensors = {}
    for key in names:
        value = context[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=device, dtype=dtype)
        else:
            value = value.to(device=device, dtype=dtype)
        if value.ndim < 1 or value.shape[0] != batch or not torch.isfinite(value).all():
            raise ValueError(f'Invalid batch/finite values for {key}: shape={tuple(value.shape)}')
        tensors[key] = value
    if tensors['koopman_state_hist_si'].shape != (batch, 5, 5):
        raise ValueError('koopman_state_hist_si must be [B,5,5]')
    if tensors['koopman_control_hist_si'].shape != (batch, 4, 2):
        raise ValueError('koopman_control_hist_si must be [B,4,2]')
    if tensors['koopman_pose0_si'].shape != (batch, 3):
        raise ValueError('koopman_pose0_si must be [B,3]')
    for key in ('koopman_payload_kg', 'koopman_road_mu'):
        if tensors[key].shape not in ((batch,), (batch, 1)):
            raise ValueError(f'{key} must be [B] or [B,1]')
    if (tensors['koopman_road_mu'] <= 0).any() or (tensors['koopman_road_mu'] > 1).any():
        raise ValueError('Road friction mu must be in (0,1]')
    if (tensors['koopman_payload_kg'] < 0).any():
        raise ValueError('Payload must be nonnegative in kg')
    return tensors


def _metadata_from_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict) or checkpoint.get('model_type') != 'K2':
        raise ValueError('Only a trained Stage-C K2 checkpoint (model_type=K2) is supported')
    required = ('model_kwargs', 'normalization', 'condition_normalization', 'model_state')
    missing = [k for k in required if k not in checkpoint]
    if missing:
        raise KeyError(f'K2 checkpoint missing {missing}')
    kwargs = checkpoint['model_kwargs']
    if int(kwargs.get('history_steps', 0)) != 5:
        raise ValueError('Expert context is exactly 5 states and 4 controls; K2 checkpoint differs')
    return {
        'model_type': 'K2',
        'model_kwargs': dict(kwargs),
        'normalization': dict(checkpoint['normalization']),
        'condition_normalization': dict(checkpoint['condition_normalization']),
        'condition_hidden': int(checkpoint.get('condition_hidden', 64)),
    }


def _make_k2(metadata):
    if not isinstance(metadata, Mapping) or metadata.get('model_type') != 'K2':
        raise ValueError('Missing/invalid K2 model reconstruction metadata')
    return ConditionalKoopman(
        'K2', condition_hidden=int(metadata.get('condition_hidden', 64)),
        **metadata['condition_normalization'],
        **metadata['model_kwargs'],
        **metadata['normalization'],
    )


class VehicleExecution(nn.Module):
    def __init__(self, cfg=None, *, dt=0.1, wheelbase_m=5.5, koopman_metadata=None):
        super().__init__()
        self.cfg = dict(cfg or {})
        self.kind = str(self.cfg.get('model', 'kinematic')).lower()
        self.dt = float(dt)
        self.wheelbase_m = float(wheelbase_m)
        if self.dt <= 0:
            raise ValueError('Positive execution dt required')
        self.koopman_metadata = None
        if self.kind not in ('kinematic', 'koopman_k2'):
            raise ValueError('execution.model must be kinematic or koopman_k2')
        if self.kind == 'koopman_k2':
            if koopman_metadata is not None:
                self.koopman_metadata = koopman_metadata
                self.koopman = _make_k2(koopman_metadata)
            else:
                filename = self.cfg.get('checkpoint')
                if not filename:
                    raise ValueError('execution.checkpoint required for koopman_k2 training')
                path = Path(filename).expanduser()
                if not path.is_absolute():
                    from highway_env.planner.diffusion.config.experiment import resolve_repo_file
                    path = resolve_repo_file(path)
                if not path.is_file():
                    raise FileNotFoundError(f'K2 checkpoint not found: {path}')
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                self.koopman_metadata = _metadata_from_checkpoint(checkpoint)
                self.koopman = _make_k2(self.koopman_metadata)
                self.koopman.load_state_dict(checkpoint['model_state'], strict=True)
            self.koopman.requires_grad_(False)
            self.koopman.eval()

    def forward(self, controls, features, context=None):
        """controls [B,M,H,2] -> poses [B,M,H+1,3], speeds [B,M,H+1]."""
        if controls.ndim != 4 or controls.shape[-1] != 2:
            raise ValueError('Vehicle execution controls must be [B,M,H,2]')
        b, m, h, _ = controls.shape
        flat = controls.reshape(b * m, h, 2)
        if self.kind == 'kinematic':
            vx = features['ego_state'][:, 0]
            poses, speeds = rollout_bicycle(flat, vx.repeat_interleave(m), self.dt, self.wheelbase_m)
            return {
                'poses': poses.reshape(b, m, h + 1, 3),
                'speeds': speeds.reshape(b, m, h + 1),
                'states': None,
                'ltr': None,
            }

        ctx = _require_context(context, K2_CONTEXT, b, device=flat.device, dtype=flat.dtype)
        repeat = lambda x: x.repeat_interleave(m, dim=0)
        conditions = torch.stack((ctx['koopman_payload_kg'].reshape(b),
                                  ctx['koopman_road_mu'].reshape(b)), dim=-1)
        initial_pose = ctx['koopman_pose0_si']
        # Keep deterministic K2 behavior during planner.train(), but preserve
        # autograd wrt *input controls*: never wrap this call in no_grad().
        self.koopman.eval()
        out = self.koopman.predict_all(
            repeat(ctx['koopman_state_hist_si']), repeat(ctx['koopman_control_hist_si']),
            flat, initial_pose=repeat(initial_pose), dt=self.dt,
            conditions=repeat(conditions),
        )
        poses_world = out['pose'].reshape(b, m, h + 1, 3)
        dx = poses_world[..., 0] - initial_pose[:, None, None, 0]
        dy = poses_world[..., 1] - initial_pose[:, None, None, 1]
        psi0 = initial_pose[:, None, None, 2]
        co, si = torch.cos(psi0), torch.sin(psi0)
        poses_local = torch.stack((co * dx + si * dy, -si * dx + co * dy,
                                   poses_world[..., 2] - psi0), dim=-1)
        states = out['states'].reshape(b, m, h + 1, 5)
        ltr = out['ltr'].reshape(b, m, h + 1)
        return {'poses': poses_local, 'speeds': states[..., 0],
                'states': states, 'ltr': ltr}
