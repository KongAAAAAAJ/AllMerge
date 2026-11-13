"""Permutation-equivariant, identity-free scoring and Top-R/farthest selection."""
from __future__ import annotations

import torch
from torch import nn

from highway_env.planner.diffusion.config.model import build_structured_diffusion_config
from highway_env.planner.diffusion.common.encoders import StructuredSceneEncoder
from highway_env.planner.diffusion.common.tensor_adapter import PlannerTensorAdapter


class AnchorSafetyScorer(nn.Module):
    """One score per control/trajectory pair; no mode embedding or mode ID."""

    def __init__(self, model_config=None):
        super().__init__()
        self.config = build_structured_diffusion_config(**(model_config or {}))
        dim = self.config.d_model
        self.adapter = PlannerTensorAdapter(self.config, torch.device('cpu'))
        self.scene_encoder = StructuredSceneEncoder(self.config, self.adapter)
        self.control_encoder = nn.Sequential(nn.Linear(40 * 2, dim), nn.LayerNorm(dim),
                                             nn.GELU(), nn.Linear(dim, dim))
        self.xy_encoder = nn.Sequential(nn.Linear(8 * 2, dim), nn.LayerNorm(dim),
                                        nn.GELU(), nn.Linear(dim, dim))
        self.head = nn.Sequential(nn.Linear(3 * dim, dim), nn.LayerNorm(dim),
                                  nn.GELU(), nn.Linear(dim, dim // 2), nn.GELU(),
                                  nn.Linear(dim // 2, 1))

    @staticmethod
    def _masked_average(tokens, padding):
        valid = (~padding).unsqueeze(-1)
        return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def forward(self, features, controls, xy):
        if controls.ndim != 4 or controls.shape[-2:] != (40, 2):
            raise ValueError('controls must be [B,M,40,2]')
        if xy.shape[:2] != controls.shape[:2] or xy.shape[-2:] != (8, 2):
            raise ValueError('xy must be [B,M,8,2]')
        device = controls.device
        for attr in ('ego_scale', 'agent_scale', 'map_scale', 'target_point_scale'):
            x = getattr(self.adapter, attr)
            if x.device != device:
                setattr(self.adapter, attr, x.to(device))
        scene = self.scene_encoder(features)
        context = (scene.ego_token.squeeze(1) + scene.target_token.squeeze(1)
                   + self._masked_average(scene.agent_tokens, scene.agent_padding_mask)
                   + self._masked_average(scene.map_tokens, scene.map_padding_mask)) / 4
        limits = controls.new_tensor([2.5, 0.08])
        ctrl_token = self.control_encoder((controls / limits).flatten(2))
        xy_token = self.xy_encoder((xy / controls.new_tensor([120.0, 24.0])).flatten(2))
        fused = torch.cat((ctrl_token, xy_token,
                           context[:, None].expand(-1, controls.shape[1], -1)), dim=-1)
        return self.head(fused).squeeze(-1)


def select_anchors(scores, controls, xy, valid, *, top_r=20, num_selected=10,
                   trajectory_weight=1.0, control_weight=1.0):
    """Safe-ranked Top-R, then normalized trajectory/control max-min spacing.

    The first selected anchor has the highest score. Remaining candidates
    maximize distance from the selected set. This uses candidate *indices*
    only as transient array positions, not learned semantic identities.
    """
    if scores.shape != valid.shape or controls.shape[:2] != scores.shape or xy.shape[:2] != scores.shape:
        raise ValueError('scores/control/xy/valid candidate dimensions differ')
    if not 1 <= num_selected <= top_r <= scores.shape[1]:
        raise ValueError('Require 1 <= num_selected <= top_r <= M')
    if (valid.sum(-1) < num_selected).any():
        raise ValueError('Fewer valid control candidates than requested output anchors')
    if trajectory_weight < 0 or control_weight < 0 or trajectory_weight + control_weight <= 0:
        raise ValueError('At least one positive distance weight is required')
    with torch.no_grad():
        scores = scores.masked_fill(~valid.bool(), -torch.inf)
        top = scores.topk(top_r, dim=1).indices
        top_valid = valid.gather(1, top)
        batch = torch.arange(scores.shape[0], device=scores.device)[:, None]
        traj = xy[batch, top] / xy.new_tensor([120.0, 24.0])
        ctrl = controls[batch, top] / controls.new_tensor([2.5, 0.08])
        dt = (traj[:, :, None] - traj[:, None]).norm(dim=-1).mean(-1)
        du = (ctrl[:, :, None] - ctrl[:, None]).norm(dim=-1).mean(-1)
        dist = trajectory_weight * dt + control_weight * du
        chosen = torch.zeros((scores.shape[0], num_selected),
                             dtype=torch.long, device=scores.device)
        min_dist = dist[:, 0, :].clone()
        used = torch.zeros_like(top_valid)
        used[:, 0] = True
        for j in range(1, num_selected):
            available = ~used & top_valid
            next_idx = min_dist.masked_fill(~available, -torch.inf).argmax(1)
            chosen[:, j] = next_idx
            used.scatter_(1, next_idx[:, None], True)
            new_dist = dist[batch.squeeze(1), next_idx]
            min_dist = torch.minimum(min_dist, new_dist)
        return top.gather(1, chosen)
