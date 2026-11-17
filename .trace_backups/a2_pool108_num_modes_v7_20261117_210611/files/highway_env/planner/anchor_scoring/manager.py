"""Independent A2 runtime adapter; reuses the original planner feature contract."""
from __future__ import annotations

from pathlib import Path
import torch

from highway_env.planner.adaptive_anchors import AdaptiveCandidateBuilder
from .model import AnchorSafetyScorer, select_anchors


class AdaptiveAnchorManager:
    def __init__(self, config, *, device='cpu'):
        self.config = dict(config)
        self.builder = AdaptiveCandidateBuilder(self.config.get('candidate_generation'))
        self.num_selected = int(self.config.get('num_selected', 10))
        self.top_r = int(self.config.get('top_r', 20))
        if not 1 <= self.num_selected <= self.top_r <= self.builder.num_candidates:
            raise ValueError('Adaptive anchors require 1<=num_selected<=top_r<=36')
        checkpoint = self.config.get('scorer_checkpoint')
        if not checkpoint:
            raise ValueError('A2 adaptive selection requires a trained scorer_checkpoint')
        from highway_env.planner.diffusion.config.experiment import resolve_repo_file
        checkpoint = resolve_repo_file(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'A2 safety scorer checkpoint missing: {checkpoint}')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload.get('architecture') != 'TRACE_Adaptive_Anchor_Safety_Scorer_V1':
            raise ValueError('Incompatible safety scorer checkpoint')
        trained_generator = payload.get('candidate_generation', {})
        if trained_generator != (self.config.get('candidate_generation') or {}):
            raise ValueError('A2 candidate generator differs from scorer training configuration')
        self.scorer = AnchorSafetyScorer(payload.get('model_config')).eval()
        self.scorer.load_state_dict(payload['model_state'], strict=True)
        for param in self.scorer.parameters():
            param.requires_grad_(False)
        self.scorer.to(device)

    @torch.no_grad()
    def select(self, features):
        device = features['ego_state'].device
        if next(self.scorer.parameters()).device != device:
            self.scorer.to(device)
        if 'a2_candidate_controls' in features:
            controls = torch.as_tensor(features['a2_candidate_controls'], device=device)
            xy = torch.as_tensor(features['a2_candidate_trajectories'], device=device)
            mask = torch.as_tensor(features['a2_candidate_valid'], device=device)
        else:
            pool = self.builder.build_batch(features)
            controls = torch.as_tensor(pool['coarse_controls'], device=device)
            xy = torch.as_tensor(pool['coarse_trajectories'], device=device)
            mask = torch.as_tensor(pool['mode_valid_mask'], device=device)
        scores = self.scorer(features, controls, xy)
        index = select_anchors(scores, controls, xy, mask,
            top_r=self.top_r, num_selected=self.num_selected,
            trajectory_weight=float(self.config.get('trajectory_weight', 1.0)),
            control_weight=float(self.config.get('control_weight', 1.0)))
        batch = torch.arange(controls.shape[0], device=device)[:, None]
        # All selected candidates are geometrically available; traffic risks
        # were ranked by scorer, not forcibly filtered before scoring.
        selected = {key: value for key, value in features.items()
                    if not key.startswith('a2_candidate_')}
        selected['coarse_controls'] = controls[batch, index]
        selected['coarse_trajectories'] = xy[batch, index]
        selected['mode_valid_mask'] = mask[batch, index]
        return selected
