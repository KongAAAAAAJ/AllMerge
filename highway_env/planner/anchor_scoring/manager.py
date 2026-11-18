"""A2 safety-scoring adapter. A1 uses planner.a1_anchor_manager."""
from __future__ import annotations

import time
import torch

from highway_env.planner.adaptive_anchors import (
    AdaptiveCandidateBuilder,
)
from .model import AnchorSafetyScorer, select_anchors
from .config import resolve_a2_config


DEFAULT_CANDIDATE_CONFIG = {
    "config_path": "configs/anchors/control_candidate_pool.yaml"
}


class AdaptiveAnchorManager:
    runtime_kind = "a2"
    """A2: safety-ranked Top-R followed by the same joint diversity selection."""

    def __init__(self, config, *, device="cpu", num_selected=None):
        self.config = resolve_a2_config(config)
        generation = self.config.get("candidate_generation", DEFAULT_CANDIDATE_CONFIG)
        self.builder = AdaptiveCandidateBuilder(generation)
        # Persist effective settings, so diffusion checkpoints do not depend on YAML changes.
        self.config['candidate_generation'] = self.builder.resolved_config
        self.num_selected = int(
            num_selected if num_selected is not None
            else self.config.get("num_selected", 10)
        )
        if self.config.get("top_r") is None:
            fraction = float(self.config.get("top_r_fraction", 0.5555556))
            if not 0.0 < fraction <= 1.0:
                raise ValueError("top_r_fraction must be in (0,1]")
            self.top_r = max(
                self.num_selected, int(round(self.builder.num_candidates * fraction))
            )
        else:
            self.top_r = max(self.num_selected, int(self.config["top_r"]))
        if not 1 <= self.num_selected <= self.top_r <= self.builder.num_candidates:
            raise ValueError(
                "A2 requires 1 <= num_selected <= top_r <= "
                f"{self.builder.num_candidates}"
            )
        checkpoint = self.config.get("scorer_checkpoint")
        if not checkpoint:
            raise ValueError("A2 adaptive selection requires a trained scorer_checkpoint")
        from highway_env.planner.diffusion.config.experiment import resolve_repo_file
        checkpoint = resolve_repo_file(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'A2 safety scorer checkpoint missing: {checkpoint}')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload.get('architecture') != 'TRACE_Adaptive_Anchor_Safety_Scorer_V1':
            raise ValueError('Incompatible safety scorer checkpoint')
        trained_generator = payload.get('candidate_generation') or {}
        # Normalize sparse legacy YAML fragments and fully resolved checkpoint
        # configs through the same builder before comparing candidate semantics.
        trained_builder = AdaptiveCandidateBuilder(trained_generator)
        trained_resolved = getattr(
            trained_builder, "resolved_config",
            getattr(getattr(trained_builder, "base", None), "config", trained_generator),
        )
        runtime_resolved = getattr(
            self.builder, "resolved_config",
            getattr(getattr(self.builder, "base", None), "config",
                    self.config.get("candidate_generation") or {}),
        )
        if dict(trained_resolved) != dict(runtime_resolved):
            raise ValueError(
                "A2 candidate generator differs from scorer training configuration "
                "after resolving defaults."
            )
        self.scorer = AnchorSafetyScorer(payload.get('model_config')).eval()
        self.scorer.load_state_dict(payload['model_state'], strict=True)
        for param in self.scorer.parameters():
            param.requires_grad_(False)
        self.scorer.to(device)
        self.last_latency_ms = 0.0
        self.last_scores = None
        self.last_selected_indices = None
        self.last_valid_counts = None

    @torch.no_grad()
    def select(self, features):
        started = time.perf_counter()
        device = features["ego_state"].device
        if next(self.scorer.parameters()).device != device:
            self.scorer.to(device)
        if "a2_candidate_controls" in features:
            controls = torch.as_tensor(features["a2_candidate_controls"], device=device)
            xy = torch.as_tensor(features["a2_candidate_trajectories"], device=device)
            mask = torch.as_tensor(features["a2_candidate_valid"], device=device)
        else:
            pool = self.builder.build_batch(features)
            controls = torch.as_tensor(pool["coarse_controls"], device=device)
            xy = torch.as_tensor(pool["coarse_trajectories"], device=device)
            mask = torch.as_tensor(pool["mode_valid_mask"], device=device)
        scores = self.scorer(features, controls, xy)
        index = select_anchors(
            scores, controls, xy, mask,
            top_r=self.top_r, num_selected=self.num_selected,
            trajectory_weight=float(self.config.get("trajectory_weight", 1.0)),
            control_weight=float(self.config.get("control_weight", 1.0)),
        )
        batch = torch.arange(controls.shape[0], device=device)[:, None]
        selected = {key: value for key, value in features.items()
                    if not key.startswith("a2_candidate_")}
        selected["coarse_controls"] = controls[batch, index]
        selected["coarse_trajectories"] = xy[batch, index]
        selected["mode_valid_mask"] = mask[batch, index]
        self.last_scores = scores.detach()
        self.last_selected_indices = index.detach()
        self.last_valid_counts = mask.sum(dim=1).detach().cpu().tolist()
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return selected
