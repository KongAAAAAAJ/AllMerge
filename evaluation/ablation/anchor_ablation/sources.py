"""Anchor-only sources: no diffusion planner or frozen K2 checkpoint needed."""
from pathlib import Path

import torch
import yaml

from highway_env.planner.diffusion.config.experiment import resolve_repo_file


class AnchorSource:
    def __init__(self, config, manager=None):
        self.anchors_config = config
        self.manager = manager

    def _prepare_anchors(self, features):
        return features if self.manager is None else self.manager.select(features)


def build_sources(methods, cfg, *, bank_path=None, scorer_path=None,
                  scorer_config='configs/train/adaptive_anchors.yaml', device='cpu'):
    """Use the same anchor managers as training, with a common safety teacher."""
    settings = cfg.get('anchor_sources', {})
    scorer_cfg_path = resolve_repo_file(scorer_config)
    scorer_cfg = yaml.safe_load(scorer_cfg_path.read_text(encoding='utf-8'))
    teacher = scorer_cfg['teacher']
    sources, artifacts = {}, {}
    for method in methods:
        if method == 'A0':
            from highway_env.planner.fixed_anchor_manager import FixedAnchorManager
            config = {'strategy': 'fixed', 'num_selected': 10,
                      'bank_path': bank_path or settings.get('a0_bank',
                          'outputs/anchor_ablation/a0_fixed_control_anchors.npz')}
            manager = FixedAnchorManager(config, **settings.get('a0_rollout', {}))
            artifacts[method] = {'bank_path': str(manager.bank_path),
                                 'bank_sha256': manager.bank_sha256}
        elif method == 'A1':
            # A1 paired controls/trajectories are already stored in expert shards.
            config, manager = {'strategy': 'dynamic'}, None
            artifacts[method] = {'source': 'dataset_paired_dynamic_anchors'}
        else:
            from highway_env.planner.anchor_scoring.manager import AdaptiveAnchorManager
            path = resolve_repo_file(scorer_path or settings.get('a2_scorer') or
                                     str(Path(scorer_cfg['output_dir']) / 'best.pt'))
            payload = torch.load(path, map_location='cpu', weights_only=True)
            trained_teacher = payload.get('teacher_config')
            if not isinstance(trained_teacher, dict) or not trained_teacher:
                raise ValueError('A2 scorer must contain teacher_config')
            if trained_teacher != teacher:
                raise ValueError('A2 scorer teacher differs from --scorer-config; use its training config')
            config = {'strategy': 'adaptive', 'num_selected': 10,
                      'scorer_checkpoint': str(path),
                      'candidate_generation': payload.get('candidate_generation', {}),
                      'top_r': int(scorer_cfg.get('top_r', 20)),
                      'trajectory_weight': float(scorer_cfg.get('trajectory_weight', 1.0)),
                      'control_weight': float(scorer_cfg.get('control_weight', 1.0))}
            manager = AdaptiveAnchorManager(config, device=device)
            artifacts[method] = {'scorer_path': str(path), 'epoch': payload.get('epoch')}
        artifacts[method]['anchors_config'] = config
        sources[method] = AnchorSource(config, manager)
    return sources, teacher, artifacts
