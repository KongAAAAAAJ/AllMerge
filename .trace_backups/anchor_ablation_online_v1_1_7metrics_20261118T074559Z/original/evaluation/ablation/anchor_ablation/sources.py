"""Anchor-only sources sharing the same K and candidate-generation config as training."""
from pathlib import Path

import torch
import yaml
from learning.config import load_yaml_config

from highway_env.planner.diffusion.config.experiment import resolve_repo_file, load_mode_yaml


class AnchorSource:
    def __init__(self, config, manager=None):
        self.anchors_config = config
        self.manager = manager

    def _prepare_anchors(self, features):
        if getattr(self.manager, 'runtime_kind', None) == 'a1' and 'coarse_controls' not in features:
            features = {**features, '_a1_online_runtime': True}
        return features if self.manager is None else self.manager.select(features)


def _resolve_num_modes(diffusion_config: str) -> int:
    path = resolve_repo_file(diffusion_config)
    raw = load_yaml_config(path)
    value = (((raw.get("control") or {}).get("model") or {}).get("num_modes", 10))
    value = int(value)
    if value <= 0:
        raise ValueError(f"control.model.num_modes must be positive, got {value}")
    return value


def build_sources(methods, cfg, *, bank_path=None, scorer_path=None,
                  scorer_config='configs/train/adaptive_anchors.yaml',
                  diffusion_config='configs/train/diffusion.yaml', device='cpu'):
    """Build A0/A1/A2 sources with a single K=control.model.num_modes."""
    settings = cfg.get('anchor_sources', {})
    scorer_cfg_path = resolve_repo_file(scorer_config)
    scorer_cfg = yaml.safe_load(scorer_cfg_path.read_text(encoding="utf-8"))
    teacher = scorer_cfg['teacher']
    num_modes = _resolve_num_modes(diffusion_config)
    candidate_generation = scorer_cfg.get(
        'candidate_generation',
        {'config_path': 'configs/anchors/control_candidate_pool.yaml'},
    )
    sources, artifacts = {}, {}
    for method in methods:
        if method == 'A0':
            from highway_env.planner.fixed_anchor_manager import FixedAnchorManager
            config = {'strategy': 'fixed', 'num_selected': num_modes,
                      'bank_path': bank_path or settings.get('a0_bank',
                          'outputs/anchor_ablation/ckpt/a0_fixed_control_anchors.npz')}
            manager = FixedAnchorManager(
                config, num_modes=num_modes, **settings.get('a0_rollout', {})
            )
            artifacts[method] = {'bank_path': str(manager.bank_path),
                                 'bank_sha256': manager.bank_sha256}
        elif method == 'A1':
            from highway_env.planner.a1_anchor_manager import A1AnchorManager
            diffusion = load_mode_yaml(diffusion_config)
            dynamics = diffusion['control']
            config = {**diffusion.get('anchors', {}), 'strategy': 'a1', 'num_selected': num_modes}
            manager = A1AnchorManager(
                config, num_selected=num_modes,
                control_steps=dynamics['horizon_steps'], dt=dynamics['dt'],
                wheelbase_m=dynamics['wheelbase_m'], max_accel_mps2=dynamics['max_accel_mps2'],
                max_steer_rad=dynamics['max_steer_rad'])
            artifacts[method] = {'source': 'shared offline/online A1AnchorBuilder',
                                 'a1_config_path': manager.config_path,
                                 'a1_parameters': manager.builder_config}
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
            config = {
                'strategy': 'adaptive', 'num_selected': num_modes,
                'scorer_checkpoint': str(path),
                'candidate_generation': payload.get('candidate_generation', {}),
                'top_r_fraction': float(scorer_cfg.get('top_r_fraction', 0.5555556)),
                'top_r': scorer_cfg.get('top_r'),
                'trajectory_weight': float(scorer_cfg.get('trajectory_weight', 1.0)),
                'control_weight': float(scorer_cfg.get('control_weight', 1.0)),
            }
            manager = AdaptiveAnchorManager(config, device=device, num_selected=num_modes)
            artifacts[method] = {
                'scorer_path': str(path), 'epoch': payload.get('epoch'),
                'candidate_pool_size': manager.builder.num_candidates,
                'top_r': manager.top_r, 'num_selected': manager.num_selected,
            }
        config['num_selected'] = num_modes
        artifacts[method]['anchors_config'] = config
        sources[method] = AnchorSource(config, manager)
    return sources, teacher, artifacts
