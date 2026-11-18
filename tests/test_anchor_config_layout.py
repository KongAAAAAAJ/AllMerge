"""Shared A2 settings must match across scorer and diffusion, and survive reload."""
from unittest.mock import patch

import pytest
import torch
import yaml

from highway_env.planner.adaptive_anchors import AdaptiveCandidateBuilder
from highway_env.planner.anchor_scoring.config import load_scorer_config, resolve_a2_config
from highway_env.planner.anchor_scoring.model import AnchorSafetyScorer
from highway_env.planner.diffusion.config.experiment import load_mode_yaml
from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner
from highway_env.planner.diffusion.training.control import load_control_checkpoint


def test_shared_policy_and_common_diffusion_settings():
    base = load_mode_yaml('configs/train/diffusion.yaml')
    fixed = load_mode_yaml('configs/train/ablation/diffusion_a0_fixed.yaml')
    adaptive = load_mode_yaml('configs/train/ablation/diffusion_a2_adaptive.yaml')
    scorer = load_scorer_config('configs/train/a2_scorer.yaml')
    policy = resolve_a2_config(adaptive['anchors'])
    for key in ('candidate_generation','top_r_fraction','trajectory_weight','control_weight'):
        assert policy[key] == scorer[key]
    for variant in (fixed, adaptive):
        for key in ('model','control','execution','losses'):
            assert variant[key] == base[key]
        assert 'a1_config' not in variant['anchors']
    assert 'scorer_checkpoint' not in base['anchors']
    assert 'top_r' not in base['anchors']


def test_a2_checkpoint_uses_effective_policy_without_reloading_yaml(tmp_path):
    small = dict(d_model=32,d_ffn=64,num_heads=4,num_modes=3,
                 num_scene_layers=0,num_denoiser_layers=1,dropout=0.)
    scorer_path = tmp_path/'scorer.pt'
    generation = AdaptiveCandidateBuilder({}).resolved_config
    scorer = AnchorSafetyScorer(small)
    torch.save(dict(architecture='TRACE_Adaptive_Anchor_Safety_Scorer_V1',
                    model_config=small,model_state=scorer.state_dict(),
                    candidate_generation=generation),scorer_path)
    policy_path = tmp_path/'a2.yaml'
    policy_path.write_text(yaml.safe_dump(dict(a2=dict(
        candidate_generation=generation,top_r_fraction=.5,
        trajectory_weight=.7,control_weight=1.3))))
    model = ControlDiffusionPlanner(small,execution_config={'model':'kinematic'},
        anchors_config=dict(strategy='adaptive',a2_config=str(policy_path),
                            scorer_checkpoint=str(scorer_path)))
    assert model.anchor_manager.top_r == 18
    assert 'a2_config' not in model.anchors_config
    checkpoint = tmp_path/'diffusion.pt'
    torch.save(dict(model_config=small,control_config={},
                    execution_config={'model':'kinematic'},
                    anchors_config=model.anchors_config,model_state=model.state_dict()),checkpoint)
    policy_path.unlink()
    with patch.object(yaml,'safe_load',side_effect=AssertionError('unexpected YAML read')):
        loaded,_ = load_control_checkpoint(checkpoint)
    assert loaded.anchor_manager.top_r == 18
    assert loaded.anchor_manager.config['trajectory_weight'] == .7


def test_malformed_policy_is_rejected(tmp_path):
    path = tmp_path/'bad.yaml'
    path.write_text('candidate_generation: {}\n')
    with pytest.raises(ValueError,match='a2 mapping'):
        resolve_a2_config({'a2_config':str(path)})


def test_policy_edits_reach_scorer_and_diffusion_without_duplicate_overrides(tmp_path):
    from highway_env.planner.diffusion.config.experiment import resolve_repo_file
    raw_training = yaml.safe_load(resolve_repo_file('configs/train/a2_scorer.yaml').read_text())
    policy_path = tmp_path/'policy.yaml'
    policy_path.write_text(yaml.safe_dump(dict(a2=dict(
        candidate_generation={},top_r_fraction=.4,trajectory_weight=.3,control_weight=.8))))
    raw_training['a2_config'] = str(policy_path)
    training_path = tmp_path/'scorer.yaml'
    training_path.write_text(yaml.safe_dump(raw_training))
    training = load_scorer_config(training_path)
    runtime = resolve_a2_config({'a2_config':str(policy_path), 'strategy':'adaptive'})
    for key in ('candidate_generation','top_r_fraction','trajectory_weight','control_weight'):
        assert training[key] == runtime[key]
