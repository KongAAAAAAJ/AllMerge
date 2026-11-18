"""The same A1 builder must serve offline data, raw evaluation and live scenes."""
import numpy as np
import pytest
import torch

from tests.test_a1_online_runtime import _live_geometry
from highway_env.planner.a1_anchor_builder import A1AnchorBuilder
from highway_env.planner.a1_anchor_manager import A1AnchorManager
from evaluation.ablation.anchor_ablation.sources import AnchorSource, build_sources


@pytest.mark.parametrize('count', [5, 10, 20])
def test_offline_online_and_raw_source_match_for_variable_k(count):
    manager = A1AnchorManager(num_selected=count)
    geometry = _live_geometry(batch=2)
    features = {'ego_state': torch.tensor([[20.] + [0.] * 13, [23.] + [0.] * 13]),
                **{key: torch.as_tensor(value) for key, value in geometry.items()}}
    generated = AnchorSource({'strategy': 'a1'}, manager)._prepare_anchors(features)
    assert generated['coarse_controls'].shape == (2, count, 40, 2)
    assert generated['coarse_trajectories'].shape == (2, count, 8, 2)
    assert generated['mode_valid_mask'].shape == (2, count)
    offline = A1AnchorBuilder(manager.builder_config).build({
        'ego_state': features['ego_state'][0].numpy(),
        **{key: value[0] for key, value in geometry.items()},
    })
    np.testing.assert_allclose(generated['coarse_controls'][0], offline['a1_coarse_controls'])
    np.testing.assert_allclose(generated['coarse_trajectories'][0], offline['a1_coarse_trajectories'])
    for i in range(2):
        single = manager.select({key:value[i:i+1] for key,value in features.items()})
        for key in ('coarse_controls','coarse_trajectories','mode_valid_mask'):
            torch.testing.assert_close(generated[key][i:i+1], single[key])


def test_anchor_evaluator_has_only_new_a1_path():
    from highway_env.planner.diffusion.config.experiment import load_mode_yaml
    models, _, artifacts = build_sources(['A1'], {})
    config = load_mode_yaml('configs/train/diffusion.yaml')
    assert isinstance(models['A1'].manager, A1AnchorManager)
    assert models['A1'].manager.num_selected == config['model'].get('num_modes', 10)
    assert models['A1'].anchors_config['strategy'] == 'a1'
    assert artifacts['A1']['source'] == 'online A1AnchorBuilder'
