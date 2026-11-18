"""Raw-only A2 cache round trips and shared A1 candidate geometry."""
import json
import numpy as np
import pytest

from tests.test_raw_anchor_dataset import create_raw
from tests.test_a1_online_runtime import _live_geometry
from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.planner.a1_anchor_builder import A1AnchorBuilder
from highway_env.planner.adaptive_anchors import AdaptiveCandidateBuilder
from highway_env.planner.anchor_scoring.training_data import (
    prepare_split_cache, build_cached_loader, CachedAnchorScoringDataset,
)


def test_raw_cache_roundtrip_and_a1_candidate_parity(tmp_path):
    root = tmp_path / 'raw'
    root.mkdir()
    create_raw(root)
    geometry = _live_geometry(batch=6)
    maps, map_valid = A1AnchorBuilder._compact_map_features(
        {key: value[0] for key, value in geometry.items()})
    for path in (root / 'episodes').glob('*.npz'):
        with np.load(path) as source:
            values = {key: source[key].copy() for key in source.files}
        values.update(geometry)
        values['map_polylines'] = np.repeat(maps[None], 6, axis=0)
        values['map_valid_mask'] = np.repeat(map_valid[None], 6, axis=0)
        values['ego_state'][:, 0] = 20.
        np.savez(path, **values)
    cfg = dict(max_samples=3, precompute_device='cpu', batch_size=2, num_workers=0,
               candidate_generation={'config_path': 'configs/anchors/control_candidate_pool.yaml'})
    cache, manifest = prepare_split_cache(root, 'train', cfg, tmp_path / 'cache')
    assert manifest['num_samples'] == 3
    assert all(item['name'].startswith('episode_') for item in manifest['episodes'])
    loader, _ = build_cached_loader(root, 'train', cfg, cache, shuffle=False)
    features, cached = next(iter(loader))
    assert 'coarse_controls' not in features
    assert np.isfinite(cached['teacher_target']).all()
    np.testing.assert_allclose(cached['teacher_target'].sum(1), 1., atol=1e-6)
    raw = RawExpertDataset(root, 'train', max_samples=3)
    sample, _, _ = raw[0]
    builder = A1AnchorBuilder({'num_selected': 20})
    maps, valid = builder._compact_map_features({key: value.numpy() for key, value in sample.items()})
    expected = builder.candidate_builder.build_single(
        ego_state=sample['ego_state'].numpy(), map_polylines=maps, map_valid_mask=valid,
        target_acceleration=float(sample['ego_state'][2]))
    np.testing.assert_array_equal(cached['coarse_controls'][0], expected['coarse_controls'])
    np.testing.assert_array_equal(cached['coarse_trajectories'][0], expected['coarse_trajectories'])
    reused, _ = prepare_split_cache(root, 'train', cfg, tmp_path / 'cache')
    assert reused == cache
    manifest['cache_version'] = 'legacy'
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='obsolete'):
        CachedAnchorScoringDataset(raw, cache)


def test_a2_rejects_legacy_shard_dataset(tmp_path):
    (tmp_path / 'shards').mkdir()
    with pytest.raises(FileNotFoundError):
        prepare_split_cache(tmp_path, 'train', {}, tmp_path / 'cache')
