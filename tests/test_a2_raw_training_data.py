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


def test_parallel_candidates_labels_progress_and_worker_independent_reuse(tmp_path, capsys):
    import torch
    from highway_env.planner.candidate_cache import build_cache
    from highway_env.planner.anchor_scoring.safety import anchor_safety_cost
    root = tmp_path / 'raw'
    root.mkdir()
    create_raw(root)
    generation = {'pool_scale': 1}
    shared = tmp_path / 'shared'
    build_cache(root, shared, generation, ('train',), workers=2)
    cfg = dict(max_samples=8, precompute_workers=2, precompute_batch_size=3,
               precompute_device='cpu', temperature=.3, candidate_generation=generation,
               teacher={'ego_length_m': 8., 'ego_width_m': 2.6})
    cache, manifest = prepare_split_cache(root, 'train', cfg, tmp_path / 'a2')
    assert manifest['num_samples'] == 8
    assert [record['count'] for record in manifest['episodes'][:2]] == [6, 2]
    arrays = {key: np.load(cache / f'{key}.npy') for key in
              ('coarse_controls', 'coarse_trajectories', 'mode_valid_mask', 'safety_total',
               'safety_collision', 'safety_road', 'safety_hard_safe', 'teacher_target')}
    for key in ('coarse_controls', 'coarse_trajectories', 'mode_valid_mask'):
        np.testing.assert_array_equal(arrays[key], np.load(shared / 'train' / f'{key}.npy')[:8])
    dataset = RawExpertDataset(root, 'train', max_samples=8)
    # Independently evaluate the unchanged teacher in one batch.
    features = {key: torch.stack([dataset[index][0][key] for index in range(8)])
                for key in dataset[0][0]}
    xy = torch.from_numpy(arrays['coarse_trajectories'])
    valid = torch.from_numpy(arrays['mode_valid_mask'])
    with torch.inference_mode():
        labels = anchor_safety_cost(features, xy, valid=valid, **cfg['teacher'])
        target = torch.softmax((-labels['total'] / cfg['temperature']).masked_fill(~valid, -torch.inf), -1)
    for name, label in (('safety_total', 'total'), ('safety_collision', 'collision'),
                        ('safety_road', 'road'), ('safety_hard_safe', 'hard_safe')):
        np.testing.assert_allclose(arrays[name], labels[label].numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(arrays['teacher_target'], target.numpy(), rtol=1e-6, atol=1e-6)
    assert 'precompute_workers' not in manifest['fingerprint_payload']
    cfg['precompute_workers'] = 1
    reused, reused_manifest = prepare_split_cache(root, 'train', cfg, tmp_path / 'a2')
    assert reused == cache and reused_manifest['fingerprint'] == manifest['fingerprint']
    output = capsys.readouterr().out
    assert 'precompute_workers=2' in output and 'done=8/8 (100.0%)' in output
    assert 'episodes=2/2' in output and 'eta=' in output and 'reuse split=train' in output


@pytest.mark.parametrize('workers', [None, 2])
def test_a2_precompute_cli_yaml_override(tmp_path, monkeypatch, workers):
    from scripts.data import train_a2_anchors
    from highway_env.planner.anchor_scoring import training, prepare_cache
    cfg = {'candidate_generation': {'pool_scale': 1}, 'precompute_workers': 3,
           'dataset_root': str(tmp_path / 'raw'), 'output_dir': str(tmp_path / 'model'),
           'cache_dir': str(tmp_path / 'cache')}
    calls = []
    def prepare(root, split, settings, cache, **kwargs):
        calls.append((split, settings['precompute_workers']))
        return cache / split, {}
    monkeypatch.setattr(training, 'load_scorer_config', lambda path: dict(cfg))
    monkeypatch.setattr(prepare_cache, 'load_scorer_config', lambda path: dict(cfg))
    monkeypatch.setattr(training, 'prepare_split_cache', prepare)
    monkeypatch.setattr(prepare_cache, 'prepare_split_cache', prepare)
    options = [] if workers is None else ['--precompute-workers', str(workers)]
    train_a2_anchors.main(['--device', 'cpu', '--prepare-cache-only', *options])
    prepare_cache.main([*options])
    assert calls == [('train', workers or 3), ('val', workers or 3)] * 2


def test_episode_pool_worker_failure_is_not_swallowed(tmp_path):
    from types import SimpleNamespace
    from highway_env.planner.candidate_cache import iter_episode_pools
    dataset = SimpleNamespace(_index=[(0, 0)], shard_paths=[tmp_path / 'missing.npz'])
    with pytest.raises(FileNotFoundError):
        list(iter_episode_pools(dataset, {'pool_scale': 1}, workers=1))


def test_episode_pool_submits_only_after_consumer_advances(monkeypatch):
    from types import SimpleNamespace
    from concurrent.futures import Future
    from highway_env.planner import candidate_cache
    submitted = []
    class Executor:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, fn, path, rows, config):
            submitted.append(path)
            future = Future()
            future.set_result(({}, {}))
            return future
    monkeypatch.setattr(candidate_cache, 'ProcessPoolExecutor', Executor)
    dataset = SimpleNamespace(_index=[(index, 0) for index in range(5)], shard_paths=list(range(5)))
    stream = candidate_cache.iter_episode_pools(dataset, {}, workers=2)
    next(stream)
    assert len(submitted) == 2
    next(stream)
    assert len(submitted) == 3
    assert len(list(stream)) == 3
    assert submitted == list(range(5))
