import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('a0_raw_builder', ROOT / 'scripts/data/build_a0_fixed_anchors.py')
a0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a0)


def history(frames):
    return np.asarray([[[step * .01, step * .0001] for step in range(int(k)-4, int(k))]
                       for k in frames], dtype=np.float32)


def test_recovers_true_controls_without_mixing_egos():
    frames = np.tile(np.arange(4, 51), 2)
    egos = np.repeat([0, 1], 47)
    h = history(frames)
    h[47:, :, 0] += .5
    windows = a0.restore_expert_windows(h, frames, egos)
    assert windows.shape == (22, 40, 2)
    np.testing.assert_allclose(windows[0, :, 0], np.arange(40) * .01, atol=1e-7)
    np.testing.assert_allclose(windows[11, :, 0], np.arange(40) * .01 + .5, atol=1e-7)


def test_gaps_and_short_episodes_are_not_padded():
    for frames in (np.arange(4, 20), np.r_[np.arange(4, 20), np.arange(30, 51)]):
        assert len(a0.restore_expert_windows(history(frames), frames, np.zeros(len(frames), int))) == 0


def test_conflicting_overlap_is_rejected():
    frames = np.arange(4, 51)
    h = history(frames)
    h[1, 0, 0] = 99
    with pytest.raises(ValueError, match='Inconsistent'):
        a0.restore_expert_windows(h, frames, np.zeros(len(frames), int))


def raw_dataset(root, n=10):
    (root/'episodes').mkdir(parents=True)
    (root/'reports').mkdir()
    records = []
    for episode in range(n):
        frames = np.arange(4, 51)
        name = f'episode_{episode:06d}.npz'
        np.savez(root/'episodes'/name, koopman_control_hist_si=history(frames),
                 frame_index=frames, ego_index=np.zeros(len(frames), int),
                 episode_index=np.full(len(frames), episode))
        records.append(dict(name=name, episode_index=episode, samples=len(frames)))
    manifest = dict(schema_version='trace_expert_raw_k2_v2', complete=True,
                    trajectory={'dt_s': .1}, episode_files=records,
                    split_ratios=[.8,.1,.1], split_seed=0)
    (root/'reports/manifest.json').write_text(json.dumps(manifest))


def test_split_matches_downstream_and_uses_only_training(tmp_path):
    raw_dataset(tmp_path)
    paths, _, audit = a0.raw_training_episodes(tmp_path)
    from data_pipeline.raw_expert_dataset import raw_episode_paths
    val, _, _ = raw_episode_paths(tmp_path, 'val')
    test, _, _ = raw_episode_paths(tmp_path, 'test')
    assert not set(paths) & (set(val) | set(test))
    windows, _ = a0.load_raw_expert_controls(tmp_path)
    assert len(windows) == len(paths) * 11
    assert audit['split_counts']['train'] == len(paths)


def test_build_raw_bank_loads_with_existing_manager(tmp_path):
    from highway_env.planner.fixed_anchor_manager import FixedAnchorManager
    raw_dataset(tmp_path)
    dest = tmp_path/'bank.npz'
    result = a0.build_fixed_anchor_bank(tmp_path, dest, num_anchors=3, epochs=2, lloyd_refine=1)
    assert result['control_source_key'] == 'koopman_control_hist_si'
    manager = FixedAnchorManager({'bank_path': str(dest)}, num_modes=3)
    assert manager.metadata['control_source_semantics'].startswith('raw_executed')
    assert manager.anchors_controls.shape == (3, 40, 2)
    windows, _ = a0.load_raw_expert_controls(tmp_path)
    for anchor in manager.anchors_controls.numpy():
        assert any(np.array_equal(anchor, window) for window in windows)
    assert result['bank_sha256'] == manager.bank_sha256


def test_exemplars_are_nearest_members_of_their_own_cluster():
    x = np.asarray([[0.], [2.], [3.], [10.], [12.]], dtype=np.float32)
    centers = np.asarray([[1.8], [11.5]], dtype=np.float32)
    indices = a0.select_cluster_exemplars(x, centers, chunk_size=2)
    np.testing.assert_array_equal(indices, [1, 4])


def test_range_filter_preserves_full_windows_and_values():
    controls = np.zeros((4,40,2),np.float32)
    controls[0,:,0] = .05
    controls[1,2,0] = .2
    controls[2,7,1] = .1
    controls[3,2,0] = .2
    controls[3,7,1] = .1
    kept,audit = a0.filter_control_range(controls,max_accel_mps2=.1,max_steer_rad=.08)
    np.testing.assert_array_equal(kept,controls[:1])
    assert kept.shape==(1,40,2)
    assert audit['excluded_windows']==3
    assert audit['acceleration_exceeded_windows']==2
    assert audit['steering_exceeded_windows']==2
    assert audit['control_clipping_used'] is False


def test_no_in_range_windows_fails_before_clustering(tmp_path, monkeypatch):
    raw_dataset(tmp_path)
    def fail(*args, **kwargs):
        raise AssertionError('Clustering must not start with no valid data')
    monkeypatch.setattr(a0,'minibatch_kmeans',fail)
    with pytest.raises(ValueError,match='Only 0 complete in-range windows'):
        a0.build_fixed_anchor_bank(tmp_path,tmp_path/'bank.npz',num_anchors=1,max_accel_mps2=.1)
