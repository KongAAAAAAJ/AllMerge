import json
import pickle
import numpy as np
import pytest

from data_pipeline.raw_expert_dataset import RawExpertDataset, raw_episode_paths
from evaluation.ablation.anchor_ablation.data import build_complete_frame_subset, make_loader, read_subset_metadata


def create_raw(root):
    (root/'episodes').mkdir()
    (root/'reports').mkdir()
    records=[]
    for episode in range(10):
        n=6
        arrays=dict(ego_state=np.zeros((n,14),np.float32),
                    agent_states=np.zeros((n,2,11),np.float32),
                    agent_valid_mask=np.ones((n,2),bool),
                    map_polylines=np.zeros((n,1,8,10),np.float32),
                    map_valid_mask=np.ones((n,1),bool), target_point=np.zeros((n,2),np.float32),
                    target_lane_polyline=np.zeros((n,8,10),np.float32),
                    expert_trajectory_xy=np.zeros((n,8,2),np.float32),
                    future_trajectory_dense=np.zeros((n,40,2),np.float32),
                    koopman_state_hist_si=np.zeros((n,5,5),np.float32),
                    koopman_control_hist_si=np.zeros((n,4,2),np.float32),
                    koopman_pose0_si=np.zeros((n,3),np.float32),
                    koopman_payload_kg=np.zeros(n,np.float32),koopman_road_mu=np.ones(n,np.float32),
                    episode_index=np.full(n,episode),frame_index=np.repeat([4,5],3),
                    ego_index=np.tile([0,1,2],2),seed=np.full(n,episode))
        from tests.test_a1_online_runtime import _live_geometry
        arrays.update(_live_geometry(batch=n))
        arrays['agent_states'][:,:,10]=1
        name=f'episode_{episode:06d}.npz'
        np.savez(root/'episodes'/name,**arrays)
        records.append(dict(name=name,episode_index=episode,samples=n))
    (root/'reports/manifest.json').write_text(json.dumps(dict(
        schema_version='trace_expert_raw_k2_v2',complete=True,trajectory={'dt_s':.1},
        episode_files=records,split_ratios=[.8,.1,.1],split_seed=0)))


def test_raw_split_and_features_do_not_fabricate_anchors(tmp_path):
    create_raw(tmp_path)
    splits={s:{p.name for p in raw_episode_paths(tmp_path,s)[0]} for s in ('train','val','test')}
    assert not splits['train'] & splits['test']
    assert not splits['train'] & splits['val']
    assert not splits['test'] & splits['val']
    ds=RawExpertDataset(tmp_path,'test')
    features,targets,meta=ds[0]
    assert 'coarse_controls' not in features
    assert 'coarse_trajectories' not in features
    assert 'target_mode' not in targets
    assert targets['trajectory'].shape==(8,2)
    assert targets['koopman_control_hist_si'].shape==(4,2)
    subset,info=build_complete_frame_subset(tmp_path,'test',num_frames=1)
    assert len(subset)==3 and info['input_format']=='raw'
    features,targets,meta=next(iter(make_loader(subset,batch_size=3)))
    assert targets['trajectory'].shape==(3,8,2)
    assert len({r['frame_index'] for r in meta})==1


def test_raw_only_rejects_old_shard_directory(tmp_path):
    (tmp_path/'shards').mkdir()
    with pytest.raises(FileNotFoundError):
        build_complete_frame_subset(tmp_path,'test')


def test_worker_serialization_does_not_copy_episode_cache(tmp_path):
    create_raw(tmp_path)
    dataset = RawExpertDataset(tmp_path, 'test')
    expected = dataset[0]
    restored = pickle.loads(pickle.dumps(dataset))
    assert dataset._cached is not None
    assert restored._cached is None and restored._cached_index is None
    actual = restored[0]
    for key in expected[0]:
        np.testing.assert_array_equal(actual[0][key], expected[0][key])
    for key in expected[1]:
        np.testing.assert_array_equal(actual[1][key], expected[1][key])
    assert actual[2] == expected[2]


def test_figure_metadata_preserves_nested_subset_order_without_scene_reads(tmp_path, monkeypatch):
    from torch.utils.data import Subset
    create_raw(tmp_path)
    dataset = RawExpertDataset(tmp_path, 'test')
    subset = Subset(Subset(dataset, [5, 0, 3, 2]), [2, 0, 1])
    expected = [dataset[i][2] for i in [3, 5, 0]]
    original_load = np.load
    reads = []

    class MetadataArchive:
        def __init__(self, path, **kwargs):
            self.archive = original_load(path, **kwargs)
            self.files = self.archive.files

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            reads.append(key)
            assert key in ('episode_index', 'frame_index', 'ego_index', 'seed')
            return self.archive[key]

    monkeypatch.setattr(np, 'load', MetadataArchive)
    assert read_subset_metadata(subset) == expected
    assert reads == ['episode_index', 'frame_index', 'ego_index', 'seed']


def test_frame_index_reads_each_npz_field_once_and_keeps_complete_groups(tmp_path, monkeypatch):
    from collections import Counter
    from evaluation.ablation.anchor_ablation import data as evaluation_data
    create_raw(tmp_path)
    dataset = RawExpertDataset(tmp_path, 'test', max_samples=5)
    monkeypatch.setattr(evaluation_data, 'RawExpertDataset', lambda *args, **kwargs: dataset)
    original_load = np.load
    reads = Counter()

    class CountedArchive:
        def __init__(self, path, **kwargs):
            self.archive = original_load(path, **kwargs)
            self.files = self.archive.files

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            reads[key] += 1
            return self.archive[key]

    monkeypatch.setattr(evaluation_data.np, 'load', CountedArchive)
    subset, info = evaluation_data.build_complete_frame_subset(tmp_path, 'test')
    assert len(subset) == 3
    assert info['selected_frames'] == 1
    assert info['incomplete_frames_skipped'] == 1
    assert subset.indices == [0, 1, 2]
    assert reads == Counter({key: 1 for key in (
        'episode_index', 'frame_index', 'ego_index', 'agent_states', 'agent_valid_mask', 'seed')})
