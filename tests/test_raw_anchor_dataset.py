import json
import numpy as np
import pytest

from data_pipeline.raw_expert_dataset import RawExpertDataset, raw_episode_paths
from evaluation.ablation.anchor_ablation.data import build_complete_frame_subset, make_loader


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
