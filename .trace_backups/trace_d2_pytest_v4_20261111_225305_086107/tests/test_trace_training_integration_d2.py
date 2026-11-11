"""Small NPZ shard -> real control trainer -> self-contained model load."""
from __future__ import annotations
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import torch
if importlib.util.find_spec('gymnasium') is None:
    pkg=types.ModuleType('highway_env')
    pkg.__path__=[str(Path(__file__).resolve().parents[1]/'highway_env')]
    sys.modules.setdefault('highway_env',pkg)

from test_trace_execution_d2 import SMALL, CTRL, features, context, create_ckpt
from scripts.training.train_diffusion_control import train,load_control_checkpoint
from highway_env.planner.diffusion.inference.control import ControlDiffusionRuntime


def test_real_shard_loader_train_checkpoint_and_runtime(tmp_path):
    root=tmp_path/'dataset'
    (root/'shards').mkdir(parents=True)
    (root/'splits').mkdir()
    feat=features(2)
    ctx=context(2)
    arrays={k:v.cpu().numpy() for k,v in feat.items()}
    arrays.update({k:v.cpu().numpy() for k,v in ctx.items()})
    arrays['koopman_ltr0']=np.zeros(2,dtype=np.float32)
    arrays['koopman_vx_init_kmh']=np.full(2,72.,dtype=np.float32)
    arrays['future_trajectory_dense']=np.zeros((2,40,2),dtype=np.float32)
    arrays['expert_trajectory_xy']=np.zeros((2,8,2),dtype=np.float32)
    arrays['dense_dt']=np.full(2,.1,dtype=np.float32)
    arrays['trajectory_horizon_s']=np.full(2,4.,dtype=np.float32)
    arrays['target_mode']=np.zeros(2,dtype=np.int64)
    arrays['target_semantic']=np.zeros(2,dtype=np.int64)
    np.savez_compressed(root/'shards'/'shard_000.npz',**arrays)
    for split in ('train','val'):
        (root/'splits'/f'{split}.txt').write_text('shard_000.npz\n')
    cp=tmp_path/'k2.pt'
    create_ckpt(cp)
    cfg=dict(output_space='control',model=SMALL,control=CTRL,
        execution=dict(model='koopman_k2',checkpoint=str(cp)),
        losses=dict(trajectory=dict(type='paper_mse',weight=1.),
                    ltr=dict(weight=0.1,limit=0.4)),
        training=dict(seed=0,batch_size=2,num_workers=0,
                      max_epochs=1,lr=1e-4,weight_decay=0.,grad_clip=1.,patience=0))
    dest=tmp_path/'output'
    best=train(cfg,root=root,output_dir=dest,max_epochs=1,max_steps=1,device='cpu')
    assert best.is_file()
    cp.unlink()
    loaded,payload=load_control_checkpoint(best)
    assert loaded.vehicle_model.kind=='koopman_k2'
    assert not any(p.requires_grad for p in loaded.vehicle_model.koopman.parameters())
    rt=ControlDiffusionRuntime(best,device='cpu',inference_steps=2)
    inputs={k:v.cpu().numpy() for k,v in {**feat,**ctx}.items()}
    output=rt.infer(inputs)
    assert output['trajectory_dense'].shape==(2,40,2)
    assert output['ltr'].shape==(2,41)
    assert np.isfinite(output['trajectory_dense']).all()
