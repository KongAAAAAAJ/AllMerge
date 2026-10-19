"""Lightweight non-CUDA checks for ranking, shape, zoom and CLI code."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
from .selectors import select_topk
from .metrics import metric_row
from .plot_utils import zoom_window
from .gallery import render_one


def main():
    xy=np.stack((np.linspace(0,40,41),np.linspace(0,2,41)),axis=-1)
    post=xy.copy();post[:,1]+=np.sin(np.linspace(0,np.pi,41))*1.0
    teacher=post.copy();teacher[:,1]+=np.sin(np.linspace(0,np.pi,41))*.2
    r=metric_row(xy,post,teacher,{'pre':-12,'post':-9,'teacher':-8},{'scenario':'curved','state_index':2,'group_id':7,'mode':1,'role':0})
    assert r['train_reward_gain']==3.0 and r['teacher_distance_improve_m']>0
    rows=[r,{**r,'state_index':3,'scenario':'merge_in','train_reward_gain':-1}]
    chosen,meta=select_topk(rows,k=2)
    assert len(chosen)==1 and chosen[0]['state_index']==2
    bounds=zoom_window(xy,post,teacher,int(r['max_dense_shift_index']))
    assert bounds[0]<bounds[1] and bounds[2]<bounds[3]
    with tempfile.TemporaryDirectory() as tmp:
        fn=Path(tmp)/'zoom.png'
        render_one(chosen[0],dict(pre_dense=xy,post_dense=post,teacher_dense=teacher),fn,dpi=80)
        assert fn.is_file() and fn.with_suffix('.pdf').is_file()
    # Test the mode-locked exact-t0 decoder without the real CUDA model.
    import torch
    from .collect import decode_paired
    class Adapter:
        def normalize_trajectory(self,x):return x
        def denormalize_trajectory(self,x):return x
    class Model:
        adapter=Adapter()
        def scene_encoder(self,features):return None
        def denoiser(self,x,t,scene,navigation_point_norm):
            assert tuple(x.shape)==(1,3,8,2) and int(t[0])==0
            return x+1,None
    features={'coarse_trajectories':torch.zeros(3,8,2)}
    out=decode_paired(Model(),features,torch.zeros(3,8,2),1,torch.device('cpu'))
    assert tuple(out.shape)==(8,2) and torch.all(out==1)
    print('[PASS] synthetic paired-t0 decoder, metrics, full-state selector, geometric zoom, PDF/PNG',flush=True)


if __name__=='__main__':main()
