"""B-1 K0 end-to-end tests. Run: python -m unittest tests.test_koopman_k0_b1 -v"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from highway_env.vehicle.koopman.dataset import discover_episodes,episode_split,KoopmanWindowDataset
from highway_env.vehicle.koopman.model import NeuralKoopmanK0,rollout_pose
from highway_env.vehicle.koopman.train import train_k0,load_checkpoint


def create_sample(root,index,steps=66):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    t=np.arange(steps)*.1
    u=np.stack([.4*np.sin(t[:-1]+index),.01*np.cos(2*t[:-1]+index)],axis=1)
    s=np.zeros((steps,5));s[0,0]=12+.1*index
    p=np.zeros((steps,3))
    for k in range(steps-1):
        s[k+1,0]=s[k,0]+.1*u[k,0]
        s[k+1,1]=.9*s[k,1]+.04*u[k,1]
        s[k+1,2]=.9*s[k,2]+.1*u[k,1]
        s[k+1,3]=.95*s[k,3]+.004*s[k,2]
        s[k+1,4]=.9*s[k,4]+.03*s[k,3]
        p[k+1,2]=p[k,2]+.1*s[k+1,2]
        p[k+1,0]=p[k,0]+.1*s[k+1,0]*np.cos(p[k+1,2])
        p[k+1,1]=p[k,1]+.1*s[k+1,0]*np.sin(p[k+1,2])
    f=root/f'case_{index:03d}.npz'
    np.savez_compressed(f,t10_s=t,state10_si=s,pose10_si=p,u_ref10_si=u,
        train_eligible=True,model_id='synthetic',payload_kg=0.,road_mu=.8,
        vx_init_kmh=60.,excitation_sha256=f'{index:064d}')
    f.with_suffix('.accepted.json').write_text(json.dumps({'data_verified':True,'status':'ok'}),encoding='utf-8')
    return f


class TestK0B1(unittest.TestCase):
    def test_prediction_and_gradient(self):
        model=NeuralKoopmanK0(history_steps=5,latent_dim=16,hidden_dim=32)
        x=torch.randn(2,5,5)
        uh=torch.randn(2,4,2)
        uf=torch.randn(2,40,2,requires_grad=True)
        y,z=model(x,uh,uf)
        self.assertEqual(y.shape,(2,41,5))
        self.assertEqual(z.shape,(2,41,16))
        pose=rollout_pose(y,torch.zeros(2,3),.1)
        self.assertEqual(pose.shape,(2,41,3))
        pose[:,-1,:2].sum().backward()
        self.assertIsNotNone(uf.grad)
        self.assertTrue(torch.isfinite(uf.grad).all())

    def test_split_and_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(6):create_sample(tmp,i)
            eps=discover_episodes([tmp],tmp)
            split=episode_split(eps,seed=42)
            self.assertEqual(sum(map(len,split.values())),6)
            self.assertFalse({e.id for e in split['train']} & {e.id for e in split['val']})
            b=KoopmanWindowDataset(eps,history_steps=5,horizon_steps=40,warmup_s=1.)[0]
            self.assertEqual(b['state_hist'].shape,(5,5))
            self.assertEqual(b['control_hist'].shape,(4,2))
            self.assertEqual(b['future_controls'].shape,(40,2))
            self.assertEqual(b['target_states'].shape,(41,5))
            self.assertTrue(torch.equal(b['state_hist'][-1],b['target_states'][0]))

    def test_training_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(5):create_sample(root/'data',i)
            cfg={'data':{'roots':['data'],'accepted_only':True,'dt_s':.1,
                         'history_steps':5,'horizon_steps':40,'warmup_s':1.,
                         'train_stride':3,'eval_stride':5,'val_fraction':.2,'test_fraction':.2},
                 'model':{'type':'K0','latent_dim':16,'hidden_dim':32},
                 'training':{'seed':7,'epochs':2,'batch_size':8,'lr':.001,
                             'patience':10,'device':'cpu','position_scale_m':5.},
                 'outputs':{'checkpoints':'ckpt'}}
            result=train_k0(cfg,root,epochs=2,device='cpu')
            self.assertTrue(Path(result['best']).is_file())
            self.assertTrue(Path(result['last']).is_file())
            self.assertTrue((root/'ckpt'/'loss.csv').is_file())
            model, ckpt=load_checkpoint(result['best'],'cpu')
            self.assertEqual(ckpt['architecture'],'NeuralKoopmanK0_v1')
            x=model.state_mean.expand(1,5,5)
            u=model.control_mean.expand(1,4,2)
            uf=model.control_mean.expand(1,40,2)
            with torch.no_grad():pred,hidden=model(x,u,uf)
            self.assertEqual(pred.shape,(1,41,5))
            self.assertEqual(hidden.shape,(1,41,16))
            self.assertTrue(torch.isfinite(pred).all())

if __name__=='__main__':unittest.main(verbosity=2)
