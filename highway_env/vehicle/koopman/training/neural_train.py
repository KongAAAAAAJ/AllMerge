"""Unified K0/K1/K2 neural optimization; model-specific wrappers call here.

All four variants use identical frozen episodes, windows, model dimensions,
losses, optimizer, and selection criterion. No TruckSim DLL is imported.
"""
from __future__ import annotations
import copy
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from highway_env.vehicle.koopman.models.base_model import NeuralKoopmanK0
from highway_env.vehicle.koopman.models.conditional_model import ConditionalKoopman
from highway_env.vehicle.koopman.data.dataset import KoopmanWindowDataset
from highway_env.vehicle.koopman.training.base_train import set_seed, choose_device, save_checkpoint
from highway_env.vehicle.koopman.data.shared_split import resolve

NEURAL_NAMES=('K0','K1-P','K1-F','K2')


def construct_model(kind, cfg, manifest):
    dc=cfg['data']; mc=cfg['model']
    kwargs={'history_steps':int(dc.get('history_steps',5)),
            'latent_dim':int(mc.get('latent_dim',32)),
            'hidden_dim':int(mc.get('hidden_dim',128))}
    norm=manifest['normalization']
    if kind=='K0':model=NeuralKoopmanK0(**kwargs,**norm)
    elif kind in ('K1-P','K1-F','K2'):
        model=ConditionalKoopman(kind,
            condition_hidden=int(mc.get('condition_hidden',64)),
            **manifest['condition_normalization'],**kwargs,**norm)
    else:raise ValueError(f'Unknown neural Koopman model: {kind}')
    return model,kwargs


def prediction(model,kind,batch,dt):
    kwargs={'initial_pose':batch['target_poses'][:,0],'dt':dt}
    if kind!='K0':kwargs['conditions']=batch['conditions']
    return model.predict_all(batch['state_hist'],batch['control_hist'],batch['future_controls'],**kwargs)


def objective(model,kind,batch,cfg):
    tr=cfg['training'];w=tr['loss_weights'];dt=float(cfg['data'].get('dt_s',.1))
    pred=prediction(model,kind,batch,dt)
    residual=(pred['states']-batch['target_states'])/model.state_std
    smooth=torch.nn.functional.smooth_l1_loss
    # Same loss for K0,K1-P,K1-F,K2, same target windows.
    recon=smooth(residual[:,0],torch.zeros_like(residual[:,0]))
    one=smooth(residual[:,1],torch.zeros_like(residual[:,1]))
    multi=smooth(residual[:,1:],torch.zeros_like(residual[:,1:]))
    err=(pred['pose'][:,1:,:2]-batch['target_poses'][:,1:,:2])/float(tr.get('position_scale_m',5.))
    traj=smooth(err,torch.zeros_like(err))
    mask=batch['ltr_valid'][:,1:]
    if mask.any():
        ltr_err=(pred['ltr'][:,1:][mask]-batch['target_ltr'][:,1:][mask])/float(tr.get('ltr_scale',.1))
        ltr=smooth(ltr_err,torch.zeros_like(ltr_err))
    else:ltr=one.new_zeros(())
    mat=(model.A-.995*torch.eye(model.latent_dim,device=model.A.device)).square().mean()
    if kind!='K0':mat=mat+1e-2*model.condition_net[-1].weight.square().mean()
    total=(float(w.get('reconstruction',.1))*recon+float(w.get('one_step',.2))*one+
           float(w.get('rollout',1.))*multi+float(w.get('trajectory',.05))*traj+
           float(w.get('ltr',.15))*ltr+float(w.get('matrix_regularization',1e-4))*mat)
    # Paper lifted consistency: Phi(h_{k+1}) should agree with predicted z_{k+1}.
    if float(w.get('dynamics',0.0))>0:
        next_states=torch.cat((batch['state_hist'][:,1:],batch['target_states'][:,1:2]),dim=1)
        next_controls=torch.cat((batch['control_hist'][:,1:],batch['future_controls'][:,:1]),dim=1)
        lifted=model.lift(next_states,next_controls)
        # Symmetric optimization; both observed lift and predicted dynamics train.
        dyn=smooth(pred['latent'][:,1],lifted)
        total=total+float(w['dynamics'])*dyn
    return total


def _loader(episodes,cfg,stride,batch_size,shuffle):
    d=cfg['data'];ds=KoopmanWindowDataset(episodes,history_steps=int(d.get('history_steps',5)),
         horizon_steps=int(d.get('horizon_steps',40)),warmup_s=float(d.get('warmup_s',1.)),
         window_stride=int(stride),dt=float(d.get('dt_s',.1)))
    if not ds:raise ValueError('No valid prediction windows in shared partition; collect longer episodes')
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=0)


def train_neural(kind,cfg,project_root,splits,manifest,epochs=None,device=None):
    if kind not in NEURAL_NAMES:raise ValueError(kind)
    training=cfg['training'];data=cfg['data']
    seed=int(training.get('seed',2026));set_seed(seed)
    device=choose_device(device or training.get('device','auto'))
    model,kwargs=construct_model(kind,cfg,manifest)
    model=model.to(device)
    bs=int(training.get('batch_size',64));assert bs>0
    train_loader=_loader(splits['train'],cfg,int(data.get('train_stride',2)),bs,True)
    val_loader=_loader(splits['val'],cfg,int(data.get('eval_stride',5)),bs,False)
    opt=torch.optim.AdamW(model.parameters(),lr=float(training.get('lr',3e-4)),
                          weight_decay=float(training.get('weight_decay',1e-5)))
    out=resolve(project_root,Path(cfg['outputs']['checkpoints'])/kind.lower().replace('-','_'))
    out.mkdir(parents=True,exist_ok=True)
    max_epochs=int(epochs if epochs is not None else training.get('epochs',60))
    if max_epochs<1:raise ValueError('epochs must be positive')
    patience=int(training.get('patience',20));best=float('inf');best_epoch=0
    records=[]
    print(f'[TRAIN {kind}] {device} split={manifest["sha256"][:12]} '
          f'windows={len(train_loader.dataset)}/{len(val_loader.dataset)}',flush=True)
    for epoch in range(1,max_epochs+1):
        losses=[]
        for optimize,loader in ((True,train_loader),(False,val_loader)):
            model.train(optimize);summed=0.;count=0
            for batch in loader:
                batch={k:v.to(device) for k,v in batch.items()}
                if optimize:
                    opt.zero_grad(set_to_none=True)
                    loss=objective(model,kind,batch,cfg)
                    if not torch.isfinite(loss):raise FloatingPointError(f'{kind} nonfinite training loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),float(training.get('grad_clip',1.)))
                    opt.step()
                else:
                    with torch.no_grad():loss=objective(model,kind,batch,cfg)
                n=len(batch['target_states']);summed+=float(loss.detach())*n;count+=n
            losses.append(summed/count)
        tr_loss,val_loss=losses
        records.append({'epoch':epoch,'train_loss':tr_loss,'val_loss':val_loss})
        checkpoint={'model_type':kind,'architecture':'Koopman_Neural_StageC',
                    'model_kwargs':kwargs,'condition_hidden':int(cfg['model'].get('condition_hidden',64)),
                    'normalization':manifest['normalization'],
                    'condition_normalization':manifest['condition_normalization'],
                    'model_state':copy.deepcopy(model.state_dict()),'epoch':epoch,
                    'split_sha256':manifest['sha256'],'split_manifest':manifest['report'],
                    'train_config':cfg,'history':records.copy()}
        save_checkpoint(out/'last.pt',checkpoint)
        if val_loss<best-1e-9:
            best=val_loss;best_epoch=epoch
            checkpoint['best_val_loss']=best
            save_checkpoint(out/'best.pt',checkpoint)
        print(f'[{kind}] epoch={epoch}/{max_epochs} train={tr_loss:.6f} '
              f'val={val_loss:.6f} best={best:.6f}',flush=True)
        if patience>0 and epoch-best_epoch>=patience:break
    with (out/'loss.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=('epoch','train_loss','val_loss'))
        writer.writeheader();writer.writerows(records)
    print(f'[TRAIN DONE] {kind} best={out/"best.pt"}',flush=True)
    return out/'best.pt'


def load_neural(checkpoint,device='cpu'):
    try:d=torch.load(checkpoint,map_location=device,weights_only=True)
    except TypeError:d=torch.load(checkpoint,map_location=device)
    kind=d['model_type']
    normal=d['normalization'];kw=d['model_kwargs']
    if kind=='K0':model=NeuralKoopmanK0(**kw,**normal)
    else:model=ConditionalKoopman(kind,condition_hidden=int(d.get('condition_hidden',64)),
              **d['condition_normalization'],**kw,**normal)
    model.load_state_dict(d['model_state'])
    return model.to(device).eval(),d
