"""Six-model Koopman paper evaluation on identical frozen episode windows.

Outputs only ADE/FDE/yaw-rate RMSE/roll RMSE/LTR RMSE; the paper's position
error curve and episode-level ADE boxplot; controlled three-case chassis plot
requires independently acquired matching-control TruckSim ground truth.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import KoopmanWindowDataset, discover_episodes
from .metrics import (PAPER_COLUMNS, MODELS, get_errors, summarize_all,
                      save_episodes, save_position_curve, save_table_row)
from .neural_train import load_neural
from .edmd_train import load_edmd
from .k0_train import choose_device
from .shared_split import resolve,load as load_shared


def slug(kind):return kind.lower().replace(' ','_').replace('-','_')


def model_checkpoint(cfg,project_root,kind):
    return resolve(project_root,Path(cfg['outputs']['checkpoints'])/slug(kind)/'best.pt')


def load_model(kind,path,device):
    if kind in ('K0','K1-P','K1-F','K2'):return load_neural(path,device)
    return load_edmd(path,device)


def _run_model(model,kind,batch,device,dt):
    kw={'initial_pose':batch['target_poses'][:,0].to(device),'dt':dt}
    if kind!='K0':kw['conditions']=batch['conditions'].to(device)
    return model.predict_all(batch['state_hist'].to(device),batch['control_hist'].to(device),
                             batch['future_controls'].to(device),**kw)


def _episode_eval(model,kind,ep,cfg,device,batch_size):
    data=cfg['data'];dt=float(data.get('dt_s',.1));h=int(data.get('horizon_steps',40))
    ds=KoopmanWindowDataset([ep],history_steps=int(data.get('history_steps',5)),
          horizon_steps=h,warmup_s=float(data.get('warmup_s',1.0)),
          window_stride=int(data.get('eval_stride',5)),dt=dt)
    if not len(ds):raise ValueError(f'{ep.id} has no complete 4s evaluation window')
    position=np.zeros(h+1,dtype=np.float64)
    fde=yaw=roll=ltr=0.0;valid_ltr=0;windows=0
    for batch in DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0):
        with torch.inference_mode():pred=_run_model(model,kind,batch,device,dt)
        errs=get_errors(pred['states'].cpu().numpy(),batch['target_states'].numpy(),
                        pred['pose'].cpu().numpy(),batch['target_poses'].numpy(),
                        pred['ltr'].cpu().numpy(),batch['target_ltr'].numpy(),batch['ltr_valid'].numpy())
        position+=errs['position_error'].sum(axis=0)
        fde+=errs['position_error'][:,-1].sum()
        yaw+=errs['yaw_sq'].sum()
        roll+=errs['roll_sq'].sum()
        m=errs['ltr_valid'];ltr+=errs['ltr_sq'][m].sum();valid_ltr+=int(m.sum())
        windows+=len(errs['position_error'])
    curve=position/windows
    if not valid_ltr:raise ValueError(f'{ep.id} lacks valid LTR labels')
    row={'episode_id':str(ep.path),'windows':windows,
         'ADE (m)':float(curve[1:].mean()),'FDE (m)':float(fde/windows),
         'Yaw Rate RMSE (rad/s)':float(np.sqrt(yaw/(windows*h))),
         'Roll RMSE (deg)':float(np.sqrt(roll/(windows*h))),
         'LTR RMSE':float(np.sqrt(ltr/valid_ltr)),
         'curve':curve.tolist()}
    if not all(np.isfinite(row[k]) for k in PAPER_COLUMNS):
        raise ValueError(f'{kind} nonfinite evaluation metrics on {ep.id}')
    return row


def evaluate_one(cfg,project_root,kind,split='test',data_root=None,device='auto',batch_size=128):
    if kind not in MODELS:raise ValueError(kind)
    eps,manifest=load_shared(cfg,project_root,data_root)
    selected=eps[split]
    if not selected:raise ValueError(f'No episodes in frozen {split} split; use val for smoke')
    device=choose_device(device)
    ckpath=model_checkpoint(cfg,project_root,kind)
    model,ck=load_model(kind,ckpath,device)
    if ck.get('split_sha256')!=manifest['sha256']:
        raise ValueError(f'{kind} checkpoint uses a different frozen partition')
    rows=[]
    for i,ep in enumerate(selected):
        rows.append(_episode_eval(model,kind,ep,cfg,device,batch_size))
        if (i+1)%25==0:print(f'[EVAL] {kind} {i+1}/{len(selected)} episodes',flush=True)
    summary,curve=summarize_all(rows)
    out=resolve(project_root,Path(cfg['outputs']['evaluation'])/split/slug(kind))
    out.mkdir(parents=True,exist_ok=True)
    save_table_row(out/'table.csv',kind,summary)
    save_episodes(out/'episode_metrics.csv',rows)
    save_position_curve(out/'position_curve_k0_compatible.csv',curve,float(cfg['data'].get('dt_s',.1)))
    (out/'summary.json').write_text(json.dumps({'model':kind,'split':split,
        'split_sha256':manifest['sha256'],'metrics':summary,
        'num_episodes':len(rows),'num_windows':sum(r['windows'] for r in rows),
        'dt_s':float(cfg['data'].get('dt_s',.1))},indent=2),encoding='utf-8')
    print(f'[EVAL DONE] {kind} {split} ADE={summary["ADE (m)"]:.4f} '
          f'FDE={summary["FDE (m)"]:.4f} LTR={summary["LTR RMSE"]:.4f}',flush=True)
    return {'summary':summary,'curve':curve,'episodes':rows,'split_sha256':manifest['sha256']}


def compare(cfg,project_root,split='test',data_root=None,device='auto',batch_size=128):
    output={}
    for kind in MODELS:
        output[kind]=evaluate_one(cfg,project_root,kind,split,data_root,device,batch_size)
    out=resolve(project_root,Path(cfg['outputs']['evaluation'])/split)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'koopman_prediction_accuracy.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['Model',*PAPER_COLUMNS])
        for kind in MODELS:w.writerow([kind,*[output[kind]['summary'][k] for k in PAPER_COLUMNS]])
    with (out/'koopman_prediction_errors_curve.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['time_s',*MODELS])
        dt=float(cfg['data'].get('dt_s',.1))
        for j in range(len(output['K0']['curve'])):
            w.writerow([j*dt,*[output[kind]['curve'][j] for kind in MODELS]])
    plot_six_models(out/'koopman_prediction_errors.pdf',output,dt)
    print(f'[COMPARE] Six models tested on SAME {split} episodes; paper table and PDF -> {out}',flush=True)
    return output


def plot_six_models(path,output,dt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10.8,3.6))
    for kind in MODELS:
        curve=output[kind]['curve']
        axs[0].plot(np.arange(len(curve))*dt,curve,linewidth=1.7,label=kind)
    axs[0].set(xlabel='Prediction horizon (s)',ylabel='Mean position error (m)',title='(a) Multistep position error')
    axs[0].legend(fontsize=7);axs[0].grid(alpha=.25)
    arrays=[[ep['ADE (m)'] for ep in output[kind]['episodes']] for kind in MODELS]
    try:
        axs[1].boxplot(arrays,tick_labels=MODELS,showfliers=True,whis=1.5)
    except TypeError:  # Matplotlib before 3.9
        axs[1].boxplot(arrays,labels=MODELS,showfliers=True,whis=1.5)
    axs[1].tick_params(axis='x',rotation=24,labelsize=8)
    axs[1].set(ylabel='Episode-level ADE (m)',title='(b) Test episode distributions')
    axs[1].grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)


def controlled_chassis(cfg,project_root,case_files,device='auto',output_path=None,offset_s=None):
    """Controlled Fig. only with three independently recorded, matching input episodes.

    A vs B: same mu, different payload. B vs C: same payload, different mu.
    Nominal reference controls, initial speed and 40-step horizon must match.
    """
    if len(case_files)!=3:raise ValueError('Exactly three [A,B,C] accepted NPZ cases required')
    ep_paths=[resolve(project_root,p) for p in case_files]
    eps=discover_episodes([str(p.parent) for p in ep_paths],project_root,require_accepted=True)
    lookup={str(e.path.resolve()):e for e in eps}
    cases=[lookup[str(p.resolve())] for p in ep_paths]
    a,b,c=cases
    def value(e,k):return float(e.info[k])
    if not(np.isclose(value(a,'road_mu'),value(b,'road_mu'),atol=1e-5) and
           not np.isclose(value(a,'payload_kg'),value(b,'payload_kg'),atol=1e-4) and
           np.isclose(value(b,'payload_kg'),value(c,'payload_kg'),atol=1e-4) and
           not np.isclose(value(b,'road_mu'),value(c,'road_mu'),atol=1e-5)):
        raise ValueError('Cases must be A/B: payload contrast, B/C: friction contrast')
    if not np.allclose([value(e,'vx_init_kmh') for e in cases],value(a,'vx_init_kmh'),atol=1e-6):
        raise ValueError('Initial speeds differ; cannot claim controlled contrast')
    dc=cfg['data'];dt=float(dc.get('dt_s',.1));h=int(dc.get('horizon_steps',40));history=int(dc.get('history_steps',5))
    offset_s=float(dc.get('warmup_s',1.0) if offset_s is None else offset_s)
    k=max(history-1,int(round(offset_s/dt)))
    if any(len(e.controls)<k+h for e in cases):raise ValueError('Controlled cases are shorter than 4s horizon')
    common=cases[0].controls[k:k+h]
    for ep in cases[1:]:
        if not np.allclose(ep.controls[k:k+h],common,atol=1e-7,rtol=0):
            raise ValueError('Control sequences DO NOT MATCH; collect matched fixed-profile TruckSim cases first')
    _,manifest=load_shared(cfg,project_root)
    selected=('Parametric EDMDc','K0','K2')
    device=choose_device(device)
    predictions={}
    for kind in selected:
        model,ck=load_model(kind,model_checkpoint(cfg,project_root,kind),device)
        if ck['split_sha256']!=manifest['sha256']:raise ValueError('Mismatch in training partitions')
        results=[]
        for ep in cases:
            from .dataset import KoopmanWindowDataset
            ds=KoopmanWindowDataset([ep],history_steps=history,horizon_steps=h,
                                    warmup_s=offset_s,window_stride=1,dt=dt)
            # Exact aligned window k, not merely first window if rounding differs.
            keys=[i for i,(_,start) in enumerate(ds.keys) if start==k]
            if not keys:raise ValueError('Selected controlled start time does not align to a valid window')
            batch={field:value[None] for field,value in ds[keys[0]].items()}
            with torch.inference_mode():out=_run_model(model,kind,batch,device,dt)
            results.append({'states':out['states'].cpu().numpy()[0],
                            'ltr':out['ltr'].cpu().numpy()[0]})
        predictions[kind]=results
    path=resolve(project_root,output_path or Path(cfg['outputs']['evaluation'])/'controlled'/'koopman_chassis_responses.pdf')
    path.parent.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,3,figsize=(12,8.5),sharex=True)
    t=np.arange(h+1)*dt
    labels=('Yaw rate (rad/s)','Roll (deg)','LTR')
    for row,ep in enumerate(cases):
        true=(ep.states[k:k+h+1,2],np.rad2deg(ep.states[k:k+h+1,3]),ep.ltr[k:k+h+1])
        for col in range(3):
            ax=axs[row,col]
            ax.plot(t,true[col],color='black',linewidth=1.7,label='TruckSim')
            for kind in selected:
                value=predictions[kind][row]['ltr'] if col==2 else predictions[kind][row]['states'][:,2 if col==0 else 3]
                if col==1:value=np.rad2deg(value)
                ax.plot(t,value,label=kind,linewidth=1.1,linestyle='--')
            ax.grid(alpha=.25)
            if row==0:ax.set_title(labels[col])
            if row==2:ax.set_xlabel('Prediction horizon (s)')
        axs[row,0].set_ylabel(f'Case {"ABC"[row]}\npayload={value_float(ep,"payload_kg"):g} kg, mu={value_float(ep,"road_mu"):g}',fontsize=8)
    axs[0,0].legend(fontsize=7,loc='best')
    fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)
    print(f'[CONTROLLED] verified identical 40-step controls and initial speed; saved {path}',flush=True)
    return path


def value_float(ep,key):return float(ep.info[key])
