"""Four manuscript-aligned open-loop figures from real saved K2 predictions.

Figures are deterministic visual summaries, not illustrative synthetic diagrams.
Representative selection is independent of prediction error or ground-truth fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PAPER_OPEN_FIGURES = (
    'open_loop_trajectory_distribution',
    'open_loop_error_statistics',
    'open_loop_control_profiles',
    'open_loop_chassis_responses',
)


def _save(fig, directory: Path, name: str):
    fig.savefig(directory / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(directory / (name + '.png'), bbox_inches='tight', dpi=220)


def plot_open_loop_figures(samples: list[dict], records: list[dict],
                           errors_by_time: list[dict], output_dir, *, dt: float,
                           stability_config: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'pdf.fonttype': 42})
    c_gt, c_pred, c_anchor, c_candidate = '#202020', '#1269A3', '#D97721', '#A6A6A6'
    samples = sorted(samples, key=lambda x: x['sample_index'])
    # Full 3x3 overview. The same fixed validation indices are selected regardless
    # of error; fewer than nine samples leave intentionally blank panels.
    fig, ax = plt.subplots(3, 3, figsize=(11.6, 8.7), layout='constrained')
    for k, a in enumerate(ax.flat):
        if k >= len(samples):
            a.axis('off')
            continue
        sample = samples[k]
        for mode, valid in enumerate(sample['candidate_valid']):
            if valid:
                xy = sample['candidate_xy'][mode]
                a.plot(xy[:, 0], xy[:, 1], color=c_candidate, lw=.75, alpha=.5)
                xy_a = sample['anchor_candidate_xy'][mode]
                a.plot(xy_a[:, 0], xy_a[:, 1], color=c_anchor,
                       lw=.7, alpha=.26, ls='--')
        gt, selected, anchor = sample['gt'], sample['selected_xy'], sample['anchor_selected_xy']
        a.plot(gt[:, 0], gt[:, 1], color=c_gt, lw=2., label='Expert')
        a.plot(selected[:, 0], selected[:, 1], color=c_pred, lw=1.8, label='Selected diffusion')
        a.plot(anchor[:, 0], anchor[:, 1], color=c_anchor, lw=1.25, ls='--',
               label='Paired control anchor')
        a.plot(0, 0, marker='o', ms=3, color='black')
        a.set_title(f"Validation index {sample['sample_index']}  |  Mode {sample['selected_mode']}")
        a.set_xlabel('Local longitudinal position (m)')
        a.set_ylabel('Local lateral position (m)')
        a.grid(alpha=.2)
        a.set_aspect('equal', adjustable='datalim')
    handles=[Line2D([0],[0],color=c_gt,lw=2,label='Expert'),
             Line2D([0],[0],color=c_pred,lw=2,label='Selected diffusion'),
             Line2D([0],[0],color=c_anchor,lw=1.4,ls='--',label='Anchor'),
             Line2D([0],[0],color=c_candidate,lw=1.,label='Valid candidates')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,-.018), ncol=4, frameon=False)
    _save(fig,output_dir,PAPER_OPEN_FIGURES[0]);plt.close(fig)

    fig, ax = plt.subplots(1,3,figsize=(12.1,3.6),layout='constrained')
    keys=['ade_m','fde_m','minade_k_m','minfde_k_m']
    values=[[r[k] for r in records if r.get(k) is not None] for k in keys]
    ax[0].boxplot(values, labels=['ADE','FDE','minADE$_K$','minFDE$_K$'],
                  patch_artist=False,showfliers=False)
    ax[0].set(ylabel='Displacement error (m)',title='(a) Prediction-error distribution')
    times=np.asarray([r['time_s'] for r in errors_by_time],dtype=float)
    means=np.asarray([r['mean_error_m'] for r in errors_by_time],dtype=float)
    sds=np.asarray([r['std_error_m'] for r in errors_by_time],dtype=float)
    ax[1].plot(times,means,color=c_pred,lw=2,label='Mean over samples and seeds')
    ax[1].fill_between(times,np.maximum(0,means-sds),means+sds,
                       alpha=.14,color=c_pred,label='Sample/seed std. dev.')
    ax[1].set(xlabel='Prediction horizon (s)',ylabel='Displacement error (m)',
              title='(b) Error versus prediction horizon')
    ax[1].legend(loc='upper left',fontsize=7,frameon=False)
    ade=np.asarray([r['ade_m'] for r in records]);minade=np.asarray([r['minade_k_m'] for r in records])
    ax[2].scatter(ade,minade,s=9,alpha=.32,color=c_pred)
    bound=max(float(np.max(ade)),float(np.max(minade)),1e-4)*1.05
    ax[2].plot([0,bound],[0,bound],ls='--',color='#666666',lw=1)
    ax[2].set(xlim=(0,bound),ylim=(0,bound),xlabel='Selected ADE (m)',
              ylabel='Oracle minADE$_K$ (m)',title='(c) Selected versus oracle coverage')
    for a in ax:a.grid(alpha=.2)
    _save(fig,output_dir,PAPER_OPEN_FIGURES[1]);plt.close(fig)

    reference=samples[0]
    control=np.asarray(reference['control']);anchor=np.asarray(reference['anchor_control'])
    t=np.arange(1,control.shape[0]+1)*dt
    fig, ax=plt.subplots(2,2,figsize=(10.0,6.5),layout='constrained')
    pairs=[
        (control[:,0],anchor[:,0],t,'Longitudinal acceleration (m/s²)','(a) Acceleration'),
        (np.rad2deg(control[:,1]),np.rad2deg(anchor[:,1]),t,
         'Front-wheel steering (deg)','(b) Front-wheel steering'),
        (np.diff(control[:,0])/dt,np.diff(anchor[:,0])/dt,t[1:],
         'Acceleration jerk (m/s³)','(c) Acceleration jerk'),
        (np.rad2deg(np.diff(control[:,1])/dt),np.rad2deg(np.diff(anchor[:,1])/dt),t[1:],
         'Steering rate (deg/s)','(d) Steering rate'),
    ]
    for a,(sig,ref,timeaxis,ylabel,title) in zip(ax.flat,pairs):
        a.plot(timeaxis,sig,color=c_pred,lw=1.8,label='Selected diffusion')
        a.plot(timeaxis,ref,color=c_anchor,ls='--',lw=1.4,label='Paired anchor')
        a.set(xlabel='Prediction time (s)',ylabel=ylabel,title=title)
        a.grid(alpha=.2)
    ax[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle(f"Control profiles, fixed validation index {reference['sample_index']}",fontsize=10)
    _save(fig,output_dir,PAPER_OPEN_FIGURES[2]);plt.close(fig)

    if 'states' not in reference or 'ltr' not in reference:
        raise ValueError('Paper chassis-response figure requires frozen Koopman K2 states and LTR')
    import torch
    from highway_env.planner.diffusion.projection.constraints import stability_limits
    state=np.asarray(reference['states']);ltr=np.asarray(reference['ltr'])
    bounds=stability_limits(torch.as_tensor(state[1:,0],dtype=torch.float64),
             torch.tensor(reference['payload'],dtype=torch.float64),
             torch.tensor(reference['mu'],dtype=torch.float64),stability_config).numpy()
    curves=[
        (np.rad2deg(state[1:,2]),np.rad2deg(bounds[:,0]),'Yaw rate (deg/s)','(a) Yaw rate'),
        (np.rad2deg(np.arctan2(state[1:,1],state[1:,0])),np.rad2deg(bounds[:,1]),
         'Sideslip angle (deg)','(b) Sideslip angle'),
        (np.rad2deg(state[1:,3]),np.rad2deg(bounds[:,2]),'Roll angle (deg)','(c) Roll angle'),
        (ltr[1:],bounds[:,3],'Lateral load-transfer ratio','(d) LTR')]
    fig,ax=plt.subplots(2,2,figsize=(10.0,6.5),layout='constrained')
    for a,(value,limit,ylabel,title) in zip(ax.flat,curves):
        if not (len(value)==len(t)==len(limit)):
            raise ValueError('K2 response and stability threshold time grids differ')
        a.plot(t,value,color=c_pred,lw=1.8,label='Frozen K2 prediction')
        a.plot(t,limit,color='#A23D3D',lw=1.,ls='--',label='Stability limits')
        a.plot(t,-limit,color='#A23D3D',lw=1.,ls='--')
        a.set(xlabel='Prediction time (s)',ylabel=ylabel,title=title)
        a.grid(alpha=.2)
    ax[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle(f"Koopman responses, fixed validation index {reference['sample_index']}",fontsize=10)
    _save(fig,output_dir,PAPER_OPEN_FIGURES[3]);plt.close(fig)

    # Save exact arrays and sample-selection provenance for figure reproduction.
    numeric_keys=[k for k,v in reference.items() if isinstance(v,np.ndarray)]
    np.savez_compressed(output_dir/'representative_samples.npz',
        **{k:np.stack([s[k] for s in samples]) for k in numeric_keys})
    (output_dir/'representative_selection.json').write_text(json.dumps([
        {'sample_index':s['sample_index'],'inference_seed':s['seed'],
         'selected_mode':s['selected_mode']} for s in samples],indent=2),encoding='utf-8')
    return {key:str(output_dir/(key+'.pdf')) for key in PAPER_OPEN_FIGURES}
