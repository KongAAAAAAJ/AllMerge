"""Detailed per-state publication-ready trajectory zooms and gallery contact sheet."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .plot_utils import draw_trajectories,zoom_panel


def render_one(row,sample,out,dpi=240):
    import matplotlib.pyplot as plt
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    pre,post,teacher=[np.asarray(sample[k]) for k in ('pre_dense','post_dense','teacher_dense')]
    fig,(ax,ax_zoom)=plt.subplots(1,2,figsize=(16,4.8),gridspec_kw={'width_ratios':[1.35,1]})
    draw_trajectories(ax,pre,post,teacher,
                      sample.get('pre_sparse'),sample.get('post_sparse'),sample.get('teacher_sparse'))
    centre=int(row['max_dense_shift_index'])
    zoom_panel(ax,ax_zoom,pre,post,teacher,centre)
    ax.set_title(f"Whole trajectory | {row['scenario']} state #{row['state_index']}\n"
                 f"Teacher G{row['group_id']} / mode {row['mode']}",fontsize=12)
    title=(f"Pretrain Reward {row['pretrain_reward']:.3f} → Posttrain {row['posttrain_reward']:.3f} "
           f"| Teacher {row['teacher_reward']:.3f} | ΔR={row['train_reward_gain']:+.3f}")
    fig.suptitle(title,fontsize=13,fontweight='bold',y=.975)
    fig.text(.5,.045,
             f"To Teacher RMS: {row['pre_teacher_dense_rms_m']:.2f} → {row['post_teacher_dense_rms_m']:.2f} m"
             f"   |   Pre/Post Dense RMS: {row['pre_post_dense_rms_m']:.2f} m"
             f"   |   Max lateral correction: {row['max_lateral_shift_m']:.2f} m",
             ha='center',fontsize=10,bbox=dict(facecolor='#f5f7f9',edgecolor='#d1d5db',pad=5))
    fig.subplots_adjust(left=.055,right=.975,bottom=.16,top=.83,wspace=.22)
    fig.savefig(out,dpi=dpi,bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
    return out


def contact_sheet(rows,samples,path):
    import matplotlib.pyplot as plt
    if not rows:return
    cols=3;nr=int(np.ceil(len(rows)/cols))
    fig,axes=plt.subplots(nr,cols,figsize=(cols*5.0,nr*3.6),squeeze=False)
    for i,row in enumerate(rows):
        ax=axes.flat[i];s=samples[int(row['state_index'])]
        p,q,t=(np.asarray(s[k]) for k in ('pre_dense','post_dense','teacher_dense'))
        draw_trajectories(ax,p,q,t,legend=i==0)
        ax.set_title(f"#{i+1} {row['scenario']} [{row['state_index']}] | ΔR={row['train_reward_gain']:+.2f}",fontsize=9)
    for ax in axes.flat[len(rows):]:ax.axis('off')
    fig.tight_layout();path=Path(path);fig.savefig(path,dpi=170);plt.close(fig)
