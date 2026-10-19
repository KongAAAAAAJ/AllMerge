"""Geometry-exact inset with dashed rectangle and correction arrows."""
from __future__ import annotations
import numpy as np

PALETTE={'pre':'#778391','post':'#2c6ecb','teacher':'#1f9a7b'}


def zoom_window(pre,post,teacher,centre,halfspan=4,margin=.35):
    lo=max(0,centre-halfspan);hi=min(len(pre),centre+halfspan+1)
    region=np.vstack((pre[lo:hi],post[lo:hi],teacher[lo:hi]))
    xymin=region.min(0);xymax=region.max(0)
    span=np.maximum(xymax-xymin,np.array([.5,.5]))
    pad=np.maximum(span*margin,np.array([.15,.15]))
    return (float(xymin[0]-pad[0]),float(xymax[0]+pad[0]),float(xymin[1]-pad[1]),float(xymax[1]+pad[1]))


def draw_trajectories(ax,pre,post,teacher,pre_sparse=None,post_sparse=None,teacher_sparse=None,legend=True):
    ax.plot(pre[:,0],pre[:,1],'--',lw=1.9,c=PALETTE['pre'],label='Pretrain (no guide)',zorder=3)
    ax.plot(post[:,0],post[:,1],'-',lw=2.1,c=PALETTE['post'],label='Posttrain (no guide)',zorder=4)
    ax.plot(teacher[:,0],teacher[:,1],'-',lw=2.1,c=PALETTE['teacher'],label='Teacher (PGD guided)',zorder=5)
    for p,c,marker in ((pre_sparse,PALETTE['pre'],'o'),(post_sparse,PALETTE['post'],'s'),(teacher_sparse,PALETTE['teacher'],'^')):
        if p is not None:ax.scatter(p[:,0],p[:,1],s=16,color=c,marker=marker,alpha=.65,zorder=6)
    ax.scatter([0],[0],s=38,color='#1f2937',marker='*',label='Ego origin',zorder=7)
    ax.set_xlabel('Ego-local X (m)');ax.set_ylabel('Ego-local Y (m)')
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.18)
    if legend:ax.legend(fontsize=8,loc='best')


def zoom_panel(ax_main,ax_zoom,pre,post,teacher,centre,halfspan=3):
    """Equal-scale standalone zoom panel, linked to real physical XY parent extent."""
    from matplotlib.patches import Rectangle,ConnectionPatch
    rect=zoom_window(pre,post,teacher,centre,halfspan=halfspan,margin=.25)
    x0,x1,y0,y1=rect
    outline=Rectangle((x0,y0),x1-x0,y1-y0,linewidth=1.6,edgecolor='#dc8733',
                      facecolor='none',linestyle='--',zorder=15)
    ax_main.add_patch(outline)
    draw_trajectories(ax_zoom,pre,post,teacher,legend=False)
    ax_zoom.set_xlim(x0,x1);ax_zoom.set_ylim(y0,y1)
    ax_zoom.set_title(f'Zoom: t={centre*.1:.1f}s | matched waypoint correction',fontsize=11)
    ax_zoom.tick_params(labelsize=9)
    correction=post[centre]-pre[centre]
    if np.linalg.norm(correction)>1e-6:
        ax_zoom.annotate('',xy=post[centre],xytext=pre[centre],
                         arrowprops=dict(arrowstyle='-|>',color='#dc793d',lw=2.8,mutation_scale=16),zorder=20)
        ax_zoom.scatter([pre[centre,0]],[pre[centre,1]],s=25,c=PALETTE['pre'],zorder=18)
        ax_zoom.scatter([post[centre,0]],[post[centre,1]],s=25,c=PALETTE['post'],zorder=18)
        ax_zoom.text(.03,.04,f'Correction vector: {np.linalg.norm(correction):.2f} m',
                     transform=ax_zoom.transAxes,fontsize=9,va='bottom',
                     bbox=dict(facecolor='white',edgecolor='#e4a25e',alpha=.92,pad=3))
    # Draw connectors in figure space from actual rectangle boundaries.
    for a,b in (((x1,y1),(0,1)),((x1,y0),(0,0))):
        conn=ConnectionPatch(xyA=a,coordsA=ax_main.transData,xyB=b,coordsB=ax_zoom.transAxes,
                             color='#dc8733',linestyle='--',lw=1.2,alpha=.75)
        ax_main.figure.add_artist(conn)
    return rect


def draw_summary(rows,out):
    import matplotlib.pyplot as plt
    import numpy as np
    out.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=[('train_reward_gain','01_reward_gain_distribution.png','Posttrain − Pretrain V6 reward'),
          ('teacher_distance_improve_m','02_teacher_distance_improve.png','Dense RMS teacher-distance reduction (m)'),
          ('max_lateral_shift_m','03_max_lateral_correction.png','Maximum lateral correction (m)')]
    for key,filename,xlabel in keys:
        x=np.array([r[key] for r in rows],dtype=float)
        fig,ax=plt.subplots(figsize=(8.2,4.6));ax.hist(x,bins=min(24,max(5,len(x)//3)),edgecolor='white')
        ax.axvline(0,c='#4b5563',lw=.9,ls='--');ax.set_xlabel(xlabel);ax.set_ylabel('State count')
        ax.set_title(f'{xlabel} | N={len(rows)}');ax.grid(alpha=.14)
        fig.tight_layout();fig.savefig(out/filename,dpi=180);plt.close(fig)
