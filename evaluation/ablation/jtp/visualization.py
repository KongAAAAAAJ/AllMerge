"""Paper 3x2 figure: overall XY, local zoom, kappa, yaw rate, roll, LTR."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def draw_projection_figure(path: Path, figure: dict, *, dt: float, kappa_lim: float):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, ConnectionPatch

    original = np.asarray(figure['nominal_world'])
    projected = np.asarray(figure['projected_world'])
    if original.shape != projected.shape or original.ndim != 3 or original.shape[-1] != 3:
        raise ValueError('Expected same-shaped [N,41,3] world trajectories')
    N, steps, _ = original.shape
    exemplar = int(np.argmax(np.linalg.norm(original[:,:, :2] - projected[:,:, :2],axis=-1).max(axis=1)))
    a, b = original[exemplar], projected[exemplar]
    times = np.arange(1, steps)*dt
    fig, axs = plt.subplots(3,2,figsize=(11,11),constrained_layout=True)
    overall, zoom, kappa_ax, yaw_ax, roll_ax, ltr_ax = axs.flat
    primary, second = '#2563eb','#ea580c'
    for i in range(N):
        color = primary if i==exemplar else '#8aa7cd'
        overall.plot(original[i,:,0],original[i,:,1],ls='--',c=color,alpha=.85,
                     label='Baseline' if i==exemplar else None)
        color = second if i==exemplar else '#edb89a'
        overall.plot(projected[i,:,0],projected[i,:,1],c=color,alpha=.85,
                     label='JTP' if i==exemplar else None)
    if figure.get('road_edge_world') is not None:
        for edge in figure['road_edge_world']:
            overall.plot(edge[:,0],edge[:,1],color='#666666',ls=':',lw=.9,alpha=.6)
    for actor in figure.get('external_actors',[]):
        overall.plot(actor[:,0],actor[:,1],ls='-.',c='#666666',alpha=.35,lw=1.0)
    differences = np.linalg.norm(a[:,:2]-b[:,:2],axis=-1)
    focus = int(np.argmax(differences))
    focus_center = .5*(a[focus,:2]+b[focus,:2])
    half_x=max(6.0, 3.*differences[focus]); half_y=max(2.5, 2.*differences[focus])
    rect = Rectangle((focus_center[0]-half_x,focus_center[1]-half_y),2*half_x,2*half_y,
                     fill=False,ls='--',lw=1.2,ec='#1f2937')
    overall.add_patch(rect)
    zoom.plot(a[:,0],a[:,1],ls='--',c=primary,lw=1.6,label='Baseline')
    zoom.plot(b[:,0],b[:,1],c=second,lw=1.6,label='JTP')
    zoom.annotate('',xy=b[focus,:2],xytext=a[focus,:2],arrowprops=dict(arrowstyle='->',color='#16804b',lw=1.8))
    zoom.scatter([a[focus,0],b[focus,0]],[a[focus,1],b[focus,1]],s=15,c=[primary,second])
    zoom.set_xlim(focus_center[0]-half_x,focus_center[0]+half_x)
    zoom.set_ylim(focus_center[1]-half_y,focus_center[1]+half_y)
    for ax in (overall,zoom):
        ax.set(xlabel='World X (m)',ylabel='World Y (m)'); ax.grid(alpha=.2)
    overall.legend(loc='best',fontsize=8); zoom.legend(loc='best',fontsize=8)
    overall.set_title('(a) Overall trajectories',fontsize=10)
    zoom.set_title(f'(b) Local refinement (truck {exemplar}, t={focus*dt:.1f}s)',fontsize=10)

    def curvature(path):
        delta = np.arctan2(np.sin(path[2:,2]-path[1:-1,2]), np.cos(path[2:,2]-path[1:-1,2]))
        ds = np.linalg.norm(path[2:,:2]-path[1:-1,:2],axis=-1)
        return delta / np.maximum(ds,1e-4)
    for curve,name,c,ls in ((curvature(a),'Baseline',primary,'--'),(curvature(b),'JTP',second,'-')):
        kappa_ax.plot(np.arange(1,steps-1)*dt,curve,c=c,ls=ls,label=name)
    kappa_ax.axhline(kappa_lim,color='black',ls=':',lw=1,label='Limit')
    kappa_ax.axhline(-kappa_lim,color='black',ls=':',lw=1)
    kappa_ax.set(title='(c) Trajectory curvature',xlabel='Time (s)',ylabel='Curvature (1/m)')

    sn = np.asarray(figure['nominal_states'])[exemplar]
    sp = np.asarray(figure['projected_states'])[exemplar]
    ln = np.asarray(figure['nominal_ltr'])[exemplar]
    lp = np.asarray(figure['projected_ltr'])[exemplar]
    bounds_n = np.asarray(figure['nominal_stability_bounds'])[exemplar]
    bounds_p = np.asarray(figure['projected_stability_bounds'])[exemplar]
    for ax,series_n,series_p,lim_n,lim_p,title,ylabel in (
        (yaw_ax,np.abs(sn[1:,2]),np.abs(sp[1:,2]),bounds_n[:,0],bounds_p[:,0],
         '(d) Yaw rate','|r| (rad/s)'),
        (roll_ax,np.abs(sn[1:,3]),np.abs(sp[1:,3]),bounds_n[:,2],bounds_p[:,2],
         '(e) Roll angle','|φ| (rad)'),
        (ltr_ax,np.abs(ln[1:]),np.abs(lp[1:]),bounds_n[:,3],bounds_p[:,3],
         '(f) Lateral load transfer ratio','|LTR|')):
        ax.plot(times,series_n,c=primary,ls='--',label='Baseline')
        ax.plot(times,series_p,c=second,label='JTP')
        ax.plot(times,lim_n,c='#333333',ls=':',label='Baseline limit')
        if not np.allclose(lim_n,lim_p):
            ax.plot(times,lim_p,c='#777777',ls='-.',label='JTP limit')
        ax.set(title=title,xlabel='Time (s)',ylabel=ylabel);ax.grid(alpha=.2)
    kappa_ax.legend(fontsize=8,ncol=3);yaw_ax.legend(fontsize=8,ncol=3)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)
