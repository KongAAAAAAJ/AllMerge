"""Manuscript figures from saved TRACE Control Diffusion + frozen-K2 outputs.

All roads are reconstructed from the evaluator's *observed* map-polylines
(centerline and per-point lane width); no synthetic or fixed-width road is drawn.
The 8+3 figures use reproducibly random cases. The K2-response distribution
uses a fixed-seed random subset (64 cases, or all if fewer are available).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PAPER_OPEN_FIGURES = (
    'open_loop_trajectory_distribution',
    'open_loop_error_statistics',
    'open_loop_control_profiles',
    'open_loop_chassis_responses',
)
ANCHOR_FIGURE = 'open_loop_anchor_distribution'


def _save(fig, directory: Path, name: str):
    fig.savefig(directory / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(directory / (name + '.png'), bbox_inches='tight', dpi=210)


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8-sig') as out:
        w=csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def _map_geometry(s):
    """Reconstruct actual sampled lane edges, and approximate outer envelope.

    map[..., :2] is the centerline, map[..., 4] is width in meters.
    Both are produced by AllMergeFeatureBuilder._sample_lane_polyline.
    Missing/invalid map data is an error, not a license to draw fabricated lanes.
    """
    mp=np.asarray(s.get('map_polylines',[]),dtype=float)
    mask=np.asarray(s.get('map_valid_mask',[]),dtype=bool)
    if mp.ndim!=3 or mp.shape[-1]<5 or mask.shape!=(mp.shape[0],):
        raise ValueError('Map-derived boundary plot requires map_polylines [N,P,>=5] and map_valid_mask [N]')
    lanes=[]
    for lane in mp[mask]:
        mid=lane[:,:2]; width=lane[:,4]
        if not (np.isfinite(mid).all() and np.isfinite(width).all() and (width>0).all()):
            continue
        d=np.gradient(mid,axis=0)
        unit=d/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-8)
        normal=np.stack([-unit[:,1],unit[:,0]],axis=-1)
        if np.linalg.norm(mid[-1]-mid[0])<1e-6:continue
        lanes.append({'center':mid, 'upper':mid+normal*width[:,None]/2.,
                      'lower':mid-normal*width[:,None]/2., 'normal':normal})
    if not lanes:
        raise ValueError('No valid lane geometry in actual dataset features')
    # The common-road outer envelope uses per-point lane centers and widths,
    # not a constant road width. It is only used when lanes are near-parallel.
    ref=next((x for x in lanes if x is not None),lanes[0])
    road_edges=None
    aligned = all(np.median(np.sum(x['normal']*ref['normal'],axis=-1))>.90 for x in lanes)
    if len(lanes)>=1 and aligned and all(len(x['center'])==len(ref['center']) for x in lanes):
        xy0=ref['center']; n=ref['normal']
        lower=np.full(len(xy0),np.inf);upper=np.full(len(xy0),-np.inf)
        for lane in lanes:
            for side in ('lower','upper'):
                signed=np.sum((lane[side]-xy0)*n,axis=-1)
                lower=np.minimum(lower,signed);upper=np.maximum(upper,signed)
        road_edges=(xy0+lower[:,None]*n,xy0+upper[:,None]*n)
    return lanes,road_edges


def _road(ax,s,*,legend=False):
    lanes, outer=_map_geometry(s)
    for lane in lanes:
        ax.plot(*lane['center'].T,color='#9BA6AC',ls=':',lw=.55,alpha=.48,zorder=0)
        for side in ('lower','upper'):
            edge=lane[side]
            ax.plot(*edge.T,color='#87939B',lw=.6,alpha=.35,zorder=0)
    if outer is not None:
        for edge in outer:
            ax.plot(*edge.T,color='#657780',lw=1.,ls='-',alpha=.7,zorder=1)
    return lanes,outer


def _trajectory_limits(s, curves, *, include_anchors=True):
    lanes, outer=_map_geometry(s)
    main=np.concatenate([np.asarray(q) for q in curves if q is not None],axis=0)
    xlo=float(min(0.0,np.min(main[:,0]))); xhi=float(np.max(main[:,0]))
    dx=max(xhi-xlo,10.)
    xlo-=0.03*dx; xhi+=0.035*dx
    relevant=[main[:,1]]
    for lane in lanes:
        for side in ('lower','upper'):
            edge=lane[side]
            points=edge[(edge[:,0]>=xlo)&(edge[:,0]<=xhi)]
            if len(points):relevant.append(points[:,1])
    candidates=np.asarray(s.get('candidate_xy',[]))
    valid=np.asarray(s.get('candidate_valid',[]),dtype=bool)
    if len(candidates) and len(valid)==len(candidates):
        candidates=candidates[valid]
        if candidates.size:
            relevant.append(np.percentile(candidates[...,1],[2,98]))
    yvals=np.concatenate(relevant)
    ymin,ymax=float(np.min(yvals)),float(np.max(yvals))
    dy=max(ymax-ymin,3.5)
    return (xlo,xhi),(ymin-.08*dy-.30,ymax+.08*dy+.30)


def _line_anchor(ax,s,color='#D97721',*,candidate_alpha=.16):
    anchors=np.asarray(s['anchor_candidate_xy'])
    valid=np.asarray(s.get('candidate_valid',np.ones(len(anchors),dtype=bool)),dtype=bool)
    for mid in np.flatnonzero(valid):
        xy=anchors[mid]
        ax.plot(xy[:,0],xy[:,1],color=color,ls='--',lw=.75,alpha=candidate_alpha,zorder=2)


def _plot_trajectory(ax,s,colors,*,show_inset=False):
    cg,cp,ca,cc=colors
    _road(ax,s)
    candidates=np.asarray(s['candidate_xy']);valid=np.asarray(s['candidate_valid'],dtype=bool)
    for xy in candidates[valid]:ax.plot(*xy.T,color=cc,lw=.7,alpha=.45,zorder=2)
    gt=np.asarray(s['gt']);selected=np.asarray(s['selected_xy']);anchor=np.asarray(s['anchor_selected_xy'])
    _line_anchor(ax,s,color=ca)
    ax.plot(*anchor.T,color=ca,ls='--',lw=1.3,alpha=.86,zorder=3)
    ax.plot(*gt.T,color=cg,lw=2.0,zorder=4)
    ax.plot(*selected.T,color=cp,lw=1.8,zorder=5)
    ax.plot([0],[0],'ko',ms=2.5,zorder=6)
    ax.plot(gt[-1,0],gt[-1,1],'s',color=cg,ms=3.5,zorder=7)
    ax.plot(selected[-1,0],selected[-1,1],'o',color=cp,ms=3.5,zorder=7)
    ax.set_xlim(*_trajectory_limits(s,[gt,selected,anchor])[0]);ax.set_ylim(*_trajectory_limits(s,[gt,selected,anchor])[1])
    ax.set_aspect('auto')  # independent physical axes reveal lateral errors in ~100m maneuvers
    ade=float(np.linalg.norm(selected-gt,axis=-1).mean())
    ax.set_title(f"idx {s['sample_index']}  mode {s['selected_mode']}  ADE {ade:.2f} m",fontsize=9)
    ax.grid(alpha=.12)
    if show_inset:
        # Inset follows the actual largest selected-vs-GT lateral disagreement.
        k=int(np.argmax(np.abs(selected[:,1]-gt[:,1])))
        xcenter=float((selected[k,0]+gt[k,0])/2.)
        ycenter=float((selected[k,1]+gt[k,1])/2.)
        max_gap=float(abs(selected[k,1]-gt[k,1]))
        ins=ax.inset_axes([.52,.50,.46,.44],facecolor='white',zorder=20)
        for q,color,style,width in [(gt,cg,'-',1.3),(selected,cp,'-',1.3),(anchor,ca,'--',.9)]:
            ins.plot(*q.T,color=color,ls=style,lw=width)
        xspan=max(12., .16*(gt[:,0].max()-gt[:,0].min()))
        ins.set_xlim(xcenter-xspan/2,xcenter+xspan/2)
        yspan=max(1.6,2.5*max_gap)
        ins.set_ylim(ycenter-yspan/2,ycenter+yspan/2)
        ins.tick_params(labelsize=6,length=1.5)
        ins.grid(alpha=.13)
        try:ax.indicate_inset_zoom(ins,edgecolor='#444444',alpha=.55)
        except TypeError:ax.indicate_inset_zoom(ins,edgecolor='#444444')


def _plot_anchor(ax,s,colors):
    cg,cp,ca,cc=colors
    _road(ax,s)
    cands=np.asarray(s['anchor_candidate_xy']);valid=np.asarray(s['candidate_valid'],dtype=bool)
    for xy in cands[valid]:ax.plot(*xy.T,lw=1.,color=ca,alpha=.42)
    # Original kinematic sparse anchors are available without K2 assumptions.
    coarse=np.asarray(s.get('anchor_kinematic_xy',[]))
    if len(coarse)==len(cands):
        for xy in coarse[valid]:ax.plot(*xy.T,color='#6E76A8',ls=':',lw=.8,alpha=.38)
    gt=np.asarray(s['gt']);pair=np.asarray(s['anchor_selected_xy'])
    ax.plot(*gt.T,color=cg,lw=1.9)
    ax.plot(*pair.T,color=ca,lw=2.,ls='--')
    ax.plot([0],[0],'ko',ms=2.5)
    target=np.asarray(s.get('target_point',[np.nan,np.nan]))
    if np.isfinite(target).all():ax.scatter([target[0]],[target[1]],marker='*',c=['#B06033'],s=23)
    xmin,xmax=_trajectory_limits(s,[gt,pair],include_anchors=True)[0]
    ymin,ymax=_trajectory_limits(s,[gt,pair],include_anchors=True)[1]
    ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),title=f"idx {s['sample_index']}  valid {valid.sum()}/{len(valid)}")
    ax.set_aspect('auto');ax.grid(alpha=.12)


def _select_three_conditions(pool, seed):
    """Randomly sample across observed maneuver / speed conditions, never by error."""
    rng=np.random.default_rng(seed)
    if len(pool)<=3:return list(pool)
    # Use goal lateral displacement (observed planner condition); if no three
    # distinct classes are present, fall back to measured initial-speed terciles.
    def cls(s):
        dy=float(np.asarray(s.get('target_point',[0,0]))[1])
        return 'lateral+' if dy>1.5 else ('lateral-' if dy < -1.5 else 'lane-keeping')
    labels=[cls(s) for s in pool]
    if len(set(labels))<3:
        v=np.asarray([float(s.get('initial_speed_mps',0)) for s in pool])
        bins=np.quantile(v,[1/3,2/3])
        labels=[f'speed-bin-{int(np.searchsorted(bins,x,side="right"))}' for x in v]
    groups={}
    for s,label in zip(pool,labels):groups.setdefault(label,[]).append(s)
    selected=[]
    for label in rng.permutation(list(groups)):
        if len(selected)>=3:break
        pick=groups[label][int(rng.integers(0,len(groups[label])))]
        selected.append(pick)
    if len(selected)<3:
        leftovers=[x for x in pool if all(x['sample_index']!=y['sample_index'] for y in selected)]
        for i in rng.permutation(len(leftovers))[:3-len(selected)]:selected.append(leftovers[int(i)])
    return selected


def _cdf(ax,values,*,label,color):
    vals=np.sort(np.asarray(values,dtype=float))
    vals=vals[np.isfinite(vals)]
    if len(vals):ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=label,color=color,lw=1.8)


def plot_open_loop_figures(samples: list[dict], records: list[dict],
                           errors_by_time: list[dict], output_dir, *, dt: float,
                           stability_config: dict, visualization_seed: int=0,
                           trajectory_count: int=8, chassis_count: int=64,
                           control_limits=(2.5,0.08)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from highway_env.planner.diffusion.projection.constraints import stability_limits
    import torch

    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,
                         'pdf.fonttype':42,'savefig.facecolor':'white'})
    cg,cp,ca,cc='#222222','#166EA5','#D87928','#AAB2B6'
    colors=(cg,cp,ca,cc)
    samples=sorted(samples,key=lambda s:s['sample_index'])
    if not samples:raise ValueError('No real K2 predictions available for figure generation')
    # The evaluator preselects a reproducible random pool. Reorder with the
    # visualization seed, without any GT error or prediction-quality ranking.
    rng=np.random.default_rng(visualization_seed)
    perm=rng.permutation(len(samples));pool=[samples[int(k)] for k in perm]
    display=pool[:min(8,trajectory_count,len(pool))]
    dynamical=pool[:min(chassis_count,len(pool))]
    controls=_select_three_conditions(pool,visualization_seed+47)
    for s in display:
        if 'map_polylines' not in s or 'map_valid_mask' not in s:
            raise ValueError('V3.2 needs actual dataset map features; rerun open-loop evaluation to generate road plots')

    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color=cg,lw=2,label='Expert'),
             Line2D([0],[0],color=cp,lw=2,label='Selected diffusion'),
             Line2D([0],[0],color=ca,lw=1.4,ls='--',label='Paired anchor (K2)'),
             Line2D([0],[0],color=cc,lw=1.,label='Valid candidates'),
             Line2D([0],[0],color='#657780',lw=1,label='Map-derived road edges')]
    fig,axis=plt.subplots(2,4,figsize=(15,7.0),layout='constrained')
    for j,a in enumerate(axis.flat):
        if j>=len(display):a.axis('off');continue
        _plot_trajectory(a,display[j],colors,show_inset=(j<4))
        if j//4==1:a.set_xlabel('Local longitudinal position (m)')
        if j%4==0:a.set_ylabel('Local lateral position (m)')
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.022),ncol=5,fontsize=8,frameon=False)
    _save(fig,out,PAPER_OPEN_FIGURES[0]);plt.close(fig)

    # Extra requested dynamic-anchor figure, keeping the four established main figures.
    fig,axis=plt.subplots(2,4,figsize=(15,6.8),layout='constrained')
    for j,a in enumerate(axis.flat):
        if j>=len(display):a.axis('off');continue
        _plot_anchor(a,display[j],colors)
        if j//4==1:a.set_xlabel('Local longitudinal position (m)')
        if j%4==0:a.set_ylabel('Local lateral position (m)')
    fig.legend(handles=[Line2D([0],[0],color=cg,lw=2,label='Expert'),
             Line2D([0],[0],color=ca,lw=1,alpha=.45,label='K2 anchor candidates'),
             Line2D([0],[0],color=ca,lw=2,ls='--',label='Diffusion-selected paired K2 anchor'),
             Line2D([0],[0],color='#6E76A8',lw=1,ls=':',label='Kinematic sparse anchor'),
             Line2D([0],[0],color='#657780',lw=1,label='Map-derived road edges')],
             loc='lower center',bbox_to_anchor=(.5,-.022),ncol=5,frameon=False,fontsize=8)
    _save(fig,out,ANCHOR_FIGURE);plt.close(fig)

    # Error statistics: scalar summary under the boxplots; retained separately as CSV.
    keys=['ade_m','fde_m','minade_k_m','minfde_k_m']
    labels=['ADE','FDE','minADE$_K$','minFDE$_K$']
    vals=[np.array([float(r[k]) for r in records if r.get(k) is not None and np.isfinite(float(r[k]))]) for k in keys]
    scalar=[]
    for key,v in zip(keys,vals):
        scalar.append(dict(metric=key,n=len(v),mean_m=float(np.mean(v)),median_m=float(np.median(v)),
                           std_m=float(np.std(v,ddof=1 if len(v)>1 else 0)),p90_m=float(np.percentile(v,90))))
    _write_csv(out/'prediction_error_statistics.csv',scalar)
    fig=plt.figure(figsize=(15.2,5.3),layout='constrained');gs=fig.add_gridspec(2,4,height_ratios=[4.,1.25],width_ratios=[1.2,1.15,1.1,1.1])
    a=fig.add_subplot(gs[0,0]);tabax=fig.add_subplot(gs[1,0]);b=fig.add_subplot(gs[:,1]);c=fig.add_subplot(gs[:,2]);d=fig.add_subplot(gs[:,3])
    try:a.boxplot(vals,tick_labels=labels,showmeans=True,meanprops={'marker':'D','markerfacecolor':cp,'markeredgecolor':cp,'markersize':4},showfliers=False)
    except TypeError:a.boxplot(vals,labels=labels,showmeans=True,showfliers=False)
    a.set(ylabel='Displacement error (m)',title='(a) Error distribution');a.tick_params(axis='x',labelsize=7)
    tabax.axis('off')
    table=tabax.table(cellText=[[f'{r[field]:.2f}' for r in scalar] for field in ('mean_m','median_m','std_m','p90_m')],
          rowLabels=['Mean','Median','Std','P90'],colLabels=['ADE','FDE','mADE','mFDE'],loc='center',cellLoc='center')
    table.auto_set_font_size(False);table.set_fontsize(6.3);table.scale(1.05,.94)
    t=np.asarray([x['time_s'] for x in errors_by_time]);mean=np.asarray([x['mean_error_m'] for x in errors_by_time]);sd=np.asarray([x['std_error_m'] for x in errors_by_time]);
    b.plot(t,mean,color=cp,lw=1.8,label='Mean');b.fill_between(t,np.maximum(mean-sd,0),mean+sd,color=cp,alpha=.15,label='± sample/seed SD')
    b.set(title='(b) Error vs horizon',xlabel='Prediction time (s)',ylabel='Displacement error (m)');b.legend(frameon=False,fontsize=7)
    ade=np.asarray([float(r['ade_m']) for r in records]); oracle=np.asarray([float(r['minade_k_m']) for r in records]);regret=np.maximum(0,ade-oracle)
    bound=1.05*max(float(np.max(ade)),float(np.max(oracle)),.2)
    c.scatter(ade,oracle,s=9,alpha=.34,color=cp);c.plot([0,bound],[0,bound],'--',color='#777777',lw=1)
    c.set(xlim=(0,bound),ylim=(0,bound),xlabel='Selected ADE (m)',ylabel='Oracle minADE$_K$ (m)',title='(c) Candidate coverage')
    _cdf(d,regret,label='Selection regret',color=cp)
    hit=float(np.mean(regret<=.1))*100.
    d.set(xlabel='Selected ADE − oracle minADE (m)',ylabel='Cumulative fraction',ylim=(0,1.02),title='(d) Mode selection quality')
    d.text(.97,.06,f'Mean regret {regret.mean():.2f} m\n≤0.10 m: {hit:.1f}%',transform=d.transAxes,
           ha='right',va='bottom',fontsize=7,bbox={'boxstyle':'round','facecolor':'white','alpha':.75,'edgecolor':'#dddddd'})
    for ax in (a,b,c,d):ax.grid(alpha=.14)
    _save(fig,out,PAPER_OPEN_FIGURES[1]);plt.close(fig)
    (out/'selection_quality.json').write_text(json.dumps(dict(
        n_sample_seed_pairs=len(regret),mean_regret_m=float(regret.mean()),
        median_regret_m=float(np.median(regret)),p90_regret_m=float(np.percentile(regret,90)),
        within_0_10m_pct=hit),indent=2),encoding='utf-8')

    # Three random distinct maneuver/speed conditions, arranged 3 columns by 4 rows.
    fig,ax=plt.subplots(4,3,figsize=(13.6,9.5),layout='constrained',sharex='col')
    max_a,max_steer=map(float,control_limits)
    for k,s in enumerate(controls):
        u=np.asarray(s['control']);ua=np.asarray(s['anchor_control'])
        t=np.arange(1,len(u)+1)*dt
        pairs=[(u[:,0],ua[:,0],t,'Acceleration (m/s²)',[-max_a,max_a]),
               (np.diff(u[:,0])/dt,np.diff(ua[:,0])/dt,t[1:],'Jerk (m/s³)',None),
               (np.rad2deg(u[:,1]),np.rad2deg(ua[:,1]),t,'Steering (deg)',[-np.rad2deg(max_steer),np.rad2deg(max_steer)]),
               (np.rad2deg(np.diff(u[:,1])/dt),np.rad2deg(np.diff(ua[:,1])/dt),t[1:],'Steering rate (deg/s)',None)]
        for r,(sig,anch,grid,ylabel,limits) in enumerate(pairs):
            a=ax[r,k]
            a.plot(grid,sig,color=cp,lw=1.55,label='Selected diffusion')
            a.plot(grid,anch,color=ca,lw=1.15,ls='--',label='Paired anchor')
            if limits is not None:
                for lim in limits:a.axhline(lim,color='#B55353',lw=.95,ls=':',label='Control bound' if lim==limits[0] else None)
            if k==0:a.set_ylabel(ylabel)
            if r==3:a.set_xlabel('Prediction time (s)')
            a.grid(alpha=.14)
        vs=float(s.get('initial_speed_mps',np.nan));dy=float(s.get('goal_lateral_m',np.nan))
        ax[0,k].set_title(f'Condition {k+1} | idx {s["sample_index"]} | v={vs:.1f} m/s | Δy={dy:+.1f} m')
    ax[0,0].legend(frameon=False,fontsize=7)
    _save(fig,out,PAPER_OPEN_FIGURES[2]);plt.close(fig)

    # Plot 64 held-out K2 samples, with honest percentiles and conservative
    # per-time physical stability thresholds across the sampled conditions.
    dyn=[]; limits=[]
    for s in dynamical:
        if 'states' not in s or 'ltr' not in s:continue
        state=np.asarray(s['states']);ltr=np.asarray(s['ltr']);h=state.shape[0]-1
        if h!=40 or len(ltr)!=41:raise ValueError('K2 states/LTR must have 41 points including initial state')
        lim=stability_limits(torch.as_tensor(state[1:,0],dtype=torch.float64),
             torch.tensor(float(s['payload']),dtype=torch.float64),
             torch.tensor(float(s['mu']),dtype=torch.float64),stability_config).numpy()
        dyn.append(np.stack([np.rad2deg(state[1:,2]),
                             np.rad2deg(np.arctan2(state[1:,1],state[1:,0])),
                             np.rad2deg(state[1:,3]),ltr[1:]],axis=-1))
        limits.append(np.stack([np.rad2deg(lim[:,0]),np.rad2deg(lim[:,1]),
                                np.rad2deg(lim[:,2]),lim[:,3]],axis=-1))
    if not dyn:raise ValueError('At least one real frozen K2 response required')
    dyn=np.stack(dyn);lim=np.stack(limits)
    fig,ax=plt.subplots(2,2,figsize=(10.7,7.2),layout='constrained')
    ylabel=['Yaw rate (deg/s)','Sideslip angle (deg)','Roll angle (deg)','LTR']
    titles=['(a) Yaw rate','(b) Sideslip angle','(c) Roll angle','(d) LTR']
    t=np.arange(1,dyn.shape[1]+1)*dt
    for k,a in enumerate(ax.flat):
        med=np.percentile(dyn[:,:,k],50,axis=0)
        p25,p75=np.percentile(dyn[:,:,k],[25,75],axis=0)
        p10,p90=np.percentile(dyn[:,:,k],[10,90],axis=0)
        a.fill_between(t,p10,p90,color=cp,alpha=.12,label='P10–P90')
        a.fill_between(t,p25,p75,color=cp,alpha=.3,label='P25–P75')
        a.plot(t,med,color=cp,lw=2.,label='Median')
        # The minimum among physical speed/payload/mu-dependent limits is a
        # conservative display envelope, never a redefined evaluation limit.
        bound=np.min(lim[:,:,k],axis=0)
        a.plot(t,bound,color='#A63C3C',ls='--',lw=1,label='Most conservative stability limit')
        a.plot(t,-bound,color='#A63C3C',ls='--',lw=1)
        a.set(ylabel=ylabel[k],xlabel='Prediction time (s)',title=titles[k]);a.grid(alpha=.15)
    ax[0,0].legend(frameon=False,fontsize=7)
    fig.suptitle(f'Frozen Koopman K2 responses: {len(dyn)} random validation samples, inference seed {dynamical[0]["seed"]}',fontsize=10)
    _save(fig,out,PAPER_OPEN_FIGURES[3]);plt.close(fig)

    # Saved figure data/provenance; never report these random samples as new tests.
    fields=['gt','candidate_xy','candidate_valid','selected_xy','anchor_candidate_xy',
            'anchor_selected_xy','control','anchor_control','states','ltr',
            'map_polylines','map_valid_mask','target_point','anchor_kinematic_xy','anchor_controls_bank']
    packed={k:np.stack([np.asarray(s[k]) for s in display]) for k in fields if all(k in s for s in display)}
    np.savez_compressed(out/'representative_samples.npz',**packed)
    selection={'visualization_seed':int(visualization_seed),'random_trajectory_indices':[int(s['sample_index']) for s in display],
          'random_control_indices':[int(s['sample_index']) for s in controls],
          'random_chassis_indices':[int(s['sample_index']) for s in dynamical],
          'sample_count':len(samples),'chassis_count':len(dyn),
          'selection_policy':'uniform random pool; condition-stratified random 3 controls; no error-based cherry picking'}
    (out/'representative_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    return {name:str(out/(name+'.pdf')) for name in (*PAPER_OPEN_FIGURES,ANCHOR_FIGURE)}
