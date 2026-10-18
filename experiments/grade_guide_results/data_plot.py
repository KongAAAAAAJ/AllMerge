#!/usr/bin/env python3
"""Independent offline V5 CSV plotter. No model, CUDA, env, or worktree imports.

Reads merged samples.csv and trajectories.csv. Statistical uncertainty is
sample STD across three *run seeds*, never individual G-group samples.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np

METHODS=('reward_gradient','random_sign','random_gaussian')
LABELS={'reward_gradient':'Reward Gradient','random_sign':'Random Sign','random_gaussian':'Random Gaussian'}
COLORS={'reward_gradient':'#0E8277','random_sign':'#D08B31','random_gaussian':'#6674A5'}


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as fd:
        yield from csv.DictReader(fd)


def fval(r,name,default=np.nan):
    try:return float(r[name])
    except (KeyError,ValueError,TypeError):return float(default)


def compact_metrics(path):
    """Keep V4 raw-proposal mean and positive-fraction definitions per seed.

    G samples and Random repetitions are scored V4 proposals, never independent
    SEEDS. The shaded uncertainty is calculated only from 3 seed-level values.
    """
    byseed=defaultdict(list)
    full_rows=0
    seeds=set()
    for r in rows(path):
        full_rows+=1
        seed=int(r['seed']);seeds.add(seed)
        method=r['method'];alpha=fval(r,'step_size_m')
        if method not in METHODS or alpha<=0:continue
        if fval(r,'reward_valid',1)<.5:continue
        gain=fval(r,'reward_gain')
        if not np.isfinite(gain):continue
        byseed[(seed,alpha,method)].append(gain)
    if len(seeds)<2:raise ValueError('Expected 3 seeds (at least 2 for STD), found '+str(seeds))
    results=[]
    for (seed,alpha,method),vs in sorted(byseed.items()):
        a=np.asarray(vs,float)
        results.append(dict(seed=seed,step_size_m=alpha,method=method,
                  mean_gain=float(a.mean()),positive_fraction=float((a>1e-6).mean()),
                  n_scored_proposals=len(a)))
    amps=sorted({x['step_size_m'] for x in results})
    out=[]
    for alpha in amps:
        for method in METHODS:
            subset=[r for r in results if r['step_size_m']==alpha and r['method']==method]
            if len(subset)!=len(seeds):
                raise ValueError(f'Missing observations at alpha={alpha} method={method}: got {len(subset)}/{len(seeds)} seeds')
            a=np.array([r['mean_gain'] for r in subset]);b=np.array([r['positive_fraction'] for r in subset])
            out.append(dict(step_size_m=alpha,method=method,n_seeds=len(seeds),
                    mean_reward_gain=float(a.mean()),std_reward_gain=float(a.std(ddof=1)),
                    mean_positive_fraction=float(b.mean()),std_positive_fraction=float(b.std(ddof=1)),
                    min_scored_proposals=int(min(r['n_scored_proposals'] for r in subset))))
    return results,out,sorted(seeds),full_rows


def write_csv(path,things):
    if not things:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as fo:
        w=csv.DictWriter(fo,fieldnames=list(things[0]))
        w.writeheader();w.writerows(things)


def make_figures(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    amps=sorted({r['step_size_m'] for r in summary})
    for key,mean,std,ylabel,filename,ylim in [
            ('reward','mean_reward_gain','std_reward_gain','Mean dense reward gain',
             '01_reward_gain_multiseed.png',None),
            ('positive','mean_positive_fraction','std_positive_fraction','Positive reward fraction',
             '02_positive_fraction_multiseed.png',(-0.04,1.04))]:
        fig,ax=plt.subplots(figsize=(9,5.5))
        for method in METHODS:
            values={r['step_size_m']:r for r in summary if r['method']==method}
            y=np.array([values[x][mean] for x in amps]); err=np.array([values[x][std] for x in amps])
            ax.plot(amps,y,'o-',color=COLORS[method],label=LABELS[method],lw=2.0,ms=4)
            lo=y-err;hi=y+err
            if ylim is not None:lo=np.clip(lo,0,1);hi=np.clip(hi,0,1)
            ax.fill_between(amps,lo,hi,color=COLORS[method],alpha=.18,linewidth=0)
        if key=='reward':ax.axhline(0,color='0.35',ls='--',lw=1)
        ax.set(xlabel='Requested mean-shift RMS (m)',ylabel=ylabel)
        if ylim:ax.set_ylim(*ylim)
        ax.grid(alpha=.2);ax.legend(loc='best',frameon=False)
        ax.set_title('Three inference seeds | line: seed mean | shade: ±1 seed STD',fontsize=12)
        fig.tight_layout();fig.savefig(out/filename,dpi=200);plt.close(fig)

    from matplotlib import colors as mcolors
    methods={m:{r['step_size_m']:r for r in summary if r['method']==m} for m in METHODS}
    heads=['Mean shift (m)','Gradient ΔR','Random Sign ΔR','Random Gauss ΔR',
           'Grad−Sign','Grad−Gauss','Grad positive']
    data=[]
    for a in amps:
        grad=methods['reward_gradient'][a];sign=methods['random_sign'][a];gauss=methods['random_gaussian'][a]
        def ms(r):return f"{r['mean_reward_gain']:+.4f} ± {r['std_reward_gain']:.4f}"
        data.append([f'{a:g}',ms(grad),ms(sign),ms(gauss),
            f"{grad['mean_reward_gain']-sign['mean_reward_gain']:+.4f}",
            f"{grad['mean_reward_gain']-gauss['mean_reward_gain']:+.4f}",
            f"{grad['mean_positive_fraction']*100:.1f}% ± {grad['std_positive_fraction']*100:.1f}%"])
    fig,ax=plt.subplots(figsize=(16,max(4.0,1.25+0.47*len(data))))
    ax.axis('off')
    widths=[.09,.17,.17,.17,.12,.12,.16]
    table=ax.table(cellText=data,colLabels=heads,loc='center',cellLoc='center',colWidths=widths)
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.6)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#D9DEE5');cell.set_linewidth(.5)
        if row==0:
            cell.set_facecolor('#263745');cell.get_text().set_color('white');cell.get_text().set_weight('bold')
        else:
            cell.set_facecolor('#F4F6F8' if row%2==0 else 'white')
            if col in (4,5):
                gain=float(data[row-1][col]);cell.get_text().set_weight('bold')
                if gain>0:cell.get_text().set_color('#12695F')
    ax.set_title('Dense reward improvement by guidance magnitude | mean ± SD across seeds',fontsize=12,pad=18)
    fig.tight_layout();fig.savefig(out/'03_reward_comparison_table.png',dpi=200,bbox_inches='tight');plt.close(fig)


def polyline_dist(points,reference):
    """Point-to-polyline Euclidean distance; not the full production reward."""
    xy=np.asarray(points,float);ref=np.asarray(reference,float)
    ref=ref[np.isfinite(ref).all(axis=1)]
    if len(ref)<2:return np.full(len(xy),np.nan)
    a,b=ref[:-1],ref[1:]
    ab=b-a;norm=np.maximum((ab**2).sum(-1),1e-12)
    t=np.clip(((xy[:,None,:]-a[None,:,:])*ab[None,:,:]).sum(-1)/norm,0,1)
    projected=a[None,:,:]+t[...,None]*ab[None,:,:]
    return np.min(np.linalg.norm(xy[:,None,:]-projected,axis=-1),axis=1)


def candidate_galleries(path,amplitude):
    """Read only the requested amplitude + baseline. One G-group per state was persisted."""
    grouped=defaultdict(dict)
    for r in rows(path):
        method=r['method'];alpha=fval(r,'step_size_m')
        if method!='baseline' and (not math.isclose(alpha,amplitude,abs_tol=1e-7) or
              method not in METHODS or int(r['random_repeat'])!=0):continue
        key=(int(r['seed']),r['scenario'],int(r['state_id']),int(r['mode']),int(r['group']))
        if method!='baseline':
            # Treat an absent/invalid reward as missing, not an improvement.
            if not np.isfinite(fval(r,'guided_reward')):continue
        grouped[key][method]=r
    result=[]
    for key,data in grouped.items():
        if not all(m in data for m in ('baseline',*METHODS)):continue
        b=data['baseline']; g=data['reward_gradient'];a=data['random_sign'];c=data['random_gaussian']
        baseline=fval(b,'guided_reward')
        ga=fval(g,'guided_reward')-baseline
        ra=fval(a,'guided_reward')-baseline;rb=fval(c,'guided_reward')-baseline
        if not np.isfinite(ga+ra+rb) or ga<=0 or ga<=max(ra,rb):continue
        # Prefer physically valid teacher, but keep candidate even if no fully safe examples exist.
        safe=fval(g,'guided_unsafe',1)<.5
        score=ga-.5*(ra+rb)
        result.append(dict(key=key,score=score,safe=safe,gain=ga,sign_gain=ra,gaussian_gain=rb,
                           records=data))
    return sorted(result,key=lambda r:(r['safe'],r['score']),reverse=True)


def select_diverse(candidates,max_gallery):
    # 2 per scenario at most, one per seed/scenario until possible; deterministic.
    chosen=[];scene_count=defaultdict(int);seed_scene=set()
    for first_pass in (True,False):
        for r in candidates:
            if r in chosen:continue
            key=r['key'];ss=(key[0],key[1])
            if first_pass and ss in seed_scene:continue
            if scene_count[key[1]]>=2:continue
            chosen.append(r);scene_count[key[1]]+=1;seed_scene.add(ss)
            if len(chosen)>=max_gallery:return chosen
    return chosen


def gallery_plots(candidates,out,amplitude,count):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,ConnectionPatch
    chosen=select_diverse(candidates,count)
    selected=[]
    for n,info in enumerate(chosen):
        key=info['key'];data=info['records']
        dense={m:np.asarray(json.loads(data[m]['dense_xy_json']),float) for m in ('baseline',*METHODS)}
        base=dense['baseline'];grad=dense['reward_gradient'];rnd=dense['random_sign'];gau=dense['random_gaussian']
        ref=np.asarray(json.loads(data['baseline']['centerline_xy_json']),float)
        # Find a contiguous region with best reduction in centerline distance *relative to random sign*.
        # If there is no positive centerline improvement, zoom the largest geometry difference
        # and state so explicitly. A local dense reward was NOT computed in V4/V5.
        dg=polyline_dist(grad,ref);dr=polyline_dist(rnd,ref)
        difference=dr-dg
        width=min(10,len(grad)-1)
        candidates_windows=[(i,float(np.nanmean(difference[i:i+width]))) for i in range(len(grad)-width+1)]
        good=[p for p in candidates_windows if np.isfinite(p[1])]
        if good and max(v for _,v in good)>1e-4:
            center,local_value=max(good,key=lambda p:p[1]);zone='centerline improvement'
        else:
            spacing=np.linalg.norm(grad-rnd,axis=1)
            center=int(np.argmax(np.convolve(spacing,np.ones(width)/width,'valid')))
            local_value=float('nan');zone='geometry difference (not local reward)'
        sl=slice(center,center+width)
        fig,axs=plt.subplots(1,2,figsize=(12.8,5.1),gridspec_kw={'width_ratios':[1.58,1]})
        ax,zoom=axs
        for m in ('baseline',*METHODS):
            xy=dense[m];color='black' if m=='baseline' else COLORS[m]
            label=('Baseline' if m=='baseline' else LABELS[m])+f' (R={fval(data[m],"guided_reward"):.3f})'
            for pane in (ax,zoom):
                pane.plot(xy[:,0],xy[:,1],color=color,lw=2.4 if m=='reward_gradient' else 1.7,
                          alpha=1 if m in ('baseline','reward_gradient') else .8,label=label if pane is ax else None)
            ax.scatter(xy[::5,0],xy[::5,1],s=8,c=color,alpha=.6)
        valid=ref[np.isfinite(ref).all(axis=1)]
        if len(valid)>1:
            ax.plot(valid[:,0],valid[:,1],ls='--',lw=1.2,color='0.6',label='Reference centerline')
            zoom.plot(valid[:,0],valid[:,1],ls='--',lw=1.2,color='0.7')
        part=np.concatenate([xy[sl] for xy in dense.values()],axis=0)
        lo=part.min(axis=0);hi=part.max(axis=0)
        padding=np.maximum((hi-lo)*.28,[1.0,.35]);lo-=padding;hi+=padding
        ax.add_patch(Rectangle(tuple(lo),*(hi-lo),fill=False,ls='--',lw=1.3,ec='#BA4A40'))
        zoom.set_xlim(lo[0],hi[0]);zoom.set_ylim(lo[1],hi[1])
        # Connector flags the exact section enlarged on the right.
        fig.add_artist(ConnectionPatch(xyA=(hi[0],hi[1]),coordsA=ax.transData,
            xyB=(0,1),coordsB=zoom.transAxes,color='#BA4A40',ls='--',alpha=.7))
        ax.set(xlabel='Ego X (m)',ylabel='Ego Y (m)',title='Full 41-point dense spline trajectory')
        zoom.set(xlabel='Ego X (m)',ylabel='Ego Y (m)',title=f'Local zoom: t={center/10:.1f}–{(center+width-1)/10:.1f} s')
        ax.grid(alpha=.2);zoom.grid(alpha=.2)
        ax.legend(loc='best',fontsize=7.2,framealpha=.9)
        note=(f'Reference-line distance: Gradient better by {local_value:.3f} m in zoom window'
              if zone=='centerline improvement' else
              'Zoom shows largest local geometric difference; not a verified local reward gain')
        ax.text(.015,-.18,note,transform=ax.transAxes,fontsize=8.5,color='#35535D',va='top')
        fig.suptitle(f'{key[1]} | seed={key[0]} state={key[2]} mode={key[3]} G={key[4]} | '
                     f'R gains: Grad {info["gain"]:+.3f}, Sign {info["sign_gain"]:+.3f}, Gauss {info["gaussian_gain"]:+.3f}',fontsize=11)
        fig.subplots_adjust(bottom=.22,top=.86,wspace=.28)
        filename=f'04_guided_trajectory_zoom_{n:02d}.png'
        fig.savefig(out/filename,dpi=210,bbox_inches='tight');plt.close(fig)
        selected.append(dict(seed=key[0],scenario=key[1],state_id=key[2],mode=key[3],group=key[4],
             step_size_m=amplitude,gradient_gain=info['gain'],random_sign_gain=info['sign_gain'],
             random_gaussian_gain=info['gaussian_gain'],gradient_safe=info['safe'],
             zoom_reason=zone,local_centerline_distance_advantage_m=local_value,image=filename))
    write_csv(out/'selected_gallery_states.csv',selected)
    return len(selected)


def plot_results(input_dir, gallery_amplitude_m=0.6, gallery_count=8, output_dir=None):
    """Read existing CSV data and regenerate charts, tables and zoomed galleries.

    No model/environment/CUDA imports. The V5 raw-proposal metrics are unchanged.
    Returns a dictionary with output metadata for programmatic callers.
    """
    root=Path(input_dir).resolve()
    out=Path(output_dir).resolve() if output_dir is not None else root
    out.mkdir(parents=True,exist_ok=True)
    if not (root/'samples.csv').is_file():
        raise FileNotFoundError(root/'samples.csv')
    seed_metrics,summary,seeds,nrows=compact_metrics(root/'samples.csv')
    write_csv(out/'seed_metrics.csv',seed_metrics)
    write_csv(out/'summary_by_amplitude.csv',summary)
    make_figures(summary,out)
    n=0
    if (root/'trajectories.csv').exists():
        candidates=candidate_galleries(root/'trajectories.csv',float(gallery_amplitude_m))
        n=gallery_plots(candidates,out,float(gallery_amplitude_m),int(gallery_count))
    print('[PLOT] seeds:',seeds,'sample rows:',nrows,'gallery cases:',n,'output:',out)
    if n==0:
        print('[INFO] No eligible gallery cases at amplitude',gallery_amplitude_m)
    return {'seeds':seeds,'rows':nrows,'galleries':n,'output_dir':str(out)}


def plot(input_dir, gallery_amplitude_m=0.6, gallery_count=8, output_dir=None):
    """Short alias for the public CSV-only plotting function."""
    return plot_results(input_dir, gallery_amplitude_m, gallery_count, output_dir)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=Path('outputs/reward_guided_mean_v5'))
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--gallery-amplitude-m',type=float,default=.6)
    p.add_argument('--gallery-count',type=int,default=8)
    args=p.parse_args(argv)
    return plot_results(args.input_dir,args.gallery_amplitude_m,args.gallery_count,args.output_dir)

if __name__=='__main__':main()
