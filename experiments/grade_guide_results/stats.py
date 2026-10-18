#!/usr/bin/env python3
"""State-clustered paired statistics and publication-ready exploratory plots.

Never count G-group members as independent scenes for confidence intervals.
"""
from __future__ import annotations
import csv,json,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np

NUMERIC=('reward_gain','baseline_reward','guided_reward','step_size_m','actual_rms_shift_m',
         'target_rms_m','input_match_rel_error','dense_rms_displacement_m',
         'dense_nonrigid_rms_m','conditional_kl_nats','guided_unsafe','base_unsafe',
         'guided_collision','guided_out_of_drivable','safe_improved','safe_regressed',
         'group','mode','random_repeat','reward_valid')


def read_rows(out):
    with (Path(out)/'samples.csv').open('r',encoding='utf-8-sig',newline='') as f:
        rr=list(csv.DictReader(f))
    for r in rr:
        for k in NUMERIC:
            if k in r:
                try:r[k]=float(r[k])
                except (TypeError,ValueError):r[k]=float('nan')
    return rr


def finite_mean(a):
    a=np.asarray(a,dtype=float)
    a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None


def summarize(rr):
    if not rr:return dict(n=0)
    reward=np.asarray([r['reward_gain'] for r in rr],dtype=float)
    valid=np.isfinite(reward)
    reward=reward[valid]
    if len(reward)==0:return dict(n=len(rr),valid_n=0)
    return dict(n=len(rr),valid_n=len(reward),mean_gain=float(reward.mean()),
          median_gain=float(np.median(reward)),positive_fraction=float((reward>1e-6).mean()),
          safe_fraction=finite_mean([1-float(r['guided_unsafe']) for r in rr if np.isfinite(r['reward_gain'])]),
          collision_fraction=finite_mean([float(r['guided_collision']) for r in rr if np.isfinite(r['reward_gain'])]),
          road_out_fraction=finite_mean([float(r['guided_out_of_drivable']) for r in rr if np.isfinite(r['reward_gain'])]),
          mean_actual_rms_shift_m=finite_mean([r['actual_rms_shift_m'] for r in rr if np.isfinite(r['reward_gain'])]),
          mean_dense_displacement_m=finite_mean([r['dense_rms_displacement_m'] for r in rr if np.isfinite(r['reward_gain'])]),
          mean_dense_shape_m=finite_mean([r['dense_nonrigid_rms_m'] for r in rr if np.isfinite(r['reward_gain'])]),
          mean_input_match_rel_error=finite_mean([r['input_match_rel_error'] for r in rr if np.isfinite(r['reward_gain'])]))


def paired_comparisons(rows):
    """For each identical state/mode/group/amplitude compare one grad with mean random repeats."""
    data=defaultdict(dict)
    for r in rows:
        if r['method']=='baseline' or r['step_size_m']==0 or not np.isfinite(r['reward_gain']):continue
        key=(r['scenario'],int(r['state_id']),int(r['mode']),int(r['group']),float(r['step_size_m']))
        data[key].setdefault(r['method'],[]).append(r['reward_gain'])
    compared=[]
    for key,methods in data.items():
        if 'reward_gradient' not in methods:continue
        grad=float(np.mean(methods['reward_gradient']))
        for typ in ('random_sign','random_gaussian'):
            if typ not in methods:continue
            rnd=float(np.mean(methods[typ]))
            compared.append(dict(scenario=key[0],state_id=key[1],mode=key[2],group=key[3],
                      alpha_m=key[4],random_type=typ,gradient_gain=grad,random_mean_gain=rnd,
                      paired_improvement=grad-rnd,gradient_win_fraction=float(np.mean(grad>np.asarray(methods[typ]))),
                      random_repeats=len(methods[typ])))
    return compared


def best_at_g(rows):
    """Fair Best@G per state/mode/amplitude/method/repetition, not Best across random restarts."""
    pools=defaultdict(list)
    for r in rows:
        if r['method']=='baseline' or r['step_size_m']==0 or not np.isfinite(r['reward_gain']):continue
        key=(r['scenario'],int(r['state_id']),int(r['mode']),float(r['step_size_m']),
             r['method'],int(r.get('random_repeat',0)))
        pools[key].append(r)
    by=defaultdict(list)
    for key,rr in pools.items():
        best=max(r.get('guided_reward',r['reward_gain']) for r in rr)
        safe=[r.get('guided_reward',r['reward_gain']) for r in rr if r['guided_unsafe']<.5]
        by[(key[3],key[4])].append((best,max(safe) if safe else float('nan'),len(rr)))
    return {str(amp):{method:{
            'n_state_mode_repetitions':len(values),
            'mean_best_reward':finite_mean([x[0] for x in values]),
            'mean_best_safe_reward':finite_mean([x[1] for x in values]),
            'any_safe_fraction':float(np.mean([np.isfinite(x[1]) for x in values])),
            'mean_valid_G':float(np.mean([x[2] for x in values]))}
           for (a,method),values in by.items() if a==amp}
         for amp in sorted({k[0] for k in by})}


def _state_bootstrap(rr,*,b=1200,seed=42):
    """CI on mean paired gradient advantage, clustered by scenario/state."""
    if not rr:return None
    groups=defaultdict(list)
    for r in rr:groups[(r['scenario'],r['state_id'])].append(float(r['paired_improvement']))
    per_state=np.asarray([np.mean(v) for v in groups.values()],dtype=float)
    if len(per_state)==0:return None
    rng=np.random.default_rng(seed)
    boot=per_state[rng.integers(0,len(per_state),(b,len(per_state)))].mean(axis=1)
    return {'state_count':int(len(per_state)),'mean_state_paired_gain':float(per_state.mean()),
            'bootstrap_95ci':list(np.quantile(boot,[.025,.975]).astype(float)),
            'state_positive_fraction':float(np.mean(per_state>0))}


def statistics(rows,errors,config,names):
    methods=('baseline','reward_gradient','random_sign','random_gaussian')
    alphas=sorted({float(r['step_size_m']) for r in rows if 'step_size_m' in r})
    by_method={m:summarize([r for r in rows if r['method']==m and (m=='baseline' or r['step_size_m']>0)])
               for m in methods}
    by_alpha={str(a):{m:summarize([r for r in rows if r['method']==m and r['step_size_m']==a])
                      for m in methods if any(r['method']==m and r['step_size_m']==a for r in rows)} for a in alphas}
    by_scenario={s:{m:summarize([r for r in rows if r['scenario']==s and r['method']==m and r['step_size_m']>0])
                   for m in methods[1:]} for s in names}
    paired=paired_comparisons(rows)
    paired_overall={m:_state_bootstrap([r for r in paired if r['random_type']==m])
                       for m in ('random_sign','random_gaussian')}
    paired_by_alpha={str(a):{m:_state_bootstrap([r for r in paired if r['random_type']==m and r['alpha_m']==a])
                                   for m in ('random_sign','random_gaussian')} for a in alphas if a>0}
    return dict(experiment='V4 larger-state frozen guidance-vs-random controlled Gaussian mean probe',
        n_rows=len(rows),n_original_base_candidates=sum(r['method']=='baseline' for r in rows),
        baseline_unscored_proposals=len(errors),nonfinite_reward_rows=sum(not np.isfinite(r['reward_gain']) for r in rows),failed_proposals_by_method=dict(Counter(e.get('method','state') for e in errors)),
        methods=by_method,by_amplitude_m=by_alpha,by_scenario=by_scenario,
        paired_gradient_minus_random=paired_overall,
        paired_by_amplitude_m=paired_by_alpha,best_at_g=best_at_g(rows),config=config,
        notes={'primary':'UNSELECTED per-proposal reward gain, not best-of-K and not accepted-only',
               'confidence_interval':'Bootstrap resamples scene states (scenario,state_id), not G candidates',
               'failure_handling':'Failed whole-batch scoring logged, absent from valid-data means; inspect failure count before interpreting large amplitudes',
               'random_sign':'Exactly same absolute gradient perturbation per coordinate => identical KL and perturbation budget',
               'random_gaussian':'Smooth Gaussian, per-axis energy matched before cap; actual magnitude mismatch explicitly logged'})


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    rr=read_rows(out)
    paired=paired_comparisons(rr)
    methods=('reward_gradient','random_sign','random_gaussian')
    amps=sorted({r['step_size_m'] for r in rr if r['step_size_m']>0})
    if not amps:raise ValueError('No valid amplitude observations')
    def series(method,k,func='mean'):
        vals=[]
        for a in amps:
            x=np.asarray([r[k] for r in rr if r['method']==method and r['step_size_m']==a],dtype=float)
            x=x[np.isfinite(x)]
            vals.append(float(np.median(x) if func=='median' else np.mean(x)) if len(x) else np.nan)
        return vals
    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    for m in methods:
        axs[0].plot(amps,series(m,'reward_gain'),'o-',label=m)
        axs[1].plot(amps,[float(np.mean([r['reward_gain']>1e-6 for r in rr if r['method']==m and r['step_size_m']==a and np.isfinite(r['reward_gain'])])) if any(r['method']==m and r['step_size_m']==a and np.isfinite(r['reward_gain']) for r in rr) else np.nan for a in amps],'o-',label=m)
    axs[0].axhline(0,ls='--',c='.5');axs[0].set(xlabel='Requested mean-shift RMS (m)',ylabel='Mean dense reward gain',title='Raw matched proposals, no selection')
    axs[1].set(xlabel='Requested mean-shift RMS (m)',ylabel='Positive reward fraction',title='Improvement frequency',ylim=(-.05,1.05))
    for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'01_reward_gradient_vs_random.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    for m in ('random_sign','random_gaussian'):
        mm=[];lo=[];hi=[]
        for a in amps:
            ci=_state_bootstrap([r for r in paired if r['random_type']==m and r['alpha_m']==a],b=500)
            if ci:
                mm.append(ci['mean_state_paired_gain']);lo.append(ci['bootstrap_95ci'][0]);hi.append(ci['bootstrap_95ci'][1])
            else:mm.append(np.nan);lo.append(np.nan);hi.append(np.nan)
        axs[0].plot(amps,mm,'o-',label='grad - '+m)
        axs[0].fill_between(amps,lo,hi,alpha=.18)
    axs[0].axhline(0,c='.5',ls='--');axs[0].set(xlabel='Requested shift RMS (m)',ylabel='Paired dense reward advantage',title='State-clustered 95% bootstrap CI')
    for m in methods:
        axs[1].plot(amps,series(m,'actual_rms_shift_m'),'o-',label=m)
    axs[1].set(xlabel='Requested shift RMS (m)',ylabel='Actual input mean-shift RMS (m)',title='Check matched perturbation budgets')
    for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'02_paired_advantage_and_budget.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    for m in methods:
        axs[0].plot(amps,series(m,'dense_rms_displacement_m'),'o-',label=m)
        axs[1].plot(amps,series(m,'dense_nonrigid_rms_m'),'o-',label=m)
    axs[0].set(xlabel='Requested input shift RMS (m)',ylabel='Final dense RMS displacement (m)',title='Final trajectory response')
    axs[1].set(xlabel='Requested input shift RMS (m)',ylabel='Dense nonrigid RMS (m)',title='Final trajectory geometric change')
    for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'03_dense_geometry_comparison.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    for m in methods:
        axs[0].plot(amps,[1-x for x in series(m,'guided_unsafe')],'o-',label=m)
        axs[1].plot(amps,series(m,'guided_out_of_drivable'),'o-',label=m)
    axs[0].set(xlabel='Requested shift RMS (m)',ylabel='Safe fraction',title='Production safety diagnosis',ylim=(-.05,1.05))
    axs[1].set(xlabel='Requested shift RMS (m)',ylabel='Road-out fraction',title='Road constraint',ylim=(-.05,1.05))
    for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'04_safety_by_method.png',dpi=170);plt.close(fig)
    scenarios=sorted({r['scenario'] for r in rr});top=max(amps)
    fig,ax=plt.subplots(figsize=(9,4.6))
    x=np.arange(len(scenarios));width=.25
    for j,m in enumerate(methods):
        y=[finite_mean([r['reward_gain'] for r in rr if r['scenario']==s and r['method']==m and r['step_size_m']==top]) for s in scenarios]
        ax.bar(x+(j-1)*width,[v if v is not None else 0 for v in y],width,label=m)
    ax.set_xticks(x,scenarios);ax.axhline(0,c='.5',lw=1);ax.set(ylabel='Mean dense reward gain',title=f'Scenario breakdown at requested {top:g}m (raw, no selection)')
    ax.grid(axis='y',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'05_scenario_large_shift.png',dpi=170);plt.close(fig)
    gallery_path=out/'gallery.json'
    if gallery_path.exists():
        gal=json.loads(gallery_path.read_text(encoding='utf-8'))
        for i,p in enumerate(gal):
            fig,axs=plt.subplots(1,3,figsize=(16,4.5))
            base=np.asarray(p['baseline_dense'],dtype=float)
            bs=np.asarray(p['baseline_sparse'],dtype=float)
            axs[0].plot(base[:,0],base[:,1],'-',lw=2.5,label=f'base R={p["baseline_reward"]:.3f}',c='black')
            axs[0].plot(bs[:,0],bs[:,1],'o',ms=3,c='black')
            for e in p['traces']:
                if e['method']=='reward_gradient' or (e['method']=='random_sign' and e['repeat']==0):
                    path=np.asarray(e['dense'],dtype=float)
                    label=f'{e["method"]} {e["amplitude_m"]:g}m R={e["reward"]:.3f}'
                    axs[0].plot(path[:,0],path[:,1],'-',lw=1,alpha=.85,label=label)
                    t=np.arange(len(path))*.1
                    axs[1].plot(t,path[:,0]-base[:,0],lw=1,alpha=.8,label=e['method']+f' {e["amplitude_m"]:g}')
                    axs[2].plot(t,path[:,1]-base[:,1],lw=1,alpha=.8,label=e['method']+f' {e["amplitude_m"]:g}')
            axs[0].set(xlabel='Ego-local X (m)',ylabel='Ego-local Y (m)',title='Dense 10Hz trajectories')
            axs[1].set(xlabel='Time (s)',ylabel='Delta X (m)',title='Longitudinal change')
            axs[2].set(xlabel='Time (s)',ylabel='Delta Y (m)',title='Lateral change')
            axs[0].legend(fontsize=5,loc='best');axs[1].legend(fontsize=5,ncol=2);axs[2].legend(fontsize=5,ncol=2)
            for a in axs:a.grid(alpha=.22)
            fig.suptitle(f'{p["scenario"]} state={p["state_id"]} group={p["group"]} mode={p["mode"]} (random state selection)')
            fig.tight_layout();fig.savefig(out/f'06_gallery_{i:02d}.png',dpi=160);plt.close(fig)
    print('[PLOT] Dense reward, state-CI, geometry, safety, scenario, randomized-state galleries saved in',out)

if __name__=='__main__':
    import sys
    plot(Path(sys.argv[1]))
