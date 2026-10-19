"""Filter and rank all states; no first-nine shortcut or double-counting reward."""
from __future__ import annotations
import numpy as np


def _rank01(vals):
    x=np.asarray(vals,dtype=float)
    if len(x)<2:return np.ones(len(x),dtype=float)
    return np.argsort(np.argsort(x,kind='mergesort'),kind='mergesort').astype(float)/(len(x)-1)


def select_topk(rows,k=12,min_gain=0.0,min_shape_m=0.08,min_closer_m=0.03,per_scene=None):
    """Selection guards are strict; zero selected rows are reported as such, never fabricated."""
    if k<=0:raise ValueError('top-k must be positive')
    eligible=[dict(r) for r in rows if r['train_reward_gain']>min_gain
              and r['teacher_distance_improve_m']>min_closer_m
              and r['pre_post_dense_rms_m']>=min_shape_m]
    if not eligible:return [],{'eligible':0,'selected':0,'min_gain':min_gain,'min_shape_m':min_shape_m,'min_closer_m':min_closer_m}
    a=_rank01([r['train_reward_gain'] for r in eligible]);b=_rank01([r['teacher_distance_improve_m'] for r in eligible])
    c=_rank01([r['max_lateral_shift_m'] for r in eligible])
    for i,r in enumerate(eligible):r['selection_score']=float(.45*a[i]+.35*b[i]+.20*c[i])
    eligible.sort(key=lambda r:(-r['selection_score'],-r['train_reward_gain'],int(r['state_index'])))
    # Greedy diversity: first pass spreads scene coverage; second fills remaining slots.
    seen=set();chosen=[];ids=set()
    if per_scene is None:per_scene=max(1,k//4)
    counts={}
    for r in eligible:
        sc=r['scenario']
        if counts.get(sc,0)<per_scene and len(chosen)<k:
            chosen.append(r);ids.add(r['state_index']);counts[sc]=counts.get(sc,0)+1;seen.add(sc)
    for r in eligible:
        if len(chosen)>=k:break
        if r['state_index'] not in ids:chosen.append(r);ids.add(r['state_index'])
    chosen.sort(key=lambda r:-r['selection_score'])
    for idx,r in enumerate(chosen,1):r['selection_rank']=idx
    return chosen,{'eligible':len(eligible),'selected':len(chosen),'scenario_counts':{s:sum(r['scenario']==s for r in chosen) for s in sorted(seen|set(r['scenario'] for r in chosen))}}
