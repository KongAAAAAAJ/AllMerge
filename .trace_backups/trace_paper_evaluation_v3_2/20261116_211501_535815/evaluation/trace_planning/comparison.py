"""Paper-ready comparison from VERIFIED executed TruckSim episode records.

Does not implement OCP or TransFuser; use their genuine TruckSim episode JSON
outputs with identical schemas and matched scene/seed/physical case.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import center_spacing_error, summarize_closed_loop_rows
from .paper_reporting import read_open_loop_runs, build_open_rows, combined_latex
from .open_loop import _write_csv

PAPER_METHODS=('Expert Planner','Cooperative OCP','TransFuser','TRACE')
PAPER_HEADERS=('Success (\\%)','Collision (\\%)','Stability Viol. (\\%)',
               'Spacing Error (m)','Fuel (L/100 km)','Planning Time (ms)')
PAPER_KEYS=('success_pct','collision_pct','stability_violation_pct',
            'spacing_error_m','fuel_l_per_100km','planning_time_ms')


def _latex_table(method_summaries):
    parts=[r'\begin{table*}[!t]',r'\centering',
           r'\caption{Comparison of Closed-Loop Platoon Motion Planning Performance}',
           r'\label{tab:planning_comparison}',r'\footnotesize',
           r'\renewcommand{\arraystretch}{1.15}',
           r'\begin{tabular}{@{}lcccccc@{}}',r'\hline',
           'Method & Success (\\%) & Collision (\\%) & Stability Viol. (\\%) & Spacing Error (m) & Fuel (L/100 km) & Planning Time (ms) \\\\',r'\hline']
    for name in PAPER_METHODS:
        score=method_summaries.get(name)
        cells=['--']*len(PAPER_KEYS) if score is None else [f'{score[k]:.2f}' for k in PAPER_KEYS]
        label=r'\textbf{TRACE}' if name=='TRACE' else name
        parts.append(label+' & '+' & '.join(cells)+r' \\')
    parts += [r'\hline',r'\end{tabular}',r'\end{table*}']
    return '\n'.join(parts)+'\n'


def _load_rows(paths):
    records=[]
    for path in paths:
        p=Path(path)
        sources=list(p.rglob('*.json')) if p.is_dir() else [p]
        for src in sources:
            if src.name.endswith('_summary.json') or src.name=='manifest.json':
                continue
            try: item=json.loads(src.read_text(encoding='utf-8'))
            except (ValueError,UnicodeError): continue
            if isinstance(item,dict) and {'method','scenario','seed','physical_case','success','collision'}.issubset(item):
                item['_source']=str(src)
                records.append(item)
    if not records:
        raise ValueError('No real executed TruckSim per-episode JSON records found')
    return records


def _validate(rows, complete_paper):
    keys=set()
    by_method=defaultdict(set)
    pairs=set()
    for r in rows:
        if r.get('execution_source')!='native_TruckSim_19_output':
            raise ValueError(f'Comparison requires native TruckSim source provenance: {r.get("_source")}')
        if r.get('status')!='complete':
            raise ValueError(f'Only completed real episodes are accepted: {r.get("_source")}')
        if r['method'] not in PAPER_METHODS:
            raise ValueError(f'Unexpected baseline label: {r["method"]}')
        if not all(k in r for k in ('stability_steps','vehicle_steps','spacing_error_sum_m',
                                    'spacing_observations','fuel_l','distance_m','planning_time_ms')):
            raise ValueError(f'Missing physically measured paper fields: {r.get("_source")}')
        if r['method']=='TRACE' and int(r.get('jtp_unexecuted_steps',0))>0:
            raise ValueError('A post-warmup control step did not execute the JTP-returned sequence; cannot compare as pure TRACE')
        if not r.get('physical_sim_sha256') or not r.get('physical_dll_sha256'):
            raise ValueError(f'Missing physical .sim/.dll checksum: {r.get("_source")}')
        key=(r['method'],r['scenario'],int(r['seed']),r['physical_case'])
        if key in keys:
            raise ValueError(f'Duplicate episode: {key}')
        keys.add(key)
        by_method[r['method']].add((r['scenario'],int(r['seed']),r['physical_case'],
                                     r['physical_sim_sha256'],r['physical_dll_sha256'],
                                     float(r['payload_kg']),float(r['road_mu'])))
        pairs.add((r['scenario'],int(r['seed']),r['physical_case']))
    if len(by_method)>1:
        first=next(iter(by_method.values()))
        for method,items in by_method.items():
            if items!=first:
                raise ValueError(f'Method {method} does not use identical (scenario,seed,physical_case) runs; '
                                 're-run missing episodes instead of comparing unmatched samples')
    if complete_paper:
        missing=set(PAPER_METHODS)-set(by_method)
        scenarios={r['scenario'] for r in rows}
        if missing:
            raise ValueError(f'Paper requires all 4 genuinely executed methods. '
                             f'Missing methods={sorted(missing)}')
    return sorted({r['scenario'] for r in rows})


def _plot_episodes(rows, scenarios, seed, output_dir):
    """One manuscript Fig. 5: actual trajectories and actual signed spacing.

    The old report plotted absolute spacing *errors*, which is not the spacing
    response required by the manuscript. Coordinates and target references
    come exclusively from native executed episodes.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[s for s in scenarios if any(r['scenario']==s and int(r['seed'])==seed for r in rows)]
    if not chosen:return None
    fig,ax=plt.subplots(len(chosen),2,figsize=(11.5,3.0*len(chosen)),squeeze=False,
                        layout='constrained')
    plotted=False
    palette={'Expert Planner':'#666666','Cooperative OCP':'#C16E38',
             'TransFuser':'#1B84B3','TRACE':'#5A44A4'}
    for n,scenario in enumerate(chosen):
        local_targets=set()
        for method in PAPER_METHODS:
            episode=next((r for r in rows if r['scenario']==scenario and
                int(r['seed'])==seed and r['method']==method),None)
            if episode is None: continue
            source=Path(episode['_source']).with_suffix('.npz')
            if not source.exists():
                raise FileNotFoundError(f'Native TruckSim plot data missing: {source}')
            with np.load(source,allow_pickle=False) as data:
                position=np.asarray(data['position_m']);heading=np.asarray(data['heading_rad']);
                time_s=np.asarray(data['time_s'])
            if len(time_s)!=len(position) or heading.shape!=position.shape[:2]:
                raise ValueError(f'Invalid recorded time grid/poses: {source}')
            if 'spacing_target_m' not in episode:
                raise ValueError(f'Explicit spacing_target_m missing in {episode["_source"]}; '
                                 'rerun with paper-evaluation version; do not assume 15 m')
            target=float(episode['spacing_target_m']);local_targets.add(target)
            tangent=np.stack((np.cos(heading[:, :-1]),np.sin(heading[:, :-1])),axis=-1)
            signed_gaps=((position[:, :-1]-position[:, 1:])*tangent).sum(axis=-1)
            # Full t=0 ... t=end signed gap is plotted, not the T-step error array.
            for car in range(position.shape[1]):
                ax[n,0].plot(position[:,car,0],position[:,car,1],color=palette[method],
                             alpha=1.-car*.2,linestyle=('-','--',':')[car%3],
                             label=method if car==0 else None)
            ax[n,1].plot(time_s,np.mean(signed_gaps,axis=1),label=method,
                         color=palette[method],lw=1.6)
            plotted=True
        if len(local_targets)>1:
            raise ValueError(f'Platoon spacing targets differ among baselines: {scenario}')
        if local_targets:
            ax[n,1].axhline(next(iter(local_targets)),color='#333333',ls='--',
                            lw=1.1,label='Target spacing')
        ax[n,0].set(title=scenario+'  executed trajectories',xlabel='World X (m)',ylabel='World Y (m)')
        ax[n,1].set(title=scenario+'  signed longitudinal spacing',
                    xlabel='Time (s)',ylabel='Mean center spacing (m)')
        for a in ax[n]:
            a.grid(alpha=0.25)
            handles,_=a.get_legend_handles_labels()
            if handles:a.legend(fontsize=8,frameon=False)
    if not plotted:
        plt.close(fig)
        return None
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'planning_comparison.pdf',bbox_inches='tight')
    fig.savefig(out/'planning_comparison.png',dpi=250,bbox_inches='tight')
    plt.close(fig)
    return str(out/'planning_comparison.pdf')

def evaluate(paths, *, output_dir, complete_paper=False, figure_seed=0,
             figure_scenarios='straight,merge_in,curved', open_loop_dirs=None):
    rows=_load_rows(paths)
    scenarios=_validate(rows,complete_paper)
    grouped=defaultdict(list)
    for row in rows:grouped[row['method']].append(row)
    sums={key:summarize_closed_loop_rows(rs) for key,rs in grouped.items()}
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    open_runs,open_verified=read_open_loop_runs(open_loop_dirs)
    open_rows=build_open_rows(open_runs)
    (out/'planning_comparison_table.tex').write_text(
        combined_latex(open_rows,sums),encoding='utf-8')
    # Keep a self-contained figures folder matching every manuscript includegraphics.
    figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    if 'F3' in open_runs:
        import shutil
        from .open_loop_visualization import PAPER_OPEN_FIGURES
        f3_dir=open_runs['F3'][1]/'figures'
        for name in PAPER_OPEN_FIGURES:
            for ext in ('pdf','png'):
                src=f3_dir/(name+'.'+ext)
                if not src.is_file():
                    raise FileNotFoundError(f'F3 paper figure missing: {src}')
                shutil.copy2(src,figures/src.name)
    _write_csv(out/'paper_metrics.csv',[{'method':m,**s} for m,s in sums.items()])
    current_scenarios=[s for s in figure_scenarios.split(',') if s in scenarios]
    graph=_plot_episodes(rows,current_scenarios,figure_seed,figures)
    ready=(set(PAPER_METHODS)==set(grouped) and set(open_runs)=={'F0','F3'}
           and graph is not None and len(current_scenarios)>=3)
    if complete_paper and not ready:
        raise ValueError('Complete manuscript requires verified F0/F3 open-loop runs, '
                         'all four genuine native TruckSim methods and three figure scenarios')
    manifest={
        'study':'Real TruckSim executed platoon motion planning',
        'complete_paper_table':bool(complete_paper),
        'complete_paper_ready':ready,
        'methods':list(grouped),'scenarios':scenarios,
        'paper_methods':list(PAPER_METHODS),
        'note':'Missing baseline values are --, never invented. Scenes/physical plants/seeds are paired exactly.',
        'figure':'Actual recorded traces for selected representative project scenarios; 3x2 trajectory/spacing layout.',
        'figure_output':graph,
        'open_loop_verified_runs':open_verified,
        'open_loop_panel_available':bool(open_rows),
        'paper_table':str(out/'planning_comparison_table.tex'),
    }
    (out/'comparison_summary.json').write_text(json.dumps(dict(metrics=sums,manifest=manifest),indent=2,ensure_ascii=False),encoding='utf-8')
    note=['# TRACE Manuscript Evaluation Results', '',
        '- `planning_comparison_table.tex`: one two-panel table for the manuscript.',
        '- `figures/planning_comparison.pdf`: native TruckSim trajectories and spacing.',
        '- `figures/open_loop_*.pdf`: four F3 nominal-K2 open-loop figures when supplied.',
        '- Unimplemented baselines remain `--` and are not substituted.',
        '- All TRACE reported episodes must exclude post-warmup Polynomial fallback.',
        '', '## Verified open-loop runs', '']
    note += [f'- {v["label"]}: {v["directory"]}' for v in open_verified] or ['- None']
    note += ['', '## Verified closed-loop methods', '']
    note += [f'- {method}: {s["episodes"]} episodes' for method,s in sums.items()]
    (out/'report.md').write_text('\n'.join(note)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False))
    return sums


def main(argv=None):
    parser=argparse.ArgumentParser(description='Aggregate four-method real TruckSim comparisons')
    parser.add_argument('--runs',action='append',required=True,
                        help='Episode JSON file or directory; repeat for each method')
    parser.add_argument('--output-dir',default='outputs/evaluation/trace_comparison')
    parser.add_argument('--complete-paper',action='store_true',help='Require 4 methods and 5 scenes')
    parser.add_argument('--figure-seed',type=int,default=0)
    parser.add_argument('--figure-scenarios',default='straight,merge_in,curved')
    parser.add_argument('--open-loop-dir',action='append',default=[],help='F0=PATH or F3=PATH; repeat')
    parser.add_argument('--eval-config',default='configs/evaluation/trace_planning.yaml')
    probe,_=parser.parse_known_args(argv)
    from .eval_config import defaults
    parser.set_defaults(**defaults(probe.eval_config,'comparison'))
    args=parser.parse_args(argv)
    evaluate(args.runs,output_dir=args.output_dir,complete_paper=args.complete_paper,
             figure_seed=args.figure_seed,figure_scenarios=args.figure_scenarios,
             open_loop_dirs=args.open_loop_dir)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
