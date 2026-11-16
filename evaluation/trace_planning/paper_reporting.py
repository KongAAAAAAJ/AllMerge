"""Manuscript-aligned two-panel paper table; no invented missing baselines.

Panel A is populated only from verified open-loop K2 records. Panel B only
from native TruckSim episode summaries. A dash marks absent verified runs.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

OPEN_METHODS = ('Dynamic control anchors (K2)',
                'Control Diffusion, F0 (K2)',
                'Control Diffusion, F3 (K2)')
CLOSED_METHODS = ('Expert Planner','Cooperative OCP','TransFuser','TRACE')
OPEN_KEYS = ('ade_m','fde_m','minade_k_m','minfde_k_m',
             'k2_stability_horizon_rate_pct','fuel_l_per_100km')
CLOSED_KEYS = ('success_pct','collision_pct','stability_violation_pct',
               'spacing_error_m','fuel_l_per_100km','planning_time_ms')


def _numbers(seeds, prefix=''):
    remap={
        'k2_stability_horizon_rate_pct':'anchor_k2_stability_horizon_rate_pct',
        'fuel_l_per_100km':'anchor_fuel_l_per_100km',
    }
    values=[]
    for key in OPEN_KEYS:
        source=(remap.get(key,prefix+key) if prefix else key)
        nums=[float(item[source]) for item in seeds if source in item and
              math.isfinite(float(item[source]))]
        if len(nums)!=len(seeds) or not nums:
            raise ValueError(f'Open-loop table requires metric {source} from every inference seed')
        mu=sum(nums)/len(nums)
        sd=(sum((v-mu)**2 for v in nums)/(len(nums)-1))**.5 if len(nums)>1 else None
        values.append((mu,sd))
    return values


def _run_signature(directory: Path):
    with (directory/'samples.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        required=('inference_seed','sample_index','shard','shard_sample_index')
        if not set(required).issubset(set(reader.fieldnames or [])):
            raise ValueError(f'{directory}: missing paired sample identification fields')
        items=[]
        for row in reader:
            items.append(tuple(row.get(k,'') for k in
                ('inference_seed','sample_index','shard','shard_sample_index',
                 'episode_index','frame_index','ego_index','seed')))
        return tuple(items)


def read_open_loop_runs(specs):
    """--open-loop-dir F0=DIR --open-loop-dir F3=DIR. Can pass just F3."""
    matched={}
    common=None
    verified=[]
    for spec in specs or []:
        if '=' not in spec:
            raise ValueError('Use --open-loop-dir F0=PATH or F3=PATH')
        name,dirname=spec.split('=',1)
        name=name.strip().upper()
        if name not in ('F0','F3') or name in matched:
            raise ValueError('Open-loop run labels must be unique F0/F3')
        directory=Path(dirname).resolve()
        summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        manifest=summary.get('manifest',{})
        if manifest.get('plant')!='frozen_koopman_k2_prediction' or manifest.get('jtp_enabled') is not False:
            raise ValueError(f'{directory}: not a pure nominal frozen K2 open-loop run')
        if manifest.get('checkpoint_architecture')!='TRACE_control_diffusion_D2':
            raise ValueError(f'{directory}: unexpected diffusion checkpoint architecture')
        source_signature=_run_signature(directory)
        signature={
          'samples':source_signature,
          'dataset':manifest.get('dataset'),
          'split':manifest.get('split'),
          'seeds':manifest.get('seeds'),
          'execution_config':manifest.get('checkpoint_execution_config'),
        }
        if common is None: common=signature
        elif common!=signature:
            raise ValueError('F0/F3 open-loop runs use unmatched samples, split, seeds or K2 execution configuration')
        _numbers(summary['per_seed'])
        _numbers(summary['per_seed'],prefix='anchor_')
        matched[name]=(summary,directory)
        verified.append({'label':name,'directory':str(directory),
                         'checkpoint':manifest.get('checkpoint'),
                         'sha256':manifest.get('checkpoint_sha256'),
                         'samples':len(source_signature)})
    return matched,verified


def build_open_rows(runs):
    result={}
    if runs:
        selected = runs.get('F3') or runs.get('F0')
        result[OPEN_METHODS[0]]=_numbers(selected[0]['per_seed'],prefix='anchor_')
    for label,display in [('F0',OPEN_METHODS[1]),('F3',OPEN_METHODS[2])]:
        if label in runs:
            result[display]=_numbers(runs[label][0]['per_seed'])
    return result


def _format_cell(value, digits=2):
    if value is None:return '--'
    if isinstance(value,tuple):
        mean,sd=value
        return (f'${mean:.{digits}f}\\pm{sd:.{digits}f}$' if sd is not None
                else f'{mean:.{digits}f}')
    return f'{float(value):.{digits}f}'


def combined_latex(open_rows, closed_rows):
    lines=[r'\begin{table*}[!t]',r'\centering',
        r'\caption{Open-loop generation accuracy and closed-loop TruckSim execution performance. Panel (a) reports 40-step frozen Koopman predictions without JTP; stability violation is the fraction of horizons with any stability breach, and fuel is estimated using VT-CPFM. Panel (b) reports native TruckSim executed-state metrics. Open-loop results are mean $\pm$ standard deviation across inference seeds. A dash denotes an unverified or unavailable result.}',
        r'\label{tab:planning_comparison}',r'\footnotesize',
        r'\renewcommand{\arraystretch}{1.13}',r'\setlength{\tabcolsep}{3.5pt}',
        r'\begin{tabular}{@{}lcccccc@{}}',r'\hline',
        r'\multicolumn{7}{@{}l}{\textbf{(a) Open-loop trajectory generation (Koopman-predicted)}} \\',
        r'\hline',
        r'\textbf{Method} & \shortstack{\textbf{ADE}\\\textbf{(m)$\downarrow$}} & \shortstack{\textbf{FDE}\\\textbf{(m)$\downarrow$}} & \shortstack{\textbf{minADE$_K$}\\\textbf{(m)$\downarrow$}} & \shortstack{\textbf{minFDE$_K$}\\\textbf{(m)$\downarrow$}} & \shortstack{\textbf{Stability Viol.}\\\textbf{(\%)$\downarrow$}} & \shortstack{\textbf{Fuel}\\\textbf{(L/100 km)$\downarrow$}} \\',
        r'\hline']
    for method in OPEN_METHODS:
        vals=open_rows.get(method)
        cells=['--']*6 if vals is None else [_format_cell(v) for v in vals]
        text=(r'\textbf{Control Diffusion, F3 (K2)}' if method==OPEN_METHODS[2] else method)
        lines.append(text+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\hline',
        r'\multicolumn{7}{@{}l}{\textbf{(b) Closed-loop platoon execution (TruckSim)}} \\',r'\hline',
        r'\textbf{Method} & \shortstack{\textbf{Success}\\\textbf{(\%)$\uparrow$}} & \shortstack{\textbf{Collision}\\\textbf{(\%)$\downarrow$}} & \shortstack{\textbf{Stability Viol.}\\\textbf{(\%)$\downarrow$}} & \shortstack{\textbf{Spacing Error}\\\textbf{(m)$\downarrow$}} & \shortstack{\textbf{Fuel}\\\textbf{(L/100 km)$\downarrow$}} & \shortstack{\textbf{Planning Time}\\\textbf{(ms)$\downarrow$}} \\',r'\hline'])
    for method in CLOSED_METHODS:
        scores=closed_rows.get(method)
        vals=['--']*6 if scores is None else [_format_cell(scores[k]) for k in CLOSED_KEYS]
        lines.append((r'\textbf{TRACE}' if method=='TRACE' else method)+' & '+' & '.join(vals)+r' \\')
    lines.extend([r'\hline',r'\end{tabular}',r'\end{table*}'])
    return '\n'.join(lines)+'\n'


def write_open_loop_panel(directory):
    directory=Path(directory)
    summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    rows={OPEN_METHODS[0]:_numbers(summary['per_seed'],prefix='anchor_')}
    # A single run cannot prove its checkpoint corresponds to F0 or F3.
    # Label assignment takes place explicitly when the closed-loop paper run
    # receives --open-loop-dir F0=... or F3=...
    (directory/'paper_open_loop_panel.json').write_text(json.dumps(
        {'single_checkpoint':summary['manifest'].get('checkpoint'),
         'anchor':{k:[v[0],v[1]] for k,v in zip(OPEN_KEYS,rows[OPEN_METHODS[0]])},
         'diffusion':{k:[v[0],v[1]] for k,v in zip(OPEN_KEYS,_numbers(summary['per_seed']))}},
        indent=2),encoding='utf-8')
