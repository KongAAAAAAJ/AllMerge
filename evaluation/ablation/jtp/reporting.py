"""CSV, JSON, paired statistics, and IEEE paper table for JTP ablation."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PAPER_COLUMNS = ('joint_feasible', 'geometric_violation', 'stability_violation')


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8-sig'); return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=lambda x:
                         x.item() if isinstance(x, np.generic) else str(x))+'\n', encoding='utf-8')


def _summary(rows):
    n = len(rows)
    if n == 0:
        raise ValueError('No rows for ablation summary')
    def pct(key): return 100. * sum(bool(r[key]) for r in rows) / n
    distinct_frames = {}
    for row in rows:
        key = (row['seed'], row['episode_index'], row['frame_index'])
        if key in distinct_frames and distinct_frames[key] != bool(row['joint_frame_feasible']):
            raise ValueError(f'Inconsistent full-platoon feasibility for {key}')
        distinct_frames[key] = bool(row['joint_frame_feasible'])
    return {
        'num_vehicles': n,
        'num_frames': len({(r['seed'],r['episode_index'],r['frame_index']) for r in rows}),
        'joint_feasibility_pct':100. * sum(distinct_frames.values()) / len(distinct_frames),
        'joint_vehicle_feasibility_pct': pct('joint_feasible'),
        'geometric_violation_pct': pct('geometric_violation'),
        'stability_violation_pct': pct('stability_violation'),
        'collision_violation_pct': pct('collision_violation'),
        'control_violation_pct': pct('control_violation'),
        'external_collision_violation_pct': pct('external_collision_violation'),
        'intertruck_collision_violation_pct': pct('intertruck_collision_violation'),
        'control_rate_violation_pct': 100.*np.mean([r['acceleration_rate_violation'] or r['steering_rate_violation'] for r in rows]),
        'deviation_m': float(np.mean([r['deviation_m'] for r in rows])) if rows[0]['method']=='Ours' else None,
        'projector_success_pct': pct('projector_success') if rows[0]['method']=='Ours' else None,
        'projection_applied_pct': pct('projection_applied') if rows[0]['method']=='Ours' else None,
        'projection_latency_ms_mean': float(np.mean([r['projection_latency_ms'] for r in rows])) if rows[0]['method']=='Ours' else None,
    }


def paired_report(rows: list[dict], *, bootstrap_repeats=2000, bootstrap_seed=2026):
    """Cluster-bootstrap scene/frame pairs (a frame includes all controlled trucks).

    Each physical episode/frame is a cluster containing all inference seeds and
    all platoon trucks. Both methods share identical keys.
    """
    keys = ('seed','episode_index','frame_index','ego_index')
    lookup = {}
    for row in rows:
        key = (row['method'],) + tuple(row[k] for k in keys)
        if key in lookup:
            raise ValueError(f'Duplicate ablation row {key}')
        lookup[key] = row
    names = [('Baseline',r) for r in rows if r['method']=='Baseline']
    if not names:
        raise ValueError('No Baseline observations')
    frames = defaultdict(list)
    for _, ref in names:
        key = tuple(ref[k] for k in keys)
        if ('Ours',)+key not in lookup:
            raise ValueError(f'Missing exact paired JTP observation: {key}')
        frames[(ref['episode_index'],ref['frame_index'])].append((ref,lookup[('Ours',)+key]))
    if len([r for r in rows if r['method']=='Ours']) != len(names):
        raise ValueError('Unmatched/duplicated Ours observations')
    baselines = [p[0] for chunk in frames.values() for p in chunk]
    projected = [p[1] for chunk in frames.values() for p in chunk]
    by_method = {'Baseline':_summary(baselines),'Ours':_summary(projected)}
    deltas = {name: by_method['Ours'][name]-by_method['Baseline'][name]
              for name in ('joint_feasibility_pct','geometric_violation_pct',
                           'stability_violation_pct','collision_violation_pct','control_violation_pct')}
    rng = np.random.default_rng(bootstrap_seed)
    frame_groups = list(frames.values())
    def _pair_deltas(pairs):
        frame_values = {}
        for a,b in pairs:
            key=(a['seed'],a['episode_index'],a['frame_index'])
            value = float(a['joint_frame_feasible'])-float(b['joint_frame_feasible'])
            if key in frame_values and value != frame_values[key]:
                raise ValueError(f'Conflicting frame feasibility in paired results {key}')
            frame_values[key]=value
        return {col: (float(np.mean(list(frame_values.values())))*100. if col=='joint_feasible'
                      else float(np.mean([float(a[col])-float(b[col]) for a,b in pairs]))*100.)
                for col in PAPER_COLUMNS}
    observed = _pair_deltas([pair for group in frame_groups for pair in group])
    array = {col: [] for col in PAPER_COLUMNS}
    for _ in range(bootstrap_repeats):
        draw = rng.integers(0,len(frame_groups),size=len(frame_groups))
        sample = [p for ix in draw for p in frame_groups[int(ix)]]
        vals = _pair_deltas(sample)
        for col in PAPER_COLUMNS: array[col].append(vals[col])
    confidence = {col: {'baseline_minus_ours_pp': observed[col],
                        'bootstrap_95ci_pp': ([float(x) for x in np.percentile(array[col],[2.5,97.5])]
                                               if bootstrap_repeats else None)}
                  for col in PAPER_COLUMNS}
    return by_method, deltas, confidence


def paper_table(by_method: dict, *, paper_ready: bool) -> str:
    """Report actual results, with a non-paper watermark for incomplete solvers."""
    caption = ('Effects of Joint Trajectory Projection on Constraint Satisfaction'
               if paper_ready else 'DIAGNOSTIC ONLY: Partial JTP Solver Constraint Satisfaction')
    lines = [r'\begin{table}[!t]',r'\centering',
             r'\caption{' + caption + '}',r'\label{tab:projection_ablation}',
             r'\scriptsize',r'\renewcommand{\arraystretch}{1.15}',r'\setlength{\tabcolsep}{2.5pt}',
             r'\begin{tabular}{@{}lcccc@{}}',r'\hline',
             r'\textbf{Method} & \shortstack{\textbf{Joint}\\\textbf{Feas. (\%)}} & '
             r'\shortstack{\textbf{Geometric}\\\textbf{Viol. (\%)}} & '
             r'\shortstack{\textbf{Stability}\\\textbf{Viol. (\%)}} & '
             r'\shortstack{\textbf{Deviation}\\\textbf{(m)}} \\',r'\hline']
    for key,label in (('Baseline','Baseline'),('Ours',r'\textbf{Ours}')):
        value = by_method[key]
        cell = [f"{value['joint_feasibility_pct']:.2f}",
                f"{value['geometric_violation_pct']:.2f}",
                f"{value['stability_violation_pct']:.2f}",
                'N/A' if key=='Baseline' else f"{value['deviation_m']:.3f}"]
        if key=='Ours':cell=[r'\textbf{'+x+'}' for x in cell]
        lines.append(label + ' & ' + ' & '.join(cell) + r' \\')
    lines.extend([r'\hline',r'\end{tabular}',r'\end{table}',''])
    return '\n'.join(lines)
