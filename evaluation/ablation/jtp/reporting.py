"""CSV, JSON, paired statistics, and IEEE paper table for JTP ablation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path: Path, rows: list[dict], columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and columns is None:
        path.write_text('', encoding='utf-8-sig')
        return
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)
    ) + '\n', encoding='utf-8')


def _summary(rows):
    n = len(rows)
    if n == 0:
        raise ValueError('No rows for ablation summary')

    def pct(key):
        return 100.0 * sum(bool(row[key]) for row in rows) / n

    distinct_frames = {}
    for row in rows:
        key = (row['seed'], row['episode_index'], row['frame_index'])
        value = bool(row['joint_frame_feasible'])
        if key in distinct_frames and distinct_frames[key] != value:
            raise ValueError(f'Inconsistent full-platoon feasibility for {key}')
        distinct_frames[key] = value

    summary = {
        'num_vehicles': n,
        'num_frames': len({(row['seed'], row['episode_index'], row['frame_index']) for row in rows}),
        'joint_feasibility_pct': 100.0 * sum(distinct_frames.values()) / len(distinct_frames),
        'joint_vehicle_feasibility_pct': pct('joint_feasible'),
        'stability_violation_pct': pct('stability_violation'),
        'control_violation_pct': pct('control_violation'),
        'control_magnitude_violation_pct': pct('control_magnitude_violation'),
    }

    for family in ('stability', 'control_magnitude'):
        summary[family+'_excess_mean_norm'] = float(np.mean([r[family+'_excess_mean_norm'] for r in rows]))
        summary['mean_vehicle_max_'+family+'_excess_norm'] = float(np.mean([r[family+'_excess_max_norm'] for r in rows]))

    if rows[0]['method'] == 'Ours':
        summary.update({
            'deviation_m': float(np.mean([row['deviation_m'] for row in rows])),
            'projection_applied_pct': pct('projection_applied'),
            'projection_control_delta_norm_mean': float(np.mean(
                [row['projection_control_delta_norm'] for row in rows])),
            'projection_control_delta_norm_p95': float(np.percentile(
                [row['projection_control_delta_norm'] for row in rows], 95)),
            'optimization_time_ms_mean': float(np.mean(
                [row['optimization_time_ms'] for row in rows])),
            'optimization_time_ms_p95': float(np.percentile(
                [row['optimization_time_ms'] for row in rows], 95)),
            'projection_iterations_mean': float(np.mean(
                [row['projection_iterations'] for row in rows])),
        })
    else:
        summary.update({
            'deviation_m': None,
            'projection_applied_pct': None,
            'projection_control_delta_norm_mean': None,
            'projection_control_delta_norm_p95': None,
            'optimization_time_ms_mean': None,
            'optimization_time_ms_p95': None,
            'projection_iterations_mean': None,
        })
    return summary


def paper_table(by_method: dict, unit: str = 'trajectory') -> str:
    """Generate the six-constraint table for the requested statistics unit."""
    groups = (
        ('Stability', (('Yaw rate satisfaction (%)', 'yaw_violation_pct'),
                       ('Sideslip angle satisfaction (%)', 'sideslip_violation_pct'),
                       ('Roll angle satisfaction (%)', 'roll_violation_pct'),
                       ('Lateral load transfer ratio (LTR) satisfaction (%)', 'ltr_violation_pct'))),
        ('Control', (('Acceleration magnitude satisfaction (%)', 'acceleration_violation_pct'),
                     ('Steering magnitude satisfaction (%)', 'steering_violation_pct'))),
        ('Overall feasibility', (('Individual vehicle-step feasibility (%)', 'feasibility_pct'),
                                 ('Synchronized platoon-step feasibility (%)', 'joint_feasibility_pct'))),
        ('Real-time', (('Mean optimization time (ms)', 'optimization_time_ms_mean'),
                       ('P95 optimization time (ms)', 'optimization_time_ms_p95'))),
    )
    lines = [
        r'\begin{table}[!t]',
        r'\centering',
        r'\caption{Per-constraint Satisfaction, Overall Feasibility, and Projection Latency (step statistics)}',
        r'\label{tab:projection_ablation_step}',
        r'\scriptsize',
        r'\renewcommand{\arraystretch}{1.12}',
        r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{@{}lcc@{}}',
        r'\hline',
        r'\textbf{Constraint / Metric} & \textbf{Baseline} & \textbf{JTP} \\',
        r'\hline',
    ]
    for group, metrics in groups:
        lines.append(r'\multicolumn{3}{@{}l}{\textbf{' + group + r'}} \\')
        for label, metric in metrics:
            cells = []
            for method in ('Baseline', 'Ours'):
                value = by_method[method].get(metric)
                if value is None:
                    cell = 'N/A'
                elif metric.endswith('_violation_pct'):
                    cell = f'{100.0-value:.2f}'
                else:
                    cell = f'{value:.2f}'
                if method == 'Ours':
                    cell = r'\textbf{' + cell + '}'
                cells.append(cell)
            lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines.extend([
        r'\hline',
        r'\multicolumn{3}{@{}l}{\footnotesize Feasibility includes stability and control magnitude only; timing is per vehicle.} \\',
        r'\hline',
        r'\end{tabular}',
        r'\end{table}',
        '',
    ])
    return '\n'.join(lines).replace('step statistics', unit+' statistics').replace('ablation_step', 'ablation_'+unit).replace('vehicle-step', 'vehicle-'+unit).replace('platoon-step', 'platoon-'+unit)
