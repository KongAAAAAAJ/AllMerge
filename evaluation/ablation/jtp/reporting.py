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
        'geometric_violation_pct': pct('geometric_violation'),
        'stability_violation_pct': pct('stability_violation'),
        'collision_violation_pct': pct('collision_violation'),
        'external_collision_violation_pct': pct('external_collision_violation'),
        'intertruck_collision_violation_pct': pct('intertruck_collision_violation'),
        'control_violation_pct': pct('control_violation'),
        'control_magnitude_violation_pct': pct('control_magnitude_violation'),
        'control_rate_violation_pct': pct('control_rate_violation'),
    }

    continuous_diagnostics = {
        'stability_excess_mean_norm': 'stability_excess_mean_norm',
        'mean_vehicle_max_stability_excess_norm': 'stability_excess_max_norm',
        'geometry_excess_mean_norm': 'geometry_excess_mean_norm',
        'mean_vehicle_max_geometry_excess_norm': 'geometry_excess_max_norm',
        'control_magnitude_excess_mean_norm': 'control_magnitude_excess_mean_norm',
        'mean_vehicle_max_control_magnitude_excess_norm': 'control_magnitude_excess_max_norm',
        'control_rate_excess_mean_norm': 'control_rate_excess_mean_norm',
        'mean_vehicle_max_control_rate_excess_norm': 'control_rate_excess_max_norm',
        'collision_penetration_mean_m': 'collision_penetration_mean_m',
        'mean_vehicle_max_collision_penetration_m': 'collision_penetration_max_m',
    }
    for summary_key, row_key in continuous_diagnostics.items():
        summary[summary_key] = float(np.mean([row[row_key] for row in rows]))

    if rows[0]['method'] == 'Ours':
        summary.update({
            'deviation_m': float(np.mean([row['deviation_m'] for row in rows])),
            'projection_applied_pct': pct('projection_applied'),
            'projection_control_delta_norm_mean': float(np.mean(
                [row['projection_control_delta_norm'] for row in rows])),
            'projection_control_delta_norm_p95': float(np.percentile(
                [row['projection_control_delta_norm'] for row in rows], 95)),
            'projection_latency_ms_mean': float(np.mean(
                [row['projection_latency_ms'] for row in rows])),
            'projection_latency_ms_p95': float(np.percentile(
                [row['projection_latency_ms'] for row in rows], 95)),
            'projection_iterations_mean': float(np.mean(
                [row['projection_iterations'] for row in rows])),
        })
    else:
        summary.update({
            'deviation_m': None,
            'projection_applied_pct': None,
            'projection_control_delta_norm_mean': None,
            'projection_control_delta_norm_p95': None,
            'projection_latency_ms_mean': None,
            'projection_latency_ms_p95': None,
            'projection_iterations_mean': None,
        })
    return summary


def paper_table(by_method: dict, unit: str = 'trajectory') -> str:
    """Paper table with an explicit statistical unit and step-only JTP timing."""
    caption = f'Effects of Joint Trajectory Projection on Constraint Satisfaction ({unit} statistics)'
    lines = [
        r'\begin{table}[!t]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{tab:projection_ablation_' + unit + '}',
        r'\scriptsize',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\setlength{\tabcolsep}{2.5pt}',
        r'\begin{tabular}{@{}l' + 'c' * (6 if unit == 'step' else 5) + r'@{}}',
        r'\hline',
        r'\textbf{Method} & \shortstack{\textbf{Joint}\\\textbf{Feas. (\%)}} & '
        r'\textbf{Feas. (\%)} & '
        r'\shortstack{\textbf{Geometric}\\\textbf{Viol. (\%)}} & '
        r'\shortstack{\textbf{Stability}\\\textbf{Viol. (\%)}} & '
        r'\shortstack{\textbf{Deviation}\\\textbf{(m)}}'
        + (r' & \shortstack{\textbf{Mean Time}\\\textbf{(ms)}}' if unit == 'step' else '') + r' \\',
        r'\hline',
    ]
    for key, label in (('Baseline', 'Baseline'), ('Ours', r'\textbf{Ours}')):
        value = by_method[key]
        def formatted(name, digits=2):
            v = value.get(name)
            return 'N/A' if v is None else f'{v:.{digits}f}'
        cells = [
            formatted('joint_feasibility_pct'),
            formatted('feasibility_pct'),
            formatted('geometric_violation_pct'),
            formatted('stability_violation_pct'),
            'N/A' if key == 'Baseline' else formatted('deviation_m', 3),
        ]
        if unit == 'step':
            cells.append('0.00' if key == 'Baseline' else formatted('projection_latency_ms_mean'))
        if key == 'Ours':
            cells = [r'\textbf{' + cell + '}' for cell in cells]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines.extend([r'\hline', r'\end{tabular}', r'\end{table}', ''])
    return '\n'.join(lines)
