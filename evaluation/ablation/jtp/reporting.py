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
    """Generate the detailed step table or the existing trajectory-level table."""
    if unit == 'step':
        groups = (
            ('Geometry', (('Road boundary satisfaction (%)', 'road_violation_pct'),
                          ('Curvature satisfaction (%)', 'curvature_violation_pct'))),
            ('Stability', (('Yaw rate satisfaction (%)', 'yaw_violation_pct'),
                           ('Sideslip angle satisfaction (%)', 'sideslip_violation_pct'),
                           ('Roll angle satisfaction (%)', 'roll_violation_pct'),
                           ('Lateral load transfer ratio (LTR) satisfaction (%)', 'ltr_violation_pct'))),
            ('Control', (('Acceleration magnitude satisfaction (%)', 'acceleration_violation_pct'),
                         ('Steering magnitude satisfaction (%)', 'steering_violation_pct'),
                         ('Acceleration rate satisfaction (%)', 'acceleration_rate_violation_pct'),
                         ('Steering rate satisfaction (%)', 'steering_rate_violation_pct'))),
            ('Overall feasibility', (('Individual vehicle-step feasibility (%)', 'feasibility_pct'),
                                     ('Synchronized platoon-step feasibility (%)', 'joint_feasibility_pct'))),
            ('Real-time', (('Mean projection latency (ms)', 'projection_latency_ms_mean'),
                           ('P95 projection latency (ms)', 'projection_latency_ms_p95'))),
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
            r'\multicolumn{3}{@{}l}{\footnotesize Collision constraints remain included in overall feasibility; latency is measured per vehicle projection.} \\',
            r'\hline',
            r'\end{tabular}',
            r'\end{table}',
            '',
        ])
        return '\n'.join(lines)

    caption = f'Effects of Joint Trajectory Projection on Constraint Satisfaction ({unit} statistics)'
    lines = [
        r'\begin{table}[!t]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{tab:projection_ablation_' + unit + '}',
        r'\scriptsize',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\setlength{\tabcolsep}{2.5pt}',
        r'\begin{tabular}{@{}lccccc@{}}',
        r'\hline',
        r'\textbf{Method} & \shortstack{\textbf{Joint}\\\textbf{Feas. (\%)}} & '
        r'\textbf{Feas. (\%)} & '
        r'\shortstack{\textbf{Geometric}\\\textbf{Viol. (\%)}} & '
        r'\shortstack{\textbf{Stability}\\\textbf{Viol. (\%)}} & '
        r'\shortstack{\textbf{Deviation}\\\textbf{(m)}} \\',
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
        if key == 'Ours':
            cells = [r'\textbf{' + cell + '}' for cell in cells]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines.extend([r'\hline', r'\end{tabular}', r'\end{table}', ''])
    return '\n'.join(lines)
