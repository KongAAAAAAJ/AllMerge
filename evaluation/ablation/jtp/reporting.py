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
        path.write_text('', encoding='utf-8-sig')
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
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


def paired_report(rows: list[dict], *, bootstrap_repeats=2000, bootstrap_seed=2026):
    """Cluster-bootstrap physical synchronized frames, preserving seeds and trucks."""
    keys = ('seed', 'episode_index', 'frame_index', 'ego_index')
    lookup = {}
    for row in rows:
        key = (row['method'],) + tuple(row[k] for k in keys)
        if key in lookup:
            raise ValueError(f'Duplicate ablation row {key}')
        lookup[key] = row

    baseline_rows = [row for row in rows if row['method'] == 'Baseline']
    if not baseline_rows:
        raise ValueError('No Baseline observations')

    frames = defaultdict(list)
    for baseline in baseline_rows:
        key = tuple(baseline[k] for k in keys)
        paired_key = ('Ours',) + key
        if paired_key not in lookup:
            raise ValueError(f'Missing exact paired JTP observation: {key}')
        frames[(baseline['episode_index'], baseline['frame_index'])].append(
            (baseline, lookup[paired_key]))

    ours_rows = [row for row in rows if row['method'] == 'Ours']
    if len(ours_rows) != len(baseline_rows):
        raise ValueError('Unmatched/duplicated Ours observations')

    baselines = [pair[0] for group in frames.values() for pair in group]
    projected = [pair[1] for group in frames.values() for pair in group]
    by_method = {'Baseline': _summary(baselines), 'Ours': _summary(projected)}
    detailed_delta_keys = (
        'joint_feasibility_pct',
        'geometric_violation_pct',
        'stability_violation_pct',
        'collision_violation_pct',
        'control_violation_pct',
        'stability_excess_mean_norm',
        'geometry_excess_mean_norm',
        'control_magnitude_excess_mean_norm',
        'control_rate_excess_mean_norm',
        'collision_penetration_mean_m',
    )
    deltas = {
        name: (by_method['Ours'][name] - by_method['Baseline'][name])
        for name in detailed_delta_keys
    }

    rng = np.random.default_rng(bootstrap_seed)
    frame_groups = list(frames.values())

    def pair_deltas(pairs):
        frame_values = {}
        for baseline, ours in pairs:
            key = (baseline['seed'], baseline['episode_index'], baseline['frame_index'])
            value = float(baseline['joint_frame_feasible']) - float(ours['joint_frame_feasible'])
            if key in frame_values and value != frame_values[key]:
                raise ValueError(f'Conflicting frame feasibility in paired results {key}')
            frame_values[key] = value
        return {
            col: (
                float(np.mean(list(frame_values.values()))) * 100.0
                if col == 'joint_feasible'
                else float(np.mean([float(a[col]) - float(b[col]) for a, b in pairs])) * 100.0
            )
            for col in PAPER_COLUMNS
        }

    observed = pair_deltas([pair for group in frame_groups for pair in group])
    samples = {col: [] for col in PAPER_COLUMNS}
    for _ in range(int(bootstrap_repeats)):
        draw = rng.integers(0, len(frame_groups), size=len(frame_groups))
        sample = [pair for index in draw for pair in frame_groups[int(index)]]
        values = pair_deltas(sample)
        for col in PAPER_COLUMNS:
            samples[col].append(values[col])

    confidence = {
        col: {
            'baseline_minus_ours_pp': observed[col],
            'bootstrap_95ci_pp': (
                [float(x) for x in np.percentile(samples[col], [2.5, 97.5])]
                if bootstrap_repeats else None
            ),
        }
        for col in PAPER_COLUMNS
    }
    return by_method, deltas, confidence


def paper_table(by_method: dict) -> str:
    """Four-metric paper table; no solver-success column or solver watermark."""
    caption = 'Effects of Joint Trajectory Projection on Constraint Satisfaction'
    lines = [
        r'\begin{table}[!t]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{tab:projection_ablation}',
        r'\scriptsize',
        r'\renewcommand{\arraystretch}{1.15}',
        r'\setlength{\tabcolsep}{2.5pt}',
        r'\begin{tabular}{@{}lcccc@{}}',
        r'\hline',
        r'\textbf{Method} & \shortstack{\textbf{Joint}\\\textbf{Feas. (\%)}} & '
        r'\shortstack{\textbf{Geometric}\\\textbf{Viol. (\%)}} & '
        r'\shortstack{\textbf{Stability}\\\textbf{Viol. (\%)}} & '
        r'\shortstack{\textbf{Deviation}\\\textbf{(m)}} \\',
        r'\hline',
    ]
    for key, label in (('Baseline', 'Baseline'), ('Ours', r'\textbf{Ours}')):
        value = by_method[key]
        cells = [
            f"{value['joint_feasibility_pct']:.2f}",
            f"{value['geometric_violation_pct']:.2f}",
            f"{value['stability_violation_pct']:.2f}",
            'N/A' if key == 'Baseline' else f"{value['deviation_m']:.3f}",
        ]
        if key == 'Ours':
            cells = [r'\textbf{' + cell + '}' for cell in cells]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines.extend([r'\hline', r'\end{tabular}', r'\end{table}', ''])
    return '\n'.join(lines)
