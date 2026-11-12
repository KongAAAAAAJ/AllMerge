"""Multi-seed statistics with properly pooled distance-normalized fuel."""
from __future__ import annotations
import csv
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
from .validation import REQUIRED_LABELS

METRICS = ('fuel_l_per_100km', 'fuel_l_per_4s', 'progress_m',
           'pvr_pct', 'ade_m', 'fde_m')

def parse_multi_checkpoint_specs(specs: list[str]):
    """Parse F0=path or F0@TRAIN_SEED=path; insist on matched training seeds."""
    result = {}
    for spec in specs:
        name, sep, value = spec.partition('=')
        if not sep or not value:
            raise ValueError(f'Expected --checkpoint F0[@train_seed]=PATH: {spec}')
        label, at, train_seed = name.partition('@')
        label = label.upper()
        if label not in REQUIRED_LABELS or (at and not train_seed.strip()):
            raise ValueError(f'Invalid method or train_seed: {name}')
        train_seed = train_seed if at else 'single'
        key = (label, train_seed)
        if key in result:
            raise ValueError(f'Duplicate checkpoint {key}')
        result[key] = Path(value)
    by_method = defaultdict(set)
    for method, tr_seed in result:
        by_method[method].add(tr_seed)
    if 'F0' not in by_method:
        raise ValueError('Must provide F0 baseline checkpoint')
    baseline_set = by_method['F0']
    for method, ids in by_method.items():
        if ids != baseline_set:
            raise ValueError(f'Unpaired training seeds: {method}={sorted(ids)}, F0={sorted(baseline_set)}')
    if len(baseline_set) > 1 and 'single' in baseline_set:
        raise ValueError('Use explicit train-seed IDs (F0@0 etc.) for repeated trainings')
    return result


def parse_seeds(value: str | list[int] | tuple[int, ...]):
    if isinstance(value, str):
        result = [int(s.strip()) for s in value.split(',') if s.strip()]
    else:
        result = [int(s) for s in value]
    if not result or any(s < 0 for s in result) or len(set(result)) != len(result):
        raise ValueError('Inference seeds must be unique nonnegative integers')
    return result


def summarize_records(rows):
    if not rows:
        raise ValueError('Empty matched evaluation sample list')
    total_fuel = math.fsum(r['fuel_l'] for r in rows)
    total_distance = math.fsum(r['distance_m'] for r in rows)
    if total_distance <= 1:
        raise ValueError('Insufficient executed distance for intensity metric')
    avg = lambda key: math.fsum(r[key] for r in rows) / len(rows)
    return {
        'n_samples': len(rows),
        'fuel_l_per_100km': 100000.0 * total_fuel / total_distance,
        'fuel_l_per_4s': avg('fuel_l'),
        'progress_m': avg('progress_m'),
        'pvr_pct': 100.0 * avg('pvr'),
        'ade_m': avg('ade_m'), 'fde_m': avg('fde_m'),
        'goal_terminal_error_m': avg('goal_terminal_error_m'),
        'speed_mae_proxy_kmh': avg('speed_mae_proxy_kmh'),
        'rms_longitudinal_jerk_m_s3': avg('rms_longitudinal_jerk_m_s3'),
        'total_fuel_l': total_fuel, 'total_executed_distance_m': total_distance,
    }


def _avg_sd(vals):
    x = np.asarray(vals, dtype=np.float64)
    if not np.isfinite(x).all() or x.size < 1:
        raise ValueError('Invalid or empty values for statistics')
    return float(np.mean(x)), float(np.std(x, ddof=1)) if x.size > 1 else 0.0


def summarize_multiseed(by_method_train_inference):
    """Each per-inference statistic is first pooled over identical dataset samples.

    If multiple independently trained checkpoints exist, the headline SD is
    across the training-seed averages; otherwise it is across inference seeds.
    This avoids treating 1000 correlated samples as 1000 independent runs.
    """
    result = {}
    details = []
    for method in sorted(by_method_train_inference, key=lambda m: REQUIRED_LABELS.index(m)):
        train_groups = by_method_train_inference[method]
        training_mean_stats = defaultdict(list)
        within_training_std = defaultdict(list)
        for train_seed, inference_groups in sorted(train_groups.items()):
            run_stats = []
            for inference_seed, samples in sorted(inference_groups.items()):
                stats = summarize_records(samples)
                details.append(dict(method=method, train_seed=train_seed,
                                    inference_seed=inference_seed, **stats))
                run_stats.append(stats)
            for metric in METRICS:
                m, sd = _avg_sd([s[metric] for s in run_stats])
                training_mean_stats[metric].append(m)
                within_training_std[metric].append(sd)
        n_train = len(train_groups)
        n_inference = len(next(iter(train_groups.values())))
        row = dict(method=method, n_train_seeds=n_train, n_inference_seeds=n_inference,
                   n_samples=len(next(iter(next(iter(train_groups.values())).values()))),
                   std_basis='training_seeds' if n_train > 1 else 'inference_seeds')
        for metric in METRICS:
            mean, std_train = _avg_sd(training_mean_stats[metric])
            std_infer = float(np.mean(within_training_std[metric]))
            row[metric] = mean
            row[metric + '_mean'] = mean
            row[metric + '_std'] = std_train if n_train > 1 else std_infer
            row[metric + '_std_training'] = std_train if n_train > 1 else 0.0
            row[metric + '_std_inference'] = std_infer
        result[method] = row
    return result, details


def summarize_per_sample_repeats(rows):
    """A seed-aware per-sample summary; avoid pseudo-independent sample replicates."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row['method'], row['sample_index'], row['shard_name'],
                 row['shard_sample_index'])][row['train_seed']].append(row)
    result = []
    fields = ('fuel_l', 'fuel_l_per_100km', 'distance_m', 'progress_m',
              'pvr', 'ade_m', 'fde_m', 'goal_terminal_error_m',
              'speed_mae_proxy_kmh', 'rms_longitudinal_jerk_m_s3')
    for (method, sample_index, shard_name, shard_idx), trainings in sorted(grouped.items()):
        record = dict(method=method, sample_index=sample_index, shard_name=shard_name,
                      shard_sample_index=shard_idx, n_train_seeds=len(trainings),
                      n_evaluation_runs=sum(len(x) for x in trainings.values()),
                      std_basis='training_seeds' if len(trainings)>1 else 'inference_seeds')
        for key in fields:
            train_means = [np.mean([r[key] for r in reps]) for reps in trainings.values()]
            mean, train_std = _avg_sd(train_means)
            infer_stds = [_avg_sd([r[key] for r in reps])[1] for reps in trainings.values()]
            record[key+'_mean'] = mean
            record[key+'_std'] = train_std if len(trainings)>1 else float(np.mean(infer_stds))
        result.append(record)
    return result


def _write_csv(path, rows):
    if not rows:
        raise ValueError('No CSV rows')
    fields = list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
