"""Unit-specific JTP reports with physical-frame cluster bootstrap."""
import numpy as np
from .distributions import KEYS, FLAGS
from .reporting import _summary, write_csv, write_json, paper_table


def step_records(collected, metadata, method, deviation):
    """Align all twelve constraints at interval-ending steps 2..H."""
    _, bad = collected
    h = deviation.shape[1]
    records = []
    for i, base in enumerate(metadata):
        for t in range(2, h+1):
            row = {**base, 'method': method, 'prediction_step': t}
            for k, flag in zip(KEYS, FLAGS):
                index = t-2 if k in ('curvature', 'acceleration_rate', 'steering_rate') else t-1
                row[flag] = bool(bad[k][i][index]) if len(bad[k][i]) else False
            for name, components in (
                    ('stability', ('yaw', 'sideslip', 'roll', 'ltr')),
                    ('geometric', ('road', 'curvature')),
                    ('collision', ('external_collision', 'intertruck_collision')),
                    ('control_magnitude', ('acceleration', 'steering')),
                    ('control_rate', ('acceleration_rate', 'steering_rate'))):
                row[name+'_violation'] = any(row[c+'_violation'] for c in components)
            row['control_violation'] = row['control_magnitude_violation'] or row['control_rate_violation']
            row['joint_feasible'] = not any(row[k+'_violation'] for k in ('stability', 'geometric', 'collision', 'control'))
            row['deviation_m'] = float(deviation[i, t-1]) if method == 'Ours' else None
            records.append(row)
    for t in range(2, h+1):
        group = [r for r in records if r['prediction_step'] == t]
        feasible = all(r['joint_feasible'] for r in group)
        for row in group:
            row['joint_frame_feasible'] = feasible
    return records


RATE_KEYS = ('joint_feasibility_pct', 'feasibility_pct', 'geometric_violation_pct',
             'stability_violation_pct', 'collision_violation_pct', 'control_violation_pct',
             'control_rate_violation_pct', 'control_magnitude_violation_pct',
             'external_collision_violation_pct', 'intertruck_collision_violation_pct')


def summary(rows, unit):
    if unit == 'trajectory' and rows:
        result = _summary(rows)
        result['feasibility_pct'] = result['joint_vehicle_feasibility_pct']
    else:
        joint = {(r['seed'], r['episode_index'], r['frame_index'], r['prediction_step']): r['joint_frame_feasible'] for r in rows}
        result = {k: None for k in RATE_KEYS}
        result['deviation_m'] = None
        if rows:
            result['joint_feasibility_pct'] = 100*float(np.mean(list(joint.values())))
            result['feasibility_pct'] = 100*float(np.mean([r['joint_feasible'] for r in rows]))
            for k in RATE_KEYS[2:]:
                result[k] = 100*float(np.mean([r[k.replace('_pct', '')] for r in rows]))
            if rows[0]['method'] == 'Ours':
                result['deviation_m'] = float(np.mean([r['deviation_m'] for r in rows]))
    result.update(statistics_unit=unit, num_samples=len(rows),
                  num_joint_samples=len({(r['seed'], r['episode_index'], r['frame_index'], r.get('prediction_step')) for r in rows}),
                  num_vehicle_trajectories=len({(r['seed'], r['episode_index'], r['frame_index'], r['ego_index']) for r in rows}))
    for flag in FLAGS:
        result[flag+'_pct'] = 100*float(np.mean([r[flag] for r in rows])) if rows else None
    return result


def report(rows, unit, repeats=2000, seed=2026):
    methods = ('Baseline', 'Ours')
    keys = ('seed', 'episode_index', 'frame_index', 'ego_index') + (('prediction_step',) if unit == 'step' else ())
    lookup = {}
    for m in methods:
        items = [r for r in rows if r['method'] == m]
        lookup[m] = {tuple(r[k] for k in keys): r for r in items}
        if len(lookup[m]) != len(items):
            raise ValueError('Duplicate statistical observations')
    if lookup['Baseline'].keys() != lookup['Ours'].keys():
        raise ValueError('Unpaired statistical observations')
    by_method = {m: summary(list(lookup[m].values()), unit) for m in methods}
    delta_keys = (*RATE_KEYS, 'deviation_m')
    deltas = {k: (by_method['Ours'].get(k)-by_method['Baseline'].get(k))
              if by_method['Ours'].get(k) is not None and by_method['Baseline'].get(k) is not None else None for k in delta_keys}
    # Preaggregate each physical frame; resampling preserves repeated clusters,
    # all seeds, vehicles, and steps without expanding millions of row objects.
    clusters = sorted({(r['episode_index'], r['frame_index']) for r in rows})
    positions = {key: i for i, key in enumerate(clusters)}
    sums = {m: np.zeros((len(clusters), len(RATE_KEYS))) for m in methods}
    counts = {m: np.zeros_like(sums[m]) for m in methods}
    for m in methods:
        seen_joint = set()
        joint_values = {}
        for r in lookup[m].values():
            i = positions[(r['episode_index'], r['frame_index'])]
            joint_key = (r['seed'], r['episode_index'], r['frame_index'], r.get('prediction_step'))
            if joint_key in joint_values and joint_values[joint_key] != r['joint_frame_feasible']:
                raise ValueError('Inconsistent synchronized feasibility')
            joint_values[joint_key] = r['joint_frame_feasible']
            if joint_key not in seen_joint:
                sums[m][i, 0] += r['joint_frame_feasible']; counts[m][i, 0] += 1
                seen_joint.add(joint_key)
            for j, k in enumerate(RATE_KEYS[1:], 1):
                sums[m][i, j] += r['joint_feasible'] if k == 'feasibility_pct' else r[k.replace('_pct', '')]
                counts[m][i, j] += 1
    samples = []
    rng = np.random.default_rng(seed)
    for _ in range(repeats if clusters else 0):
        indices = rng.integers(0, len(clusters), len(clusters))
        rates = {m: 100*sums[m][indices].sum(0)/counts[m][indices].sum(0) for m in methods}
        samples.append(rates['Ours']-rates['Baseline'])
    confidence = {k: {'ours_minus_baseline_pp': deltas[k],
                      'bootstrap_95ci_pp': np.percentile(np.asarray(samples)[:, j], [2.5, 97.5]).tolist() if samples else None}
                  for j, k in enumerate(RATE_KEYS)}
    return by_method, deltas, confidence


def write_reports(output, rows, unit, seeds, repeats, bootstrap_seed):
    directory = output / unit
    directory.mkdir(parents=True, exist_ok=True)
    summaries, deltas, confidence = report(rows, unit, repeats, bootstrap_seed)
    empty_columns = ['seed', 'episode_index', 'frame_index', 'ego_index', 'method',
                     'prediction_step', *FLAGS, 'stability_violation', 'geometric_violation',
                     'collision_violation', 'control_magnitude_violation', 'control_rate_violation',
                     'control_violation', 'joint_feasible', 'joint_frame_feasible', 'deviation_m']
    write_csv(directory/f'jtp_per_vehicle_{unit}.csv', rows, empty_columns if not rows else None)
    write_csv(directory/f'jtp_summary_{unit}.csv', [{'method': m, **v} for m, v in summaries.items()])
    per_seed = []
    for seed in seeds:
        by_method, _, _ = report([r for r in rows if r['seed'] == seed], unit, 0, bootstrap_seed)
        per_seed.extend({'seed': seed, 'method': m, **v} for m, v in by_method.items())
    write_csv(directory/f'jtp_per_seed_{unit}.csv', per_seed)
    write_json(directory/f'jtp_paired_statistics_{unit}.json', {'statistics_unit': unit,
               'difference_ours_minus_baseline_pp': deltas, 'paired_cluster_bootstrap': confidence,
               'resampling_unit': 'physical episode/frame, preserving seeds, trucks and steps'})
    (directory/f'projection_ablation_table_{unit}.tex').write_text(paper_table(summaries, unit), encoding='utf-8')
    return {'summary': summaries, 'paired_deltas': deltas,
            'prediction_steps': '2..H' if unit == 'step' else '1..H; control rates 2..H',
            'denominator': 'synchronized platoon steps for Joint; vehicle steps otherwise' if unit == 'step' else 'synchronized platoon trajectories for Joint; vehicle trajectories otherwise'}
