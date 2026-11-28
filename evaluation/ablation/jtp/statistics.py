"""Unit-specific JTP reports with physical-frame cluster bootstrap."""
import numpy as np
from .distributions import KEYS, FLAGS
from .reporting import _summary, write_csv, write_json, paper_table
from .payload import PAYLOAD_GROUPS


def step_records(collected, metadata, method, deviation):
    _, bad = collected
    records = []
    for i, base in enumerate(metadata):
        for t in range(deviation.shape[1]):
            row = {**base, 'method': method, 'prediction_step': t+1,
                   **{flag: bool(bad[key][i, t]) for key, flag in zip(KEYS, FLAGS)}}
            row['stability_violation'] = any(row[flag] for flag in FLAGS[:4])
            row['control_magnitude_violation'] = any(row[flag] for flag in FLAGS[4:])
            row['control_violation'] = row['control_magnitude_violation']
            row['joint_feasible'] = not (row['stability_violation'] or row['control_violation'])
            row['deviation_m'] = float(deviation[i,t]) if method == 'Ours' else None
            records.append(row)
    for t in range(1, deviation.shape[1]+1):
        joint = all(r['joint_feasible'] for r in records if r['prediction_step'] == t)
        for row in records:
            if row['prediction_step'] == t:
                row['joint_frame_feasible'] = joint
    return records


RATE_KEYS = ('joint_feasibility_pct', 'feasibility_pct', 'stability_violation_pct',
             'control_violation_pct', 'control_magnitude_violation_pct') + tuple(flag+'_pct' for flag in FLAGS)


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
                values = [r[k[:-4]] for r in rows if r[k[:-4]] is not None]
                result[k] = 100*float(np.mean(values)) if values else None
            if rows[0]['method'] == 'Ours':
                result['deviation_m'] = float(np.mean([r['deviation_m'] for r in rows]))
    result.update(statistics_unit=unit, num_samples=len(rows),
                  num_joint_samples=len({(r['seed'], r['episode_index'], r['frame_index'], r.get('prediction_step')) for r in rows}),
                  num_vehicle_trajectories=len({(r['seed'], r['episode_index'], r['frame_index'], r['ego_index']) for r in rows}))
    for flag in FLAGS:
        values = [r[flag] for r in rows if r[flag] is not None]
        result[flag+'_pct'] = 100*float(np.mean(values)) if values else None
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
                value = r['joint_feasible'] if k == 'feasibility_pct' else r[k[:-4]]
                if value is not None:
                    sums[m][i, j] += value
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


def add_projection_timing(summaries, frames):
    """Summarize cumulative OSQP solve time per vehicle projection."""
    values = np.asarray([
        float(optimization_time)
        for frame in frames
        for optimization_time in frame['optimization_time_ms_by_vehicle']
    ], dtype=np.float64)
    summaries['Baseline'].update(
        optimization_time_ms_mean=0.0,
        optimization_time_ms_p95=0.0,
        optimization_time_ms_max=0.0,
        num_vehicle_projections=0,
    )
    summaries['Ours'].update(
        optimization_time_ms_mean=float(values.mean()) if values.size else None,
        optimization_time_ms_p95=float(np.percentile(values, 95)) if values.size else None,
        optimization_time_ms_max=float(values.max()) if values.size else None,
        num_vehicle_projections=int(values.size),
    )


def write_reports(output, rows, unit, seeds, repeats, bootstrap_seed, frame_diagnostics):
    directory = output / unit
    directory.mkdir(parents=True, exist_ok=True)
    summaries, deltas, confidence = report(rows, unit, repeats, bootstrap_seed)
    add_projection_timing(summaries, frame_diagnostics)
    empty_columns = ['seed', 'episode_index', 'frame_index', 'ego_index', 'method',
                     'prediction_step', *FLAGS, 'stability_violation', 'control_magnitude_violation',
                     'control_violation', 'joint_feasible', 'joint_frame_feasible', 'deviation_m']
    write_csv(directory/f'jtp_per_vehicle_{unit}.csv', rows, empty_columns if not rows else None)
    write_csv(directory/f'jtp_summary_{unit}.csv', [{'method': m, **v} for m, v in summaries.items()])
    per_seed = []
    for seed in seeds:
        by_method, _, _ = report([r for r in rows if r['seed'] == seed], unit, 0, bootstrap_seed)
        add_projection_timing(by_method, [f for f in frame_diagnostics if f['seed'] == seed])
        per_seed.extend({'seed': seed, 'method': m, **v} for m, v in by_method.items())
    write_csv(directory/f'jtp_per_seed_{unit}.csv', per_seed)
    write_json(directory/f'jtp_paired_statistics_{unit}.json', {'statistics_unit': unit,
               'difference_ours_minus_baseline_pp': deltas, 'paired_cluster_bootstrap': confidence,
               'resampling_unit': 'physical episode/frame, preserving seeds, trucks and steps'})
    by_payload, paired_payload = {}, {}
    for group in PAYLOAD_GROUPS:
        group_summary, group_delta, group_confidence = report(
            [r for r in rows if r['payload_group'] == group], unit, repeats, bootstrap_seed)
        add_projection_timing(group_summary, [f for f in frame_diagnostics if f['payload_group'] == group])
        by_payload[group] = group_summary
        paired_payload[group] = dict(difference_ours_minus_baseline_pp=group_delta,
                                     paired_cluster_bootstrap=group_confidence)
    write_csv(directory/f'jtp_summary_{unit}_payload.csv',
              [dict(payload_group=g, method=m, **v) for g, methods in by_payload.items() for m, v in methods.items()])
    write_json(directory/f'jtp_paired_statistics_{unit}_payload.json',
               dict(statistics_unit=unit, resampling_unit='physical episode/frame, preserving seeds, trucks and steps',
                    by_payload=paired_payload))
    (directory/f'projection_ablation_table_{unit}.tex').write_text(paper_table(by_payload, unit), encoding='utf-8')
    result = {'summary': summaries, 'paired_deltas': deltas, 'by_payload': by_payload,
            'prediction_steps': '1..H',
            'denominator': 'synchronized platoon prediction steps for Joint; valid vehicle prediction steps per constraint otherwise' if unit == 'step' else 'synchronized platoon trajectories for Joint; vehicle trajectories otherwise'}
    result['timing'] = {
        'unit': 'ms_per_vehicle_projection',
        'num_vehicle_projections': sum(
            len(frame['optimization_time_ms_by_vehicle']) for frame in frame_diagnostics),
        'num_synchronized_frames': len(frame_diagnostics),
        'definition': (
            'Mean and P95 of cumulative OSQP solve time per vehicle, including all '
            'iterations and failed or timed-out QPs. Excludes '
            'linearization, QP construction, data transfer, backtracking and diagnostics. '
            'Baseline adds no optimization time; each QP has its own time limit.'
        ),
        'per_seed': {
            str(seed): {
                r['method']: r['optimization_time_ms_mean']
                for r in per_seed if r['seed'] == seed
            } for seed in seeds
        },
        'per_seed_p95': {
            str(seed): {
                r['method']: r.get('optimization_time_ms_p95')
                for r in per_seed if r['seed'] == seed
            } for seed in seeds
        },
    }
    return result
