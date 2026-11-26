"""Deterministic, independent selection of the three case-study examples."""
import numpy as np


KINDS = ('geometry', 'stability', 'control')
METRICS = {
    'geometry': ('geometric_violation', 'road_excess_mean_norm', 'geometry_excess_max_norm'),
    'stability': ('stability_violation', 'stability_excess_mean_norm', 'stability_excess_max_norm'),
    'control': ('control_violation', 'control_rate_excess_mean_norm', 'control_rate_excess_max_norm'),
}
VALUES = {'geometry': ('road', 'curvature'), 'stability': ('yaw', 'beta', 'roll', 'ltr'),
          'control': ('acceleration_rate', 'steering_rate')}


def violation_steps(kind, bad, i):
    if kind == 'geometry':
        result = bad['road'][i].copy()
        result[1:] |= bad['curvature'][i]
        return result
    return np.logical_or.reduce([bad[k][i] for k in VALUES[kind]])


def candidate_rank(kind, before, after, before_count, after_count, change):
    flag, mean_key, max_key = METRICS[kind]
    count_gain = before_count-after_count
    mean_gain = before[mean_key]-after[mean_key]
    peak_gain = before[max_key]-after[max_key]
    originally_violated = before['road_violation'] if kind == 'geometry' else (
        before['control_rate_violation'] if kind == 'control' else before[flag])
    if originally_violated and not after[flag]:
        return (3, before[max_key], before[mean_key], change), 'complete_repair'
    if originally_violated and (count_gain > 0 or mean_gain > 0):
        return (2, count_gain, mean_gain, peak_gain), 'partial_improvement'
    if not before[flag] and not after[flag]:
        return (1, change, 0., 0.), 'remains_feasible'
    return (0, count_gain, mean_gain, peak_gain), 'least_degradation'


def frame_candidates(pair, collected, meta, seed, episode, frame, cfg):
    scales = {'road': cfg['geometry']['road_clearance_m'],
              'curvature': cfg['geometry']['curvature_limit_per_m'],
              **dict(zip(('yaw', 'beta', 'roll', 'ltr'), cfg['projection']['residual_scales']['stability'])),
              'acceleration_rate': cfg['control']['acceleration_rate_max_mps3'],
              'steering_rate': cfg['control']['steering_rate_max_rad_s']}
    result = {}
    for kind in KINDS:
        candidates = []
        for i, item in enumerate(meta):
            counts = [int(violation_steps(kind, collected[m][1], i).sum()) for m in ('Baseline', 'Ours')]
            change = max((float(np.max(np.abs(collected['Baseline'][0][k][i]-collected['Ours'][0][k][i]))) / scales[k]
                          for k in VALUES[kind] if len(collected['Baseline'][0][k][i])), default=0.)
            rank, grade = candidate_rank(kind, pair['before'][i], pair['after'][i], *counts, change)
            identity = (seed, episode, frame, int(item['ego_index']))
            candidates.append({'rank': rank, 'grade': grade, 'identity': identity,
                               'exemplar_index': i, 'before_count': counts[0], 'after_count': counts[1]})
        result[kind] = max(candidates, key=candidate_key)
    return result


def candidate_key(candidate):
    return candidate['rank'] + tuple(-int(v) for v in candidate['identity'])


def save_case(path, figure):
    arrays = {k: np.asarray(v) for k, v in figure.items()
              if k not in ('road_edge_world', 'road_safety_edge_world')}
    for key in ('road_edge_world', 'road_safety_edge_world'):
        values = figure[key]
        lengths = np.asarray([len(v) for v in values], dtype=np.int32)
        packed = np.full((len(values), int(lengths.max()), 2), np.nan)
        for i, value in enumerate(values):
            packed[i, :len(value)] = value
        arrays[key] = packed
        arrays[key+'_lengths'] = lengths
    np.savez_compressed(path, **arrays)
