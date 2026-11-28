"""Deterministic, independent selection of stability and control case-study examples."""
import numpy as np


KINDS = ('stability', 'control')
VALUES = {'stability': ('yaw', 'beta', 'roll', 'ltr'),
          'control': ('acceleration', 'steering')}


def violation_steps(kind, bad, i):
    return np.logical_or.reduce([bad[k][i] for k in VALUES[kind]])


def candidate_rank(kind, before, after, before_count, after_count, change):
    flag = kind + '_violation'
    prefix = 'stability' if kind == 'stability' else 'control_magnitude'
    mean_key, max_key = prefix+'_excess_mean_norm', prefix+'_excess_max_norm'
    count_gain = before_count-after_count
    mean_gain = before[mean_key]-after[mean_key]
    peak_gain = before[max_key]-after[max_key]
    if before[flag] and not after[flag]:
        return (3, before[max_key], before[mean_key], change), 'complete_repair'
    if before[flag] and (count_gain > 0 or mean_gain > 0):
        return (2, count_gain, mean_gain, peak_gain), 'partial_improvement'
    if not before[flag] and not after[flag]:
        return (1, change, 0., 0.), 'remains_feasible'
    return (0, count_gain, mean_gain, peak_gain), 'least_degradation'


def frame_candidates(pair, collected, meta, seed, episode, frame, cfg):
    scales = dict(zip(('yaw', 'beta', 'roll', 'ltr'), cfg['projection']['residual_scales']['stability']))
    scales.update(acceleration=cfg['control']['acceleration_max_mps2']-cfg['control']['acceleration_min_mps2'],
                  steering=cfg['control']['steering_max_rad']-cfg['control']['steering_min_rad'])
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
    np.savez_compressed(path, **{key: np.asarray(value) for key,value in figure.items()})
