"""Deterministic, independent selection of stability and control case-study examples."""
import numpy as np


STABILITY_KEYS = ('yaw', 'beta', 'roll', 'ltr')
CONTROL_KEYS = ('acceleration', 'steering')


def control_candidate_rank(before, after, before_count, after_count, change):
    mean_key, max_key = 'control_magnitude_excess_mean_norm', 'control_magnitude_excess_max_norm'
    count_gain = before_count-after_count
    mean_gain = before[mean_key]-after[mean_key]
    peak_gain = before[max_key]-after[max_key]
    if before['control_violation'] and not after['control_violation']:
        return (3, before[max_key], before[mean_key], change), 'complete_repair'
    if before['control_violation'] and (count_gain > 0 or mean_gain > 0):
        return (2, count_gain, mean_gain, peak_gain), 'partial_improvement'
    if not before['control_violation'] and not after['control_violation']:
        return (1, change, 0., 0.), 'remains_feasible'
    return (0, count_gain, mean_gain, peak_gain), 'least_degradation'


def stability_candidate_rank(before, after, before_bound, after_bound, before_count, after_count, scale):
    before, after = np.abs(before), np.abs(after)
    excess_before = np.maximum(before-before_bound, 0.)/scale
    excess_after = np.maximum(after-after_bound, 0.)/scale
    count_gain = before_count-after_count
    mean_gain = float(excess_before.mean()-excess_after.mean())
    peak_gain = float(excess_before.max()-excess_after.max())
    change = float(np.max(np.abs(before-after)))/scale
    if before_count and not after_count:
        return (3, float(excess_before.max()), float(excess_before.mean()), change, 0.), 'complete_repair'
    if before_count and (count_gain > 0 or mean_gain > 0):
        return (2, count_gain, mean_gain, peak_gain, 0.), 'partial_improvement'
    if not before_count and not after_count:
        response_peak_gain = float(before.max()-after.max())/scale
        response_mean_gain = float(before.mean()-after.mean())/scale
        if response_peak_gain > 0 or (response_peak_gain == 0 and response_mean_gain > 0):
            return (1, 1., response_peak_gain, response_mean_gain, -change), 'remains_feasible'
        return (1, 0., -change, 0., 0.), 'remains_feasible'
    return (0, count_gain, mean_gain, peak_gain, -change), 'least_degradation'


def frame_candidates(pair, collected, meta, seed, episode, frame, cfg, bounds):
    scales = dict(zip(STABILITY_KEYS, cfg['projection']['residual_scales']['stability']))
    scales.update(acceleration=cfg['control']['acceleration_max_mps2']-cfg['control']['acceleration_min_mps2'],
                  steering=cfg['control']['steering_max_rad']-cfg['control']['steering_min_rad'])
    result = {}
    for key in (*STABILITY_KEYS, 'control'):
        candidates = []
        for i, item in enumerate(meta):
            keys = CONTROL_KEYS if key == 'control' else (key,)
            counts = [int(np.logical_or.reduce([collected[m][1][k][i] for k in keys]).sum())
                      for m in ('Baseline', 'Ours')]
            if key == 'control':
                change = max(float(np.max(np.abs(collected['Baseline'][0][k][i]-collected['Ours'][0][k][i]))) / scales[k]
                             for k in CONTROL_KEYS)
                rank, grade = control_candidate_rank(pair['before'][i], pair['after'][i], *counts, change)
            else:
                channel = STABILITY_KEYS.index(key)
                rank, grade = stability_candidate_rank(
                    collected['Baseline'][0][key][i], collected['Ours'][0][key][i],
                    bounds['Baseline'][i,:,channel], bounds['Ours'][i,:,channel], *counts, scales[key])
            identity = (seed, episode, frame, int(item['ego_index']))
            candidates.append({'rank': rank, 'grade': grade, 'identity': identity,
                               'exemplar_index': i, 'before_count': counts[0], 'after_count': counts[1]})
        result[key] = max(candidates, key=candidate_key)
    return result


def candidate_key(candidate):
    return candidate['rank'] + tuple(-int(v) for v in candidate['identity'])


def save_case(path, figure):
    np.savez_compressed(path, **{key: np.asarray(value) for key,value in figure.items()})
