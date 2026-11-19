"""Select explicit A2 advantage examples for figures, never for metric averages."""
from __future__ import annotations

import random


def sample_figure_candidates(metadata, *, count: int, seed: int):
    """Sample observed ego scenes without using any model result."""
    if count < 1:
        raise ValueError('figure candidate sample count must be positive')
    keys = sorted({tuple(int(meta[k]) for k in ('episode_index', 'frame_index', 'ego_index'))
                   for meta in metadata})
    return sorted(random.Random(seed).sample(keys, min(count, len(keys))))


def select_safety_improvement_scenes(payloads, *, count: int):
    """Rank A2/A1 safety gaps, preferring distinct episodes and unique frames.

    A trajectory unsafe for both collision and road is counted once in the
    primary score. Separate road/collision counts break ties. All counts use
    the same safety teacher and validity masks as the quantitative evaluation.
    """
    if count < 1:
        raise ValueError('figure scene count must be positive')
    common = set(payloads['A1']) & set(payloads['A2'])
    for scenes in payloads.values():
        common &= set(scenes)
    ranked = []
    for key in sorted(common):
        a1, a2 = (payloads[method][key]['safety'] for method in ('A1', 'A2'))
        ranked.append({
            'episode_index': key[0], 'frame_index': key[1], 'ego_index': key[2],
            'A1': a1, 'A2': a2,
            'unsafe_anchor_reduction': a1['unsafe_count'] - a2['unsafe_count'],
            'road_unsafe_reduction': a1['road_unsafe_count'] - a2['road_unsafe_count'],
            'collision_unsafe_reduction': a1['collision_unsafe_count'] - a2['collision_unsafe_count'],
        })
    ranked.sort(key=lambda row: (
        -row['unsafe_anchor_reduction'], -row['road_unsafe_reduction'],
        -row['collision_unsafe_reduction'], row['A2']['unsafe_count'],
        row['episode_index'], row['frame_index'], row['ego_index']))
    selected, episodes, frames = [], set(), set()
    # Do not replace a positive example with a weaker scene merely to obtain
    # another episode. Within each tier, prefer different episodes first.
    for positive_only, distinct_episodes in ((True, True), (True, False), (False, True), (False, False)):
        for row in ranked:
            if positive_only and row['unsafe_anchor_reduction'] <= 0:
                continue
            key = tuple(row[k] for k in ('episode_index', 'frame_index', 'ego_index'))
            if key[:2] in frames or (distinct_episodes and key[0] in episodes):
                continue
            if len(selected) >= count:
                break
            selected.append(key)
            episodes.add(key[0])
            frames.add(key[:2])
    selected_set = set(selected)
    for row in ranked:
        row['selected'] = tuple(row[k] for k in ('episode_index', 'frame_index', 'ego_index')) in selected_set
    return selected, ranked
