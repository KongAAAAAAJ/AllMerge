"""Stratified payload-group sampling and reproducible figure example selection."""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset

def parse_payloads(value):
    result = [float(v.strip()) for v in value.split(',')] if isinstance(value, str) else [float(v) for v in value]
    if len(result) != 3 or any(not math.isfinite(x) or x < 0 for x in result) or len(set(result)) != 3:
        raise ValueError('Exactly three distinct nonnegative payloads required, e.g. 0,3500,7000')
    return result


class IndexedSubset(Dataset):
    """Preserve NPZ shard provenance and indices while stratifying the split."""
    def __init__(self, source, indices):
        self.source = source
        self.source_indices = [int(i) for i in indices]
        self._index = [source._index[i] for i in self.source_indices]
        self.shard_paths = source.shard_paths

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self, i):
        return self.source[self.source_indices[i]]


def select_payload_samples(ds, payloads, *, max_samples=0, sample_seed=2026, tolerance_kg=1.0):
    """Scan all split shards for actual payload values; select without replacement.

    max_samples=0 selects all matching samples. Otherwise sample as evenly as
    possible across the three payloads. Do not take the dataset's first N items:
    they may belong to only one loading condition.
    """
    if tolerance_kg < 0 or not math.isfinite(tolerance_kg):
        raise ValueError('Invalid payload tolerance')
    if max_samples < 0 or 0 < max_samples < 3:
        raise ValueError('--num-samples must be 0 (all) or >=3 for three payloads')
    if any(abs(a-b) <= 2*tolerance_kg for i,a in enumerate(payloads) for b in payloads[i+1:]):
        raise ValueError('Payload groups overlap at the requested tolerance')
    by_payload = {float(x): [] for x in payloads}
    # Dataset index is shard-major; avoid expensive __getitem__ and per-sample
    # decompression solely for payload discovery.
    by_shard = defaultdict(list)
    for global_idx, (shard_idx, row_idx) in enumerate(ds._index):
        by_shard[shard_idx].append((global_idx, row_idx))
    for shard_idx, entries in sorted(by_shard.items()):
        with np.load(ds.shard_paths[shard_idx], allow_pickle=False) as shard:
            if 'koopman_payload_kg' not in shard.files:
                raise KeyError(f'Missing koopman_payload_kg in {ds.shard_paths[shard_idx]}')
            values = np.asarray(shard['koopman_payload_kg']).reshape(-1)
        for global_idx, row_idx in entries:
            value = float(values[row_idx])
            matches = [p for p in payloads if abs(value-p) <= tolerance_kg]
            if len(matches) == 1:
                by_payload[matches[0]].append(global_idx)
    missing = [p for p in payloads if not by_payload[p]]
    if missing:
        raise ValueError(f'No samples for payload(s) {missing}; available groups: '
                         + str({k:len(v) for k,v in by_payload.items()}))
    rng = np.random.default_rng(int(sample_seed))
    if max_samples == 0:
        chosen = {p: list(by_payload[p]) for p in payloads}
    else:
        # Round-robin quota allocation, compensating for short groups.
        quota = {p:0 for p in payloads}
        available = sum(len(v) for v in by_payload.values())
        needed = min(max_samples, available)
        for _ in range(needed):
            eligible = [p for p in payloads if quota[p] < len(by_payload[p])]
            p = min(eligible, key=lambda v: (quota[v], payloads.index(v)))
            quota[p] += 1
        chosen = {p: rng.choice(by_payload[p], size=quota[p], replace=False).tolist()
                  for p in payloads}
    selected = sorted(i for ids in chosen.values() for i in ids)
    offset = {source_idx: local_idx for local_idx, source_idx in enumerate(selected)}
    groups = {p: sorted(offset[i] for i in chosen[p]) for p in payloads}
    return IndexedSubset(ds, selected), groups, {p: len(by_payload[p]) for p in payloads}


def select_figure_samples(groups, payloads, *, figure_seed=2026):
    rng = np.random.default_rng(int(figure_seed))
    return {p: int(rng.choice(groups[p])) for p in payloads}
