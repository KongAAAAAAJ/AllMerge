"""Anchor-independent raw episode loading and reproducible episode splits."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def raw_episode_paths(root: Path, split: str):
    """Shared immutable raw inventory and episode split for A0 and evaluation."""
    if split not in ("train", "val", "test"):
        raise ValueError("Raw split must be train, val, or test")
    manifest_path = root / 'reports/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest.get('complete') or manifest.get('schema_version') != 'trace_expert_raw_k2_v2':
        raise ValueError('Expected a completed trace_expert_raw_k2_v2 dataset')
    if not np.isclose(float(manifest['trajectory']['dt_s']), 0.1):
        raise ValueError('Raw A0 controls require a 0.1 s observation grid')
    records = manifest['episode_files']
    paths = sorted((root / 'episodes').glob('episode_*.npz'))
    if len(paths) != len(records) or {p.name for p in paths} != {r['name'] for r in records}:
        raise ValueError('Raw episode inventory differs from manifest')
    if len({r['episode_index'] for r in records}) != len(records):
        raise ValueError('Duplicate raw episode IDs')
    # Match expert_dataset.split_shards: one final shard per episode, same sort,
    # RandomState shuffle, reserved splits and largest-remainder allocation.
    by_name = {r['name']: r for r in records}
    paths.sort(key=lambda p: f"shard_{int(by_name[p.name]['episode_index']):06d}.npz")
    ratios = np.asarray(manifest['split_ratios'], dtype=float)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or (ratios < 0).any() or ratios.sum() <= 0:
        raise ValueError('Invalid raw split ratios')
    ratios /= ratios.sum()
    active = [i for i, ratio in enumerate(ratios) if ratio > 0]
    counts = np.zeros(3, dtype=int)
    if len(paths) >= len(active):
        counts[active] = 1
        exact = (len(paths) - len(active)) * ratios
        extras = np.floor(exact).astype(int)
        counts += extras
        order = sorted(active, key=lambda i: exact[i] - extras[i], reverse=True)
        for i in order[:len(paths) - int(counts.sum())]:
            counts[i] += 1
    else:
        counts[active[:len(paths)]] = 1
    indices = np.arange(len(paths))
    np.random.RandomState(int(manifest['split_seed'])).shuffle(indices)
    offset = {"train": 0, "val": int(counts[0]), "test": int(counts[0] + counts[1])}[split]
    count = int(counts[("train", "val", "test").index(split)])
    selected = [paths[i] for i in indices[offset:offset+count]]
    if not selected:
        raise ValueError(f'Raw dataset has no {split} episodes')
    return selected, by_name, {
        'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'split_seed': int(manifest['split_seed']), 'split_ratios': ratios.tolist(),
        'selected_episode_files': [p.name for p in selected],
        **({'training_episode_files': [p.name for p in selected]} if split == 'train' else {}),
        'split_counts': dict(zip(('train', 'val', 'test'), counts.astype(int).tolist())),
        'split_policy': 'episode-level; matches one-episode-per-shard split_shards',
    }



class RawExpertDataset(Dataset):
    """Observed scenes and expert targets; never synthesizes A1 features or labels."""
    FEATURE_KEYS = ('ego_state', 'agent_states', 'agent_valid_mask', 'map_polylines',
                    'map_valid_mask', 'target_point', 'target_lane_polyline')
    ROAD_KEYS = ('road_lane_geometry', 'road_lane_point_mask', 'road_lane_mask',
                 'road_current_lane_slot', 'road_target_lane_slot',
                 'road_lane_connections', 'road_lane_meta')
    CONTEXT_KEYS = ('koopman_state_hist_si', 'koopman_control_hist_si', 'koopman_pose0_si',
                    'koopman_payload_kg', 'koopman_road_mu')

    def __init__(self, root, split='test', max_samples=None):
        self.shard_paths, records, self.split_manifest = raw_episode_paths(Path(root), split)
        self._index = []
        self._cached_index = None
        self._cached = None
        for file_index, path in enumerate(self.shard_paths):
            with np.load(path, allow_pickle=False) as data:
                required = (*self.FEATURE_KEYS, *self.ROAD_KEYS, *self.CONTEXT_KEYS, 'expert_trajectory_xy',
                            'future_trajectory_dense', 'episode_index', 'frame_index', 'ego_index')
                missing = [key for key in required if key not in data.files]
                if missing:
                    raise ValueError(f'{path.name}: missing raw fields {missing}')
                count = len(data['expert_trajectory_xy'])
                if count != int(records[path.name]['samples']):
                    raise ValueError(f'{path.name}: sample count differs from raw manifest')
                if not np.all(data['episode_index'] == records[path.name]['episode_index']):
                    raise ValueError(f'{path.name}: episode ID differs from raw manifest')
                for key, shape in (('expert_trajectory_xy', (count,8,2)),
                                   ('future_trajectory_dense', (count,40,2)),
                                   ('koopman_state_hist_si', (count,5,5)),
                                   ('koopman_control_hist_si', (count,4,2))):
                    if data[key].shape != shape:
                        raise ValueError(f'{path.name}: {key} must have shape {shape}')
            remaining = count if max_samples is None else max(0, min(count, max_samples-len(self._index)))
            self._index.extend((file_index, row) for row in range(remaining))
            if max_samples is not None and len(self._index) >= max_samples:
                break

    def __len__(self):
        return len(self._index)

    def __getstate__(self):
        # Windows workers should load their own episode, not receive a copy of
        # the parent's potentially large decompressed scene/road cache.
        state = self.__dict__.copy()
        state['_cached_index'] = None
        state['_cached'] = None
        return state

    def __getitem__(self, index):
        file_index, row = self._index[index]
        if self._cached_index != file_index:
            with np.load(self.shard_paths[file_index], allow_pickle=False) as data:
                names = (*self.FEATURE_KEYS, *self.ROAD_KEYS, *self.CONTEXT_KEYS, 'expert_trajectory_xy',
                         'future_trajectory_dense', 'episode_index', 'frame_index', 'ego_index',
                         'seed', 'dense_dt', 'trajectory_horizon_s')
                self._cached = {key: np.array(data[key], copy=True) for key in names if key in data.files}
            self._cached_index = file_index
        data = self._cached
        tensor = lambda key: torch.as_tensor(np.array(data[key][row], copy=True))
        features = {key: (tensor(key).bool() if key.endswith('_mask') else tensor(key).float())
                    for key in self.FEATURE_KEYS}
        for key in self.ROAD_KEYS:
            if key in data:
                value = tensor(key)
                features[key] = (value.bool() if key.endswith('_mask') or key == 'road_lane_connections'
                                 else value.long() if key.endswith('_slot') else value.float())
        targets = {'trajectory': tensor('expert_trajectory_xy').float(),
                   'trajectory_dense': tensor('future_trajectory_dense').float(),
                   **{key: tensor(key).float() for key in self.CONTEXT_KEYS}}
        for key in ('dense_dt', 'trajectory_horizon_s'):
            if key in data:
                targets[key] = tensor(key).float()
        metadata = {key: int(data[key][row]) for key in ('episode_index', 'frame_index', 'ego_index', 'seed') if key in data}
        metadata.update(shard_path=str(self.shard_paths[file_index]), shard_sample_index=row)
        return features, targets, metadata
