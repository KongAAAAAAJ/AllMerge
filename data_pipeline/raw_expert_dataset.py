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

    CONTROLLED_EGO_INDICES = (0, 1, 2)

    def __init__(self, root, split='test', max_samples=None, include_platoon_reference=False):
        self.shard_paths, records, self.split_manifest = raw_episode_paths(Path(root), split)
        self.include_platoon_reference = bool(include_platoon_reference)
        self._index = []
        self._frame_rows = []
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
                frame_rows = {}
                if self.include_platoon_reference:
                    for row, (frame, ego) in enumerate(zip(data['frame_index'], data['ego_index'])):
                        frame = int(frame)
                        ego = int(ego)
                        if ego not in self.CONTROLLED_EGO_INDICES:
                            raise ValueError(f'{path.name}: unexpected controlled ego_index={ego}')
                        group = frame_rows.setdefault(frame, {})
                        if ego in group:
                            raise ValueError(f'{path.name}: duplicate ego={ego} at frame={frame}')
                        group[ego] = row
                    expected = set(self.CONTROLLED_EGO_INDICES)
                    incomplete = {frame: sorted(set(rows)) for frame, rows in frame_rows.items()
                                  if set(rows) != expected}
                    if incomplete:
                        raise ValueError(f'{path.name}: incomplete synchronized platoon frames: '
                                         f'{dict(list(incomplete.items())[:5])}')
            self._frame_rows.append(frame_rows)
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

    @staticmethod
    def _partner_poses_in_ego_frame(ego_pose, partner_pose, partner_future):
        ego_pose = np.asarray(ego_pose, dtype=np.float64)
        partner_pose = np.asarray(partner_pose, dtype=np.float64)
        partner_future = np.asarray(partner_future, dtype=np.float64)
        if ego_pose.shape != (3,) or partner_pose.shape != (3,) or partner_future.shape != (40, 2):
            raise ValueError('Platoon reference pose/trajectory shapes are invalid')
        if not (np.isfinite(ego_pose).all() and np.isfinite(partner_pose).all()
                and np.isfinite(partner_future).all()):
            raise ValueError('Platoon reference contains NaN/Inf')

        ce, se = np.cos(ego_pose[2]), np.sin(ego_pose[2])
        cp, sp = np.cos(partner_pose[2]), np.sin(partner_pose[2])
        partner_world = partner_future @ np.asarray([[cp, sp], [-sp, cp]]) + partner_pose[:2]
        world_delta = partner_world - ego_pose[:2]
        future_ego = world_delta @ np.asarray([[ce, -se], [se, ce]])
        initial_delta = partner_pose[:2] - ego_pose[:2]
        initial_ego = initial_delta @ np.asarray([[ce, -se], [se, ce]])
        full_xy = np.concatenate((initial_ego[None], future_ego), axis=0)
        delta = np.diff(full_xy, axis=0)
        initial_heading = np.arctan2(np.sin(partner_pose[2] - ego_pose[2]),
                                     np.cos(partner_pose[2] - ego_pose[2]))
        future_heading = []
        heading = initial_heading
        for displacement in delta:
            if np.linalg.norm(displacement) > 1e-3:
                heading = float(np.arctan2(displacement[1], displacement[0]))
            future_heading.append(heading)
        heading = np.concatenate(([initial_heading], np.asarray(future_heading)))
        return np.concatenate((full_xy, heading[:, None]), axis=-1).astype(np.float32)

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
        if self.include_platoon_reference:
            frame = int(data['frame_index'][row])
            ego_index = int(data['ego_index'][row])
            ego_pose = np.asarray(data['koopman_pose0_si'][row], dtype=np.float32)
            partner_indices = [value for value in self.CONTROLLED_EGO_INDICES if value != ego_index]
            partner_rows = [self._frame_rows[file_index][frame][value] for value in partner_indices]
            partner_poses = [
                self._partner_poses_in_ego_frame(
                    ego_pose,
                    data['koopman_pose0_si'][partner_row],
                    data['future_trajectory_dense'][partner_row],
                )
                for partner_row in partner_rows
            ]
            targets['platoon_reference_poses'] = torch.as_tensor(
                np.stack(partner_poses, axis=0), dtype=torch.float32
            )
        for key in ('dense_dt', 'trajectory_horizon_s'):
            if key in data:
                targets[key] = tensor(key).float()
        metadata = {key: int(data[key][row]) for key in ('episode_index', 'frame_index', 'ego_index', 'seed') if key in data}
        metadata.update(shard_path=str(self.shard_paths[file_index]), shard_sample_index=row)
        return features, targets, metadata
