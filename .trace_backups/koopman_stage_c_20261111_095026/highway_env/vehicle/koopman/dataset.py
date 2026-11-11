"""Episode-level train/val splits and 10-Hz supervised 4-s windows.

This module does not import collect, TruckSim or the native DLL. It reads only
independently verified/accepted .npz artifacts. Splits happen BEFORE extracting
possibly overlapping windows to avoid temporal leakage.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


class Episode:
    def __init__(self, path, states, controls, poses, time_s, info, ltr=None):
        self.path = Path(path)
        self.states = states
        self.controls = controls
        self.poses = poses
        self.time_s = time_s
        self.info = info
        self.ltr = np.asarray(ltr, dtype=np.float32) if ltr is not None else np.full(len(states), np.nan, dtype=np.float32)

    @property
    def id(self):
        return self.path.stem


def discover_episodes(roots, project_root, require_accepted=True, expected_dt=0.1,
                      strict=False):
    """Snapshot completed files; skip staging/incomplete/invalid artifacts."""
    paths = {}
    for root in roots:
        p = Path(root)
        if not p.is_absolute(): p = Path(project_root) / p
        if not p.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {p}")
        for file in p.rglob('*.npz'):
            paths[file] = p
    episodes = []
    skipped = []
    for file in sorted(paths):
        if any(part.startswith('.') or part.lower() == 'staging' for part in file.relative_to(paths[file]).parts):
            continue
        try:
            if require_accepted:
                marker = file.with_suffix('.accepted.json')
                if not marker.is_file():
                    continue  # still collecting, or earlier unaccepted data
                accepted = json.loads(marker.read_text(encoding='utf-8'))
                if accepted.get('data_verified') is not True or accepted.get('status') != 'ok':
                    raise ValueError('Accepted marker is not verified/ok')
            with np.load(file, allow_pickle=False) as npz:
                if not bool(npz['train_eligible'].item()):
                    continue
                states = np.asarray(npz['state10_si'], dtype=np.float32)
                controls = np.asarray(npz['u_ref10_si'], dtype=np.float32)
                poses = np.asarray(npz['pose10_si'], dtype=np.float32)
                ts = np.asarray(npz['t10_s'], dtype=np.float64)
                if 'ltr10' not in npz:
                    raise ValueError('Missing TruckSim LTR label ltr10; paper K0 requires LTR supervision')
                ltr = np.asarray(npz['ltr10'], dtype=np.float32)
                info = {k: npz[k].item() for k in ('model_id','payload_kg','road_mu','vx_init_kmh') if k in npz}
                info['excitation_sha256'] = str(npz['excitation_sha256'].item()) if 'excitation_sha256' in npz else None
            if states.ndim != 2 or states.shape != (len(ts),5): raise ValueError('state shape')
            if poses.shape != (len(ts),3): raise ValueError('pose shape')
            if ltr.shape != (len(ts),): raise ValueError('ltr shape')
            if not np.isfinite(ltr).any(): raise ValueError('No valid LTR samples for supervised K0')
            if controls.shape != (len(ts)-1,2): raise ValueError('control alignment')
            if len(ts) < 2 or not np.allclose(np.diff(ts), expected_dt, rtol=0, atol=1e-5):
                raise ValueError('time alignment / dt mismatch')
            if not (np.isfinite(states).all() and np.isfinite(controls).all() and np.isfinite(poses).all()):
                raise ValueError('nonfinite states, controls, poses')
            episodes.append(Episode(file.resolve(), states, controls, poses, ts, info, ltr=ltr))
        except (OSError, KeyError, ValueError, RuntimeError, TypeError) as exc:
            skipped.append((str(file), str(exc)))
            if strict: raise ValueError(f"Rejected dataset {file}: {exc}") from exc
    if skipped:
        warnings.warn(f"Skipped {len(skipped)} corrupt/incomplete episode(s); first: {skipped[0]}")
    if not episodes:
        raise ValueError('No verified train-eligible episodes found. Check data.roots and accepted_only.')
    return episodes


def episode_split(episodes, val_fraction=.15, test_fraction=.10, seed=1234):
    if len(episodes) < 2:
        raise ValueError('Need at least TWO distinct accepted episodes to train/validate without leakage.')
    if not (0 < val_fraction < 1 and 0 <= test_fraction < 1 and val_fraction+test_fraction < 1):
        raise ValueError('Invalid train/validation/test ratios')
    order = np.random.default_rng(seed).permutation(len(episodes))
    n_val = max(1, int(round(len(order)*val_fraction)))
    n_test = int(round(len(order)*test_fraction)) if len(order) >= 3 else 0
    n_test = min(n_test, len(order)-n_val-1)
    n_val = min(n_val, len(order)-n_test-1)
    groups = {
        'train': [episodes[int(i)] for i in order[n_val+n_test:]],
        'val': [episodes[int(i)] for i in order[:n_val]],
        'test': [episodes[int(i)] for i in order[n_val:n_val+n_test]],
    }
    return groups


def compute_normalization(episodes):
    states = np.concatenate([ep.states for ep in episodes], axis=0).astype(np.float64)
    controls = np.concatenate([ep.controls for ep in episodes], axis=0).astype(np.float64)
    # Training data only. Fixed floor guards approximately constant channels.
    return {'state_mean': states.mean(axis=0).astype(np.float32).tolist(),
            'state_std': np.maximum(states.std(axis=0), 1e-4).astype(np.float32).tolist(),
            'control_mean': controls.mean(axis=0).astype(np.float32).tolist(),
            'control_std': np.maximum(controls.std(axis=0), 1e-4).astype(np.float32).tolist()}


class KoopmanWindowDataset(Dataset):
    def __init__(self, episodes, history_steps=5, horizon_steps=40, warmup_s=1.0,
                 window_stride=1, dt=0.1):
        self.episodes = list(episodes)
        self.history_steps = int(history_steps)
        self.horizon_steps = int(horizon_steps)
        if self.history_steps < 1 or self.horizon_steps < 1 or window_stride < 1:
            raise ValueError('Invalid history_steps, horizon_steps or window_stride')
        self.keys = []
        for ei, ep in enumerate(self.episodes):
            start = max(self.history_steps-1, int(math.ceil(warmup_s / dt - 1e-8)))
            latest = len(ep.states) - 1 - self.horizon_steps
            for s in range(start, latest+1, window_stride):
                self.keys.append((ei,s))

    def __len__(self): return len(self.keys)

    def __getitem__(self, i):
        ei, k = self.keys[i]
        ep = self.episodes[ei]
        h, t = self.history_steps, self.horizon_steps
        return {
            'state_hist': torch.from_numpy(ep.states[k-h+1:k+1].copy()),
            'control_hist': torch.from_numpy(ep.controls[k-h+1:k].copy()),
            'future_controls': torch.from_numpy(ep.controls[k:k+t].copy()),
            'target_states': torch.from_numpy(ep.states[k:k+t+1].copy()),
            'target_poses': torch.from_numpy(ep.poses[k:k+t+1].copy()),
            'target_ltr': torch.from_numpy(np.nan_to_num(ep.ltr[k:k+t+1], nan=0.0, posinf=0.0, neginf=0.0).copy()),
            'ltr_valid': torch.from_numpy(np.isfinite(ep.ltr[k:k+t+1]).copy()),
        }


def describe_splits(splits, history_steps, horizon_steps, warmup_s, window_stride, dt):
    result = {}
    for name, episodes in splits.items():
        ds = KoopmanWindowDataset(episodes,history_steps,horizon_steps,warmup_s,window_stride,dt)
        result[name] = {'episodes': len(episodes), 'windows': len(ds),
                        'ids': [str(ep.path) for ep in episodes]}
    return result
