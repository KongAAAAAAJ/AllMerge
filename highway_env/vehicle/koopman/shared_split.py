"""Shared, immutable episode-level partition for all six Koopman candidates.

The split is generated from accepted files only. Once written, new completed
TruckSim episodes do not silently alter any train/validation/test population.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .dataset import discover_episodes, episode_split, compute_normalization, describe_splits


def resolve(project_root, name):
    p = Path(name).expanduser()
    return p if p.is_absolute() else Path(project_root) / p


def discover(cfg, project_root, data_root=None):
    dc = cfg['data']
    roots = [data_root] if data_root else dc['roots']
    return discover_episodes(roots, project_root,
        require_accepted=bool(dc.get('accepted_only', True)),
        expected_dt=float(dc.get('dt_s', 0.1)), strict=bool(dc.get('strict', False)))


def _file_id(ep):
    return str(ep.path.resolve())


def create(cfg, project_root, data_root=None, force=False):
    dest = resolve(project_root, cfg['split']['path'])
    if dest.exists() and not force:
        raise FileExistsError(f'Split already exists: {dest}; use it unchanged or choose a new split path')
    episodes = discover(cfg, project_root, data_root)
    dc = cfg['data']
    splits = episode_split(episodes, float(dc.get('val_fraction', 0.15)),
                           float(dc.get('test_fraction', 0.1)), int(cfg['training'].get('seed', 2026)))
    if not splits['train'] or not splits['val']:
        raise ValueError('A shared split requires at least one train and one validation episode')
    norm = compute_normalization(splits['train'])
    cs = np.asarray([[float(e.info['payload_kg']), float(e.info['road_mu'])] for e in splits['train']], dtype=np.float64)
    condition_norm = {'condition_mean':cs.mean(axis=0).astype(np.float32).tolist(),
                      'condition_std':np.maximum(cs.std(axis=0), [1000.0, 0.1]).astype(np.float32).tolist()}
    items = {k:[_file_id(ep) for ep in group] for k,group in splits.items()}
    ids = [path for group in items.values() for path in group]
    if len(set(ids))!=len(ids):raise ValueError('Shared split contains duplicate paths')
    signature = hashlib.sha256(json.dumps(items, sort_keys=True).encode('utf-8')).hexdigest()
    report = describe_splits(splits, int(dc.get('history_steps',5)), int(dc.get('horizon_steps',40)),
                            float(dc.get('warmup_s',1.)), int(dc.get('train_stride',2)),float(dc.get('dt_s',.1)))
    payload = {'version':1,'sha256':signature,'splits':items,'normalization':norm,
               'condition_normalization':condition_norm,'report':report,
               'data':{k:dc[k] for k in ('dt_s','history_steps','horizon_steps','warmup_s','train_stride','eval_stride') if k in dc}}
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.writing')
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(dest)
    print(f'[SPLIT] frozen {signature[:12]} train/val/test={len(splits["train"])}/{len(splits["val"])}/{len(splits["test"])} -> {dest}', flush=True)
    return payload


def load(cfg, project_root, data_root=None):
    dest=resolve(project_root,cfg['split']['path'])
    if not dest.exists():
        raise FileNotFoundError(f'No shared episode split: {dest}; run `split` once before training')
    payload=json.loads(dest.read_text(encoding='utf-8'))
    items=payload['splits']
    signature=hashlib.sha256(json.dumps(items,sort_keys=True).encode('utf-8')).hexdigest()
    if signature != payload['sha256']:raise ValueError('Shared split signature mismatch')
    available={str(e.path.resolve()):e for e in discover(cfg,project_root,data_root)}
    # Do not include newly collected episodes; insist all frozen members are present.
    missing=[path for ids in items.values() for path in ids if path not in available]
    if missing:raise FileNotFoundError(f'{len(missing)} frozen accepted episodes missing. First: {missing[0]}')
    splits={k:[available[x] for x in items[k]] for k in ('train','val','test')}
    return splits,payload
