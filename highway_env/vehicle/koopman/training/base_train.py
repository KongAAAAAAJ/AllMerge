"""Shared training utilities and the standalone K0 training implementation.

Path resolution, device selection, seeding and checkpoint saving are shared
by the unified trainers and evaluators. The standalone K0 routines retain
episode-held-out validation and LTR supervision for existing callers.

Does not load the TruckSim DLL or modify the running collector.
"""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from highway_env.vehicle.koopman.data.dataset import (discover_episodes, episode_split, KoopmanWindowDataset,
                      compute_normalization, describe_splits)
from highway_env.vehicle.koopman.models.base_model import NeuralKoopmanK0, rollout_pose


def resolve_path(project_root, path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(project_root) / path


def load_config(filename):
    with Path(filename).open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('Expected a YAML mapping')
    for name in ('data', 'model', 'training', 'outputs'):
        if not isinstance(cfg.get(name), dict):
            raise ValueError(f'Missing K0 config section {name}')
    if str(cfg['model'].get('type', 'K0')).upper() != 'K0':
        raise ValueError('Koopman paper K0 trainer only supports model.type=K0')
    if abs(float(cfg['data'].get('dt_s', .1)) - .1) > 1e-8:
        raise ValueError('Current collected data and 4s horizon use 10Hz (dt_s=0.1)')
    if int(cfg['data'].get('horizon_steps', 40)) != 40:
        raise ValueError('Koopman paper evaluation requires 4s/40-step rollout')
    return cfg


def choose_device(name):
    if name == 'auto': name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_episodes(cfg, root, data_root=None):
    dc = cfg['data']
    return discover_episodes([data_root] if data_root else dc.get('roots', []), root,
        require_accepted=bool(dc.get('accepted_only', True)),
        expected_dt=float(dc.get('dt_s', .1)), strict=bool(dc.get('strict', False)))


def make_dataset(episodes, cfg, stride):
    dc = cfg['data']
    return KoopmanWindowDataset(episodes, history_steps=int(dc.get('history_steps', 5)),
        horizon_steps=40, warmup_s=float(dc.get('warmup_s', 1.)),
        window_stride=int(stride), dt=float(dc.get('dt_s', .1)))


def loss_parts(model, batch, cfg):
    pred = model.predict_all(batch['state_hist'], batch['control_hist'],
                             batch['future_controls'],batch['target_poses'][:,0],
                             dt=float(cfg['data'].get('dt_s',.1)))
    states, latent = pred['states'], pred['latent']
    target = batch['target_states']
    residual = (states - target) / model.state_std
    smooth = torch.nn.functional.smooth_l1_loss
    zero = lambda x: torch.zeros_like(x)
    one = smooth(residual[:, 1], zero(residual[:, 1]))
    multi = smooth(residual[:, 1:], zero(residual[:, 1:]))
    recon = smooth(residual[:, 0], zero(residual[:, 0]))
    pos_err = (pred['pose'][:, 1:, :2] - batch['target_poses'][:, 1:, :2]) / float(cfg['training'].get('position_scale_m', 5.))
    traj = smooth(pos_err, zero(pos_err))
    # Regularize toward the initialized near-identity dynamics; no spectral-radius guarantee.
    regularizer = (model.A - .995 * torch.eye(model.latent_dim, device=model.A.device)).square().mean()
    w = cfg['training'].get('loss_weights', {})
    total = (float(w.get('one_step', .2))*one + float(w.get('rollout', 1.))*multi +
             float(w.get('reconstruction', .1))*recon + float(w.get('trajectory', .05))*traj +
             float(w.get('matrix_regularization', 1e-4))*regularizer)
    # LTR is independently supervised from TruckSim wheel loads.
    # Invalid labels are masked, never replaced with predicted/truth surrogates.
    mask = batch['ltr_valid'][:, 1:]
    if mask.any():
        ltr_scale = float(cfg['training'].get('ltr_scale', 0.1))
        if ltr_scale <= 0: raise ValueError('ltr_scale must be positive')
        ltr_resid = (pred['ltr'][:, 1:][mask] - batch['target_ltr'][:, 1:][mask]) / ltr_scale
        ltr_loss = smooth(ltr_resid, torch.zeros_like(ltr_resid))
        total = total + float(w.get('ltr', 0.15)) * ltr_loss
    return total


def run_epoch(model, loader, cfg, device, optimizer=None):
    model.train(optimizer is not None)
    total = 0.; count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_parts(model, batch, cfg)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite K0 loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['training'].get('grad_clip', 1.)))
            optimizer.step()
        else:
            with torch.no_grad(): loss = loss_parts(model, batch, cfg)
        n = batch['target_states'].shape[0]
        total += loss.detach().item()*n
        count += n
    if count == 0: raise ValueError('No 4-second windows in the data loader')
    return total/count


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.writing')
    torch.save(state, temp)
    os.replace(temp, path)


def load_checkpoint(path, device='cpu'):
    # weights_only avoids arbitrary pickle loading when supported by PyTorch.
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model = NeuralKoopmanK0(**ckpt['model_kwargs'], **ckpt['normalization'])
    model.load_state_dict(ckpt['model_state'], strict=True)
    return model.to(device).eval(), ckpt


def train_k0(cfg, project_root, data_root=None, epochs=None, device=None):
    train_cfg, dc = cfg['training'], cfg['data']
    seed = int(train_cfg.get('seed', 2026))
    set_seed(seed)
    device = choose_device(device or train_cfg.get('device', 'auto'))
    episodes = get_episodes(cfg, project_root, data_root)
    split = episode_split(episodes, float(dc.get('val_fraction', .15)),
                          float(dc.get('test_fraction', .10)), seed)
    train_ds = make_dataset(split['train'], cfg, int(dc.get('train_stride', 2)))
    val_ds = make_dataset(split['val'], cfg, int(dc.get('eval_stride', 5)))
    if not train_ds or not val_ds:
        raise ValueError('No valid 4s windows: use accepted episodes longer than history+warmup+4s')
    norm = compute_normalization(split['train'])
    kwargs = {'history_steps': int(dc.get('history_steps', 5)),
              'latent_dim': int(cfg['model'].get('latent_dim', 32)),
              'hidden_dim': int(cfg['model'].get('hidden_dim', 128)),
              }
    model = NeuralKoopmanK0(**kwargs, **norm).to(device)
    if not all(np.isfinite(ep.ltr).any() for ep in split['train']+split['val']):
        raise ValueError('LTR is required for both training and validation; check NPZ ltr10')
    batch_size = int(train_cfg.get('batch_size', 64))
    if batch_size < 1: raise ValueError('batch_size must be >=1')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    optim = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get('lr', 3e-4)),
                              weight_decay=float(train_cfg.get('weight_decay', 1e-5)))
    max_epochs = int(epochs if epochs is not None else train_cfg.get('epochs', 60))
    if max_epochs < 1: raise ValueError('epochs must be >=1')
    out = resolve_path(project_root, cfg['outputs'].get('checkpoints', 'outputs/koopman/checkpoints/k0_paper'))
    out.mkdir(parents=True, exist_ok=True)
    report = describe_splits(split, kwargs['history_steps'], 40,
                             float(dc.get('warmup_s', 1.)), int(dc.get('train_stride', 2)), .1)
    (out/'split_manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[K0] device={device} accepted={len(episodes)} episodes '
          f"train/val/test={len(split['train'])}/{len(split['val'])}/{len(split['test'])} "
          f'windows={len(train_ds)}/{len(val_ds)}', flush=True)
    best, best_epoch = float('inf'), 0
    history = []
    patience = int(train_cfg.get('patience', 20))
    for epoch in range(1, max_epochs+1):
        train_loss = run_epoch(model, train_loader, cfg, device, optim)
        val_loss = run_epoch(model, val_loader, cfg, device)
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        history.append(row)
        state = {'architecture':'NeuralKoopmanK0_Paper_v3', 'model_kwargs':kwargs,
                 'normalization':norm, 'model_state':copy.deepcopy(model.state_dict()),
                 'epoch': epoch, 'train_config':cfg, 'split_manifest':report,
                 'history':copy.deepcopy(history)}
        save_checkpoint(out/'last.pt', state)
        if val_loss < best - 1e-9:
            best, best_epoch = val_loss, epoch
            state['best_val_loss'] = best
            save_checkpoint(out/'best.pt', state)
        print(f'[K0] epoch={epoch:03d}/{max_epochs} train={train_loss:.6f} '
              f'val={val_loss:.6f} best={best:.6f}', flush=True)
        if patience > 0 and epoch-best_epoch >= patience:
            print(f'[K0] early_stop best_epoch={best_epoch}',flush=True)
            break
    with (out/'loss.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=('epoch','train_loss','val_loss'))
        writer.writeheader(); writer.writerows(history)
    best_model, _ = load_checkpoint(out/'best.pt', device)
    sample = next(iter(val_loader))
    with torch.no_grad():
        prediction, latent = best_model(sample['state_hist'].to(device),
            sample['control_hist'].to(device), sample['future_controls'].to(device))
    assert prediction.shape == (sample['state_hist'].shape[0], 41, 5)
    assert latent.shape == (sample['state_hist'].shape[0], 41, kwargs['latent_dim'])
    assert best_model.decode_ltr(latent).shape == (sample['state_hist'].shape[0], 41)
    print(f'[K0 DONE] best={out/"best.pt"} last={out/"last.pt"} '
          f'prediction={tuple(prediction.shape)}',flush=True)
    return {'best':str(out/'best.pt'), 'last':str(out/'last.pt'),
            'best_val_loss':best, 'prediction_shape':tuple(prediction.shape)}
