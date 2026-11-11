"""Train 40-step direct-control diffusion ONLY with expert trajectory loss.

Input: expert future_trajectory_dense [B,40,2]. No inverse dynamics, pseudo
control labels, control regression loss or mode classification loss.
"""
from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline.expert_dataset import AllMergeExpertDatasetConfig, AllMergeExpertShardDataset
from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner
from highway_env.planner.diffusion.config.experiment import load_mode_yaml, resolve_repo_file


def unpack(batch, device):
    features, targets = batch[:2]
    features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
    if 'trajectory_dense' not in targets:
        raise RuntimeError('Control-space training requires future_trajectory_dense [40,2]')
    # Explicitly ignore target_mode: this training path uses ONLY trajectory GT.
    return features, targets['trajectory_dense'].to(device)


def _make_loader(dataset_root, split, batch_size, workers, max_samples, shuffle):
    ds = AllMergeExpertShardDataset(AllMergeExpertDatasetConfig(
        dataset_root=Path(dataset_root), split=split,
        max_samples=(int(max_samples) if max_samples else None)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def _run_pass(model, loader, optimizer, device, max_batches=None, clip_norm=1.0):
    training = optimizer is not None
    model.train(training)
    totals = {'loss': 0.0, 'position_loss': 0.0}
    # Fixed validation noise gives comparable model-selection scores per epoch.
    val_generator = None
    if not training:
        val_generator = torch.Generator(device=device)
        val_generator.manual_seed(2048)
    samples = 0
    for j, batch in enumerate(loader):
        if max_batches is not None and j >= max_batches:
            break
        features, xy = unpack(batch, device)
        with torch.set_grad_enabled(training):
            result = model.forward_train(features, xy, generator=val_generator)
            loss = result['loss']
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite trajectory-only training loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
        n = int(xy.shape[0]); samples += n
        for key in totals:
            totals[key] += float(result[key].detach()) * n
    if not samples:
        raise RuntimeError('No train/validation examples processed')
    return {key: value / samples for key, value in totals.items()}


def train(cfg, *, root, output_dir, train_split=None, val_split=None,
          max_steps=None, max_epochs=None, limit_samples=0, device=None):
    if cfg.get('output_space') != 'control':
        raise ValueError('train_diffusion_control requires output_space: control')
    tr = cfg['training']
    seed = int(tr.get('seed', 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_name = device or tr.get('device', 'auto')
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    selected_device = torch.device(device_name)
    model = ControlDiffusionPlanner(cfg.get('model'), cfg.get('control')).to(selected_device)
    batch_size = int(tr.get('batch_size', 8))
    workers = int(tr.get('num_workers', 0))
    train_loader = _make_loader(root, train_split or tr.get('train_split', 'train'),
                                batch_size, workers, limit_samples, True)
    val_loader = _make_loader(root, val_split or tr.get('val_split', 'val'),
                              batch_size, workers, limit_samples, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr.get('lr', 1e-4)),
                                  weight_decay=float(tr.get('weight_decay', 1e-4)))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved_config.yaml').write_text(
        __import__('yaml').safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    history = []
    best = float('inf')
    best_epoch = 0
    epochs = int(max_epochs if max_epochs is not None else tr.get('max_epochs', 50))
    limit = int(max_steps or 0)
    batch_limit = limit if limit > 0 else None
    for epoch in range(1, epochs + 1):
        train_metrics = _run_pass(model, train_loader, optimizer, selected_device, batch_limit,
                                  float(tr.get('grad_clip', 1.0)))
        val_metrics = _run_pass(model, val_loader, None, selected_device, batch_limit)
        row = {'epoch': epoch, **{f'train_{k}': v for k, v in train_metrics.items()},
               **{f'val_{k}': v for k, v in val_metrics.items()}}
        history.append(row)
        ckpt = {
            'architecture': 'TRACE_control_diffusion_D1_trajectory_only',
            'model_config': cfg.get('model', {}), 'control_config': cfg.get('control', {}),
            'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'epoch': epoch, 'history': history,
        }
        torch.save(ckpt, output / 'last.pt')
        if val_metrics['loss'] < best:
            best = val_metrics['loss']; best_epoch = epoch
            torch.save(ckpt, output / 'best.pt')
        print(f'[D1 TRAJ ONLY] epoch={epoch}/{epochs} '
              f'train_xy={train_metrics["loss"]:.6f} '
              f'val_xy={val_metrics["loss"]:.6f}', flush=True)
        if int(tr.get('patience', 0)) > 0 and epoch - best_epoch >= int(tr['patience']):
            break
    with (output / 'loss.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    print(f'[D1 DONE] best={output / "best.pt"}', flush=True)
    return output / 'best.pt'


def load_control_checkpoint(checkpoint, device='cpu'):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model = ControlDiffusionPlanner(payload['model_config'], payload['control_config'])
    model.load_state_dict(payload['model_state'])
    return model.to(device).eval(), payload


def main(argv=None):
    parser = argparse.ArgumentParser(description='Trajectory-only control diffusion D1')
    parser.add_argument('--config', default='configs/train/diffusion.yaml')
    parser.add_argument('--dataset-root', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--train-split', default=None)
    parser.add_argument('--val-split', default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--limit-samples', type=int, default=0)
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)
    cfg = load_mode_yaml(args.config)
    tr = cfg['training']
    root = resolve_repo_file(args.dataset_root or tr['dataset_root'])
    output = resolve_repo_file(args.output_dir or tr['output_dir'])
    train(cfg, root=root, output_dir=output, train_split=args.train_split,
          val_split=args.val_split, max_steps=args.max_steps,
          max_epochs=args.max_epochs, limit_samples=args.limit_samples,
          device=args.device)


if __name__ == '__main__':
    main()
