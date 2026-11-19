"""Train TRACE control-space diffusion with configurable execution and losses.

Input: expert future_trajectory_dense [B,40,2]. No inverse dynamics or
pseudo controls. K2 requires observed history, payload and road friction.
"""
from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner
from highway_env.planner.diffusion.config.experiment import load_mode_yaml, resolve_repo_file


def unpack(batch, device):
    features, targets = batch[:2]
    features = {k: v.to(device, non_blocking=True) for k, v in features.items()}
    if 'trajectory_dense' not in targets:
        raise RuntimeError('Control-space training requires future_trajectory_dense [40,2]')
    # No inverse-control labels or mode CE target. K2 context remains observed input.
    context = {k: v.to(device, non_blocking=True) for k, v in targets.items()
               if k.startswith('koopman_') and isinstance(v, torch.Tensor)}
    return features, targets['trajectory_dense'].to(device), context


def _make_loader(dataset_root, split, batch_size, workers, max_samples, shuffle):
    ds = RawExpertDataset(Path(dataset_root), split=split,
        max_samples=(int(max_samples) if max_samples else None))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def _run_pass(model, loader, optimizer, device, max_batches=None, clip_norm=1.0):
    training = optimizer is not None
    model.train(training)
    totals = {k: 0.0 for k in ('loss', 'position_loss', 'trajectory_loss',
                                  'ltr_loss', 'fuel_loss', 'progress_loss')}
    # Fixed validation noise gives comparable model-selection scores per epoch.
    val_generator = None
    if not training:
        val_generator = torch.Generator(device=device)
        val_generator.manual_seed(2048)
    samples = 0
    for j, batch in enumerate(loader):
        if max_batches is not None and j >= max_batches:
            break
        features, xy, context = unpack(batch, device)
        with torch.set_grad_enabled(training):
            result = model.forward_train(features, xy, generator=val_generator, context=context)
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


def _restore_training_state(path, model, optimizer, cfg):
    """Restore a completed epoch, refusing changes to the trained experiment."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    for key, expected in (
        ('model_config', cfg.get('model', {})),
        ('control_config', cfg.get('control', {})),
        ('execution_config', model.execution_config),
        ('loss_config', model.loss_config),
        ('anchors_config', model.anchors_config),
    ):
        if saved.get(key) != expected:
            raise ValueError(f'Resume checkpoint {key} differs from the current configuration; '
                             'restore the original experiment settings and anchor artifacts')
    history = saved.get('history')
    epoch = saved.get('epoch')
    if ('optimizer_state' not in saved or not history or not isinstance(epoch, int)
            or epoch < 1 or history[-1].get('epoch') != epoch):
        raise ValueError('Full resume requires optimizer_state, epoch and matching training history')
    if any(not np.isfinite(row.get('val_loss', float('nan'))) for row in history):
        raise ValueError('Resume checkpoint history contains missing or invalid validation losses')
    model.load_state_dict(saved['model_state'], strict=True)
    optimizer.load_state_dict(saved['optimizer_state'])
    return saved


def _write_history(path, history):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train(cfg, *, root, output_dir, train_split=None, val_split=None,
          max_steps=None, max_epochs=None, limit_samples=0, device=None, resume=None):
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
    model = ControlDiffusionPlanner(
        cfg.get('model'), cfg.get('control'),
        execution_config=cfg.get('execution'), loss_config=cfg.get('losses'),
                anchors_config=cfg.get('anchors'),
            ).to(selected_device)
    # Explicitly exclude frozen K2 parameters from the optimizer.
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(tr.get('lr', 1e-4)),
                                  weight_decay=float(tr.get('weight_decay', 1e-4)))
    resume_path = resolve_repo_file(resume) if resume else None
    saved = _restore_training_state(resume_path, model, optimizer, cfg) if resume_path else None
    history = list(saved['history']) if saved else []
    start_epoch = int(saved['epoch']) + 1 if saved else 1
    best_row = min(history, key=lambda row: row['val_loss']) if history else None
    best = float(best_row['val_loss']) if best_row else float('inf')
    best_epoch = int(best_row['epoch']) if best_row else 0
    epochs = int(max_epochs if max_epochs is not None else tr.get('max_epochs', 50))
    if start_epoch > epochs:
        raise ValueError(f'Resume epoch {start_epoch - 1} already reaches max_epochs={epochs}; '
                         'max_epochs is the total epoch limit')
    batch_size = int(tr.get('batch_size', 8))
    workers = int(tr.get('num_workers', 0))
    train_loader = _make_loader(root, train_split or tr.get('train_split', 'train'),
                                batch_size, workers, limit_samples, True)
    val_loader = _make_loader(root, val_split or tr.get('val_split', 'val'),
                              batch_size, workers, limit_samples, False)
    # Audit observed execution context before the first optimizer step.
    from highway_env.planner.diffusion.common.execution import K2_CONTEXT
    first_targets = train_loader.dataset[0][1]
    required = K2_CONTEXT if model.vehicle_model.kind == 'koopman_k2' else ()
    if model._weight('fuel') > 0:
        required = tuple(set(required) | {'koopman_payload_kg'})
    missing = [key for key in required if key not in first_targets]
    if missing:
        raise KeyError(f'Selected raw dataset is incompatible with execution/loss config: missing {missing}')
    output = Path(output_dir)
    best_source = None
    if saved and output.resolve() != resume_path.parent.resolve():
        best_source = resume_path if best_epoch == saved['epoch'] else resume_path.parent / 'best.pt'
        if not best_source.is_file():
            raise FileNotFoundError(f'Original best checkpoint is required in the new output directory: {best_source}')
    output.mkdir(parents=True, exist_ok=True)
    if best_source is not None:
        shutil.copy2(best_source, output / 'best.pt')
    (output / 'resolved_config.yaml').write_text(
        __import__('yaml').safe_dump({**cfg, 'anchors': model.anchors_config},
                                    allow_unicode=True, sort_keys=False), encoding='utf-8')
    if saved:
        _write_history(output / 'loss.csv', history)
        print(f'[TRACE RESUME] checkpoint={resume_path} completed_epoch={start_epoch - 1} '
              f'next_epoch={start_epoch} best_epoch={best_epoch} best_val={best:.6f}', flush=True)
    limit = int(max_steps or 0)
    batch_limit = limit if limit > 0 else None
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = _run_pass(model, train_loader, optimizer, selected_device, batch_limit,
                                  float(tr.get('grad_clip', 1.0)))
        val_metrics = _run_pass(model, val_loader, None, selected_device, batch_limit)
        row = {'epoch': epoch, **{f'train_{k}': v for k, v in train_metrics.items()},
               **{f'val_{k}': v for k, v in val_metrics.items()}}
        history.append(row)
        improved = val_metrics['loss'] < best
        if improved:
            best = val_metrics['loss']; best_epoch = epoch
        ckpt = {
            'architecture': ('TRACE_control_diffusion_D1_trajectory_only'
                             if model.vehicle_model.kind == 'kinematic' and not cfg.get('losses')
                             else 'TRACE_control_diffusion_D2'),
            'model_config': cfg.get('model', {}), 'control_config': cfg.get('control', {}),
            'execution_config': model.execution_config,
            'loss_config': model.loss_config,
            'anchors_config': model.anchors_config,
            'koopman_metadata': model.vehicle_model.koopman_metadata,
            'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'epoch': epoch, 'history': history,
            'best_val_loss': best, 'best_epoch': best_epoch,
        }
        torch.save(ckpt, output / 'last.pt')
        if improved:
            torch.save(ckpt, output / 'best.pt')
        _write_history(output / 'loss.csv', history)
        print(f'[TRACE CONTROL] epoch={epoch}/{epochs} '
              f'train_total={train_metrics["loss"]:.6f} '
              f'val_total={val_metrics["loss"]:.6f} '
              f'traj={val_metrics["trajectory_loss"]:.6f} '
              f'ltr={val_metrics["ltr_loss"]:.6f} '
              f'fuel={val_metrics["fuel_loss"]:.6f} '
              f'prog={val_metrics["progress_loss"]:.6f}', flush=True)
        if int(tr.get('patience', 0)) > 0 and epoch - best_epoch >= int(tr['patience']):
            break
    print(f'[D1 DONE] best={output / "best.pt"}', flush=True)
    return output / 'best.pt'


def load_control_checkpoint(checkpoint, device='cpu'):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model = ControlDiffusionPlanner(
        payload['model_config'], payload['control_config'],
        execution_config=payload.get('execution_config'),
        loss_config=payload.get('loss_config'),
        koopman_metadata=payload.get('koopman_metadata'),
                anchors_config=payload.get('anchors_config'),
            )
    if any(key.startswith('denoiser.mode_embedding.') for key in payload['model_state']):
        raise RuntimeError('Legacy mode_embedding checkpoint is incompatible; retrain the identity-free control planner')
    model.load_state_dict(payload['model_state'])
    return model.to(device).eval(), payload


def main(argv=None):
    parser = argparse.ArgumentParser(description='Trajectory-only control diffusion D1')
    parser.add_argument('--config', default='configs/train/diffusion.yaml')
    parser.add_argument('--dataset-root', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--resume', default=None, help='Resume model, optimizer and epoch history from last.pt')
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
          device=args.device, resume=args.resume)


if __name__ == '__main__':
    main()
