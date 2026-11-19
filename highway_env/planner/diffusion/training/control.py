"""Train TRACE control diffusion with K2 execution-space safety objectives."""
from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner
from highway_env.planner.diffusion.config.experiment import load_mode_yaml, resolve_repo_file


METRIC_KEYS = (
    'loss', 'trajectory_loss',
    'background_collision_loss', 'intertruck_collision_loss',
    'ttc_loss', 'ttc_background_loss', 'ttc_intertruck_loss',
    'road_boundary_loss', 'ltr_loss', 'fuel_loss', 'progress_loss',
    'background_collision_rate', 'intertruck_collision_rate',
    'ttc_background_violation_rate', 'ttc_intertruck_violation_rate',
    'road_violation_rate', 'background_min_gap_m',
    'intertruck_min_gap_m', 'max_road_violation_m',
)


SAFETY_ARCHITECTURE = 'TRACE_control_diffusion_safety_v1'


def unpack(batch, device):
    features, targets = batch[:2]
    features = {key: value.to(device, non_blocking=True) for key, value in features.items()}
    context = {
        key: value.to(device, non_blocking=True)
        for key, value in targets.items()
        if key.startswith('koopman_') and isinstance(value, torch.Tensor)
    }
    safety_targets = {
        'platoon_reference_poses': targets['platoon_reference_poses'].to(
            device, non_blocking=True
        )
    }
    return features, targets['trajectory_dense'].to(device, non_blocking=True), context, safety_targets


def _make_loader(dataset_root, split, batch_size, workers, max_samples, shuffle,
                 include_platoon_reference=False):
    dataset = RawExpertDataset(
        Path(dataset_root), split=split,
        max_samples=(int(max_samples) if max_samples else None),
        include_platoon_reference=include_platoon_reference,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
    )


def _run_pass(model, loader, optimizer, device, max_batches=None, clip_norm=1.0):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in METRIC_KEYS}
    val_generator = None
    if not training:
        val_generator = torch.Generator(device=device)
        val_generator.manual_seed(2048)
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        features, expert_xy, context, safety_targets = unpack(batch, device)
        with torch.set_grad_enabled(training):
            result = model.forward_train(
                features, expert_xy, safety_targets,
                generator=val_generator, context=context,
            )
            loss = result['loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite TRACE safety-aware training loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
        count = int(expert_xy.shape[0])
        samples += count
        for key in totals:
            totals[key] += float(result[key].detach()) * count
    if samples == 0:
        raise RuntimeError('No train/validation examples processed')
    return {key: value / samples for key, value in totals.items()}



def _restore_training_state(path, model, optimizer, cfg):
    """Restore a complete checkpoint produced by TRACE Safety-aware Loss V1."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('architecture') != SAFETY_ARCHITECTURE:
        raise ValueError(
            'Resume requires a TRACE Safety-aware Diffusion Training V1 checkpoint'
        )

    for key, expected in (
        ('model_config', cfg.get('model', {})),
        ('control_config', cfg.get('control', {})),
        ('execution_config', model.execution_config),
        ('loss_config', model.loss_config),
        ('anchors_config', model.anchors_config),
        ('koopman_metadata', model.vehicle_model.koopman_metadata),
    ):
        if saved.get(key) != expected:
            raise ValueError(
                f'Resume checkpoint {key} differs from the current configuration; '
                'restore the original Safety-aware experiment settings and K2 checkpoint'
            )

    history = saved['history']
    epoch = saved['epoch']
    if not isinstance(history, list) or not history or not isinstance(epoch, int) or epoch < 1:
        raise ValueError('Safety-aware resume requires a completed epoch and training history')
    if [row.get('epoch') for row in history] != list(range(1, epoch + 1)):
        raise ValueError('Safety-aware resume history must contain every completed epoch in order')
    if any(not np.isfinite(row['val_loss']) for row in history):
        raise ValueError('Safety-aware resume history contains a non-finite validation loss')

    best_val = float(saved['best_val_loss'])
    best_epoch = int(saved['best_epoch'])
    best_row = min(history, key=lambda row: row['val_loss'])
    if best_epoch != int(best_row['epoch']) or not np.isclose(
        best_val, float(best_row['val_loss']), rtol=1e-7, atol=1e-10
    ):
        raise ValueError('Safety-aware resume best-checkpoint metadata disagrees with history')
    if 'optimizer_state' not in saved or 'model_state' not in saved:
        raise ValueError('Safety-aware resume requires model_state and optimizer_state')

    model.load_state_dict(saved['model_state'], strict=True)
    optimizer.load_state_dict(saved['optimizer_state'])
    return saved


def _write_history(path, history):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train(cfg, *, root, output_dir, train_split=None, val_split=None,
          max_steps=None, max_epochs=None, limit_samples=0, device=None, resume=None):
    if cfg.get('output_space') != 'control':
        raise ValueError('train_diffusion_control requires output_space: control')
    training_cfg = cfg['training']
    seed = int(training_cfg.get('seed', 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_name = device or training_cfg.get('device', 'auto')
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    selected_device = torch.device(device_name)

    model = ControlDiffusionPlanner(
        cfg['model'], cfg['control'],
        execution_config=cfg['execution'],
        loss_config=cfg['losses'],
        anchors_config=cfg['anchors'],
    ).to(selected_device)
    model._validate_objectives()
    if model.vehicle_model.kind != 'koopman_k2':
        raise ValueError('Safety-aware TRACE training requires frozen Koopman K2 execution')
    if any(parameter.requires_grad for parameter in model.vehicle_model.koopman.parameters()):
        raise RuntimeError('Koopman K2 parameters must remain frozen')
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training_cfg.get('lr', 1e-4)),
        weight_decay=float(training_cfg.get('weight_decay', 1e-4)),
    )

    resume_path = resolve_repo_file(resume) if resume else None
    saved = _restore_training_state(resume_path, model, optimizer, cfg) if resume_path else None
    history = list(saved['history']) if saved else []
    start_epoch = int(saved['epoch']) + 1 if saved else 1
    best = float(saved['best_val_loss']) if saved else float('inf')
    best_epoch = int(saved['best_epoch']) if saved else 0

    epochs = int(max_epochs if max_epochs is not None else training_cfg.get('max_epochs', 50))
    if start_epoch > epochs:
        raise ValueError(
            f'Resume epoch {start_epoch - 1} already reaches max_epochs={epochs}; '
            'max_epochs is the total epoch limit'
        )

    batch_size = int(training_cfg.get('batch_size', 8))
    workers = int(training_cfg.get('num_workers', 0))
    train_loader = _make_loader(
        root, train_split or training_cfg.get('train_split', 'train'),
        batch_size, workers, limit_samples, True, include_platoon_reference=True,
    )
    val_loader = _make_loader(
        root, val_split or training_cfg.get('val_split', 'val'),
        batch_size, workers, limit_samples, False, include_platoon_reference=True,
    )

    from highway_env.planner.diffusion.common.execution import K2_CONTEXT
    first_features, first_targets, _ = train_loader.dataset[0]
    missing = [key for key in K2_CONTEXT if key not in first_targets]
    if missing:
        raise KeyError(f'Raw training dataset is missing K2 context fields: {missing}')
    if 'platoon_reference_poses' not in first_targets:
        raise KeyError('Raw training dataset is missing synchronized expert platoon references')
    if first_features['road_lane_geometry'].shape[-1] != 9:
        raise ValueError('Road geometry must use the 9-channel TRACE lane representation')

    output = Path(output_dir)
    best_source = None
    if saved and output.resolve() != resume_path.parent.resolve():
        best_source = (
            resume_path if best_epoch == int(saved['epoch'])
            else resume_path.parent / 'best.pt'
        )
        if not best_source.is_file():
            raise FileNotFoundError(
                f'Original best checkpoint is required in the new output directory: {best_source}'
            )

    output.mkdir(parents=True, exist_ok=True)
    if best_source is not None:
        shutil.copy2(best_source, output / 'best.pt')

    resolved = {**cfg, 'anchors': model.anchors_config}
    import yaml
    (output / 'resolved_config.yaml').write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding='utf-8'
    )
    if saved:
        _write_history(output / 'loss.csv', history)
        print(
            f'[TRACE RESUME] checkpoint={resume_path} completed_epoch={start_epoch - 1} '
            f'next_epoch={start_epoch} best_epoch={best_epoch} best_val={best:.6f}',
            flush=True,
        )

    batch_limit = int(max_steps) if max_steps else None
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = _run_pass(
            model, train_loader, optimizer, selected_device, batch_limit,
            float(training_cfg.get('grad_clip', 1.0)),
        )
        val_metrics = _run_pass(
            model, val_loader, None, selected_device, batch_limit,
        )
        row = {
            'epoch': epoch,
            **{f'train_{key}': value for key, value in train_metrics.items()},
            **{f'val_{key}': value for key, value in val_metrics.items()},
        }
        history.append(row)
        improved = val_metrics['loss'] < best
        if improved:
            best = val_metrics['loss']
            best_epoch = epoch

        checkpoint = {
            'architecture': SAFETY_ARCHITECTURE,
            'model_config': cfg['model'],
            'control_config': cfg['control'],
            'execution_config': model.execution_config,
            'loss_config': model.loss_config,
            'anchors_config': model.anchors_config,
            'koopman_metadata': model.vehicle_model.koopman_metadata,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'history': history,
            'best_val_loss': best,
            'best_epoch': best_epoch,
        }
        torch.save(checkpoint, output / 'last.pt')
        if improved:
            torch.save(checkpoint, output / 'best.pt')
        _write_history(output / 'loss.csv', history)
        print(
            f'[TRACE SAFETY] epoch={epoch}/{epochs} '
            f'train={train_metrics["loss"]:.5f} val={val_metrics["loss"]:.5f} '
            f'traj={val_metrics["trajectory_loss"]:.4f} '
            f'bg_col={val_metrics["background_collision_loss"]:.4f} '
            f'intertruck={val_metrics["intertruck_collision_loss"]:.4f} '
            f'ttc={val_metrics["ttc_loss"]:.4f} road={val_metrics["road_boundary_loss"]:.4f} '
            f'CR_bg={val_metrics["background_collision_rate"]:.3f} '
            f'CR_platoon={val_metrics["intertruck_collision_rate"]:.3f} '
            f'TTC_bg={val_metrics["ttc_background_violation_rate"]:.3f} '
            f'TTC_platoon={val_metrics["ttc_intertruck_violation_rate"]:.3f} '
            f'DAC_viol={val_metrics["road_violation_rate"]:.3f}',
            flush=True,
        )
        if int(training_cfg.get('patience', 0)) > 0 and epoch - best_epoch >= int(training_cfg['patience']):
            break

    print(f'[TRACE SAFETY DONE] best={output / "best.pt"}', flush=True)
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
    model.load_state_dict(payload['model_state'], strict=True)
    return model.to(device).eval(), payload


def main(argv=None):
    parser = argparse.ArgumentParser(description='TRACE safety-aware control diffusion training')
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
    training_cfg = cfg['training']
    root = resolve_repo_file(args.dataset_root or training_cfg['dataset_root'])
    output = resolve_repo_file(args.output_dir or training_cfg['output_dir'])
    train(
        cfg, root=root, output_dir=output,
        train_split=args.train_split, val_split=args.val_split,
        max_steps=args.max_steps, max_epochs=args.max_epochs,
        limit_samples=args.limit_samples, device=args.device,
    )


if __name__ == '__main__':
    main()
