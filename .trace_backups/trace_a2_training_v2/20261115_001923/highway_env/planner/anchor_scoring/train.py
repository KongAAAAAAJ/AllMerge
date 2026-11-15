"""Offline TRACE A2 safety scorer train/eval, independent of diffusion training.

Example:
 python -m highway_env.planner.anchor_scoring.train --config configs/train/adaptive_anchors.yaml --max-batches 2
 python -m highway_env.planner.anchor_scoring.train --config configs/train/adaptive_anchors.yaml --evaluate-only --checkpoint outputs/anchor_scoring/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data_pipeline.expert_dataset import AllMergeExpertDatasetConfig, AllMergeExpertShardDataset
from highway_env.planner.adaptive_anchors import AdaptiveCandidateBuilder
from highway_env.planner.diffusion.config.experiment import resolve_repo_file
from .model import AnchorSafetyScorer, select_anchors
from .safety import anchor_safety_cost


def _loader(root, split, cfg, shuffle):
    dataset = AllMergeExpertShardDataset(AllMergeExpertDatasetConfig(
        dataset_root=root, split=split, max_samples=cfg.get('max_samples')))
    workers = int(cfg.get('num_workers', 0))
    return DataLoader(dataset, batch_size=int(cfg.get('batch_size', 16)),
                      num_workers=workers, shuffle=shuffle,
                      persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())


def run_epoch(model, loader, candidate_builder, cfg, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    report = {name: 0.0 for name in ('loss', 'safe_ratio', 'oracle_safe_ratio',
                                     'top_r_safe_recall', 'label_entropy', 'score_gap')}
    count = 0
    top_r, ka = int(cfg['top_r']), int(cfg['num_selected'])
    teacher = cfg.get('teacher', {})
    temperature = float(cfg.get('temperature', 0.3))
    if temperature <= 0:
        raise ValueError('temperature must be > 0')
    for batch_index, (features, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        pool = candidate_builder.build_batch(features)
        features = {name: value.to(device) for name, value in features.items()}
        control = torch.as_tensor(pool['coarse_controls'], device=device)
        xy = torch.as_tensor(pool['coarse_trajectories'], device=device)
        valid = torch.as_tensor(pool['mode_valid_mask'], device=device)
        with torch.no_grad():
            labels = anchor_safety_cost(features, xy, valid=valid, **teacher)
            teacher_logits = (-labels['total'] / temperature).masked_fill(~valid, -torch.inf)
            target = F.softmax(teacher_logits, dim=-1)
        with torch.set_grad_enabled(training):
            scores = model(features, control, xy)
            log_prob = F.log_softmax(scores.masked_fill(~valid, -torch.inf), dim=-1)
            loss = -(target * log_prob.masked_fill(~valid, 0.0)).sum(-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite anchor-scoring loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get('grad_clip', 1.0)))
                optimizer.step()
        with torch.no_grad():
            idx = select_anchors(scores.detach(), control, xy, valid,
                     top_r=top_r, num_selected=ka,
                     trajectory_weight=float(cfg.get('trajectory_weight', 1.0)),
                     control_weight=float(cfg.get('control_weight', 1.0)))
            selected_safe = labels['hard_safe'].gather(1, idx)
            # Softmax target normalization is over valid candidate pool only.
            sorted_ids = scores.detach().masked_fill(~valid, -torch.inf).topk(top_r, dim=1).indices
            total_safe = labels['hard_safe'].sum(1).clamp_min(1)
            recall = labels['hard_safe'].gather(1, sorted_ids).sum(1) / total_safe
            optimal_safe = torch.minimum(labels['hard_safe'].sum(1),
                                         torch.full_like(total_safe, ka)).float() / ka
            values = {
                'loss': loss.detach(),
                'safe_ratio': selected_safe.float().mean(),
                'oracle_safe_ratio': optimal_safe.mean(),
                'top_r_safe_recall': recall.mean(),
                'label_entropy': (-(target * target.clamp_min(1e-12).log()).sum(1)).mean(),
                'score_gap': scores.detach().masked_fill(~valid, 0).std(dim=1).mean(),
            }
            n = int(xy.shape[0]); count += n
            for key, value in values.items():
                report[key] += float(value) * n
    if count == 0:
        raise RuntimeError('No samples processed')
    return {name: value / count for name, value in report.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description='Train/evaluate TRACE adaptive safety scorer')
    p.add_argument('--config', default='configs/train/adaptive_anchors.yaml')
    p.add_argument('--dataset-root', default=None)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--max-epochs', type=int, default=None)
    p.add_argument('--max-batches', type=int, default=None)
    p.add_argument('--device', default='auto')
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--checkpoint', default=None)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(resolve_repo_file(args.config).read_text(encoding='utf-8'))
    if not isinstance(cfg, dict):
        raise ValueError('Scorer config must be a mapping')
    seed = int(cfg.get('seed', 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    root = resolve_repo_file(args.dataset_root or cfg['dataset_root'])
    output_dir = resolve_repo_file(args.output_dir or cfg.get('output_dir', 'outputs/anchor_scoring'))
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = AdaptiveCandidateBuilder(cfg.get('candidate_generation'))
    train_loader = None if args.evaluate_only else _loader(root, cfg.get('train_split', 'train'), cfg, True)
    val_loader = _loader(root, cfg.get('val_split', 'val'), cfg, False)
    model_cfg = cfg.get('model', {})
    model = AnchorSafetyScorer(model_cfg).to(device)
    if args.checkpoint:
        saved = torch.load(resolve_repo_file(args.checkpoint), map_location='cpu', weights_only=True)
        if saved.get('architecture') != 'TRACE_Adaptive_Anchor_Safety_Scorer_V1':
            raise ValueError('Incompatible checkpoint')
        model.load_state_dict(saved['model_state'], strict=True)
    if args.evaluate_only:
        metrics = run_epoch(model, val_loader, generator, cfg, device,
                            max_batches=args.max_batches)
        (output_dir / 'evaluation.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(json.dumps(metrics, indent=2))
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get('lr', 1e-4)),
                                  weight_decay=float(cfg.get('weight_decay', 1e-4)))
    best = float('inf')
    history = []
    epochs = int(args.max_epochs if args.max_epochs is not None else cfg.get('epochs', 30))
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, generator, cfg, device,
                                  optimizer=optimizer, max_batches=args.max_batches)
        val_metrics = run_epoch(model, val_loader, generator, cfg, device,
                                max_batches=args.max_batches)
        history.append({'epoch': epoch, **{'train_' + k: v for k,v in train_metrics.items()},
                        **{'val_' + k: v for k,v in val_metrics.items()}})
        ckpt = {'architecture': 'TRACE_Adaptive_Anchor_Safety_Scorer_V1',
                'model_config': model_cfg, 'candidate_generation': cfg.get('candidate_generation', {}),
                'model_state': model.state_dict(), 'epoch': epoch,
                'temperature': float(cfg.get('temperature', 0.3)),
                'teacher_config': cfg.get('teacher', {})}
        torch.save(ckpt, output_dir / 'last.pt')
        if val_metrics['loss'] < best:
            best = val_metrics['loss']
            torch.save(ckpt, output_dir / 'best.pt')
        print(f'[A2] epoch={epoch}/{epochs} train_CE={train_metrics["loss"]:.5f} '
              f'val_CE={val_metrics["loss"]:.5f} val_safe={val_metrics["safe_ratio"]:.3f} '
              f'oracle={val_metrics["oracle_safe_ratio"]:.3f}', flush=True)
    with (output_dir / 'history.csv').open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    (output_dir / 'resolved_config.yaml').write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print(f'[A2] best scorer: {output_dir / "best.pt"}')

if __name__ == '__main__':
    main()
