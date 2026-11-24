"""Read-only TRACE diagnostics for road curvature and lateral endpoint bias.

This script never writes checkpoints or modifies training/configuration files.
It reports dataset statistics and, when a checkpoint is supplied, the selected
Diffusion trajectory endpoint error under the deployed Candidate Scoring selector.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.planner.diffusion.training.control import load_control_checkpoint


def signed_polyline_curvature(polyline: np.ndarray) -> np.ndarray:
    """Three-point signed curvature [1/m] from target-lane XY samples."""
    p = np.asarray(polyline, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2:
        return np.empty(0, dtype=np.float64)
    p = p[:, :2]
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 3:
        return np.empty(0, dtype=np.float64)
    # Remove duplicate/near-duplicate samples before differentiating.
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-4]
    p = p[keep]
    if len(p) < 3:
        return np.empty(0, dtype=np.float64)
    a = p[1:-1] - p[:-2]
    b = p[2:] - p[1:-1]
    c = p[2:] - p[:-2]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    valid = denom > 1e-8
    out = np.full(len(denom), np.nan, dtype=np.float64)
    out[valid] = 2.0 * cross[valid] / denom[valid]
    return out[np.isfinite(out)]


def summarize(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'count': 0}
    return {
        'count': int(len(x)), 'mean': float(np.mean(x)), 'std': float(np.std(x)),
        'median': float(np.median(x)), 'p05': float(np.quantile(x, .05)),
        'p95': float(np.quantile(x, .95)), 'min': float(np.min(x)), 'max': float(np.max(x)),
        'positive_ratio': float(np.mean(x > 1e-6)),
        'negative_ratio': float(np.mean(x < -1e-6)),
        'near_zero_ratio': float(np.mean(np.abs(x) <= 1e-6)),
    }


def collate_context(targets, device):
    return {k: v.to(device) for k, v in targets.items()
            if k.startswith('koopman_') and isinstance(v, torch.Tensor)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root', default='outputs/expert_dataset/expert_trucksim_k2_raw_100k')
    ap.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    ap.add_argument('--checkpoint', default=None, help='Optional trained ControlDiffusionPlanner checkpoint')
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=2048)
    ap.add_argument('--output-dir', default='outputs/diagnostics/lateral_bias')
    args = ap.parse_args(argv)

    root = Path(args.dataset_root)
    if not root.is_absolute(): root = ROOT / root
    out = Path(args.output_dir)
    if not out.is_absolute(): out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    dataset = RawExpertDataset(root, split=args.split,
                               max_samples=args.max_samples or None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=args.device.startswith('cuda'))
    model = None
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_absolute(): ckpt = ROOT / ckpt
        model, _ = load_control_checkpoint(ckpt, device=args.device)
        model.eval()

    curvatures, expert_y, target_y = [], [], []
    pred_y, pred_error, pred_fde, pred_mode = [], [], [], []
    records = []
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    offset = 0
    with torch.inference_mode():
        for features, targets, metadata in loader:
            bsz = len(targets['trajectory_dense'])
            f_np = features['target_lane_polyline'].cpu().numpy()
            expert = targets['trajectory_dense'].cpu().numpy()
            target = features['target_point'].cpu().numpy()
            for j in range(bsz):
                curv = signed_polyline_curvature(f_np[j])
                curvatures.extend(curv.tolist())
                expert_y.append(float(expert[j, -1, 1]))
                target_y.append(float(target[j, 1]))

            if model is not None:
                fdev = {k: v.to(args.device) for k, v in features.items()}
                context = collate_context(targets, args.device)
                result = model.infer_multimodal(fdev, generator=generator, context=context)
                selected = result['trajectory_mode_idx'].cpu().numpy()
                poses = result['trajectory_dense'].cpu().numpy()
                expert_end = expert[:, -1, :]
                for j in range(bsz):
                    yhat = float(poses[j, -1, 1])
                    ygt = float(expert_end[j, 1])
                    pred_y.append(yhat)
                    pred_error.append(yhat-ygt)
                    pred_fde.append(float(np.linalg.norm(poses[j, -1]-expert_end[j])))
                    pred_mode.append(int(selected[j]))
            for j in range(bsz):
                def item(v):
                    if isinstance(v, torch.Tensor): return v[j].item()
                    return int(v[j]) if isinstance(v, np.ndarray) else v
                rec = {
                    'sample_index': offset+j,
                    'episode_index': item(metadata['episode_index']),
                    'frame_index': item(metadata['frame_index']),
                    'ego_index': item(metadata['ego_index']),
                    'expert_terminal_y_m': float(expert[j, -1, 1]),
                    'target_point_y_m': float(target[j, 1]),
                    'target_minus_expert_y_m': float(target[j, 1]-expert[j, -1, 1]),
                    'mean_signed_lane_curvature_1pm': float(np.mean(signed_polyline_curvature(f_np[j]))) if len(signed_polyline_curvature(f_np[j])) else float('nan'),
                }
                if model is not None:
                    rec.update(pred_terminal_y_m=pred_y[-bsz+j],
                               pred_terminal_lateral_error_m=pred_error[-bsz+j],
                               pred_terminal_fde_m=pred_fde[-bsz+j],
                               selected_mode=pred_mode[-bsz+j])
                records.append(rec)
            offset += bsz
            if offset % 1000 == 0: print(f'[diagnostics] processed {offset}/{len(dataset)}', flush=True)

    stats = {
        'dataset_root': str(root), 'split': args.split, 'num_samples': len(dataset),
        'road_signed_curvature_1pm': summarize(np.asarray(curvatures)),
        'road_mean_signed_curvature_per_sample_1pm': summarize(np.asarray([r['mean_signed_lane_curvature_1pm'] for r in records])),
        'road_curvature_class_counts': {
            'left_positive_gt_0p001_1pm': int(sum(r['mean_signed_lane_curvature_1pm'] > 1e-3 for r in records)),
            'approximately_straight_abs_le_0p001_1pm': int(sum(abs(r['mean_signed_lane_curvature_1pm']) <= 1e-3 for r in records)),
            'right_negative_lt_minus_0p001_1pm': int(sum(r['mean_signed_lane_curvature_1pm'] < -1e-3 for r in records)),
            'threshold_1pm': 1e-3,
        },
        'expert_terminal_lateral_displacement_m': summarize(np.asarray(expert_y)),
        'target_point_lateral_position_m': summarize(np.asarray(target_y)),
        'target_minus_expert_terminal_lateral_m': summarize(np.asarray(target_y)-np.asarray(expert_y)),
    }
    if model is not None:
        grouped = {}
        for label, predicate in (
            ('left', lambda v: v > 1e-3),
            ('straight', lambda v: abs(v) <= 1e-3),
            ('right', lambda v: v < -1e-3),
        ):
            selected_rows = [r for r in records if predicate(r['mean_signed_lane_curvature_1pm'])]
            grouped[label] = {
                'num_samples': len(selected_rows),
                'terminal_lateral_error_m': summarize(np.asarray([r['pred_terminal_lateral_error_m'] for r in selected_rows])),
                'terminal_fde_m': summarize(np.asarray([r['pred_terminal_fde_m'] for r in selected_rows])),
            }
        stats['prediction_error_by_road_curvature_class'] = grouped
        stats.update({
            'predicted_terminal_lateral_position_m': summarize(np.asarray(pred_y)),
            'predicted_terminal_lateral_error_m': summarize(np.asarray(pred_error)),
            'predicted_terminal_fde_m': summarize(np.asarray(pred_fde)),
            'selected_mode_histogram': {str(i): int(np.sum(np.asarray(pred_mode)==i))
                                        for i in range(model.scene_config.num_modes)},
            'checkpoint': str(ckpt),
        })
    (out/'summary.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    with (out/'samples.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    fig, axes = plt.subplots(1, 3 if model is not None else 2, figsize=(15, 4.3), layout='constrained')
    axes[0].hist(np.asarray(curvatures), bins=80); axes[0].set_title('Target-lane signed curvature'); axes[0].set_xlabel('Curvature [1/m]')
    axes[1].hist(np.asarray(expert_y), bins=60); axes[1].set_title('Expert terminal lateral displacement'); axes[1].set_xlabel('Terminal y [m]')
    if model is not None:
        axes[2].hist(np.asarray(pred_error), bins=60); axes[2].axvline(0, color='k', linestyle='--')
        axes[2].set_title('Predicted terminal lateral error'); axes[2].set_xlabel('y_pred - y_expert [m]')
    for ax in axes: ax.grid(alpha=.25)
    fig.savefig(out/'distributions.png', dpi=180); plt.close(fig)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f'[DONE] {out.resolve()}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
