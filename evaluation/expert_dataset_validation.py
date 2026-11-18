"""Validate raw scenes and online A1 generation without a stored anchor bank."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.planner.a1_anchor_manager import A1AnchorManager
from highway_env.planner.diffusion.config.experiment import load_mode_yaml, resolve_repo_file


def validate_dataset(root, *, split='test', diffusion_config='configs/train/diffusion.yaml',
                     batch_size=16, max_samples=None):
    dataset = RawExpertDataset(root, split=split, max_samples=max_samples)
    cfg = load_mode_yaml(diffusion_config)
    dynamics = cfg['control']
    manager = A1AnchorManager(cfg.get('anchors'), num_selected=cfg['model'].get('num_modes', 10),
        control_steps=dynamics['horizon_steps'], dt=dynamics['dt'],
        wheelbase_m=dynamics['wheelbase_m'], max_accel_mps2=dynamics['max_accel_mps2'],
        max_steer_rad=dynamics['max_steer_rad'])
    count = 0
    for features, targets, _ in DataLoader(dataset, batch_size=batch_size, num_workers=0):
        for key, value in {**features, **targets}.items():
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise ValueError(f'Raw dataset contains nonfinite {key}')
        prepared = manager.select(features)
        assert prepared['mode_valid_mask'].all()
        count += len(features['ego_state'])
    if count == 0:
        raise ValueError('Selected raw split is empty')
    return dict(schema_version='trace_expert_raw_k2_v2', dataset_root=str(Path(root).resolve()),
                split=split, samples_checked=count, num_modes=manager.num_selected,
                a1_generation='online', a1_parameters=manager.builder_config,
                episode_split=dataset.split_manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default='outputs/expert_dataset/expert_trucksim_k2_raw_100k')
    parser.add_argument('--split', choices=('train','val','test'), default='test')
    parser.add_argument('--diffusion-config', default='configs/train/diffusion.yaml')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-loader-samples', type=int, default=0)
    parser.add_argument('--report-path', type=Path)
    args = parser.parse_args(argv)
    root = resolve_repo_file(args.dataset_root)
    result = validate_dataset(root, split=args.split, diffusion_config=args.diffusion_config,
                              batch_size=args.batch_size, max_samples=args.max_loader_samples or None)
    path = args.report_path or root/'reports/dataset_validation.json'
    path = resolve_repo_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
