"""Unified YAML-controlled trajectory/control diffusion training entry point.

Trajectory mode launches the unchanged original training script. Control mode
launches trajectory-only control diffusion; no pseudo-control labels are used.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from highway_env.planner.diffusion.config.experiment import load_mode_yaml, resolve_repo_file


def main(argv=None):
    parser = argparse.ArgumentParser(description='AllMerge YAML diffusion mode training')
    parser.add_argument('--config', default='configs/train/diffusion.yaml')
    parser.add_argument('--dataset-root', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--limit-samples', type=int, default=0)
    args = parser.parse_args(argv)
    cfg = load_mode_yaml(args.config)
    mode = cfg['output_space']
    if mode == 'control':
        from scripts.training.train_diffusion_control import train
        tr = cfg['training']
        root = resolve_repo_file(args.dataset_root or tr['dataset_root'])
        output = resolve_repo_file(args.output_dir or tr['output_dir'])
        print('[DIFFUSION MODE] control (40-step acceleration/steering; configurable dynamics and objectives)', flush=True)
        train(cfg, root=root, output_dir=output, max_steps=args.max_steps,
              max_epochs=args.max_epochs, limit_samples=args.limit_samples,
              device=args.device)
        return 0

    original = resolve_repo_file(args.config)
    command = [sys.executable, str(ROOT / 'scripts/training/train_diffusion_pretrain.py'),
               '--config', str(original)]
    for field, value in [('dataset-root', args.dataset_root),
                         ('output-dir', args.output_dir), ('max-steps', args.max_steps),
                         ('max-epochs', args.max_epochs), ('device', args.device),
                         ('limit-train-samples', args.limit_samples or None),
                         ('limit-val-samples', args.limit_samples or None)]:
        if value is not None:
            command.extend(['--' + field, str(value)])
    print('[DIFFUSION MODE] trajectory (original 8-waypoint training code)', flush=True)
    return subprocess.call(command, cwd=str(ROOT))


if __name__ == '__main__':
    raise SystemExit(main())
