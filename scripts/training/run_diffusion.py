"""Train or resume control-space diffusion from a YAML experiment config."""
from __future__ import annotations

import argparse
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
    parser.add_argument('--resume', default=None, help='Resume model, optimizer and epoch history from last.pt')
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--limit-samples', type=int, default=0)
    args = parser.parse_args(argv)
    cfg = load_mode_yaml(args.config)
    mode = cfg['output_space']
    if mode != 'control':
        raise ValueError('Trajectory-space pretraining is retired; use output_space: control')
    from highway_env.planner.diffusion.training.control import train
    tr = cfg['training']
    root = resolve_repo_file(args.dataset_root or tr['dataset_root'])
    output = resolve_repo_file(args.output_dir or tr['output_dir'])
    print('[DIFFUSION MODE] control (40-step acceleration/steering; configurable dynamics and objectives)', flush=True)
    train(cfg, root=root, output_dir=output, max_steps=args.max_steps,
          max_epochs=args.max_epochs, limit_samples=args.limit_samples,
          device=args.device, resume=args.resume)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
