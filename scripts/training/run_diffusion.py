"""Train or resume control-space diffusion from a YAML experiment config."""
from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import time
import warnings
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from highway_env.planner.diffusion.config.experiment import load_mode_yaml, resolve_repo_file


def _checkpoint_epoch(path):
    import torch
    return int(torch.load(path, map_location='cpu', weights_only=True)['epoch'])


def _run_attempt(command):
    """A fresh interpreter owns CUDA and DataLoader resources for each attempt."""
    process = subprocess.Popen(command, cwd=ROOT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
        start_new_session=os.name != 'nt')
    try:
        return process.wait()
    except KeyboardInterrupt:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait()
        raise


def _auto_resume(cfg, train_args, initial_resume):
    """Normal exits (including patience) finish; only failed attempts restart."""
    output = Path(train_args['output_dir'])
    if initial_resume is None and output.exists() and any(output.glob('*.pt')):
        raise ValueError('--auto-resume from scratch requires an output directory without checkpoints')
    output.mkdir(parents=True, exist_ok=True)
    target = train_args['max_epochs']
    completed = _checkpoint_epoch(initial_resume) if initial_resume else 0
    last = output / 'last.pt'
    # An explicit initial resume may differ from an existing output checkpoint.
    # Do not adopt the old output file unless this run replaces it.
    old_last_stamp = (last.stat().st_mtime_ns, last.stat().st_size) if last.exists() else None
    snapshot = output / 'auto_resume_run.json'
    snapshot.write_text(json.dumps({'config': cfg, 'train_args': train_args,
                                   'initial_resume': initial_resume}, indent=2), encoding='utf-8')
    resume = initial_resume
    attempt = no_progress = 0
    try:
        while completed < target:
            attempt += 1
            command = [sys.executable, '-u', str(Path(__file__).resolve()),
                       '--_auto-worker', str(snapshot)]
            if resume:
                command.extend(['--resume', str(resume)])
            print(f'[AUTO RESUME] attempt={attempt} resume={resume or "scratch"} '
                  f'completed={completed}/{target} no_progress={no_progress}/3', flush=True)
            code = _run_attempt(command)
            latest = None
            if last.exists():
                stamp = (last.stat().st_mtime_ns, last.stat().st_size)
                if stamp != old_last_stamp or (initial_resume and Path(initial_resume) == last):
                    latest = last
            epoch = _checkpoint_epoch(latest) if latest else completed
            advanced = epoch > completed
            completed = max(completed, epoch)
            if code == 0:
                print(f'[AUTO RESUME] attempt={attempt} exit=0 completed={completed}/{target}; '
                      'normal completion (including patience early stopping)', flush=True)
                return 0
            no_progress = 0 if advanced else no_progress + 1
            print(f'[AUTO RESUME] attempt={attempt} exit={code} completed={completed}/{target} '
                  f'no_progress={no_progress}/3', flush=True)
            if completed >= target:
                break
            if no_progress >= 3:
                print('[AUTO RESUME] stopped: three failed attempts without a new saved epoch', flush=True)
                return 1
            resume = str(latest) if latest else initial_resume
            print(f'[AUTO RESUME] restarting in 10s from {resume or "scratch"}', flush=True)
            time.sleep(10)
        print(f'[AUTO RESUME] target reached: completed={completed}/{target}', flush=True)
        return 0
    except KeyboardInterrupt:
        print('[AUTO RESUME] interrupted by user; no restart', flush=True)
        return 130


def main(argv=None):
    parser = argparse.ArgumentParser(description='AllMerge YAML diffusion mode training')
    parser.add_argument('--auto-resume', action='store_true', help='Restart failed training processes from last.pt; preserve patience')
    parser.add_argument('--_auto-worker', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--config', default='configs/train/diffusion.yaml')
    parser.add_argument('--dataset-root', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--anchor-cache-dir', default=None, help='Read shared A1/A2 candidate pools')
    parser.add_argument('--resume', default=None, help='Resume model, optimizer and epoch history from last.pt')
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--limit-samples', type=int, default=0)
    args = parser.parse_args(argv)
    if args._auto_worker:
        run = json.loads(Path(args._auto_worker).read_text(encoding='utf-8'))
        from highway_env.planner.diffusion.training.control import train
        train(run['config'], **run['train_args'], resume=args.resume)
        return 0
    initial = resolve_repo_file(args.resume) if args.resume else None
    if initial:
        snapshot = initial.parent / 'training_run.json'
        if snapshot.is_file():
            run = json.loads(snapshot.read_text(encoding='utf-8'))
            if load_mode_yaml(args.config) != run['yaml_config']:
                warnings.warn(
                    'Current YAML differs from the original training configuration; '
                    'resuming with training_run.json. Current YAML changes are ignored.',
                    RuntimeWarning, stacklevel=2)
            print(f'[TRACE RESUME CONFIG] using original parameters from {snapshot}; '
                  'current training overrides are ignored', flush=True)
            if args.auto_resume:
                return _auto_resume(run['config'], run['train_args'], str(initial))
            from highway_env.planner.diffusion.training.control import train
            train(run['config'], **run['train_args'], resume=str(initial))
            return 0
        warnings.warn(
            'Missing training_run.json: original startup parameters are unavailable. '
            'Using the current YAML and CLI overrides for this resume and saving a new snapshot; '
            'checkpoint compatibility checks still apply.', RuntimeWarning, stacklevel=2)
    cfg = load_mode_yaml(args.config)
    mode = cfg['output_space']
    if mode != 'control':
        raise ValueError('Trajectory-space pretraining is retired; use output_space: control')
    from highway_env.planner.diffusion.training.control import train
    tr = cfg['training']
    root = resolve_repo_file(args.dataset_root or tr['dataset_root'])
    output = resolve_repo_file(args.output_dir or (initial.parent if initial else tr['output_dir']))
    import torch
    device = args.device or tr.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_args = {
        'root': str(root), 'output_dir': str(output),
        'max_steps': args.max_steps,
        'max_epochs': args.max_epochs if args.max_epochs is not None else int(tr.get('max_epochs', 50)),
        'limit_samples': args.limit_samples, 'device': device,
        'anchor_cache_dir': str(resolve_repo_file(args.anchor_cache_dir)) if args.anchor_cache_dir else None,
    }
    fixed_cfg = copy.deepcopy(cfg)
    fixed_cfg['training'].update(dataset_root=str(root), output_dir=str(output),
                                 max_epochs=train_args['max_epochs'], device=device)
    if args.auto_resume and initial is None and output.exists() and any(output.glob('*.pt')):
        raise ValueError('--auto-resume from scratch requires an output directory without checkpoints')
    output.mkdir(parents=True, exist_ok=True)
    if initial and (output / 'training_run.json').exists():
        raise FileExistsError('training_run.json already exists in the output directory; '
                              'remove it explicitly before adopting different resume parameters')
    (output / 'training_run.json').write_text(
        json.dumps({'config': fixed_cfg, 'train_args': train_args, 'yaml_config': cfg}, indent=2), encoding='utf-8')
    if args.auto_resume:
        return _auto_resume(fixed_cfg, train_args, str(initial) if initial else None)
    print('[DIFFUSION MODE] control (40-step acceleration/steering; configurable dynamics and objectives)', flush=True)
    if initial:
        train(fixed_cfg, **train_args, resume=str(initial))
        return 0
    train(cfg, root=root, output_dir=output, max_steps=args.max_steps,
          max_epochs=args.max_epochs, limit_samples=args.limit_samples,
          device=args.device, resume=args.resume,
          anchor_cache_dir=resolve_repo_file(args.anchor_cache_dir) if args.anchor_cache_dir else None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
