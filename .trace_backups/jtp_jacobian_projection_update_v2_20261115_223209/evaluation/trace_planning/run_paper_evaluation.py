"""Unified TRACE paper-metric evaluation on representative registered scenarios.

Runs only genuinely implemented planners (TRACE and Expert Planner) in native
TruckSim. Cooperative OCP and TransFuser remain '--' until their actual planner
adapters are integrated; this runner never fabricates baseline measurements.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_SCENARIOS = 'straight,merge_in,curved'
AVAILABLE_METHODS = ('TRACE', 'Expert Planner')


def main(argv=None):
    p = argparse.ArgumentParser(description='Unified real-TruckSim paper evaluation')
    p.add_argument('--checkpoint', required=True, help='TRACE ControlDiffusion checkpoint')
    p.add_argument('--config', default='configs/train/diffusion.yaml')
    p.add_argument('--methods', default='TRACE,Expert Planner',
                   help='Currently implemented: TRACE,Expert Planner')
    p.add_argument('--scenarios', default=DEFAULT_SCENARIOS,
                   help='Comma-separated registered project scenarios')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--trucksim-case', default='L4_MU08')
    p.add_argument('--duration', type=float, default=5.0)
    p.add_argument('--group-action', type=int, default=3)
    p.add_argument('--koopman-config', default='configs/koopman/collect.yaml')
    p.add_argument('--jtp-config', default='configs/projection/jtp.yaml')
    p.add_argument('--jtp-runtime-config', default='configs/projection/jtp_runtime.yaml')
    p.add_argument('--copy-dlls', action='store_true')
    p.add_argument('--allow-fallback', action='store_true')
    p.add_argument('--figure-seed', type=int, default=0)
    p.add_argument('--figure-scenarios', default='straight,merge_in,curved')
    p.add_argument('--output-dir', default='outputs/evaluation/TRACE_paper_v2')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args(argv)
    methods = [x.strip() for x in a.methods.split(',') if x.strip()]
    scenarios = [x.strip() for x in a.scenarios.split(',') if x.strip()]
    bad = sorted(set(methods) - set(AVAILABLE_METHODS))
    if bad:
        p.error(f'Planner adapters not implemented in this source snapshot: {bad}. '
                f'Available genuine adapters: {AVAILABLE_METHODS}.')
    if 'TRACE' not in methods:
        p.error('TRACE must be included; --checkpoint is required.')
    if a.episodes < 1 or not scenarios:
        p.error('--episodes must be positive and --scenarios non-empty')
    root = Path(a.output_dir)
    run_dirs = []
    for method in methods:
        tag = method.replace(' ', '_')
        run_dir = root / 'episodes' / tag
        run_dirs.append(run_dir)
        cmd = [sys.executable, '-m', 'evaluation.trace_planning.closed_loop',
               '--method', method, '--config', a.config,
               '--scenarios', ','.join(scenarios), '--episodes', str(a.episodes),
               '--seed-start', str(a.seed_start), '--trucksim-case', a.trucksim_case,
               '--duration', str(a.duration), '--group-action', str(a.group_action),
               '--koopman-config', a.koopman_config, '--jtp-config', a.jtp_config,
               '--jtp-runtime-config', a.jtp_runtime_config,
               '--output-dir', str(run_dir)]
        if method == 'TRACE':
            cmd += ['--checkpoint', a.checkpoint]
        if a.copy_dlls: cmd.append('--copy-dlls')
        if a.allow_fallback: cmd.append('--allow-fallback')
        if a.check_only: cmd.append('--check-only')
        print('[RUN]', ' '.join(cmd), flush=True)
        if not a.check_only:
            subprocess.run(cmd, check=True)
    if a.check_only:
        print('[PASS] Unified evaluation preflight completed. No TruckSim episode was run.')
        return 0
    cmp_cmd = [sys.executable, '-m', 'evaluation.trace_planning.comparison']
    for d in run_dirs: cmp_cmd += ['--runs', str(d)]
    cmp_cmd += ['--output-dir', str(root / 'paper_results'),
                '--figure-seed', str(a.figure_seed),
                '--figure-scenarios', a.figure_scenarios]
    print('[AGGREGATE]', ' '.join(cmp_cmd), flush=True)
    subprocess.run(cmp_cmd, check=True)
    print(f'[DONE] Results: {root / "paper_results"}')
    print('[NOTE] Cooperative OCP and TransFuser remain blank until their real TruckSim adapters are supplied.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
