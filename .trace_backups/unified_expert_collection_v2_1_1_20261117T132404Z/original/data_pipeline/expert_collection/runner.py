"""Resumeable multi-episode TruckSim expert collection with native crash isolation.

One episode per OS process, all three TruckSim DLL instances unique.  A Python
heap-corruption exit after writing valid data is logged, but a missing/invalid
shard is never accepted. No os._exit, no reusing VehicleSim sessions.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.expert_collection.trucksim import load_collection_config, physical_cases


def parser(argv=None):
    from .config import parse_args
    return parse_args(argv)


def validate_shards(root: Path):
    """Validate complete, atomic shard files before accepting a native worker."""
    paths = sorted((root/'shards').glob('shard_*.npz'))
    total = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            required = ('expert_trajectory_xy','future_trajectory_dense',
                        'koopman_state_hist_si','koopman_control_hist_si',
                        'koopman_pose0_si','koopman_payload_kg', 'koopman_road_mu',
                        'episode_index')
            missing = sorted(set(required)-set(z.files))
            if missing:
                raise ValueError(f'Invalid {path}: missing {missing}')
            n = int(z['expert_trajectory_xy'].shape[0])
            shapes = {
                'future_trajectory_dense': (n,40,2),
                'koopman_state_hist_si': (n,5,5),
                'koopman_control_hist_si': (n,4,2),
                'koopman_pose0_si': (n,3),
            }
            for key, shape in shapes.items():
                if z[key].shape != shape or not np.isfinite(z[key]).all():
                    raise ValueError(f'Invalid {path}: {key} expected finite {shape}')
            for key in ('koopman_payload_kg','koopman_road_mu','episode_index'):
                if len(z[key]) != n:
                    raise ValueError(f'Invalid {path}: {key} length mismatch')
            total += n
    return total, len(paths)


def _dump_json_atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def main(args=None):
    args = args if args is not None else parser()
    if args.target_samples <= 0 or args.duration <= 0 or args.max_episodes <= 0:
        raise ValueError('target_samples, duration and max_episodes must be positive')
    if args.retries < 1:
        raise ValueError('retries must be >= 1')
    koopman_path = (ROOT / args.koopman_config).resolve()
    models = list(physical_cases(load_collection_config(koopman_path)))
    if not models:
        raise RuntimeError('No verified physical TruckSim cases')
    dataset = (ROOT / args.output_root / args.dataset_name).resolve()
    if args.check_only:
        for case in models:
            if not Path(case['simfile']).is_file() or not Path(case['dll']).is_file():
                raise FileNotFoundError(f"Missing physical model {case['model_id']}")
        print(f'[PLAN] {len(models)} payload/friction plants; 3 independent DLLs/episode; '
              f'4 unchanged scenarios; target_samples={args.target_samples}; '
              f'output={dataset}')
        return 0
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset/'shards').mkdir(exist_ok=True)
    # Parse the calibrated engine map CSV ONCE in this batch controller.
    # Workers load a small read-only .npy copy instead of parsing the CSV.
    reference = load_collection_config(koopman_path).get('reference_controller', {})
    engine_csv = reference.get('engine_map_path')
    if engine_csv is None:
        engine_csv = ROOT/'highway_env/vehicle/trucksim_simulation/engine_map_4455kg.csv'
    else:
        engine_csv = Path(engine_csv)
        if not engine_csv.is_absolute(): engine_csv = ROOT/engine_csv
    engine_map = np.loadtxt(engine_csv, delimiter=',',skiprows=1)
    if (engine_map.ndim != 2 or engine_map.shape[1] != 12 or
            not np.isfinite(engine_map).all() or
            np.any(np.diff(engine_map[:,0]) <= 0)):
        raise ValueError('Invalid engine map RPM x 11 throttle CSV')
    cache_path = dataset/'engine_map_cache.npy'
    if cache_path.exists():
        previous = np.load(cache_path, allow_pickle=False)
        if not np.array_equal(previous, engine_map):
            raise RuntimeError('Engine map differs from prior run. Use a fresh dataset name.')
    else:
        np.save(cache_path, engine_map)
    logs = dataset/'native_logs'
    logs.mkdir(exist_ok=True)
    progress_file = dataset/'collection_progress.json'
    audit_csv = dataset/'episode_audit.csv'
    accepted, shards = validate_shards(dataset)
    if not args.resume and (shards or progress_file.exists()):
        raise FileExistsError('Existing TruckSim expert dataset: pass --resume or use a new name')
    if args.resume and progress_file.exists():
        progress = json.loads(progress_file.read_text(encoding='utf-8'))
        next_episode = int(progress['next_episode_index'])
    elif args.resume and shards:
        # The shard split contract is one episode per shard.
        existing_max = -1
        for path in (dataset/'shards').glob('shard_*.npz'):
            with np.load(path, allow_pickle=False) as z:
                existing_max = max(existing_max, int(np.max(z['episode_index'])))
        next_episode = existing_max + 1
    else:
        next_episode = int(args.start_episode_index)
    if not audit_csv.exists():
        with audit_csv.open('w', encoding='utf-8', newline='') as out:
            csv.writer(out).writerow(['episode','case','scenario','accepted_samples',
                                       'returncode','attempt','reason'])
    attempts = 0
    while accepted < args.target_samples and attempts < args.max_episodes:
        episode = next_episode
        case = models[episode % len(models)]
        scenarios = getattr(args, 'scenarios', ['straight','curved','merge_in','merge_out'])
        scenario = scenarios[episode % len(scenarios)] if args.scenario == 'all' else args.scenario
        ok = False
        final_status = 'failed'
        for attempt in range(1, args.retries+1):
            cmd = [sys.executable, '-u', '-X', 'faulthandler',
                   str(ROOT/'scripts/data/collect_expert_dataset.py'), '--worker',
                   '--trucksim-koopman-config', str(koopman_path),
                   '--max-episodes','1', '--episode-index',str(episode), '--defer-split',
                   '--scenario',scenario,'--duration',str(args.duration),
                   '--start-seed',str(args.start_seed),'--target-samples',str(args.target_samples),
                   '--samples-per-shard',str(args.samples_per_shard),
                   '--group-action',str(getattr(args, 'group_action', 3)),
                   '--train-split-ratio',str(getattr(args, 'train_split_ratio', .8)),
                   '--val-split-ratio',str(getattr(args, 'val_split_ratio', .1)),
                   '--test-split-ratio',str(getattr(args, 'test_split_ratio', .1)),
                   '--split-seed',str(getattr(args, 'split_seed', 0)),
                   '--output-root',str((ROOT/args.output_root).resolve()),
                   '--engine-map-cache',str(cache_path),
                   '--dataset-name',args.dataset_name,'--resume']
            if args.prepare_dll_instances:
                cmd.append('--copy-trucksim-dll-instances')
            print(f'[EPISODE] index={episode} plant={case["model_id"]} '
                  f'scenario={scenario} attempt={attempt} accepted={accepted}/{args.target_samples}',flush=True)
            status = -1
            stdout_path = logs/f'episode_{episode:06d}_attempt{attempt}.stdout.log'
            stderr_path = logs/f'episode_{episode:06d}_attempt{attempt}.stderr.log'
            with stdout_path.open('w',encoding='utf-8') as out, stderr_path.open('w',encoding='utf-8') as err:
                try:
                    p = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err,
                                       timeout=args.timeout_s, check=False)
                    status = p.returncode
                except subprocess.TimeoutExpired:
                    status = -999
            try:
                new_count, new_shards = validate_shards(dataset)
                if new_count > accepted:
                    final_status = 'accepted_after_exit_'+str(status) if status else 'accepted'
                    with audit_csv.open('a',encoding='utf-8',newline='') as out:
                        csv.writer(out).writerow([episode,case['model_id'],scenario,
                                                  new_count-accepted,status,attempt,final_status])
                    accepted, shards = new_count,new_shards
                    ok = True
                    break
                final_status=f'no_complete_shard_exit_{status}'
            except Exception as error:
                final_status=f'failed_validation_{type(error).__name__}_{error}'
            print(f'[WORKER FAIL] {final_status}; logs: {stderr_path}',flush=True)
        if not ok:
            with audit_csv.open('a',encoding='utf-8',newline='') as out:
                csv.writer(out).writerow([episode,case['model_id'],scenario,0,status,args.retries,final_status])
        next_episode += 1
        attempts += 1
        _dump_json_atomic(progress_file, {'next_episode_index':next_episode,
                 'accepted_samples':accepted,'shard_count':shards,
                 'target_samples':args.target_samples,
                 'koopman_config':str(koopman_path), 'schema':'trucksim_k2_v1'})
    if shards:
        # Each shard contains one episode (worker writes then exits).
        splits_cmd=[sys.executable,str(ROOT/'scripts/data/split_expert_dataset.py'),
                    '--dataset-root',str(dataset),
                    '--train-ratio',str(getattr(args, 'train_split_ratio', .8)),
                    '--val-ratio',str(getattr(args, 'val_split_ratio', .1)),
                    '--test-ratio',str(getattr(args, 'test_split_ratio', .1)),
                    '--seed',str(getattr(args, 'split_seed', 0))]
        split_run=subprocess.run(splits_cmd,cwd=ROOT,check=False)
        if split_run.returncode:
            raise RuntimeError('Could not create final episode-disjoint shard splits')
    _dump_json_atomic(dataset/'reports_manifest_trucksim.json', {
       'plant':'TruckSim longitudinal_lateral','num_physical_cases':len(models),
       'scenarios':'unchanged polynomial four scenarios','samples':accepted,
       'shards':shards,'target':args.target_samples,'last_episode':next_episode-1,
       'schema_version':'trucksim_k2_v1','context_keys':[
           'koopman_state_hist_si','koopman_control_hist_si','koopman_pose0_si',
           'koopman_ltr0','koopman_payload_kg','koopman_road_mu']})
    print(f'[DONE] accepted {accepted}/{args.target_samples} samples in {shards} shards')
    return 0 if accepted >= args.target_samples else 2


if __name__=='__main__':
    raise SystemExit(main())
