#!/usr/bin/env python3
"""Run 3 reproducible inference seeds; merge *unselected* candidate results into CSV.

State / DDIM noise / random-control seeds are independently varied. The same
checkpoint, reward implementation, amplitudes and model architecture are fixed.
"""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path


def get_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/reward_guided_mean_v5'))
    p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--samples-per-scenario',type=int,default=20)
    p.add_argument('--group-size',type=int,default=8)
    p.add_argument('--step-sizes-m',default='0,0.10,0.30,0.60,1.0,2.0,4.0,8.0,12.0,16.0')
    p.add_argument('--gallery-amplitudes-m',default='0.10,0.30,0.60,1.0,2.0')
    p.add_argument('--random-repeats',type=int,default=2)
    p.add_argument('--gradient-axis',choices=('xy','x','y'),default='xy')
    p.add_argument('--max-x-shift-m',type=float,default=20)
    p.add_argument('--max-y-shift-m',type=float,default=6)
    p.add_argument('--max-rms-shift-m',type=float,default=16)
    p.add_argument('--scenario',choices=('all','straight','curved','merge_in','merge_out'),default='all')
    p.add_argument('--resume',action='store_true',help='Skip completed seed folders only when configuration matches')
    p.add_argument('--plot-only',action='store_true',help='Render plots from existing combined CSV; no CUDA needed')
    p.add_argument('--no-plot',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    seeds=[int(v.strip()) for v in args.seeds.split(',') if v.strip()]
    if not seeds or len(seeds)!=len(set(seeds)) or any(s<0 for s in seeds):p.error('Seeds must be unique nonnegative integers')
    if args.samples_per_scenario<1 or args.group_size<2 or args.random_repeats<1:p.error('Invalid sizes')
    return args,seeds


def combine_csv(paths,destination,*,seed_field='seed'):
    # Stream rows to keep memory bounded even for large multiseed scans.
    fields=None;counts={}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('w',encoding='utf-8-sig',newline='') as fd:
        writer=None
        for seed,path in paths:
            if not path.is_file():raise FileNotFoundError(f'Seed {seed} has no file: {path}')
            with path.open(encoding='utf-8-sig',newline='') as fi:
                reader=csv.DictReader(fi)
                cols=reader.fieldnames
                if not cols or seed_field not in cols:raise ValueError(f'Missing {seed_field} column in {path}')
                if fields is None:
                    fields=[seed_field]+[n for n in cols if n!=seed_field]
                    writer=csv.DictWriter(fd,fieldnames=fields)
                    writer.writeheader()
                if set(cols)!=set(fields):raise ValueError(f'CSV schema mismatch in {path}')
                count=0
                for row in reader:
                    if row[seed_field]!=str(seed):
                        raise ValueError(f'Seed mismatch expected {seed}, got {row[seed_field]} in {path}')
                    writer.writerow(row)
                    count+=1
                if count==0:raise ValueError(f'Empty seed CSV {path}')
                counts[str(seed)]=count
    return counts


def main(argv=None):
    args,seeds=get_args(argv)
    root=Path(args.root).resolve()
    if not (root/'highway_env/planner/diffusion/grpo/sampling.py').is_file():
        raise FileNotFoundError('Not a guideDistill worktree: '+str(root))
    out=args.output_dir.resolve()
    if args.plot_only:
        from .data_plot import plot_results
        plot_results(out)
        return
    if args.checkpoint is None:raise ValueError('--checkpoint is required; use the same frozen ckpt for all seeds')
    ckpt=args.checkpoint.resolve()
    if not args.dry_run and not ckpt.is_file():raise FileNotFoundError(ckpt)
    out.mkdir(parents=True,exist_ok=True)
    plan=[]
    for seed in seeds:
        # Distinct from other scenario offsets (100000) and random candidate indexes.
        base=seed*1000003
        config=dict(seed=seed,environment_seed=20261017+base,noise_seed=70261017+base,
                    random_seed=90261017+base,gallery_seed=80261017+base,
                    checkpoint=str(ckpt),root=str(root),
                    scenario=args.scenario,samples_per_scenario=args.samples_per_scenario,
                    group_size=args.group_size,step_sizes_m=args.step_sizes_m,
                    gallery_amplitudes_m=args.gallery_amplitudes_m,
                    random_repeats=args.random_repeats,gradient_axis=args.gradient_axis,
                    max_x_shift_m=args.max_x_shift_m,max_y_shift_m=args.max_y_shift_m,
                    max_rms_shift_m=args.max_rms_shift_m)
        folder=out/f'seed_{seed}'
        marker=folder/'RUN_COMPLETE.json'
        if args.resume and marker.is_file():
            old=json.loads(marker.read_text(encoding='utf-8'))
            if old!=config:raise RuntimeError(f'Seed {seed} marked complete with DIFFERENT config: {marker}')
            if all((folder/n).exists() for n in ['samples.csv','trajectories.csv','summary.json']):
                print('[RESUME] seed',seed,'completed');plan.append((seed,folder,config,True));continue
        plan.append((seed,folder,config,False))
    for seed,folder,conf,skip in plan:
        if skip:continue
        command=[sys.executable,'-m','experiments.grade_duide_results.collector',
            '--root',str(root),'--checkpoint',str(ckpt),'--output-dir',str(folder),
            '--scenario',args.scenario,'--samples-per-scenario',str(args.samples_per_scenario),
            '--group-size',str(args.group_size),'--step-sizes-m',args.step_sizes_m,
            '--gallery-amplitudes-m',args.gallery_amplitudes_m,
            '--max-x-shift-m',str(args.max_x_shift_m),'--max-y-shift-m',str(args.max_y_shift_m),
            '--max-rms-shift-m',str(args.max_rms_shift_m),
            '--random-repeats',str(args.random_repeats),
            '--gradient-axis',args.gradient_axis,
            '--run-seed',str(seed),'--seed',str(conf['environment_seed']),
            '--noise-seed',str(conf['noise_seed']), '--random-seed',str(conf['random_seed']),
            '--gallery-seed',str(conf['gallery_seed']), '--no-plots']
        print('[SEED]',seed,' '.join(command),flush=True)
        if args.dry_run:continue
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'RUN_COMPLETE.json').unlink(missing_ok=True)
        subprocess.run(command,cwd=str(root),check=True)
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if summary.get('run_seed')!=seed:raise RuntimeError('Seed id mismatch')
        (folder/'RUN_COMPLETE.json').write_text(json.dumps(conf,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.dry_run:
        print('[DRY RUN] Planned seeds:',seeds);return
    for fname in ('samples.csv','trajectories.csv'):
        counts=combine_csv([(seed,out/f'seed_{seed}'/fname) for seed in seeds],out/fname)
        print('[MERGED]',out/fname,counts,flush=True)
    # Save metadata separate from the primary CSV; no metrics changed.
    (out/'seed_manifest.json').write_text(json.dumps({'seeds':seeds,'settings':[p[2] for p in plan]},indent=2),encoding='utf-8')
    if not args.no_plot:
        from .data_plot import plot_results
        plot_results(out)
    print('[PASS] Three-seed V5 complete; CSV is the plotting single source of truth:',out)

if __name__=='__main__':main()
