#!/usr/bin/env python3
"""Analyze how a 10k Student absorbs its frozen-Pretrain PGD teacher.

Use exact paired terminal t=0 input: comparisons are for the denoiser's
mode-locked response, NOT new independent full-chain diffusion sampling.
All writes stay in experiments/teacher_guide_train_results/outputs.
"""
from __future__ import annotations
import argparse,os,sys
from pathlib import Path

MODULE_DIR=Path(__file__).resolve().parent
ROOT=MODULE_DIR.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if str(MODULE_DIR.parent) not in sys.path:sys.path.insert(0,str(MODULE_DIR.parent))
from teacher_guide_train_results.io_utils import (digest,load_teachers,ensure_teacher_source,
                                     validate_reward_config,rows_csv,json_out)
from teacher_guide_train_results.collect import run_collect
from teacher_guide_train_results.selectors import select_topk
from teacher_guide_train_results.gallery import render_one,contact_sheet
from teacher_guide_train_results.plot_utils import draw_summary


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-file',type=Path,required=True)
    parser.add_argument('--pretrain-ckpt',type=Path,required=True)
    parser.add_argument('--posttrain-ckpt',type=Path,required=True)
    parser.add_argument('--run-name',default='online_pgd_10k')
    parser.add_argument('--top-k',type=int,default=12)
    parser.add_argument('--max-states',type=int,default=0,help='0: all eligible states; 3: quick smoke')
    parser.add_argument('--min-gain',type=float,default=0.)
    parser.add_argument('--min-shape-m',type=float,default=.05)
    parser.add_argument('--min-closer-m',type=float,default=0.)
    parser.add_argument('--replay-tol-m',type=float,default=.002)
    parser.add_argument('--device',default='auto',choices=('auto','cpu','cuda'))
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--skip-scene-verification',action='store_true',
                        help='For diagnostic use only; report records this reduced check')
    parser.add_argument('--output-dir',type=Path,help='Must remain within this module directory')
    return parser.parse_args()


def main():
    args=cli()
    if args.top_k<1 or args.max_states<0 or args.replay_tol_m<=0:raise ValueError('Invalid count / tolerance')
    for attr in ('teacher_file','pretrain_ckpt','posttrain_ckpt'):
        p=getattr(args,attr).expanduser().resolve()
        if not p.is_file():raise FileNotFoundError(f'--{attr.replace("_","-")}: {p}')
        setattr(args,attr,p)
    out=(args.output_dir or (MODULE_DIR/'outputs'/args.run_name)).expanduser().resolve()
    if out!=MODULE_DIR and MODULE_DIR not in out.parents:
        raise RuntimeError('Output must stay inside experiments/teacher_guide_train_results/')
    import torch
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA selected but unavailable')
    teachers=load_teachers(args.teacher_file)
    sha=ensure_teacher_source(teachers['meta'],args.pretrain_ckpt)
    cfg=validate_reward_config(teachers['meta'])
    count=sum(bool(s.get('teachers')) for s in teachers['states'])
    if not count:raise RuntimeError('Teacher bank has no safety-accepted states')
    print('[preflight] teacher states=',count,'teacher_origin=initial_frozen_pretrain',
          'source_sha=',sha[:16], 'device=',device,flush=True)
    print('[preflight] posttrain=',args.posttrain_ckpt,flush=True)
    if args.preflight_only:return 0
    rows,samples,skipped=run_collect(teachers,args.pretrain_ckpt,args.posttrain_ckpt,cfg,device,
        replay_tol=args.replay_tol_m,limit=args.max_states,verify_scene=not args.skip_scene_verification)
    if not rows:raise RuntimeError('No states could be reproduced; inspection needed: '+str(skipped[:5]))
    selected,info=select_topk(rows,k=args.top_k,min_gain=args.min_gain,
                              min_shape_m=args.min_shape_m,min_closer_m=args.min_closer_m)
    out.mkdir(parents=True,exist_ok=True)
    rows_csv(out/'per_state_metrics.csv',rows)
    rows_csv(out/'selected_topk.csv',selected,columns=list(selected[0]) if selected else list(rows[0]))
    rows_csv(out/'skipped_states.csv',skipped,columns=['state_index','reason'])
    draw_summary(rows,out)
    gallery_dir=out/'topk_gallery';gallery_dir.mkdir(exist_ok=True)
    for r in selected:
        path=gallery_dir/f"rank_{r['selection_rank']:02d}_{r['scenario']}_state_{r['state_index']:04d}.png"
        render_one(r,samples[r['state_index']],path)
        print('[figure]',path,flush=True)
    if selected:contact_sheet(selected,samples,out/'topk_contact_sheet.png')
    else:print('[NOTICE] No states meet all strict improvement thresholds; all-state CSV and histograms saved. '
               'Consider inspecting rows or lowering --min-shape-m, never auto-select negative improvements.',flush=True)
    import numpy as np
    summary={'experiment':'teacher_guide_train_results_v1','kind':'paired_t0_teacher_absorption',
        'reward':'unified_v6','reward_source':'DifferentiableTaskReward.components[balanced] (same PGD V6 objective)',
        'safety_scope':'cached teacher safety gate; reconstructed road margin/curvature per trajectory; '
                       'collision and TTC NOT recomputed',
        'source_teacher_meta':{k:teachers['meta'].get(k) for k in ('kind','checkpoint_sha256','teacher_source',
            'source_step','noise_pairing_schema','reward_config_sha256','teacher_count')},
        'pretrain_sha256':sha,'posttrain_sha256':digest(args.posttrain_ckpt),'teacher_sha256':digest(args.teacher_file),
        'paired_replay_tolerance_m':args.replay_tol_m,'scene_verified':not args.skip_scene_verification,
        'total_accepted_teacher_states':count,'state_count_evaluated':len(rows),'skipped':len(skipped),
        'selection':info,'mean_train_reward_gain':float(np.mean([r['train_reward_gain'] for r in rows])),
        'mean_teacher_distance_improve_m':float(np.mean([r['teacher_distance_improve_m'] for r in rows])),
        'note':'Paired t0 tests terminal denoiser on source frozen input; not a full DDIM rollout performance claim.'}
    json_out(out/'summary.json',summary)
    print('[OK]',out,'evaluated',len(rows),'selected',len(selected),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
