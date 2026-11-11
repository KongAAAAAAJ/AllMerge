"""Unified six-model evaluation and strictly controlled 3-case paper chart.

  python highway_env/vehicle/koopman/scripts/run_evaluation.py compare --split val
  python highway_env/vehicle/koopman/scripts/run_evaluation.py evaluate --model K2 --split test
  python highway_env/vehicle/koopman/scripts/run_evaluation.py controlled --cases case_A.npz case_B.npz case_C.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description='Stage C shared Koopman paper evaluation')
    p.add_argument('command',choices=('evaluate','compare','controlled'))
    p.add_argument('--config',default=str(ROOT/'configs'/'koopman'/'experiment.yaml'))
    p.add_argument('--model',choices=('K0','K1-P','K1-F','K2','EDMDc','Parametric EDMDc'),default='K0')
    p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--data-root',default=None)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--chassis-seed',type=int,default=None,help='Reproducible 3-window chassis plot sampling seed')
    p.add_argument('--cases',nargs=3,default=None,metavar=('CASE_A','CASE_B','CASE_C'))
    p.add_argument('--output',default=None)
    p.add_argument('--offset-s',type=float,default=None)
    args=p.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    from highway_env.vehicle.koopman.evaluation.evaluate import (evaluate_one,compare,controlled_chassis)
    if args.command=='evaluate':
        evaluate_one(cfg,ROOT,args.model,args.split,args.data_root,args.device,args.batch_size)
    elif args.command=='compare':
        compare(cfg,ROOT,args.split,args.data_root,args.device,args.batch_size,args.chassis_seed)
    else:
        if not args.cases:p.error('--cases CASE_A CASE_B CASE_C is required for controlled')
        controlled_chassis(cfg,ROOT,args.cases,args.device,args.output,args.offset_s)
    return 0

if __name__=='__main__':raise SystemExit(main())
