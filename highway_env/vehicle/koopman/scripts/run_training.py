"""Six-model training orchestrator with one frozen, shared episode split.

Examples (execute from all_merge/):
  python highway_env/vehicle/koopman/scripts/run_training.py split
  python highway_env/vehicle/koopman/scripts/run_training.py train --model K0 --epochs 2
  python highway_env/vehicle/koopman/scripts/run_training.py train --model all
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description='Shared training and splits for six Koopman models')
    p.add_argument('command',choices=('split','train'))
    p.add_argument('--config',default=str(ROOT/'configs'/'koopman'/'experiment.yaml'))
    p.add_argument('--model',default='all',choices=('all','K0','K1-P','K1-F','K2','EDMDc','Parametric EDMDc'))
    p.add_argument('--data-root',default=None)
    p.add_argument('--epochs',type=int,default=None)
    p.add_argument('--device',default='auto',choices=('auto','cpu','cuda'))
    args=p.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    from highway_env.vehicle.koopman.evaluation.metrics import MODELS
    from highway_env.vehicle.koopman.data.shared_split import create,load
    if args.command=='split':
        create(cfg,ROOT,args.data_root)
        return 0
    splits,manifest=load(cfg,ROOT,args.data_root)
    kinds=MODELS if args.model=='all' else [args.model]
    for kind in kinds:
        if kind=='K0':
            from highway_env.vehicle.koopman.training.neural_train import train_neural
            train_neural('K0',cfg,ROOT,splits,manifest,epochs=args.epochs,device=args.device)
        elif kind in ('K1-P','K1-F','K2'):
            from highway_env.vehicle.koopman.training.conditional_train import train_conditioned
            train_conditioned(kind,cfg,ROOT,splits,manifest,epochs=args.epochs,device=args.device)
        else:
            from highway_env.vehicle.koopman.training.edmd_train import train_edmd
            train_edmd(kind,cfg,ROOT,splits,manifest)
    return 0

if __name__=='__main__':raise SystemExit(main())
