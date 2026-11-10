"""K0 paper B-2: new training/check/evaluation CLI (never loads B-1 checkpoints).

Run from D:\KONG_Files\AllMerge, with --data-root relative to all_merge/.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description='TRACE Koopman paper K0: train, check, evaluate')
    p.add_argument('command', choices=['train','check','evaluate'])
    p.add_argument('--config',default=str(ROOT/'configs'/'koopman_k0_paper.yaml'))
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--data-root',default=None,help='Accepted NPZ root relative to all_merge/')
    p.add_argument('--epochs',type=int,default=None)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--output-dir',default=None)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--examples',type=int,default=3)
    args=p.parse_args()
    import torch
    from highway_env.vehicle.koopman.train import (load_config,train_k0,load_checkpoint,
                                                    resolve_path,choose_device)
    cfg=load_config(args.config)
    default_ckpt=Path(cfg['outputs']['checkpoints'])/'best.pt'
    ckpt_path=resolve_path(ROOT,args.checkpoint or default_ckpt)
    if args.command=='train':
        train_k0(cfg,ROOT,data_root=args.data_root,epochs=args.epochs,device=args.device)
    elif args.command=='check':
        model, ckpt=load_checkpoint(ckpt_path,choose_device(args.device))
        n=model.history_steps
        x=model.state_mean.expand(2,n,5).clone()
        h=model.control_mean.expand(2,n-1,2).clone()
        u=model.control_mean.expand(2,40,2).clone()
        with torch.no_grad():
            out=model.predict_all(x,h,u,torch.zeros(2,3,device=x.device),.1)
        assert out['states'].shape==(2,41,5)
        assert out['ltr'].shape==(2,41)
        assert out['pose'].shape==(2,41,3)
        assert all(torch.isfinite(v).all() for v in out.values())
        print(f"[K0 PAPER CHECK] epoch={ckpt['epoch']} states={tuple(out['states'].shape)} "
              f"ltr={tuple(out['ltr'].shape)} pose={tuple(out['pose'].shape)} finite=True")
    else:
        from highway_env.vehicle.koopman.evaluate import evaluate_k0
        evaluate_k0(ckpt_path,ROOT,split=args.split,data_root=args.data_root,
            output_dir=args.output_dir,device=args.device,batch_size=args.batch_size,
            examples=args.examples)
    return 0

if __name__=='__main__':raise SystemExit(main())
