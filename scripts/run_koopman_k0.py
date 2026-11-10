"""Isolated K0 B-1 CLI; never imports or changes the running TruckSim collector.

From D:\\KONG_Files\\AllMerge:
  python all_merge\\scripts\\run_koopman_k0.py train --data-root all_merge\\outputs\\koopman\\data\\YOUR_BATCH --epochs 2
  python all_merge\\scripts\\run_koopman_k0.py check --checkpoint outputs\\koopman\\checkpoints\\k0\\best.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def main():
    p = argparse.ArgumentParser(description='TRACE Koopman Stage B-1: K0 model/training')
    p.add_argument('command', choices=['train','check'])
    p.add_argument('--config', default=str(ROOT/'configs'/'koopman_k0.yaml'))
    p.add_argument('--data-root', default=None, help='Collection directory; overrides YAML data.roots')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default=None)
    p.add_argument('--checkpoint', default=None)
    args = p.parse_args()
    from highway_env.vehicle.koopman.train import load_config, train_k0, load_checkpoint, resolve_path
    cfg = load_config(args.config)
    if args.command == 'train':
        train_k0(cfg, ROOT, data_root=args.data_root, epochs=args.epochs, device=args.device)
    else:
        import torch
        checkpoint = resolve_path(ROOT,args.checkpoint or str(Path(cfg['outputs'].get('checkpoints', 'outputs/koopman/checkpoints/k0'))/'best.pt'))
        dev = args.device or 'cpu'
        if dev == 'auto': dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, ckpt = load_checkpoint(checkpoint, dev)
        n = model.history_steps
        x = model.state_mean.expand(2,n,5).clone()
        h = model.control_mean.expand(2,n-1,2).clone()
        u = model.control_mean.expand(2,40,2).clone()
        with torch.no_grad(): y,z = model(x,h,u)
        assert y.shape == (2,41,5) and z.shape == (2,41,model.latent_dim)
        assert torch.isfinite(y).all()
        print(f'[K0 CHECK] checkpoint={checkpoint} epoch={ckpt["epoch"]} '
              f'states={tuple(y.shape)} latent={tuple(z.shape)} finite=True')
    return 0

if __name__ == '__main__': raise SystemExit(main())
