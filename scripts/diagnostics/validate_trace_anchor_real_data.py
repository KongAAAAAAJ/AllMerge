"""Check online A1 batch/single parity and A2 inference on held-out raw scenes."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import json
import torch
from torch.utils.data import DataLoader, Subset
from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.planner.a1_anchor_manager import A1AnchorManager
from highway_env.planner.anchor_scoring.manager import AdaptiveAnchorManager
from highway_env.planner.diffusion.config.experiment import resolve_repo_file, load_mode_yaml


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--scorer-checkpoint', required=True)
    p.add_argument('--diffusion-config', default='configs/train/diffusion.yaml')
    p.add_argument('--a1-config', default='configs/anchors/a1.yaml')
    p.add_argument('--max-episodes', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--top-r', type=int, default=60)
    p.add_argument('--device', default='auto')
    p.add_argument('--output', default='outputs/anchor_ablation/real_data_runtime_validation.json')
    args = p.parse_args(argv)
    cfg = load_mode_yaml(args.diffusion_config)
    count = int(cfg['model'].get('num_modes', 10))
    dev = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                       else 'cpu' if args.device == 'auto' else args.device)
    a1 = A1AnchorManager({'a1_config': args.a1_config}, num_selected=count)
    checkpoint = resolve_repo_file(args.scorer_checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    a2 = AdaptiveAnchorManager({'scorer_checkpoint': str(checkpoint),
        'candidate_generation': saved['candidate_generation'], 'top_r': args.top_r},
        num_selected=count, device=dev)
    dataset = RawExpertDataset(resolve_repo_file(args.dataset_root), split='test')
    indices = [i for i,(file_index,_) in enumerate(dataset._index)
               if not args.max_episodes or file_index < args.max_episodes]
    checked = 0
    for features, _, _ in DataLoader(Subset(dataset,indices), batch_size=args.batch_size):
        features = {k:v.to(dev) for k,v in features.items()}
        batch = a1.select(features)
        for i in range(len(features['ego_state'])):
            single = a1.select({k:v[i:i+1] for k,v in features.items()})
            for key in ('coarse_controls','coarse_trajectories','mode_valid_mask'):
                torch.testing.assert_close(batch[key][i:i+1], single[key])
        selected = a2.select(features)
        assert selected['coarse_controls'].shape[1:] == (count,40,2)
        assert torch.isfinite(selected['coarse_controls']).all()
        checked += len(features['ego_state'])
    if not checked:
        raise ValueError('No raw scenes checked')
    result = dict(raw_samples_checked=checked, num_modes=count,
                  a1_batch_single_parity=True, a2_online=True, split=dataset.split_manifest)
    output=resolve_repo_file(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
