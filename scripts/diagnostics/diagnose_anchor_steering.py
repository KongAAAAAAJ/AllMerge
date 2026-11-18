"""Plot A0/A1 controls and rollouts for one current raw expert scene."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from highway_env.planner.a1_anchor_manager import A1AnchorManager
from highway_env.planner.fixed_anchor_manager import FixedAnchorManager
from highway_env.planner.diffusion.config.experiment import load_mode_yaml, resolve_repo_file
from data_pipeline.raw_expert_dataset import raw_episode_paths


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', default='outputs/expert_dataset/expert_trucksim_k2_raw_100k')
    p.add_argument('--a0-bank', default='outputs/anchor_ablation/ckpt/A0_anchors/A0_control_anchors.npz')
    p.add_argument('--diffusion-config', default='configs/train/diffusion.yaml')
    p.add_argument('--episode', type=int, default=469)
    p.add_argument('--frame', type=int, default=9)
    p.add_argument('--ego', type=int, default=0)
    p.add_argument('--output-dir', default='outputs/anchor_ablation/diagnostics/steering')
    args=p.parse_args(argv)
    root=resolve_repo_file(args.dataset_root)
    inventories = [raw_episode_paths(root, split)[0] for split in ('train','val','test')]
    paths=[path for group in inventories for path in group]
    row=None
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            if not np.any(data['episode_index']==args.episode):
                continue
            ids=np.flatnonzero((data['episode_index']==args.episode)&
                               (data['frame_index']==args.frame)&(data['ego_index']==args.ego))
            if len(ids)==1:
                row={key:np.array(data[key][ids[0]],copy=True) for key in data.files}
                break
    if row is None:
        raise ValueError('Requested raw episode/frame/ego was not found')
    cfg=load_mode_yaml(args.diffusion_config)
    dynamics=cfg['control']
    count=int(cfg['model'].get('num_modes',10))
    common=dict(dt=dynamics['dt'],wheelbase_m=dynamics['wheelbase_m'],
                max_accel_mps2=dynamics['max_accel_mps2'],max_steer_rad=dynamics['max_steer_rad'])
    a1=A1AnchorManager(cfg['anchors'],num_selected=count,**common)
    a0=FixedAnchorManager({'bank_path':args.a0_bank},num_modes=count,**common)
    features={'ego_state':torch.from_numpy(row['ego_state'][None])}
    from data_pipeline.raw_expert_dataset import RawExpertDataset
    features.update({key:torch.as_tensor(row[key][None]) for key in RawExpertDataset.ROAD_KEYS})
    output=resolve_repo_file(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    report={}
    for i,(name,manager) in enumerate((('A0',a0),('A1',a1))):
        result=manager.select(features)
        controls=result['coarse_controls'][0].numpy()
        xy=result['coarse_trajectories'][0].numpy()
        for u,trajectory in zip(controls,xy):
            axes[i,0].plot(trajectory[:,0],trajectory[:,1])
            axes[i,1].plot(np.arange(40)*.1,u[:,0])
            axes[i,2].plot(np.arange(40)*.1,np.rad2deg(u[:,1]))
        axes[i,0].plot(row['expert_trajectory_xy'][:,0],row['expert_trajectory_xy'][:,1],'k--')
        for ax,title in zip(axes[i],('Trajectory (m)','Acceleration (m/s2)','Steering (deg)')):
            ax.set_title(f'{name}: {title}'); ax.grid(alpha=.25)
        report[name]=dict(num_anchors=len(controls),
            acceleration_range_mps2=[float(controls[:,:,0].min()),float(controls[:,:,0].max())],
            steering_range_rad=[float(controls[:,:,1].min()),float(controls[:,:,1].max())])
    fig.savefig(output/'anchor_steering.pdf',bbox_inches='tight'); plt.close(fig)
    (output/'anchor_steering.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
