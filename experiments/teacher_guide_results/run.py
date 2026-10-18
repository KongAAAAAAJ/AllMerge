"""PGD teacher quality experiment/visualization. This is NOT a separate trainer.

Launch native code via `train_grpo.py --method post_guide_distillation --pgd-stage collect`.
Afterwards inspect frozen paired teacher artifact and draw representative state panels.
"""
from __future__ import annotations
import argparse,csv,json,sys,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def visualize(teacher_file:Path, output:Path, limit:int=9):
    import torch,numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(ROOT))
    from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
    from highway_env.planner.diffusion.trajectory_mode_reward.spline_dense import evaluate_dense_spline
    data=torch.load(teacher_file,map_location='cpu',weights_only=True)
    states=[s for s in data['states'] if s['teachers']]
    if not states:raise RuntimeError('No eligible teachers to visualize')
    out=output;out.mkdir(parents=True,exist_ok=True)
    cols=['state','scene','pretrain_teacher_ade_m','max_delta_x_m','max_delta_y_m',
          'reward_gain','teacher_count']
    rows=[]
    fig,axes=plt.subplots(3,3,figsize=(16,14))
    for idx,s in enumerate(states[:limit]):
        t=max(s['teachers'],key=lambda o:o['reward_after']);pre=t['pretrain_xy'];target=t['target']
        vel=s['features']['ego_state'][:2].detach().float()
        with torch.no_grad():
            p,_v,_a=evaluate_dense_spline(pre.unsqueeze(0),TrajectoryModeRewardConfig(),initial_velocity_xy=vel)
            q,_v,_a=evaluate_dense_spline(target.unsqueeze(0),TrajectoryModeRewardConfig(),initial_velocity_xy=vel)
        P,Q=p[0].numpy(),q[0].numpy()
        dx=(Q-P).max(axis=0)
        delta=target.numpy()-pre.numpy()
        rows.append({'state':idx,'scene':s['scenario'],
                     'pretrain_teacher_ade_m':float(np.linalg.norm(Q-P,axis=1).mean()),
                     'max_delta_x_m':float(np.abs(Q[:,0]-P[:,0]).max()),
                     'max_delta_y_m':float(np.abs(Q[:,1]-P[:,1]).max()),
                     'reward_gain':float(t['reward_gain']),
                     'teacher_count':len(s['teachers'])})
        a=axes.flat[idx]
        a.plot(P[:,0],P[:,1],'--',label='Pretrain 41pts')
        a.plot(Q[:,0],Q[:,1],'-',label='Teacher 41pts')
        a.scatter(pre[:,0],pre[:,1],s=10)
        a.scatter(target[:,0],target[:,1],s=13)
        a.set_title(f"{s['scenario']} #{idx}  Gain {t['reward_gain']:+.3f}")
        a.axis('equal');a.grid(alpha=.2)
        if idx==0:a.legend(fontsize=8)
    for a in axes.flat[min(len(states),limit):]:a.axis('off')
    fig.tight_layout();fig.savefig(out/'pretrain_teacher_3x3.png',dpi=160);plt.close(fig)
    with (out/'teacher_quality.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps({'teacher_states':len(states),'plotted':len(rows),
        'teacher_file':str(teacher_file),'projector_parity':'unverified_original_W1_missing'},indent=2))
    print('[TEACHER GUIDE] plots and metrics:',out)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--method',choices=('a2_v2','b1_adaptive','b2_warm_search'),default='b2_warm_search')
    p.add_argument('--states',type=int,default=20)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/posttrain_v6/teacher_guide_results')
    p.add_argument('--teacher-file',type=Path)
    a=p.parse_args()
    target=(a.teacher_file or a.output/a.method/'teachers.pt').resolve()
    if not target.is_file():
        cmd=[sys.executable,str(ROOT/'train_grpo.py'),'--config',str(a.config.resolve()),
             '--method','post_guide_distillation','--pgd-stage','collect',
             '--pgd-strategy',a.method,'--pgd-train-states',str(a.states),
             '--pgd-teacher-file',str(target)]
        subprocess.run(cmd,check=True)
    visualize(target,a.output/a.method)
if __name__=='__main__':main()
