import numpy as np
import torch
from evaluation.ablation.jtp.distributions import collect,DistributionRecorder,KEYS,FLAGS,draw
from evaluation.ablation.jtp.metrics import evaluate_group
from highway_env.planner.diffusion.projection import load_jtp_config
from tests.test_joint_trajectory_projection import _context,StraightLineExecution


def test_six_constraint_flags_match_evaluator_and_ignore_rates():
    cfg=load_jtp_config()
    u=torch.zeros(1,40,2);u[:,::2,0]=2.;u[:,1::2,0]=-2.
    context=_context();rollout=StraightLineExecution()(u,context)
    ev={'tolerances':{'stability_rad':1e-5,'ltr':1e-5,'control_si':1e-5}}
    values,bad=collect(u,rollout,context,cfg,ev)
    rows=evaluate_group(u,rollout['trajectory'],rollout['states'],rollout['ltr'],context,cfg,ev)
    assert len(KEYS)==6 and rows[0]['joint_feasible']
    assert all(values[k].shape==(1,40) for k in KEYS)
    assert all(bool(bad[k].any())==rows[0][flag] for k,flag in zip(KEYS,FLAGS))


def test_distribution_roundtrip_and_figures(tmp_path):
    cfg=load_jtp_config();context=_context();u=torch.zeros(1,40,2)
    ev={'tolerances':{'stability_rad':1e-5,'ltr':1e-5,'control_si':1e-5}}
    rollout=StraightLineExecution()(u,context)
    collected=collect(u,rollout,context,cfg,ev)
    rows=evaluate_group(u,rollout['trajectory'],rollout['states'],rollout['ltr'],context,cfg,ev)
    recorder=DistributionRecorder(cfg,ev)
    for method in ('Baseline','Ours'):
        recorder.add(method,collected,0,1,2,[0],rows)
    path=recorder.save(tmp_path)
    with np.load(path,allow_pickle=False) as data:
        assert len(data.files)==36
        assert data['Ours_yaw_ids'][:,4].tolist()==list(range(1,41))
    draw(path,tmp_path)
    assert (tmp_path/'jtp_distributions_stability_control.pdf').is_file()
