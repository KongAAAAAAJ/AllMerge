"""JTP ablation contract smoke tests (no TruckSim or trained checkpoint needed)."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

# The existing highway_env package registers Gymnasium environments on import.
# Normal project test environments already include gymnasium.
pytest.importorskip('gymnasium')

from highway_env.planner.diffusion.projection.types import RoadGeometry
from evaluation.ablation.jtp.data import TrafficActor
from evaluation.ablation.jtp.metrics import evaluate_group, trajectory_deviation_m
from evaluation.ablation.jtp.reporting import paired_report, paper_table
from evaluation.ablation.jtp.runner import _report_capabilities

ROOT = Path(__file__).resolve().parents[1]


def _fixture(vehicles=2, road_width=20.):
    cfg=yaml.safe_load((ROOT/'configs/projection/jtp.yaml').read_text(encoding='utf-8'))
    ev=yaml.safe_load((ROOT/'configs/evaluation/jtp_ablation.yaml').read_text(encoding='utf-8'))
    H=cfg['execution']['horizon_steps']
    controls=torch.zeros(vehicles,H,2)
    world=torch.zeros(vehicles,H+1,3)
    states=torch.zeros(vehicles,H+1,5)
    states[...,0]=15.
    context={'koopman_payload_kg':torch.zeros(vehicles),
             'koopman_road_mu':torch.full((vehicles,),0.8),
             'koopman_control_hist_si':torch.zeros(vehicles,4,2)}
    for i in range(vehicles):
        world[i,:,0]=torch.arange(H+1)*1.5
        world[i,:,1]=i*8.
    road=RoadGeometry(torch.tensor([[[0.,0.],[120.,0.]]]),road_width,road_width)
    return controls,world,states,torch.zeros(vehicles,H+1),context,[road]*vehicles,cfg,ev


def test_identical_feasible_trajectory():
    x,poses,states,ltr,ctx,roads,cfg,ev=_fixture()
    result=evaluate_group(x,poses,states,ltr,ctx,roads,[],cfg,ev)
    assert all(r['joint_feasible'] for r in result)
    assert np.allclose(trajectory_deviation_m(poses.numpy(),poses.numpy()),0.)


def test_dynamic_stability_and_geometry_violations():
    x,poses,states,ltr,ctx,roads,cfg,ev=_fixture(vehicles=1)
    states[0,9,2]=9.
    ltr[0,10]=.95
    poses[0,15,1]=30.
    result=evaluate_group(x,poses,states,ltr,ctx,roads,[],cfg,ev)[0]
    assert result['yaw_violation'] and result['ltr_violation']
    assert result['road_violation'] and result['geometric_violation']
    assert not result['joint_feasible']


def test_collision_and_control_rates_are_counted_even_if_projector_ignores_them():
    x,poses,states,ltr,ctx,roads,cfg,ev=_fixture(vehicles=2)
    poses[1,:,1]=1.0
    x[0,0,0]=1.
    result=evaluate_group(x,poses,states,ltr,ctx,roads,[],cfg,ev)
    assert all(r['intertruck_collision_violation'] for r in result)
    assert result[0]['acceleration_rate_violation']
    assert not any(r['joint_feasible'] for r in result)


def test_external_actor_constant_velocity_collisions():
    x,poses,states,ltr,ctx,roads,cfg,ev=_fixture(vehicles=1)
    actor=TrafficActor(poses[0].numpy().copy(),length_m=5.,width_m=2.,source_ego=0)
    result=evaluate_group(x,poses,states,ltr,ctx,roads,[actor],cfg,ev)[0]
    assert result['external_collision_violation'] and not result['joint_feasible']


def test_paired_summary_and_diagnostic_table():
    rows=[]
    for idx in range(3):
        for method in ('Baseline','Ours'):
            ours=method=='Ours'
            rows.append(dict(method=method,seed=0,episode_index=0,frame_index=idx,ego_index=0,
                joint_feasible=ours,joint_frame_feasible=ours,geometric_violation=not ours,stability_violation=not ours,
                collision_violation=False,control_violation=False,
                external_collision_violation=False,intertruck_collision_violation=False,
                acceleration_rate_violation=False,steering_rate_violation=False,
                deviation_m=.2 if ours else None,projector_success=ours,
                projection_applied=ours,projection_latency_ms=5 if ours else 0))
    summary,deltas,ci=paired_report(rows,bootstrap_repeats=20)
    assert summary['Ours']['joint_feasibility_pct']==100
    assert deltas['joint_feasibility_pct']==100
    assert 'DIAGNOSTIC ONLY' in paper_table(summary,paper_ready=False)
    assert not _report_capabilities()['paper_ready']
