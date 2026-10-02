from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from highway_env.envs.scenarios.curved_lane_change_env import CurvedLaneChangeEnv
from highway_env.envs.scenarios.merge_in_env import MergeInEnv
from highway_env.envs.scenarios.merge_out_env import MergeOutEnv
from highway_env.envs.scenarios.straight_lane_change_env import StraightLaneChangeEnv
from highway_env.planner.geometry import ego_to_world_point, world_to_ego_point
from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
from highway_env.planner.diffusion.trajectory_mode_reward.geometry import (
    _dense_local_trajectories,
    local_to_world,
    road_margin_series,
)

SCENARIOS = {
    'straight': StraightLaneChangeEnv,
    'curved': CurvedLaneChangeEnv,
    'merge_in': MergeInEnv,
    'merge_out': MergeOutEnv,
}


@dataclass
class AlignmentRow:
    scenario: str
    sample_index: int
    env_seed: int
    vehicle_role: int
    target_lane_index: str
    pose_translation_drift_m: float
    pose_heading_drift_deg: float
    target_lane_snapshot_mean_abs_lateral_m: float
    target_lane_snapshot_max_abs_lateral_m: float
    target_lane_poststep_mean_abs_lateral_m: float
    target_lane_poststep_max_abs_lateral_m: float
    expert_min_margin_snapshot_m: float
    expert_min_margin_poststep_m: float
    expert_offroad_snapshot: int
    expert_offroad_poststep: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Check planner-target geometry against reward geometry in the exact planning frame.'
    )
    p.add_argument('--scenario', choices=('all', *SCENARIOS.keys()), default='all')
    p.add_argument('--samples-per-scenario', type=int, default=25)
    p.add_argument('--states-per-gallery', type=int, default=9)
    p.add_argument('--group-action', type=int, default=3)
    p.add_argument('--vehicle-role', type=int, choices=(0,1,2), default=0)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--visualization-seed', type=int, default=0)
    p.add_argument('--max-seed-attempts', type=int, default=1000)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/planner_reward_alignment_check'))
    return p.parse_args()


def _scenario_order(args: argparse.Namespace) -> list[str]:
    return list(SCENARIOS) if args.scenario == 'all' else [str(args.scenario)]


def _state_seed(base: int, scenario_index: int, attempt: int) -> int:
    return int(base + scenario_index * 100000 + attempt)


def _vehicle_pose(vehicle: Any) -> np.ndarray:
    heading = getattr(vehicle, 'heading', getattr(vehicle, 'heading_theta', 0.0))
    return np.asarray([float(vehicle.position[0]), float(vehicle.position[1]), float(heading)], dtype=np.float64)


def _lane_lateral_errors(world_xy: np.ndarray, lane: Any) -> np.ndarray:
    vals=[]
    for point in np.asarray(world_xy):
        _, lat = lane.local_coordinates(point[:2])
        vals.append(abs(float(lat)))
    return np.asarray(vals, dtype=np.float64)


def _lane_local_centerlines(env: Any, snapshot_pose: np.ndarray, xmin=-25.0, xmax=140.0) -> list[np.ndarray]:
    graph=getattr(env.road.network,'graph',{})
    out=[]
    seen=set()
    for outgoing in graph.values():
        if not isinstance(outgoing,dict): continue
        for lanes in outgoing.values():
            if not isinstance(lanes,(list,tuple)): continue
            for lane in lanes:
                if id(lane) in seen: continue
                seen.add(id(lane))
                ss=np.linspace(0.0,float(lane.length),120)
                world=np.asarray([lane.position(float(s),0.0) for s in ss],dtype=np.float32)
                local=world_to_ego_point(world,snapshot_pose[:2],float(snapshot_pose[2]))
                mask=(local[:,0]>=xmin)&(local[:,0]<=xmax)&(np.abs(local[:,1])<30.0)
                if int(mask.sum())>=2:
                    out.append(local[mask])
    return out


def _extract_expert(env: Any) -> np.ndarray:
    alignment=getattr(env,'latest_expert_alignment',None)
    if not isinstance(alignment,dict) or 'expert_trajectory_xy' not in alignment:
        raise RuntimeError('latest_expert_alignment/expert_trajectory_xy missing; Polynomial aligned baseline is required')
    arr=np.asarray(alignment['expert_trajectory_xy'],dtype=np.float64)
    if arr.shape!=(3,8,2):
        raise RuntimeError(f'unexpected expert_trajectory_xy shape {arr.shape}')
    return arr


def _collect(args: argparse.Namespace):
    cfg=TrajectoryModeRewardConfig()
    rows: list[AlignmentRow]=[]
    plot_data: dict[str,list[dict[str,Any]]]={name:[] for name in _scenario_order(args)}
    for sidx,sname in enumerate(_scenario_order(args)):
        env_cls=SCENARIOS[sname]
        collected=0; attempts=0
        while collected<int(args.samples_per_scenario):
            if attempts>=int(args.max_seed_attempts):
                raise RuntimeError(f'cannot collect enough states for {sname}')
            seed=_state_seed(args.seed,sidx,attempts); attempts+=1
            env=env_cls(config={'show_trajectories':False,'show_future_trajectories':False},render_mode=None)
            try:
                env.reset(seed=seed)
                _,_,terminated,truncated,_=env.step(int(args.group_action))
                if terminated or truncated: continue
                snap=getattr(env,'latest_planner_reward_state_snapshot',None)
                if not isinstance(snap,dict):
                    raise RuntimeError('planning-time reward snapshot missing; install frame snapshot fix first')
                role=int(args.vehicle_role)
                snapshot_pose=np.asarray(snap['controlled_poses'],dtype=np.float64)[role]
                current_pose=_vehicle_pose(env.controlled_vehicles[role])
                target_indices=snap.get('target_lane_indices',())
                if len(target_indices)<=role or target_indices[role] is None:
                    raise RuntimeError('target lane index missing from planner snapshot')
                target_idx=tuple(target_indices[role])
                lane=env.road.network.get_lane(target_idx)
                feat=np.asarray(env.latest_planner_features['target_lane_polyline'],dtype=np.float64)
                target_local=feat[role,:,0:2]
                target_world_snapshot=ego_to_world_point(target_local,snapshot_pose[:2],float(snapshot_pose[2]))
                target_world_post=ego_to_world_point(target_local,current_pose[:2],float(current_pose[2]))
                lat_snap=_lane_lateral_errors(target_world_snapshot,lane)
                lat_post=_lane_lateral_errors(target_world_post,lane)

                expert=_extract_expert(env)
                dense_local,_=_dense_local_trajectories(expert[None],cfg)
                local=dense_local[0,role]
                world_snap=local_to_world(local,snapshot_pose)
                world_post=local_to_world(local,current_pose)
                margin_snap=road_margin_series(world_snap,env.road,cfg,tracking_aware=True)
                margin_post=road_margin_series(world_post,env.road,cfg,tracking_aware=True)
                row=AlignmentRow(
                    scenario=sname,
                    sample_index=collected,
                    env_seed=seed,
                    vehicle_role=role,
                    target_lane_index=repr(target_idx),
                    pose_translation_drift_m=float(np.linalg.norm(current_pose[:2]-snapshot_pose[:2])),
                    pose_heading_drift_deg=float(np.rad2deg(math.atan2(math.sin(current_pose[2]-snapshot_pose[2]),math.cos(current_pose[2]-snapshot_pose[2])))),
                    target_lane_snapshot_mean_abs_lateral_m=float(np.mean(lat_snap)),
                    target_lane_snapshot_max_abs_lateral_m=float(np.max(lat_snap)),
                    target_lane_poststep_mean_abs_lateral_m=float(np.mean(lat_post)),
                    target_lane_poststep_max_abs_lateral_m=float(np.max(lat_post)),
                    expert_min_margin_snapshot_m=float(np.min(margin_snap)),
                    expert_min_margin_poststep_m=float(np.min(margin_post)),
                    expert_offroad_snapshot=int(np.any(margin_snap[1:]<0.0)),
                    expert_offroad_poststep=int(np.any(margin_post[1:]<0.0)),
                )
                rows.append(row)
                # post-step interpretation expressed back in the correct snapshot-local frame
                target_post_in_snapshot=world_to_ego_point(target_world_post,snapshot_pose[:2],float(snapshot_pose[2]))
                expert_post_in_snapshot=world_to_ego_point(world_post[:,:2],snapshot_pose[:2],float(snapshot_pose[2]))
                plot_data[sname].append({
                    'row':row,
                    'lane_lines':_lane_local_centerlines(env,snapshot_pose),
                    'target_local':target_local.copy(),
                    'target_post_in_snapshot':target_post_in_snapshot,
                    'expert_local':local[:,0:2].copy(),
                    'expert_post_in_snapshot':expert_post_in_snapshot,
                })
                print(
                    f'[align] scenario={sname} sample={collected+1:03d}/{args.samples_per_scenario} '
                    f'dxy={row.pose_translation_drift_m:.3f}m dh={row.pose_heading_drift_deg:+.3f}deg '
                    f'target_lat(correct/post)={row.target_lane_snapshot_max_abs_lateral_m:.3f}/{row.target_lane_poststep_max_abs_lateral_m:.3f}m '
                    f'expert_margin(correct/post)={row.expert_min_margin_snapshot_m:.3f}/{row.expert_min_margin_poststep_m:.3f}m '
                    f'off={row.expert_offroad_snapshot}/{row.expert_offroad_poststep}'
                )
                collected+=1
            finally:
                env.close()
    return rows,plot_data


def _write_csv(path:Path,rows:list[AlignmentRow]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader(); writer.writerows(asdict(r) for r in rows)


def _write_summary(path:Path, rows:list[AlignmentRow]):
    keys=[
        'pose_translation_drift_m','pose_heading_drift_deg',
        'target_lane_snapshot_max_abs_lateral_m','target_lane_poststep_max_abs_lateral_m',
        'expert_min_margin_snapshot_m','expert_min_margin_poststep_m',
        'expert_offroad_snapshot','expert_offroad_poststep',
    ]
    out=[]
    for scenario in sorted({r.scenario for r in rows}):
        subset=[r for r in rows if r.scenario==scenario]
        d={'scenario':scenario,'n':len(subset)}
        for k in keys:
            vals=np.asarray([float(getattr(r,k)) for r in subset])
            d[f'mean_{k}']=float(np.mean(vals))
            d[f'max_{k}']=float(np.max(vals))
        out.append(d)
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for d in out:
        for k in d:
            if k not in fields: fields.append(k)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out)


def _plot(args:argparse.Namespace, plot_data:dict[str,list[dict[str,Any]]]):
    rng=np.random.default_rng(int(args.visualization_seed))
    pdir=args.output_dir/'plots'; pdir.mkdir(parents=True,exist_ok=True)
    for scenario,states in plot_data.items():
        if not states: continue
        count=min(int(args.states_per_gallery),len(states))
        ids=sorted(rng.choice(len(states),size=count,replace=False).tolist())
        chosen=[states[i] for i in ids]
        fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
        for ax in axes.ravel(): ax.set_visible(False)
        for ax,item in zip(axes.ravel(),chosen):
            ax.set_visible(True)
            for line in item['lane_lines']:
                ax.plot(line[:,0],line[:,1],linewidth=0.8,alpha=.45)
            t=item['target_local']; tp=item['target_post_in_snapshot']; e=item['expert_local']; ep=item['expert_post_in_snapshot']; row=item['row']
            ax.plot(t[:,0],t[:,1],linewidth=2.2,label='target lane | planning pose')
            ax.plot(tp[:,0],tp[:,1],'--',linewidth=1.7,label='same target | post-step pose')
            ax.plot(e[:,0],e[:,1],linewidth=2.0,label='expert | planning pose')
            ax.plot(ep[:,0],ep[:,1],'--',linewidth=1.5,label='expert | post-step pose')
            ax.scatter([0],[0],s=24)
            ax.set_aspect('equal'); ax.grid(True,alpha=.18); ax.tick_params(labelsize=7)
            ax.set_title(
                f'id={row.sample_index:02d} seed={row.env_seed}  dxy={row.pose_translation_drift_m:.2f}m  dh={row.pose_heading_drift_deg:+.2f}°\n'
                f'target max|lat| correct/post={row.target_lane_snapshot_max_abs_lateral_m:.2f}/{row.target_lane_poststep_max_abs_lateral_m:.2f}m\n'
                f'expert min margin correct/post={row.expert_min_margin_snapshot_m:.2f}/{row.expert_min_margin_poststep_m:.2f}m off={row.expert_offroad_snapshot}/{row.expert_offroad_poststep}',fontsize=8)
        handles,labels=axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=4,fontsize=9)
        fig.suptitle(f'Planner–Reward Frame Alignment | {scenario} | role={args.vehicle_role}',fontsize=12)
        out=pdir/f'planner_reward_alignment_{scenario}_3x3.png'
        fig.savefig(out,dpi=180); plt.close(fig); print('[write]',out)


def main()->int:
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    rows,plots=_collect(args)
    _write_csv(args.output_dir/'planner_reward_alignment_states.csv',rows)
    _write_summary(args.output_dir/'planner_reward_alignment_summary.csv',rows)
    _plot(args,plots)
    print('[OK] Planner–Reward Alignment Check complete')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
