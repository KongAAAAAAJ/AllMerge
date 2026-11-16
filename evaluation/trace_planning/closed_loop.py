"""Real TruckSim closed-loop TRACE/Expert scenario evaluation.

The four registered scenario classes are available. Emergency braking and lane
narrowing, plus OCP/TransFuser baselines, require separate implementations.
Each native TruckSim episode runs in its own subprocess to avoid DLL reuse.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .metrics import center_spacing_error, stability_flags, summarize_closed_loop_rows
from .open_loop import _write_csv

SCENARIOS = ('straight','curved','merge_in','merge_out')


def _scenario_class(name):
    from highway_env.envs.scenarios.straight_lane_change_env import StraightLaneChangeEnv
    from highway_env.envs.scenarios.curved_lane_change_env import CurvedLaneChangeEnv
    from highway_env.envs.scenarios.merge_in_env import MergeInEnv
    from highway_env.envs.scenarios.merge_out_env import MergeOutEnv
    return dict(straight=StraightLaneChangeEnv, curved=CurvedLaneChangeEnv,
                merge_in=MergeInEnv, merge_out=MergeOutEnv)[name]


def _case(config_file, case_id):
    from data_pipeline.expert_collection.trucksim import load_collection_config, physical_cases
    cfg = load_collection_config(config_file)
    cases = {c['model_id']:c for c in physical_cases(cfg)}
    if case_id not in cases:
        raise ValueError(f'Unknown physical model {case_id}; available: {list(cases)[:8]}...')
    return cfg, cases[case_id]


def _make_single_reset_class(env_cls, fixed_seed):
    # AbstractEnv.__init__ calls reset() once. A second native TruckSim reset
    # would reinitialize DLL state unsafely, so seed the *first* call and reject
    # a second reset instead of following the kinematic stress-test pattern.
    class SeededOneShot(env_cls):
        def reset(self, *, seed=None, options=None):
            if getattr(self, '_trace_native_initialized', False):
                raise RuntimeError('TruckSim single-episode process must never reset twice')
            self._trace_native_initialized = True
            return super().reset(seed=int(fixed_seed), options=options)
    return SeededOneShot


def _make_env_config(name, method, *, cfg_path, checkpoint,
                     physical_cfg, physical_case, copy_dlls,
                     jtp_cfg_path, duration_s):
    from data_pipeline.expert_collection.trucksim import scenario_trucksim_config
    from highway_env.planner.diffusion.config.experiment import resolve_repo_file
    scenario_cls = _scenario_class(name)
    scenario_defaults = scenario_cls.default_config()
    planner = deepcopy(scenario_defaults['Planner'])
    if method == 'TRACE':
        if not checkpoint:
            raise ValueError('TRACE closed-loop requires --checkpoint')
        planner['Diffusion'] = {
            'config_yaml':str(resolve_repo_file(cfg_path)),
            'checkpoint':str(resolve_repo_file(checkpoint)),
            'shadow_enabled':True,
            'output_space':'control',
            'jtp_enabled':True,
            'jtp_config':str(resolve_repo_file(jtp_cfg_path)),
            'jtp_control_execution_enabled':True,
            'diffusion_execution_enabled':False,
            'diffusion_spline_enabled':False,
        }
        planner['ExpertAlignment']['enabled'] = False
    overrides = scenario_trucksim_config(physical_cfg, physical_case, 3,
                                           copy_instances=copy_dlls)
    # Runtime K2 must receive externally verified conditions. These values are
    # read from the precise selected TruckSim physical plant, not inferred.
    overrides['TruckSim']['payload_kg'] = float(physical_case['payload_kg'])
    overrides['TruckSim']['road_mu'] = float(physical_case['road_mu'])
    overrides.update(Planner=planner, duration=float(duration_s),
                     offscreen_rendering=False)
    return overrides


def _snapshot(env):
    from data_pipeline.expert_collection.trucksim import snapshot
    sign = float(env.config.get('TruckSim',{}).get('wheel_compression_sign',1))
    poses, headings, states, ltrs = [], [], [], []
    for v in env.controlled_vehicles:
        s, pose, ltr = snapshot(v, sign)
        poses.append(pose[:2]);headings.append(pose[2]);states.append(s);ltrs.append(ltr)
    return np.asarray(poses),np.asarray(headings),np.asarray(states),np.asarray(ltrs)


def worker(*, scenario, method, seed, checkpoint, config, jtp_config,
           koopman_config, trucksim_case,
           duration, group_action, output_dir, copy_dlls):
    from highway_env.planner.diffusion.config.experiment import resolve_repo_file
    from highway_env.planner.diffusion.projection.config import load_jtp_config
    from highway_env.planner.diffusion.common.fuel import FuelParameters,vt_cpfm_from_states
    from highway_env.planner.diffusion.config.experiment import load_mode_yaml
    from highway_env.planner.diffusion.training.control import load_control_checkpoint

    physical_cfg, physical_case = _case(resolve_repo_file(koopman_config),trucksim_case)
    original_sim=Path(physical_case['simfile'])
    original_dll=Path(physical_case['dll'])
    for item in (original_sim,original_dll):
        if not item.is_file():
            raise FileNotFoundError(f'Real TruckSim simulation/DLL file missing: {item}')
    original_sim_sha256=hashlib.sha256(original_sim.read_bytes()).hexdigest()
    original_dll_sha256=hashlib.sha256(original_dll.read_bytes()).hexdigest()
    mode_cfg = load_mode_yaml(config)
    if mode_cfg['output_space'] != 'control':
        raise ValueError('TRACE paper evaluation config must select control output')
    fuel = FuelParameters(**mode_cfg['losses']['fuel']['parameters'])
    fuel.validate()
    if method == 'TRACE':
        model, _ = load_control_checkpoint(resolve_repo_file(checkpoint), device='cpu')
        if model.vehicle_model.kind != 'koopman_k2':
            raise ValueError('TRACE closed loop requires trained K2 control checkpoint')
        del model
    stability = load_jtp_config(resolve_repo_file(jtp_config))
    env_cls = _make_single_reset_class(_scenario_class(scenario),seed)
    overrides = _make_env_config(scenario, method, cfg_path=config,
        checkpoint=checkpoint,physical_cfg=physical_cfg,physical_case=physical_case,
        copy_dlls=copy_dlls,jtp_cfg_path=jtp_config,
        duration_s=duration)
    env = None
    try:
        env = env_cls(config=overrides,render_mode=None)
        if len(env.controlled_vehicles) != 3 or not all(hasattr(v,'trucksim_model') for v in env.controlled_vehicles):
            raise RuntimeError('Closed-loop measurements require 3 native TruckSim vehicles')
        time_s = [0.0]
        pose,heading,state,ltr = _snapshot(env)
        positions,headings,states,ltrs = [pose],[heading],[state],[ltr]
        durations=[]; jtp_projected_control_steps=0; jtp_unexecuted_steps=0; planning_steps=0
        jtp_status_counts={}; jtp_context_status_counts={}; jtp_first_failures=[]
        jtp_step_trace=[]
        original = env.action_type.act
        def timed_act(action):
            if torch.cuda.is_available():torch.cuda.synchronize()
            t0=time.perf_counter()
            result=original(action)
            if torch.cuda.is_available():torch.cuda.synchronize()
            durations.append((time.perf_counter()-t0)*1000)
            return result
        env.action_type.act = timed_act
        final_info = {}
        max_steps=int(math.ceil(float(duration)*float(env.config['policy_frequency'])))
        finished=False
        for j in range(max_steps):
            _,_,terminated,truncated,info=env.step(int(group_action))
            planning_steps+=1
            if method=='TRACE':
                source_ok = all(
                    getattr(v, 'latest_control_source', None)
                    == 'Diffusion-Joint-Trajectory-Projection-Control'
                    for v in env.controlled_vehicles
                )
                used = source_ok
                if used:
                    jtp_projected_control_steps += 1
                elif j >= 4:
                    jtp_unexecuted_steps += 1  # first four policy steps are K2 history warmup
                context_status = str(getattr(env, 'latest_diffusion_jtp_context_status', 'unset'))
                projection_status = str(getattr(env, 'latest_diffusion_jtp_status', 'unset'))
                jtp_context_status_counts[context_status] = jtp_context_status_counts.get(context_status, 0) + 1
                jtp_status_counts[projection_status] = jtp_status_counts.get(projection_status, 0) + 1
                diagnostics = getattr(env, 'latest_diffusion_jtp_diagnostics', {}) or {}
                final_metrics = diagnostics.get('final_metrics', {}) if isinstance(diagnostics, dict) else {}
                initial_metrics = diagnostics.get('initial_metrics', {}) if isinstance(diagnostics, dict) else {}
                metric_source = final_metrics or initial_metrics or {}
                compact_metrics = {k: metric_source.get(k) for k in (
                    'projector_constraints_satisfied', 'max_violation', 'max_violation_by_group',
                    'violation_counts', 'active_counts'
                ) if k in metric_source}
                step_record = {
                    'step': int(j), 'time_s': float(env.time),
                    'executed_jtp_control': bool(used),
                    'vehicle_projector_constraints_satisfied': [
                        bool(getattr(v, 'latest_diffusion_jtp_projector_constraints_satisfied', False))
                        for v in env.controlled_vehicles],
                    'vehicle_projection_applied': [
                        bool(getattr(v, 'latest_diffusion_jtp_applied', False))
                        for v in env.controlled_vehicles],
                    'vehicle_status': [str(getattr(v, 'latest_diffusion_jtp_status', 'unset'))
                                       for v in env.controlled_vehicles],
                    'vehicle_control_source': [str(getattr(v, 'latest_control_source', 'unset'))
                                               for v in env.controlled_vehicles],
                    'context_status': context_status,
                    'projection_status': projection_status,
                    'diagnostic_metrics': compact_metrics,
                    'history_state_counts': diagnostics.get('history_state_counts') if isinstance(diagnostics, dict) else None,
                    'history_control_counts': diagnostics.get('history_control_counts') if isinstance(diagnostics, dict) else None,
                }
                if len(jtp_step_trace) < 100:
                    jtp_step_trace.append(step_record)
                if not used and j >= 4 and len(jtp_first_failures) < 8:
                    jtp_first_failures.append(step_record)
            pose,heading,state,ltr=_snapshot(env)
            time_s.append(float(env.time))
            positions.append(pose);headings.append(heading)
            states.append(state);ltrs.append(ltr)
            final_info=info
            if terminated or truncated:
                finished=True
                break
        if not finished and planning_steps!=max_steps:
            raise RuntimeError('Episode exited unexpectedly')
        pos=np.stack(positions);heads=np.stack(headings)
        state=np.stack(states).transpose(1,0,2)
        ltr=np.stack(ltrs).transpose(1,0)
        num_vehicles=pos.shape[1]
        payload=float(physical_case['payload_kg']);mu=float(physical_case['road_mu'])
        flags=stability_flags(torch.tensor(state,dtype=torch.float32),
               torch.tensor(ltr,dtype=torch.float32),
               torch.full((num_vehicles,),payload),torch.full((num_vehicles,),mu),stability)
        violation=int(flags['any'].sum().item())
        n_steps=int(flags['any'].numel())
        spacing_goal=float(env.config['traffic_randomization']['platoon_spacing'])
        gap_error=center_spacing_error(pos[1:],heads[1:],spacing_goal)
        dt=float(1./env.config['simulation_frequency'])
        if abs(dt-0.1)>1e-8:
            raise ValueError('The current VT-CPFM implementation requires 10 Hz executed states')
        speeds=torch.tensor(state[:,:,0],dtype=torch.float32)
        fuel_l=float(vt_cpfm_from_states(speeds,torch.full((num_vehicles,),payload),fuel,dt=dt)['fuel_total'].sum().item())
        distance_m=float(np.linalg.norm(np.diff(pos,axis=0),axis=-1).sum())
        done=bool(final_info.get('task_success',False))
        collision=any(v.crashed for v in env.controlled_vehicles)
        offroad=any(not v.on_road for v in env.controlled_vehicles)
        # Discard first four K2 history warmup cycles in the reported planning latency.
        latency=durations[4:] if len(durations)>4 else durations
        if not latency:
            raise RuntimeError('No planner timing samples')
        result=dict(
            method=method,scenario=scenario,seed=int(seed),physical_case=trucksim_case,
            execution_source='native_TruckSim_19_output',
            physical_sim_sha256=original_sim_sha256,
            physical_dll_sha256=original_dll_sha256,
            payload_kg=payload,road_mu=mu,status='complete',
            success=bool(done and not collision and not offroad),
            collision=bool(collision),offroad=bool(offroad),
            stability_steps=violation,vehicle_steps=n_steps,
            spacing_target_m=spacing_goal,
            spacing_error_sum_m=float(gap_error.sum()),
            spacing_observations=int(gap_error.size),
            fuel_l=fuel_l,distance_m=distance_m,
            planning_time_ms=float(np.mean(latency)),
            planning_time_sum_ms=float(np.sum(latency)),
            planning_time_count=len(latency),
            planning_steps=planning_steps,
            jtp_projected_control_steps=jtp_projected_control_steps,jtp_unexecuted_steps=jtp_unexecuted_steps,
            jtp_projection_status_counts=jtp_status_counts,
            jtp_context_status_counts=jtp_context_status_counts,
            jtp_first_failures=jtp_first_failures,
            jtp_step_trace=jtp_step_trace,
            duration_s=float(env.time),
            task_success_raw=done,
            note=('TRACE uses Polynomial only during the initial K2 history warmup; all subsequent control sequences come from JTP'
                  if method=='TRACE' else 'Expert aligned Polynomial planner'),
        )
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
        name=f'{method.replace(" ","_")}_{scenario}_{trucksim_case}_seed{seed}'
        np.savez_compressed(out/f'{name}.npz',time_s=np.array(time_s),
                            position_m=pos,heading_rad=heads,state_si=state,ltr=ltr,
                            spacing_abs_error_m=gap_error,
                            planning_time_ms=np.array(durations))
        (out/f'{name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    finally:
        if env is not None: env.close()


def _runner(args):
    from highway_env.planner.diffusion.config.experiment import resolve_repo_file
    wanted=[s.strip() for s in args.scenarios.split(',') if s.strip()]
    if not wanted or any(s not in SCENARIOS for s in wanted):
        raise ValueError(f'Available scenario classes are {SCENARIOS}. '
                          'Emergency braking and lane narrowing are NOT implemented in this snapshot.')
    from data_pipeline.expert_collection.trucksim import load_collection_config, physical_cases
    cfg=load_collection_config(resolve_repo_file(args.koopman_config))
    cases={c['model_id']:c for c in physical_cases(cfg)}
    if args.trucksim_case not in cases:
        raise ValueError(f'Unknown TruckSim physical case: {args.trucksim_case}')
    from highway_env.planner.diffusion.config.experiment import load_mode_yaml
    train_cfg=load_mode_yaml(args.config)
    if train_cfg['output_space']!='control':
        raise ValueError('Expected output_space: control')
    if args.method=='TRACE' and not args.checkpoint:
        raise ValueError('TRACE requires a control Diffusion checkpoint')
    if args.check_only:
        print(json.dumps(dict(mode='TRUCKSIM_REAL_CLOSED_LOOP',method=args.method,
             scenario_classes=wanted,physical_case=cases[args.trucksim_case],
             seeds=list(range(args.seed_start,args.seed_start+args.episodes)),
             expected_first_step='Native TruckSim, no double reset',
             notes=['All simulation and native DLL paths must exist on Windows',
                    'Only four scenarios exist in this source snapshot',
                    'TRACE executes every returned JTP control sequence after K2 history warmup',
                    'Projection exceptions stop the episode; no fallback policy is used']),
             indent=2,ensure_ascii=False))
        return 0
    out=resolve_repo_file(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for scenario in wanted:
        for seed in range(args.seed_start,args.seed_start+args.episodes):
            # One native solver instance / process / episode. Subprocess is
            # blocking to avoid DLL concurrency across physical cases.
            cmd=[sys.executable,'-m','evaluation.trace_planning.closed_loop','--_worker',
                 '--method',args.method,'--scenario',scenario,'--seed',str(seed),
                 '--config',args.config,'--jtp-config',args.jtp_config,
                 '--koopman-config',args.koopman_config,
                 '--trucksim-case',args.trucksim_case,
                 '--duration',str(args.duration),'--group-action',str(args.group_action),
                 '--output-dir',str(out)]
            if args.checkpoint:cmd+=['--checkpoint',args.checkpoint]
            if args.copy_dlls:cmd+=['--copy-dlls']
            print('[TRUCKSIM EPISODE]',args.method,scenario,seed,flush=True)
            result=subprocess.run(cmd,cwd=str(Path(__file__).resolve().parents[2]))
            if result.returncode!=0:
                raise RuntimeError('TruckSim episode failed. No aggregate was produced. '
                                   f'Scenario={scenario}, seed={seed}; see child traceback')
            row_file=out/f'{args.method.replace(" ","_")}_{scenario}_{args.trucksim_case}_seed{seed}.json'
            all_rows.append(json.loads(row_file.read_text(encoding='utf-8')))
            _write_csv(out/'episodes.csv',all_rows)
    result=summarize_closed_loop_rows(all_rows)
    if args.method == 'TRACE' and any(r['jtp_projected_control_steps'] == 0 for r in all_rows):
        failed = [r for r in all_rows if r['jtp_projected_control_steps'] == 0]
        details = [{
            'scenario': r.get('scenario'), 'seed': r.get('seed'),
            'projection_status_counts': r.get('jtp_projection_status_counts', {}),
            'context_status_counts': r.get('jtp_context_status_counts', {}),
            'first_failures': r.get('jtp_first_failures', [])[:3],
        } for r in failed]
        raise RuntimeError(
            'At least one TRACE episode did not execute JTP controls after K2 warmup. '
            'The episode is invalid for TRACE comparison. '
            'JTP diagnostics: ' + json.dumps(details, ensure_ascii=False, default=str))
    if args.method == 'TRACE' and any(r['jtp_unexecuted_steps'] > 0 for r in all_rows):
        raise RuntimeError(
            'A post-warmup control step did not execute the JTP-returned control sequence. '
            'No fallback or hybrid policy is permitted in the TRACE evaluation.')
    manifest=dict(evaluation_type='trucksim_closed_loop',method=args.method,
         physical_case=args.trucksim_case,payload_kg=cases[args.trucksim_case]['payload_kg'],
         road_mu=cases[args.trucksim_case]['road_mu'],
         scenarios=wanted,seed_start=args.seed_start,episodes_per_scenario=args.episodes,
         config=str(resolve_repo_file(args.config)),
         checkpoint=str(resolve_repo_file(args.checkpoint)) if args.checkpoint else None,
         stability_thresholds=str(resolve_repo_file(args.jtp_config)),
         success_definition='scenario task_success AND no collision/offroad',
         collision_definition='fraction of episodes with any controlled-vehicle collision',
         stability_definition='vehicle-time-step violation of any JTP yaw/sideslip/roll/LTR limit based on executed TruckSim signals',
         spacing_definition='mean abs(center longitudinal distance projected on preceding vehicle heading - commanded 15m gap)',
         fuel_definition='VT-CPFM from actual TruckSim longitudinal speed and physical payload, total L / actual executed distance',
         planning_time_definition='Mean wall-clock env.action_type.act across complete platoon, warmup cycles excluded. GPU synchronized.',
         caveat='Only the four existing scenario classes; OCP/TransFuser and five-scene paper table require later integration.',
         jtp_output_policy='Use every returned JTP control sequence; projection exceptions abort the episode; no fallback policy')
    report=dict(metrics=result,manifest=manifest)
    (out/'closed_loop_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description='Real TruckSim closed-loop TRACE / Expert evaluation')
    parser.add_argument('--method',choices=('TRACE','Expert Planner'),default='TRACE')
    parser.add_argument('--checkpoint')
    parser.add_argument('--config',default='configs/train/diffusion.yaml')
    parser.add_argument('--scenarios',default=','.join(SCENARIOS))
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--seed-start',type=int,default=0)
    parser.add_argument('--trucksim-case',default='L4_MU08')
    parser.add_argument('--koopman-config',default='configs/koopman/collect.yaml')
    parser.add_argument('--jtp-config',default='configs/projection/jtp.yaml')
    parser.add_argument('--duration',type=float,default=5.)
    parser.add_argument('--group-action',type=int,default=3)
    parser.add_argument('--output-dir',default='outputs/evaluation/trace_closed_loop')
    parser.add_argument('--copy-dlls',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    # Private subprocess arguments: one native process owns one single episode.
    parser.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--scenario',choices=SCENARIOS,help=argparse.SUPPRESS)
    parser.add_argument('--seed',type=int,help=argparse.SUPPRESS)
    parser.add_argument('--eval-config',default='configs/evaluation/trace_planning.yaml')
    probe,_=parser.parse_known_args(argv)
    from .eval_config import defaults
    parser.set_defaults(**defaults(probe.eval_config,'closed_loop'))
    args=parser.parse_args(argv)
    if args.episodes<1 or args.duration<=0:
        parser.error('episodes and duration must be positive')
    if args._worker:
        if args.scenario is None or args.seed is None:
            parser.error('_worker must set scenario and seed')
        worker(scenario=args.scenario,method=args.method,seed=args.seed,
               checkpoint=args.checkpoint,config=args.config,jtp_config=args.jtp_config,
               koopman_config=args.koopman_config,trucksim_case=args.trucksim_case,
               duration=args.duration,group_action=args.group_action,
               output_dir=args.output_dir,copy_dlls=args.copy_dlls)
        return 0
    return _runner(args)


if __name__=='__main__':
    raise SystemExit(main())
