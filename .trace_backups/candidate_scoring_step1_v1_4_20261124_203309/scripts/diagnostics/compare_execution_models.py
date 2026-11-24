"""Compare identical TRACE [ax, front-wheel-delta] controls in three plants.

Kinematic Bicycle and Koopman K2 run locally. TruckSim is optional at runtime
but requires the user's Windows TruckSim .sim/.dll and calibrated collection
configuration. It uses the project's ReferenceActuator mapping, not a guessed
linear acceleration-to-throttle conversion. No training code/checkpoint is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.planner.diffusion.common.kinematics import rollout_bicycle
from highway_env.planner.diffusion.training.control import load_control_checkpoint


def curvature_from_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3: return np.full(len(xy), np.nan)
    d1 = np.gradient(xy, axis=0)
    d2 = np.gradient(d1, axis=0)
    den = np.maximum((d1[:, 0]**2+d1[:, 1]**2)**1.5, 1e-8)
    return (d1[:, 0]*d2[:, 1]-d1[:, 1]*d2[:, 0])/den


def wrap_angle(x):
    return np.arctan2(np.sin(x), np.cos(x))


def run_trucksim(controls, initial_vx_mps, simfile, dll, collect_config, expected_payload, expected_mu):
    if sys.platform != 'win32':
        raise RuntimeError('TruckSim native DLL execution requires Windows. Run this option on the local Windows/TruckSim machine.')
    from highway_env.vehicle.trucksim_simulation.Trucksim_Simulation_ltr import TrucksimSimulationLTR
    from highway_env.vehicle.koopman.data.collect import ReferenceActuator, load_engine_map_for_collection

    cfg = yaml.safe_load(Path(collect_config).read_text(encoding='utf-8'))
    units = cfg['units']
    controller_cfg = cfg['reference_controller']
    engine_map = load_engine_map_for_collection(controller_cfg)
    dt_ref = float(cfg['excitation']['controller_dt_s'])
    actuator = ReferenceActuator(controller_cfg, units, dt_ref, engine_map=engine_map)
    sim = TrucksimSimulationLTR(simfile, dll)
    sim.start({'vx_init': float(initial_vx_mps)*3.6})
    times, xy_world, yaw_world, exports_log = [], [], [], []
    try:
        if int(sim.configuration['n_import']) != 3 or int(sim.configuration['n_export']) != 19:
            raise ValueError('Expected the validated TRACE TruckSim interface: 3 imports / 19 exports.')
        dt_native = float(sim.get_time_step())
        stride_f = dt_ref/dt_native
        stride = round(stride_f)
        if stride < 1 or not np.isclose(stride_f, stride, atol=1e-6):
            raise ValueError(f'controller dt {dt_ref} must be an integer multiple of TruckSim dt {dt_native}')
        x = np.asarray(sim.get_export_array(), dtype=float)
        if x.shape != (19,) or not np.isfinite(x).all(): raise ValueError('Invalid initial TruckSim export array')
        actual_vx = x[2]/3.6
        if abs(actual_vx-initial_vx_mps) > 2.0/3.6:
            raise ValueError(f'Initial TruckSim speed {actual_vx:.3f} m/s differs from requested {initial_vx_mps:.3f} m/s')
        # Payload and mu are physically configured in the selected .sim file;
        # metadata checks are explicit and do not pretend to alter the native model.
        t0 = float(sim.current_time)
        X0, Y0, psi0 = float(x[4]), float(x[5]), np.deg2rad(float(x[10]))
        times.append(t0); xy_world.append([X0, Y0]); yaw_world.append(psi0); exports_log.append(x.copy())
        last_native = np.zeros(3, dtype=float)
        n_native = int(round((len(controls)*dt_ref)/dt_native))
        for k in range(n_native):
            ref_idx = min(k//stride, len(controls)-1)
            if k % stride == 0:
                last_native = actuator.control(controls[ref_idx], x)
                if not np.isfinite(last_native).all(): raise ValueError('Non-finite TruckSim actuator command')
                if not (0 <= last_native[0] <= 100 and 0 <= last_native[1] <= 10):
                    raise ValueError(f'Native actuator outside expected range: {last_native}')
            t_next = t0+(k+1)*dt_native
            status, x = sim.run(t_next, last_native, x)
            times.append(float(sim.current_time)); xy_world.append([float(x[4]), float(x[5])])
            yaw_world.append(np.deg2rad(float(x[10]))); exports_log.append(x.copy())
            if status: raise RuntimeError(f'TruckSim stopped early with status={status}')
        xy_world = np.asarray(xy_world)
        delta = xy_world-np.asarray([X0,Y0])
        c,s=np.cos(psi0),np.sin(psi0)
        local=np.column_stack((c*delta[:,0]+s*delta[:,1], -s*delta[:,0]+c*delta[:,1]))
        yaw_local=wrap_angle(np.asarray(yaw_world)-psi0)
        tgrid=np.arange(len(controls)+1)*dt_ref
        reltime=np.asarray(times)-t0
        # interpolate only after preserving raw native-rate output in CSV.
        xy_grid=np.column_stack([np.interp(tgrid,reltime,local[:,i]) for i in range(2)])
        yaw_grid=np.interp(tgrid,reltime,np.unwrap(yaw_local))
        return tgrid, xy_grid, yaw_grid, {
            'native_dt_s':dt_native, 'reference_dt_s':dt_ref,
            'payload_expected_kg':expected_payload, 'road_mu_expected':expected_mu,
            'simfile':str(Path(simfile).resolve()), 'dll':str(Path(dll).resolve()),
            'trucksim_raw_samples':len(times), 'trucksim_export_order':[
                'ax','ay','vx','vy','Xo','Yo','Roll','AVx','Pitch','AVy','Yaw','AVz','AV_Eng',
                'Steer_L1','Steer_R1','Fz_L1','Fz_L2','Fz_R1','Fz_R2'],
            'note':'Payload and road friction are configured in the .sim file. Confirm they match the expected metadata before interpreting results.',
            '_raw_exports':np.asarray(exports_log), '_raw_time_s':np.asarray(times),
        }
    finally:
        # Existing project documents a native termination crash for this DLL.
        # This one-shot process intentionally skips native termination.
        sim.stop(native_terminate=False)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root',default='outputs/expert_dataset/expert_trucksim_k2_raw_100k')
    ap.add_argument('--split',choices=['train','val','test'],default='test')
    ap.add_argument('--sample-index',type=int,default=0)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--controls-npy',default=None,help='Optional exact [40,2] controls; otherwise use selected checkpoint mode')
    ap.add_argument('--mode-index',type=int,default=None,help='Optional fixed candidate mode; default is deployed goal-endpoint selection')
    ap.add_argument('--seed',type=int,default=2048)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--trucksim-simfile',default=None)
    ap.add_argument('--trucksim-dll',default=None)
    ap.add_argument('--collect-config',default='configs/koopman/collect.yaml')
    ap.add_argument('--expected-payload-kg',type=float,default=None)
    ap.add_argument('--expected-road-mu',type=float,default=None)
    ap.add_argument('--output-dir',default='outputs/diagnostics/execution_models')
    args=ap.parse_args(argv)
    root=Path(args.dataset_root); root=root if root.is_absolute() else ROOT/root
    ckpt=Path(args.checkpoint); ckpt=ckpt if ckpt.is_absolute() else ROOT/ckpt
    dataset=RawExpertDataset(root,split=args.split)
    if not 0<=args.sample_index<len(dataset): raise IndexError(f'sample-index must be in [0,{len(dataset)-1}]')
    features,targets,meta=dataset[args.sample_index]
    device=torch.device(args.device)
    model,_=load_control_checkpoint(ckpt,device=device); model.eval()
    f={k:v.unsqueeze(0).to(device) for k,v in features.items()}
    ctx={k:v.unsqueeze(0).to(device) for k,v in targets.items() if k.startswith('koopman_')}
    if args.controls_npy:
        controls=np.load(args.controls_npy).astype(np.float32)
        if controls.shape!=(40,2): raise ValueError('--controls-npy must contain [40,2] [ax_mps2, front_delta_rad]')
        selected_mode=None
    else:
        gen=torch.Generator(device=device); gen.manual_seed(args.seed)
        with torch.inference_mode(): result=model.infer_multimodal(f,generator=gen,context=ctx)
        if args.mode_index is None:
            selected_mode=int(result['trajectory_mode_idx'][0].item())
        else:
            selected_mode=int(args.mode_index)
        if not 0<=selected_mode<result['control_candidates'].shape[1]: raise ValueError('mode-index out of range')
        controls=result['control_candidates'][0,selected_mode].detach().cpu().numpy()
    controls=torch.as_tensor(controls,dtype=torch.float32,device=device).reshape(1,1,40,2)
    vx0=float(targets['koopman_state_hist_si'][-1,0].item())
    dt=float(model.ctrl_cfg.dt); wheelbase=float(model.ctrl_cfg.wheelbase_m)
    with torch.inference_mode():
        kin_pose,_=rollout_bicycle(controls.reshape(1,40,2),torch.tensor([vx0],device=device),dt,wheelbase)
        k2_pred=model.vehicle_model(controls,f,ctx)
        k2_pose=k2_pred['poses'][0,0]
    kin=kin_pose[0].cpu().numpy(); k2=k2_pose.cpu().numpy()
    tgrid=np.arange(41)*dt
    methods={'Kinematic Bicycle':(kin[:,:2],kin[:,2]),'Koopman K2':(k2[:,:2],k2[:,2])}
    report={'sample':{k:int(v) for k,v in meta.items() if isinstance(v,(int,np.integer))},
            'initial_speed_mps':vx0,'dt_s':dt,'horizon_s':40*dt,'selected_mode':selected_mode,
            'controls_shape':[40,2], 'control_order':['ax_mps2','front_wheel_delta_rad'],
            'expert_terminal_xy_m':targets['trajectory_dense'][-1].tolist(),
            'koopman_payload_kg':float(targets['koopman_payload_kg'].item()),
            'koopman_road_mu':float(targets['koopman_road_mu'].item())}
    if args.trucksim_simfile or args.trucksim_dll:
        if not (args.trucksim_simfile and args.trucksim_dll): raise ValueError('Provide both --trucksim-simfile and --trucksim-dll')
        payload=float(targets['koopman_payload_kg'].item()); mu=float(targets['koopman_road_mu'].item())
        if args.expected_payload_kg is None or args.expected_road_mu is None:
            raise ValueError('TruckSim comparison requires --expected-payload-kg and --expected-road-mu as explicit confirmation of the selected .sim physical setup')
        if abs(args.expected_payload_kg-payload)>1e-3: raise ValueError('Expected payload does not match selected dataset sample')
        if abs(args.expected_road_mu-mu)>1e-6: raise ValueError('Expected road mu does not match selected dataset sample')
        t_ts,xy_ts,yaw_ts,ts_meta=run_trucksim(controls[0,0].cpu().numpy(),vx0,args.trucksim_simfile,args.trucksim_dll,args.collect_config,payload,mu)
        methods['TruckSim']=(xy_ts,yaw_ts)
        raw_exports=ts_meta.pop('_raw_exports'); raw_times=ts_meta.pop('_raw_time_s')
        report['trucksim']=ts_meta
        import csv as _csv
        raw_path=Path(args.output_dir); raw_path=raw_path if raw_path.is_absolute() else ROOT/raw_path
        raw_path.mkdir(parents=True,exist_ok=True)
        signal_names=['ax_g','ay_g','vx_kmh','vy_kmh','Xo_m','Yo_m','roll_deg','AVx_deg_s','pitch_deg','AVy_deg_s','yaw_deg','AVz_deg_s','engine_rpm','steer_L1_deg','steer_R1_deg','Fz_L1','Fz_L2','Fz_R1','Fz_R2']
        with (raw_path/'trucksim_raw_exports.csv').open('w',newline='',encoding='utf-8') as _fh:
            _writer=_csv.writer(_fh); _writer.writerow(['time_s',*signal_names]); _writer.writerows([[t,*r] for t,r in zip(raw_times,raw_exports)])
    out=Path(args.output_dir); out=out if out.is_absolute() else ROOT/out; out.mkdir(parents=True,exist_ok=True)
    import csv as _csv_controls
    with (out/'input_controls.csv').open('w',newline='',encoding='utf-8') as _fh:
        _writer=_csv_controls.writer(_fh); _writer.writerow(['time_s','ax_mps2','front_wheel_delta_rad'])
        _writer.writerows([[i*dt,float(u[0]),float(u[1])] for i,u in enumerate(controls[0,0].cpu().numpy())])
    expert=targets['trajectory_dense'].numpy()
    rows=[]
    for name,(xy,yaw) in methods.items():
        curv=curvature_from_xy(xy)
        expert_xy=np.vstack((np.zeros((1,2),dtype=float), expert))
        lateral_error=xy[:,1]-expert_xy[:,1] if len(xy)==41 else np.full(len(xy),np.nan)
        report[name]={
            'terminal_lateral_displacement_m':float(xy[-1,1]),
            'terminal_yaw_rad':float(yaw[-1]),'terminal_yaw_deg':float(np.rad2deg(yaw[-1])),
            'terminal_lateral_error_vs_expert_m':float(xy[-1,1]-expert[-1,1]),
            'max_abs_lateral_error_vs_expert_m':float(np.nanmax(np.abs(lateral_error))),
            'mean_abs_lateral_error_vs_expert_m':float(np.nanmean(np.abs(lateral_error))),
            'max_abs_curvature_1pm':float(np.nanmax(np.abs(curv))),
            'mean_abs_curvature_1pm':float(np.nanmean(np.abs(curv))),
        }
        for i,(tt,point,yy,kk,ee) in enumerate(zip(tgrid,xy,yaw,curv,lateral_error)):
            rows.append({'method':name,'time_s':float(tt),'x_m':float(point[0]),'y_m':float(point[1]),
                         'yaw_rad':float(yy),'yaw_deg':float(np.rad2deg(yy)),
                         'curvature_1pm':float(kk),'lateral_error_vs_expert_m':float(ee)})
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    import csv
    with (out/'trajectories.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for name,(xy,yaw) in methods.items():
        axes[0,0].plot(xy[:,0],xy[:,1],label=name)
        axes[0,1].plot(tgrid,np.rad2deg(yaw),label=name)
        axes[1,0].plot(tgrid,xy[:,1]-np.r_[0.0,expert[:,1]],label=name)
        axes[1,1].plot(tgrid,curvature_from_xy(xy),label=name)
    axes[0,0].plot(expert_xy[:,0],expert_xy[:,1],'k--',label='Expert')
    axes[0,0].set_title('Executed trajectories');axes[0,0].set_xlabel('Longitudinal position [m]');axes[0,0].set_ylabel('Lateral position [m]')
    axes[0,1].set_title('Yaw response');axes[0,1].set_xlabel('Time [s]');axes[0,1].set_ylabel('Yaw [deg]')
    axes[1,0].axhline(0,color='k',linestyle='--',linewidth=.8);axes[1,0].set_title('Lateral position error vs expert');axes[1,0].set_xlabel('Time [s]');axes[1,0].set_ylabel('Lateral error [m]')
    axes[1,1].set_title('Trajectory curvature');axes[1,1].set_xlabel('Time [s]');axes[1,1].set_ylabel('Curvature [1/m]')
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(out/'execution_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2,ensure_ascii=False));print(f'[DONE] {out.resolve()}')
    return 0

if __name__=='__main__': raise SystemExit(main())
