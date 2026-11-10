"""Stage B-2: paper-specific *K0* evaluation without TruckSim.

Checkpoint-frozen episode partitions, 40-step free rollout, exact five paper
metrics (including the separately supervised LTR head),
and two strictly labeled K0 diagnostic figures. Future six-model experiments
will use the same metrics functions and matching test episodes.
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import discover_episodes, KoopmanWindowDataset
from .model import rollout_pose
from .metrics import (get_errors, summarize_episode_errors, summarize_all,
                      save_table_row, save_episodes, save_position_curve,
                      plot_errors, plot_chassis_examples)
from .train import choose_device, load_checkpoint, resolve_path


def _key(path):
    return os.path.normcase(os.path.normpath(str(Path(path).resolve())))


def _checkpoint_episodes(checkpoint, episodes, split):
    manifest = checkpoint.get('split_manifest', {})
    item = manifest.get(split)
    if not isinstance(item,dict) or 'ids' not in item:
        raise ValueError(f'Checkpoint lacks frozen {split} episode ids; evaluation must not resplit')
    ids = item['ids']
    if not ids:
        raise ValueError(f'Frozen {split} split is empty. Use --split val for smoke or retrain with more episodes for a test split')
    available = {_key(e.path):e for e in episodes}
    selected = []
    for fp in ids:
        k=_key(fp)
        if k not in available:
            raise ValueError(f'Checkpoint {split} episode missing from accepted data snapshot: {fp}. '
                             'Pass the SAME --data-root used at training; do not resplit or mix new episodes.')
        selected.append(available[k])
    if len(set(map(id,selected))) != len(selected):
        raise ValueError('Duplicate episode in checkpoint split')
    other_keys=set()
    for k in ('train','val','test'):
        if k!=split: other_keys.update(_key(fp) for fp in manifest.get(k,{}).get('ids',[]))
    if any(_key(ep.path) in other_keys for ep in selected):
        raise ValueError('Leakage: checkpoint split overlaps training/other splits')
    return selected


def _sample_example(model, ep, ds, device, dt):
    # Select an unmodified 4-second control snippet, same inputs for GT and K0.
    batch = ds[0]
    with torch.inference_mode():
        inputs=lambda key:batch[key][None].to(device)
        result=model.predict_all(inputs('state_hist'),inputs('control_hist'),inputs('future_controls'))
        pred=result['states'].cpu().numpy()[0]
        pltr=result['ltr'].cpu().numpy()[0]
    true=batch['target_states'].numpy()
    tltr=batch['target_ltr'].numpy()
    mask=batch['ltr_valid'].numpy()
    label=(f"{ep.id}\nload={ep.info.get('payload_kg','?')} kg, "
           f"mu={ep.info.get('road_mu','?')}, v0={ep.info.get('vx_init_kmh','?')} km/h")
    return {'label':label, 'episode_id':ep.id, 'pred_states':pred,'true_states':true,
            'pred_ltr':pltr, 'true_ltr':np.where(mask,tltr,np.nan)}


def evaluate_k0(checkpoint_path, project_root, split='val', data_root=None,
                output_dir=None, device='auto', batch_size=128, examples=3):
    """Evaluate using the checkpoint's exact episode split and train configuration."""
    model, checkpoint = load_checkpoint(checkpoint_path, choose_device(device))
    model.eval()
    train_cfg = checkpoint.get('train_config')
    if not isinstance(train_cfg,dict) or not isinstance(train_cfg.get('data'),dict):
        raise ValueError('Checkpoint missing paper K0 data configuration')
    dc=train_cfg['data']
    dt=float(dc.get('dt_s',0.1))
    horizon=int(dc.get('horizon_steps',40))
    if abs(dt-.1)>1e-8 or horizon!=40:
        raise ValueError('Paper validation requires 10 Hz and 40 steps / 4 seconds')
    roots = [data_root] if data_root is not None else dc.get('roots',[])
    if not roots:
        raise ValueError('No data root; supply --data-root')
    eps=discover_episodes(roots,project_root,require_accepted=bool(dc.get('accepted_only',True)),
        expected_dt=dt,strict=bool(dc.get('strict',False)))
    heldout=_checkpoint_episodes(checkpoint,eps,split)
    if batch_size<1:raise ValueError('batch_size must be positive')
    if output_dir is None:
        output_dir=Path(project_root)/'outputs'/'koopman'/'evaluation'/'k0'/split
    output_dir=resolve_path(project_root,output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    results=[];ex=[]
    for ep in heldout:
        ds=KoopmanWindowDataset([ep],history_steps=int(dc.get('history_steps',5)),
            horizon_steps=horizon,warmup_s=float(dc.get('warmup_s',1.0)),
            window_stride=int(dc.get('eval_stride',5)),dt=dt)
        if not ds:
            warnings.warn(f'No usable 4s prediction window in {ep.id}');continue
        sum_position=np.zeros(horizon+1,dtype=np.float64)
        sum_fde=0.;sum_yaw_sq=0.;sum_roll_sq=0.;sum_ltr_sq=0.;count_ltr=0;count_windows=0
        for batch in DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0):
            inputs=lambda key:batch[key].to(next(model.parameters()).device)
            with torch.inference_mode():
                result=model.predict_all(inputs('state_hist'),inputs('control_hist'),
                    inputs('future_controls'),inputs('target_poses')[:,0],dt)
                states,poses,pred_ltr=result['states'],result['pose'],result['ltr']
            errors=get_errors(states.cpu().numpy(),batch['target_states'].numpy(),
                              poses.cpu().numpy(),batch['target_poses'].numpy(),
                              pred_ltr.cpu().numpy(),
                              batch['target_ltr'].numpy(),batch['ltr_valid'].numpy())
            pe=errors['position_error'];n=len(pe)
            sum_position+=pe.sum(axis=0);sum_fde+=pe[:,-1].sum()
            sum_yaw_sq+=errors['yaw_sq'].sum();sum_roll_sq+=errors['roll_sq'].sum()
            if errors['ltr_sq'] is not None:
                mask=errors['ltr_valid'];sum_ltr_sq+=errors['ltr_sq'][mask].sum();count_ltr+=mask.sum()
            count_windows+=n
        if count_windows==0:continue
        curve=sum_position/count_windows
        n_future=count_windows*horizon
        row={'episode_id':ep.id,'windows':count_windows,
             'ADE (m)':float(curve[1:].mean()),'FDE (m)':float(sum_fde/count_windows),
             'Yaw Rate RMSE (rad/s)':float(np.sqrt(sum_yaw_sq/n_future)),
             'Roll RMSE (deg)':float(np.sqrt(sum_roll_sq/n_future)),
             'LTR RMSE':float(np.sqrt(sum_ltr_sq/count_ltr)) if count_ltr>0 else None,
             'curve':curve.tolist()}
        if not all(np.isfinite(row[k]) for k in ('ADE (m)','FDE (m)','Yaw Rate RMSE (rad/s)','Roll RMSE (deg)')):
            raise ValueError(f'Nonfinite metrics in {ep.id}')
        results.append(row)
        if len(ex)<int(examples):ex.append(_sample_example(model,ep,ds,next(model.parameters()).device,dt))
    if not results:
        raise ValueError(f'No usable {split} 4-s windows')
    table,curve=summarize_all(results)
    if table['LTR RMSE'] is None:
        raise ValueError('No valid LTR targets for held-out evaluation; paper requires LTR RMSE')
    save_table_row(output_dir/'table_k0.csv','K0',table)
    save_episodes(output_dir/'episode_metrics.csv',results)
    save_position_curve(output_dir/'position_error_curve.csv',curve,dt)
    plot_errors(output_dir/'koopman_prediction_errors_k0_diagnostic.pdf',curve,results,dt)
    if ex:plot_chassis_examples(output_dir/'koopman_chassis_responses_k0_diagnostic.pdf',ex,dt)
    report={'model':'K0','checkpoint':str(Path(checkpoint_path).resolve()),
            'checkpoint_epoch':int(checkpoint['epoch']), 'split':split,'num_episodes':len(results),
            'num_windows':int(sum(r['windows'] for r in results)),
            'horizon_steps':horizon,'dt_s':dt,'metrics_episode_equal_weight':table,
            'ltr_available':True,
            'status':'diagnostic_only_not_six_model_paper_comparison',
            'trajectory_reconstruction':'body yaw-rate approximate heading-rate; trapezoidal velocity integration',
            'position_metrics_exclude_t0':True,
            'ltr_definition':'supervised latent LTR head; invalid TruckSim labels masked',
            'plots_note':'K0 only; chassis cases may use different randomized control sequences',
            'split_source':'checkpoint frozen manifest; no resplitting'}
    (output_dir/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    ltr_display = '{:.5f}'.format(table['LTR RMSE'])
    print(f'[K0 EVAL] {split} {len(results)} episodes; {report["num_windows"]} windows '
          f'ADE={table["ADE (m)"]:.5f} m FDE={table["FDE (m)"]:.5f} m '
          f'LTR={ltr_display}',flush=True)
    print(f'[K0 EVAL] saved {output_dir}',flush=True)
    return report
