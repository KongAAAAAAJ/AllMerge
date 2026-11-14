"""The single supported TRACE Fuel-Aware open-loop evaluation flow.

F0 versus validation-selected fuel-aware method, grouped by three payloads.
All reported fuel estimates are frozen Koopman K2 + calibrated VT-CPFM;
there is no closed-loop success-rate metric.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_pipeline.expert_dataset import AllMergeExpertDatasetConfig,AllMergeExpertShardDataset
from highway_env.planner.diffusion.common.execution import K2_CONTEXT
from highway_env.planner.diffusion.config.experiment import load_mode_yaml,resolve_repo_file
from highway_env.planner.diffusion.training.control import load_control_checkpoint
from .validation import (REQUIRED_LABELS,check_ablation_objectives,
    check_checkpoint_compatibility,checkpoint_hash,fuel_parameters_from_yaml,
    k2_signature,resolve_device)
from .metrics import evaluate_sample_metrics
from .statistics import (METRICS,_write_csv,parse_multi_checkpoint_specs,
    parse_seeds,summarize_multiseed,summarize_per_sample_repeats)
from .sampling import parse_payloads,select_payload_samples,select_figure_samples
from .reporting import TABLE_METRICS,draw_payload_figure,latex_payload_table

def _infer_one(model, ds, *, seed, train_seed, method, device,
               batch_size, num_workers, params, threshold, fig_indices):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == 'cuda')
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    rows, figures = [], {}
    global_index = 0
    with torch.inference_mode():
        for f_cpu, t_cpu in loader:
            features = {k: v.to(device, non_blocking=True) for k,v in f_cpu.items()}
            missing = [k for k in K2_CONTEXT if k not in t_cpu]
            if missing:
                raise KeyError(f'Expert shard missing K2 context: {missing}')
            if 'trajectory_dense' not in t_cpu:
                raise KeyError('Expert shard missing 40-step trajectory_dense')
            context = {k: t_cpu[k].to(device, non_blocking=True) for k in K2_CONTEXT}
            truth = t_cpu['trajectory_dense'].to(device)
            prediction = model.infer_multimodal(features, generator=rng, context=context)
            metrics = evaluate_sample_metrics(prediction, {'trajectory_dense':truth},
                features, context, fuel_params=params, dt=model.ctrl_cfg.dt,
                progress_threshold_m=threshold)
            for j in range(len(truth)):
                idx = global_index
                shard_id, shard_sample_index = ds._index[idx]
                scalar = lambda key: float(metrics[key][j].detach().cpu().item())
                row = dict(method=method, train_seed=train_seed, inference_seed=int(seed),
                    sample_index=idx, original_sample_index=ds.source_indices[idx],
                    shard_name=ds.shard_paths[shard_id].name,
                    shard_sample_index=shard_sample_index,
                    selected_mode=int(metrics['selected_mode'][j].item()),
                    fuel_l=scalar('fuel_l'), distance_m=scalar('distance_m'),
                    fuel_l_per_100km=scalar('fuel_l_per_100km'),
                    progress_m=scalar('progress_m'), pvr=scalar('pvr'),
                    ade_m=scalar('ade_m'), fde_m=scalar('fde_m'),
                    goal_terminal_error_m=scalar('goal_terminal_error_m'),
                    speed_mae_proxy_kmh=scalar('speed_mae_proxy_kmh'),
                    rms_longitudinal_jerk_m_s3=scalar('rms_longitudinal_jerk_m_s3'),
                    initial_speed_kmh=float(context['koopman_state_hist_si'][j,-1,0].item())*3.6,
                    payload_kg=float(context['koopman_payload_kg'][j].item()),
                    road_mu=float(context['koopman_road_mu'][j].item()))
                rows.append(row)
                if idx in fig_indices:
                    xy = prediction['pose'][j, :, :2].detach().cpu().numpy().copy()
                    gt_future_xy = truth[j].detach().cpu().numpy()
                    # Expert dense trajectory is ego-local. Its t=0 point is (0, 0);
                    # koopman_pose0_si belongs to the world coordinate system.
                    gtxy = np.concatenate((np.zeros_like(gt_future_xy[:1]), gt_future_xy), axis=0)
                    figures[idx] = dict(speeds_kmh=metrics['speeds_kmh'][j].detach().cpu().numpy().copy(),
                                        xy=xy, gt_xy=gtxy)
                global_index += 1
    return rows, figures


def _validate_control_anchor_dataset(
    dataset,
    *,
    dataset_root,
    expected_modes=10,
    expected_control_steps=40,
    expected_trajectory_steps=8,
    configured_dataset_root=None,
):
    """Fail early unless every selected shard contains paired control anchors.

    Control-space diffusion consumes the physical control sequence that was
    generated together with each coarse XY anchor. The legacy expert dataset
    schema intentionally keeps ``coarse_controls`` optional for trajectory
    planners, but it is mandatory for TRACE control-space evaluation.
    """
    if len(dataset) <= 0:
        raise ValueError(
            f"Fuel-aware control evaluation found no samples in dataset: {dataset_root}"
        )

    missing_shards = []
    for shard_path in dataset.shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            if "coarse_controls" not in shard.files:
                missing_shards.append(Path(shard_path).name)

    if missing_shards:
        preview = ", ".join(missing_shards[:5])
        suffix = " ..." if len(missing_shards) > 5 else ""
        configured_hint = ""
        if configured_dataset_root is not None:
            configured_hint = (
                f"\nThe active YAML control.training.dataset_root resolves to: "
                f"{configured_dataset_root}"
                "\nRemove the explicit --dataset-root argument to use that "
                "configured dataset, or pass it explicitly."
            )
        raise ValueError(
            "Incompatible dataset for TRACE control-space Fuel-Aware evaluation."
            f"\nSelected dataset: {dataset_root}"
            f"\nMissing coarse_controls in {len(missing_shards)}/"
            f"{len(dataset.shard_paths)} selected shards: {preview}{suffix}"
            "\nThe current control-diffusion model requires paired dynamic "
            "anchors: coarse_controls [N,10,40,2], generated together with "
            "coarse_trajectories [N,10,8,2]."
            "\nThis dataset contains only XY anchors and cannot provide the "
            "original control anchors. Do not reconstruct controls by "
            "inverting coarse_trajectories."
            f"{configured_hint}"
        )

    features, _ = dataset[0]
    expected_control_shape = (
        int(expected_modes), int(expected_control_steps), 2
    )
    expected_trajectory_shape = (
        int(expected_modes), int(expected_trajectory_steps), 2
    )
    expected_mask_shape = (int(expected_modes),)

    actual_control_shape = tuple(features["coarse_controls"].shape)
    actual_trajectory_shape = tuple(features["coarse_trajectories"].shape)
    actual_mask_shape = tuple(features["mode_valid_mask"].shape)

    if actual_control_shape != expected_control_shape:
        raise ValueError(
            "Invalid paired control-anchor shape in selected dataset: "
            f"coarse_controls={actual_control_shape}, "
            f"expected={expected_control_shape}; dataset={dataset_root}"
        )
    if actual_trajectory_shape != expected_trajectory_shape:
        raise ValueError(
            "Invalid paired trajectory-anchor shape in selected dataset: "
            f"coarse_trajectories={actual_trajectory_shape}, "
            f"expected={expected_trajectory_shape}; dataset={dataset_root}"
        )
    if actual_mask_shape != expected_mask_shape:
        raise ValueError(
            "Invalid anchor validity-mask shape in selected dataset: "
            f"mode_valid_mask={actual_mask_shape}, "
            f"expected={expected_mask_shape}; dataset={dataset_root}"
        )
    if not torch.isfinite(features["coarse_controls"]).all():
        raise ValueError(
            f"coarse_controls contains NaN/Inf in dataset sample 0: {dataset_root}"
        )
    if not torch.isfinite(features["coarse_trajectories"]).all():
        raise ValueError(
            f"coarse_trajectories contains NaN/Inf in dataset sample 0: {dataset_root}"
        )

    return {
        "strategy": "dynamic_paired_control_anchors",
        "coarse_controls_shape": list(expected_control_shape),
        "coarse_trajectories_shape": list(expected_trajectory_shape),
        "mode_valid_mask_shape": list(expected_mask_shape),
        "checked_shards": len(dataset.shard_paths),
        "dataset_root": str(dataset_root),
    }


def _grouped_summary(records, groups, payloads):
    indices_to_payload = {int(idx):float(p) for p,ids in groups.items() for idx in ids}
    nested = {p:defaultdict(lambda: defaultdict(dict)) for p in payloads}
    for method, trains in records.items():
        for train_seed, inference in trains.items():
            for seed, rows in inference.items():
                gr = {p:[] for p in payloads}
                for row in rows:
                    gr[indices_to_payload[row['sample_index']]].append(row)
                for p in payloads:
                    if not gr[p]:
                        raise RuntimeError(f'Empty payload group {p} for {method}@{train_seed}/{seed}')
                    nested[p][method][train_seed][seed] = gr[p]
    summary_rows, byseed_rows = [], []
    for p in payloads:
        summary, details = summarize_multiseed(nested[p])
        for method, values in summary.items():
            summary_rows.append({'payload_kg':p, **values})
        for row in details:
            byseed_rows.append({'payload_kg':p, **row})
    return summary_rows, byseed_rows


def eval_payload_ablation(checkpoints, *, yaml_config, dataset_root,
    split='val', num_samples=99, batch_size=2, num_workers=0, device='auto',
    inference_seeds=(0,1,2,3,4), output_dir, paper_ours='F3',
    progress_threshold_m=60., payloads=(0.,3500.,7000.),
    payload_tolerance_kg=1., sample_seed=2026, figure_seed=2027):
    payloads = parse_payloads(payloads)
    inferred = parse_seeds(inference_seeds)
    cps = checkpoints if all(isinstance(k,tuple) for k in checkpoints) else {
        (k,'single'):v for k,v in checkpoints.items()}
    methods = [m for m in REQUIRED_LABELS if any(k[0]==m for k in cps)]
    if 'F0' not in methods or paper_ours not in methods or paper_ours == 'F0':
        raise ValueError('Require baseline F0 and selected fuel-aware checkpoint')
    if num_samples < 0 or 0 < num_samples < 3 or batch_size < 1 or num_workers < 0:
        raise ValueError('Invalid evaluation size or batch setting')
    train_ids = sorted({s for _,s in cps})
    for tr in train_ids:
        if any((m,tr) not in cps for m in methods):
            raise ValueError(f'Missing paired training checkpoint for {tr}')
    params = fuel_parameters_from_yaml(yaml_config)
    dev = resolve_device(device)
    root = resolve_repo_file(dataset_root)
    source = AllMergeExpertShardDataset(AllMergeExpertDatasetConfig(
        dataset_root=root,split=split,max_samples=None))
    cfg = load_mode_yaml(yaml_config)
    configured_root = resolve_repo_file(cfg['training']['dataset_root'])
    model_cfg = cfg.get('model') or {}
    control_cfg = cfg.get('control') or {}
    anchor_contract = _validate_control_anchor_dataset(
        source,
        dataset_root=root,
        expected_modes=int(model_cfg.get('num_modes', 10)),
        expected_control_steps=int(control_cfg.get('horizon_steps', 40)),
        expected_trajectory_steps=int(model_cfg.get('horizon_steps', 8)),
        configured_dataset_root=configured_root,
    )
    print(
        '[ANCHOR DATASET] paired controls verified: '
        f"controls={anchor_contract['coarse_controls_shape']} "
        f"trajectories={anchor_contract['coarse_trajectories_shape']} "
        f"shards={anchor_contract['checked_shards']} root={root}",
        flush=True,
    )
    ds, groups, available = select_payload_samples(source,payloads,max_samples=num_samples,
        sample_seed=sample_seed,tolerance_kg=payload_tolerance_kg)
    selected = select_figure_samples(groups,payloads,figure_seed=figure_seed)
    selected_ids = set(selected.values())
    checkpoint_info, reference = {}, None
    # Strictly compare all checkpoints before the first evaluation.
    for tr in train_ids:
        for method in methods:
            cp=resolve_repo_file(cps[(method,tr)])
            if not cp.is_file():
                raise FileNotFoundError(cp)
            model,payload=load_control_checkpoint(cp,device='cpu')
            check_ablation_objectives(method,model.loss_config,params)
            meta=dict(model_config=payload['model_config'],
                      control_config=payload['control_config'],
                      k2_signature=k2_signature(model),
                      mode_selection=model.loss_config.get('mode_selection'))
            if reference is not None:
                check_checkpoint_compatibility(reference,meta,f'{method}@{tr}')
            else:
                reference=meta
                if meta['mode_selection']!='goal_endpoint':
                    raise ValueError('Expected goal_endpoint training')
            if method != 'F0':
                minimum=float(model.loss_config['progress']['minimum_progress_m'])
                if not math.isclose(minimum,progress_threshold_m,abs_tol=1e-7):
                    raise ValueError(f'{method}@{tr} progress threshold {minimum} differs from {progress_threshold_m}')
            checkpoint_info[f'{method}@{tr}']=dict(checkpoint=str(cp),sha256=checkpoint_hash(cp),
                epoch=payload.get('epoch'),training_loss_config=model.loss_config)
            del model
    out=resolve_repo_file(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    nested=defaultdict(lambda:defaultdict(dict))
    all_rows=[]
    traces={p:{'F0':[] ,paper_ours:[]} for p in payloads}
    for tr in train_ids:
        for method in methods:
            cp=resolve_repo_file(cps[(method,tr)])
            model,_=load_control_checkpoint(cp,device='cpu')
            model.to(dev).eval()
            for seed in inferred:
                print(f'[PAYLOAD FUEL EVAL] {method}@{tr} inference_seed={seed} n={len(ds)}',flush=True)
                rows, figs=_infer_one(model,ds,seed=seed,train_seed=tr,method=method,device=dev,
                    batch_size=batch_size,num_workers=num_workers,params=params,
                    threshold=progress_threshold_m,fig_indices=selected_ids)
                if len(rows)!=len(ds):
                    raise RuntimeError('Dataset item count mismatch')
                nested[method][tr][seed]=rows
                all_rows.extend(rows)
                if method in ('F0',paper_ours):
                    for p,idx in selected.items():
                        traces[p][method].append(figs[idx])
            del model
            if dev.type=='cuda':
                torch.cuda.empty_cache()
    expected=[(r['shard_name'],r['shard_sample_index']) for r in nested['F0'][train_ids[0]][inferred[0]]]
    for method in methods:
        for tr in train_ids:
            for seed in inferred:
                got=[(r['shard_name'],r['shard_sample_index']) for r in nested[method][tr][seed]]
                if got!=expected:
                    raise RuntimeError(f'Mismatched samples in {method}@{tr}/seed{seed}')
    overall_summary, byseed = summarize_multiseed(nested)
    payload_summary, byseed_payload = _grouped_summary(nested,groups,payloads)
    _write_csv(out/'metrics_per_sample.csv',all_rows)
    _write_csv(out/'metrics_per_sample_summary.csv',summarize_per_sample_repeats(all_rows))
    _write_csv(out/'metrics_summary.csv',list(overall_summary.values()))
    _write_csv(out/'metrics_by_seed.csv',byseed)
    _write_csv(out/'metrics_summary_by_payload.csv',payload_summary)
    _write_csv(out/'metrics_by_seed_payload.csv',byseed_payload)
    (out/'fuel_ablation_table.tex').write_text(
        latex_payload_table(payload_summary,payloads,paper_ours),encoding='utf-8')
    figure_details={}
    for p,idx in selected.items():
        s,row_id=ds._index[idx]
        example=next(r for r in all_rows if r['sample_index']==idx)
        figure_details[str(p)]=dict(local_sample_index=idx,
            original_sample_index=ds.source_indices[idx],
            shard=ds.shard_paths[s].name,shard_sample_index=row_id,
            actual_payload_kg=example['payload_kg'],road_mu=example['road_mu'],
            initial_speed_kmh=example['initial_speed_kmh'])
    draw_payload_figure(out/'fuel_ablation.pdf',traces,payloads,paper_ours,
                        reference['control_config']['dt'],fig_details=figure_details)
    manifest=dict(study='TRACE payload-stratified fuel-aware open-loop ablation V3',
        evaluation_type='open_loop',success_metric=False,
        paper_metrics=TABLE_METRICS,auxiliary_metrics=METRICS,
        methods=methods,paper_comparison=['F0',paper_ours],
        payloads_kg=payloads,payload_tolerance_kg=payload_tolerance_kg,
        full_split_counts_by_payload={str(k):v for k,v in available.items()},
        selected_counts_by_payload={str(k):len(v) for k,v in groups.items()},
        figure_selection='One reproducibly random, independent sample per payload; different rows need not share a maneuver',
        figure_examples=figure_details,sample_seed=sample_seed,figure_seed=figure_seed,
        training_seed_ids=train_ids,inference_seeds=inferred,
        std_basis='Time-aligned inference/training repetitions, mean +/- 1 SD; spatial SD projected onto mean path normal',
        sample_ids=[{'original_index':i,'shard':ds.shard_paths[ds._index[j][0]].name,
                     'shard_sample_index':ds._index[j][1]} for j,i in enumerate(ds.source_indices)],
        split=split,eval_samples=len(ds),dataset_root=str(root),
        anchor_contract=anchor_contract,
        fuel_parameters=vars(params),frozen_k2_signature=reference['k2_signature'],
        configuration=str(resolve_repo_file(yaml_config)),
        checkpoint_info=checkpoint_info,
        fuel_definition='100000 * sum(VT-CPFM fuel [L]) / sum(K2 executed XY distance [m]) for each run',
        fuel_note='Estimated fuel via K2+VT-CPFM, not measured by TruckSim',
        warning='Equal planner initialization and identical source episodes must be guaranteed externally; checkpoint hashes cannot prove them.')
    (out/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[PAYLOAD FUEL EVAL DONE]',out,flush=True)
    for r in payload_summary:
        print(f"[{r['payload_kg']:g} kg] {r['method']} " +
              ' '.join(f'{k}={r[k]:.3f}' for k in TABLE_METRICS),flush=True)
    return dict(summary_by_payload=payload_summary,summary=overall_summary,
                manifest=manifest,output_dir=out)


def main(argv=None):
    parser=argparse.ArgumentParser(description='TRACE fuel-aware payload-stratified open-loop evaluator V3; no Success')
    parser.add_argument('--checkpoint',action='append',required=True,
        help='F0=path and F3=path; independent training: F0@0=path, F3@0=path etc.')
    parser.add_argument('--config',default='configs/train/diffusion.yaml')
    parser.add_argument('--dataset-root', help=('Control-space evaluation dataset; each selected shard must contain paired coarse_controls [N,10,40,2]. Defaults to control.training.dataset_root in the YAML.'))
    parser.add_argument('--split',choices=('train','val','test'),default='val')
    parser.add_argument('--num-samples',type=int,default=99,
        help='Total stratified sample count; 0 uses all matching samples')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--num-workers',type=int,default=0)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--seeds',default='0,1,2,3,4')
    parser.add_argument('--paper-ours',choices=('F1','F2','F3'),default='F3')
    parser.add_argument('--payloads',default='0,3500,7000')
    parser.add_argument('--payload-tolerance-kg',type=float,default=1.0)
    parser.add_argument('--sample-seed',type=int,default=2026)
    parser.add_argument('--figure-seed',type=int,default=2027)
    parser.add_argument('--progress-threshold-m',type=float,default=60.)
    parser.add_argument('--output-dir',default='outputs/trace_fuel_ablation/open_loop_eval_v3')
    args=parser.parse_args(argv)
    specs=parse_multi_checkpoint_specs(args.checkpoint)
    cfg=load_mode_yaml(args.config)
    dsroot=args.dataset_root or cfg['training']['dataset_root']
    eval_payload_ablation(specs,yaml_config=args.config,dataset_root=dsroot,
        split=args.split,num_samples=args.num_samples,batch_size=args.batch_size,
        num_workers=args.num_workers,device=args.device,
        inference_seeds=parse_seeds(args.seeds),output_dir=args.output_dir,
        paper_ours=args.paper_ours,progress_threshold_m=args.progress_threshold_m,
        payloads=parse_payloads(args.payloads),payload_tolerance_kg=args.payload_tolerance_kg,
        sample_seed=args.sample_seed,figure_seed=args.figure_seed)
    return 0
