"""Recorded TruckSim expert comparison on complete, paired future windows."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .distributions import collect
from .reporting import paper_table, write_csv, write_json
from .statistics import step_records, summary, add_projection_timing
from .payload import PAYLOAD_GROUPS, payload_group


def expert_step_records(dataset_root, model_rows, cfg, ev):
    frames = defaultdict(dict)
    for row in model_rows:
        key = (row['episode_index'], row['frame_index'])
        frames[key].setdefault(row['seed'], set()).add(row['ego_index'])
    by_episode = defaultdict(list)
    for episode, frame in sorted(frames):
        by_episode[episode].append(frame)
    records, complete, excluded = [], [], []
    unique_steps = 0
    horizon = int(cfg['execution']['horizon_steps'])
    for episode, frame_ids in by_episode.items():
        path = Path(dataset_root)/'episodes'/f'episode_{episode:06d}.npz'
        with np.load(path, allow_pickle=False) as shard:
            data = {key: shard[key] for key in (
                'frame_index', 'episode_step', 'ego_index', 'koopman_state_hist_si',
                'koopman_control_hist_si', 'koopman_ltr0', 'koopman_payload_kg', 'koopman_road_mu')}
        current = {(int(frame),int(ego)): i for i,(frame,ego) in enumerate(zip(data['frame_index'],data['ego_index']))}
        timeline = {(int(step),int(ego)): i for i,(step,ego) in enumerate(zip(data['episode_step'],data['ego_index']))}
        for frame in frame_ids:
            seeds = frames[(episode,frame)]
            egos = sorted(next(iter(seeds.values())))
            indices, missing = [], []
            for ego in egos:
                base = current[(frame,ego)]
                step = int(data['episode_step'][base])
                absent = [t for t in range(1,horizon+1) if (step+t,ego) not in timeline]
                if absent:
                    missing.append(dict(ego_index=ego, missing_future_steps=absent))
                else:
                    indices.append([base, *[timeline[(step+t,ego)] for t in range(1,horizon+1)]])
            if missing:
                excluded.append(dict(episode_index=episode,frame_index=frame,missing_by_vehicle=missing))
                continue
            indices = np.asarray(indices)
            # At state k+t, the last history control drove state k+t-1 -> k+t.
            controls = torch.from_numpy(data['koopman_control_hist_si'][indices[:,1:],-1])
            rollout = dict(states=torch.from_numpy(data['koopman_state_hist_si'][indices,-1]),
                           ltr=torch.from_numpy(data['koopman_ltr0'][indices]))
            context = {key: torch.from_numpy(data[key][indices[:,0]])
                       for key in ('koopman_payload_kg','koopman_road_mu')}
            collected = collect(controls,rollout,context,cfg,ev)
            physical_rows = step_records(collected,[dict(seed=0,episode_index=episode,frame_index=frame,ego_index=ego,
                payload_kg=float(context['koopman_payload_kg'][i]),
                payload_group=payload_group(context['koopman_payload_kg'][i]))
                for i, ego in enumerate(egos)], 'Expert', np.zeros((len(egos),horizon)))
            unique_steps += len(physical_rows)
            for seed in sorted(seeds):
                records.extend({**row,'seed':seed} for row in physical_rows)
            complete.append([episode,frame])
    coverage = dict(source='Recorded TruckSim future states, LTR and applied reference controls; no K2 rollout',
                    prediction_steps=f'1..{horizon}', horizon_steps=horizon,
                    selected_physical_frames=len(frames), complete_physical_frames=len(complete),
                    complete_frame_keys=complete, excluded_frames=excluded,
                    unique_expert_vehicle_steps=unique_steps,
                    dataset_root=str(Path(dataset_root).resolve()),
                    stability_settings=cfg['stability'],
                    control_bounds={k:v for k,v in cfg['control'].items() if '_min_' in k or '_max_' in k},
                    tolerances=ev['tolerances'],
                    inference_seeds=sorted({r['seed'] for r in model_rows}),
                    note='Expert truth is computed once per physical frame and repeated for pairing with inference seeds')
    coverage['by_payload'] = {
        group: dict(selected_physical_frames=len({(r['episode_index'],r['frame_index'])
                                                 for r in model_rows if r['payload_group']==group}),
                    complete_physical_frames=len({(r['episode_index'],r['frame_index'])
                                                 for r in records if r['payload_group']==group}),
                    unique_expert_vehicle_steps=len({(r['episode_index'],r['frame_index'],r['ego_index'],r['prediction_step'])
                                                     for r in records if r['payload_group']==group}))
        for group in PAYLOAD_GROUPS}
    return records, coverage


def write_expert_comparison(output, dataset_root, model_rows, frame_diagnostics, cfg, ev):
    expert_rows, coverage = expert_step_records(dataset_root,model_rows,cfg,ev)
    directory = Path(output)/'step'
    write_json(directory/'jtp_expert_coverage_step.json',coverage)
    if not expert_rows:
        print('[Expert] No selected frame has complete recorded future truth; no comparison table generated',flush=True)
        return coverage
    complete = {tuple(key) for key in coverage['complete_frame_keys']}
    paired = [row for row in model_rows if (row['episode_index'],row['frame_index']) in complete]
    paired.extend(expert_rows)
    methods = ('Expert','Baseline','Ours')
    summaries = {method: summary([r for r in paired if r['method']==method],'step') for method in methods}
    add_projection_timing(summaries,[f for f in frame_diagnostics if (f['episode_index'],f['frame_index']) in complete])
    summaries['Expert'].update(optimization_time_ms_mean=None,optimization_time_ms_p95=None,
                               num_physical_frames=coverage['complete_physical_frames'])
    write_csv(directory/'jtp_per_vehicle_step_expert.csv',paired)
    write_csv(directory/'jtp_summary_step_expert.csv',[dict(method=method,**summaries[method]) for method in methods])
    by_payload = {}
    for group in PAYLOAD_GROUPS:
        group_summary = {method: summary([r for r in paired if r['method']==method and r['payload_group']==group], 'step')
                         for method in methods}
        add_projection_timing(group_summary,[f for f in frame_diagnostics
            if f['payload_group']==group and (f['episode_index'],f['frame_index']) in complete])
        group_summary['Expert'].update(optimization_time_ms_mean=None, optimization_time_ms_p95=None,
            num_physical_frames=coverage['by_payload'][group]['complete_physical_frames'])
        by_payload[group] = group_summary
    write_csv(directory/'jtp_summary_step_expert_payload.csv',
              [dict(payload_group=g,method=m,**v) for g, by_method in by_payload.items() for m,v in by_method.items()])
    (directory/'projection_ablation_table_step_expert.tex').write_text(paper_table(by_payload,'step',expert=True),encoding='utf-8')
    print(f"[Expert] Paired comparison: {coverage['complete_physical_frames']}/{coverage['selected_physical_frames']} complete frames",flush=True)
    return coverage
