"""Paired synchronized frame sampling and K2-compatible world geometry for JTP ablation."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.planner.diffusion.projection.constraints import stability_limits
from .payload import PAYLOAD_GROUPS, PAYLOAD_KG, payload_group


def _expert_stability_bad(state, ltr, payload, mu, jtp_config, tolerances):
    state = torch.as_tensor(state, dtype=torch.float64)
    limits = stability_limits(state[0], torch.as_tensor(payload, dtype=torch.float64),
                              torch.as_tensor(mu, dtype=torch.float64), jtp_config)
    values = torch.stack((state[2], torch.atan2(state[1], state[0]), state[3],
                          state.new_tensor(ltr)))
    tolerance = state.new_tensor([tolerances['stability_rad']] * 3 + [tolerances['ltr']])
    return bool((values.abs() - limits > tolerance).any())


def selected_frames(dataset_root: Path, split: str, *, num_frames: int, sample_seed: int,
                    jtp_config: dict, tolerances: dict):
    """Sample up to N frames per stratum at rounded 70% unstable; N=0 selects all."""
    ds = RawExpertDataset(dataset_root, split=split)
    by_shard = defaultdict(list)
    for global_i, (shard_idx, sample_i) in enumerate(ds._index):
        by_shard[int(shard_idx)].append((global_i, int(sample_i)))

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    meta = {}
    for shard_idx, shard_path in enumerate(ds.shard_paths):
        rows = by_shard.get(shard_idx, [])
        if not rows:
            continue
        with np.load(shard_path, allow_pickle=False) as shard:
            states = shard['koopman_state_hist_si'][:, -1]
            ltr = shard['koopman_ltr0']
            payload = shard['koopman_payload_kg']
            mu = shard['koopman_road_mu']
            for global_i, sample_i in rows:
                episode = int(np.asarray(shard["episode_index"][sample_i]).item())
                frame = int(np.asarray(shard["frame_index"][sample_i]).item())
                ego = int(np.asarray(shard["ego_index"][sample_i]).item())
                key = (episode, frame)
                groups[key].append(global_i)
                meta[global_i] = {
                    "episode_index": episode,
                    "frame_index": frame,
                    "ego_index": ego,
                    "payload_group": payload_group(payload[sample_i]),
                    "stability_violation": _expert_stability_bad(
                        states[sample_i], ltr[sample_i], payload[sample_i], mu[sample_i],
                        jtp_config, tolerances),
                }

    complete = []
    for key, indices in groups.items():
        ego_ids = [meta[i]["ego_index"] for i in indices]
        if len(ego_ids) == len(set(ego_ids)):
            complete.append((key, sorted(indices, key=lambda i: meta[i]["ego_index"])))
    if not complete:
        raise RuntimeError(f"No complete synchronized frames found in split={split}")

    complete.sort(key=lambda x: x[0])
    selected, strata = [], {}
    rng = np.random.default_rng(int(sample_seed))
    for group in PAYLOAD_GROUPS:
        pool = [frame for frame in complete if meta[frame[1][0]]['payload_group'] == group]
        violating = [frame for frame in pool if any(meta[i]['stability_violation'] for i in frame[1])]
        compliant = [frame for frame in pool if not any(meta[i]['stability_violation'] for i in frame[1])]
        if num_frames:
            count = min(num_frames, len(pool))
            while count:
                n_violating = math.floor(count * .70 + .5)
                n_compliant = count - n_violating
                if n_violating <= len(violating) and n_compliant <= len(compliant):
                    break
                count -= 1
            if not count:
                raise ValueError(f'{group} payload has no frames satisfying the rounded 70% stability quota')
            if count < num_frames:
                print(f'[JTP] WARNING: {group} payload sampling reduced from {num_frames} to {count} frames: '
                      f'{n_violating} violating + {n_compliant} compliant; '
                      f'actual violating ratio={100*n_violating/count:.2f}%', flush=True)
        else:
            n_violating, n_compliant = len(violating), len(compliant)
        chosen = ([violating[int(i)] for i in rng.choice(len(violating), n_violating, replace=False)] +
                  [compliant[int(i)] for i in rng.choice(len(compliant), n_compliant, replace=False)]
                  if num_frames else pool)
        selected.extend(chosen)
        strata[group] = dict(payload_kg=list(PAYLOAD_KG[group]),
                            requested_frames=num_frames,
                            violating_frames_available=len(violating), compliant_frames_available=len(compliant),
                            selected_frames=len(chosen), selected_violating_frames=n_violating,
                            selected_compliant_frames=n_compliant,
                            actual_expert_stability_fraction=n_violating/len(chosen) if chosen else None,
                            selected_frame_keys=[list(frame[0]) for frame in sorted(chosen)])
    selected.sort(key=lambda frame: frame[0])

    samples = []
    selected_keys = []
    for key, indices in selected:
        entries = []
        for index in indices:
            entries.append(ds[index])
        samples.append((key, entries))
        selected_keys.append(key)

    manifest = {
        "split": split,
        "input_format": "current_raw_trace_expert_dataset",
        "dataset_samples_in_split": len(ds),
        "complete_frames_available": len(complete),
        "selected_frames": len(selected),
        "selected_samples": sum(len(x[1]) for x in samples),
        "sample_seed": int(sample_seed),
        "selected_frame_keys": [list(x) for x in selected_keys],
        "frame_sampling": "payload_stratified_expert_current_stability_70pct" if num_frames > 0 else "all_frames",
        "sampling_basis": "Any vehicle current yaw rate, sideslip, roll or LTR violation; expert truth, not model prediction",
        "expert_stability_fraction": .70,
        "requested_frames_per_payload": num_frames,
        "payload_strata": strata,
        "selected_expert_stability_violating_frames": sum(s['selected_violating_frames'] for s in strata.values()),
        "selected_expert_stability_compliant_frames": sum(s['selected_compliant_frames'] for s in strata.values()),
    }
    return samples, manifest


def collate_frame(entries: list[tuple[dict, dict, dict]], device):
    features = {k: torch.stack([e[0][k] for e in entries]).to(device)
                for k in entries[0][0] if all(k in e[0] for e in entries)}
    from highway_env.planner.diffusion.common.execution import K2_CONTEXT
    targets = {}
    for key in K2_CONTEXT:
        if any(key not in e[1] for e in entries):
            raise KeyError(f'Missing synchronized K2 field {key}')
        targets[key] = torch.stack([e[1][key] for e in entries]).to(device, dtype=torch.float32)
    return features, targets, [e[2] for e in entries]


def local_to_world(local_pose: np.ndarray, pose0: np.ndarray) -> np.ndarray:
    c, s = math.cos(float(pose0[2])), math.sin(float(pose0[2]))
    x = pose0[0] + c * local_pose[..., 0] - s * local_pose[..., 1]
    y = pose0[1] + s * local_pose[..., 0] + c * local_pose[..., 1]
    return np.stack((x, y, pose0[2] + local_pose[..., 2]), axis=-1)


def world_to_local(world: np.ndarray, pose0: np.ndarray) -> np.ndarray:
    c, s = math.cos(float(pose0[2])), math.sin(float(pose0[2]))
    dx, dy = world[..., 0] - pose0[0], world[..., 1] - pose0[1]
    return np.stack((c * dx + s * dy, -s * dx + c * dy,
                     np.arctan2(np.sin(world[..., 2] - pose0[2]), np.cos(world[..., 2] - pose0[2]))), axis=-1)
