"""Paired synchronized frame sampling and K2-compatible world geometry for JTP ablation."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_pipeline.raw_expert_dataset import RawExpertDataset
from highway_env.planner.diffusion.projection.types import RoadGeometry
from highway_env.planner.diffusion.projection.constraints import stability_limits


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
    """Sample 60% expert-current stability-violating frames, or all when N=0."""
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
    violating = [frame for frame in complete
                 if any(meta[i]['stability_violation'] for i in frame[1])]
    compliant = [frame for frame in complete
                 if not any(meta[i]['stability_violation'] for i in frame[1])]
    if num_frames > 0:
        n_violating = math.floor(num_frames * .60 + .5)
        n_compliant = num_frames - n_violating
        if len(violating) < n_violating or len(compliant) < n_compliant:
            raise ValueError(
                f'Expert stability sampling needs {n_violating} violating and '
                f'{n_compliant} compliant frames; available: '
                f'{len(violating)} violating and {len(compliant)} compliant')
        rng = np.random.default_rng(int(sample_seed))
        selected = [violating[int(i)] for i in rng.choice(
            len(violating), size=n_violating, replace=False)]
        selected += [compliant[int(i)] for i in rng.choice(
            len(compliant), size=n_compliant, replace=False)]
        selected.sort(key=lambda frame: frame[0])
    else:
        selected = complete
        n_violating, n_compliant = len(violating), len(compliant)

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
        "frame_sampling": "expert_current_stability_60pct" if num_frames > 0 else "all_frames",
        "sampling_basis": "Any vehicle current yaw rate, sideslip, roll or LTR violation; expert truth, not model prediction",
        "expert_stability_fraction": .60,
        "selected_expert_stability_violating_frames": n_violating,
        "selected_expert_stability_compliant_frames": n_compliant,
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


def build_road_geometry(features: dict[str, torch.Tensor], pose0: np.ndarray):
    """Construct one same-road corridor in world space from feature-builder channels.

    Uses only current/adjacent lanes (not unrelated or disconnected lanes).
    Returns corridor and local plotting geometry. Invalid/ambiguous road data is
    explicitly rejected rather than silently counting as feasible.
    """
    polylines = np.asarray(features['map_polylines'].detach().cpu(), dtype=np.float64)
    valid = np.asarray(features['map_valid_mask'].detach().cpu(), dtype=bool).reshape(-1)
    if polylines.ndim != 3 or polylines.shape[-1] < 10:
        raise ValueError(f'Invalid map feature shape: {polylines.shape}')
    indices = [int(i) for i in np.flatnonzero(valid)
               if np.isfinite(polylines[i]).all() and np.all(polylines[i, :, 4] > 0)]
    current = [i for i in indices if polylines[i, 0, 6] > .5]
    if len(current) != 1:
        raise ValueError('Expected exactly one valid current lane for geometric evaluation')
    same = [i for i in indices if i == current[0] or polylines[i, 0, 7] > .5
            or polylines[i, 0, 8] > .5]
    ref = polylines[current[0]]
    angle = np.arctan2(ref[:, 2], ref[:, 3])
    normal = np.column_stack((-np.sin(angle), np.cos(angle)))
    lateral = {i: float(np.median(np.sum((polylines[i, :, :2] - ref[:, :2]) * normal, axis=1)))
               for i in same}
    left, right = max(same, key=lateral.get), min(same, key=lateral.get)
    def edge(idx, factor):
        p = polylines[idx]
        yaw = np.arctan2(p[:, 2], p[:, 3])
        return p[:, :2] + factor * .5 * p[:, 4, None] * np.column_stack((-np.sin(yaw), np.cos(yaw)))
    left_xy, right_xy = edge(left, 1.), edge(right, -1.)
    center_local = .5 * (left_xy + right_xy)
    half_width = .5 * np.linalg.norm(left_xy - right_xy, axis=-1)
    keep = [0]
    for i in range(1, len(center_local)):
        if np.linalg.norm(center_local[i] - center_local[keep[-1]]) > 1.e-5:
            keep.append(i)
    center_local, half_width = center_local[keep], half_width[keep]
    if len(center_local) < 2 or np.any(~np.isfinite(half_width)) or np.any(half_width <= 0):
        raise ValueError('Degenerate road corridor')
    target = [i for i in indices if polylines[i, 0, 9] > .5]
    if target and target[0] not in same:
        raise ValueError('Target lane lies on a different road; current-road approximation invalid')
    zero = np.zeros((len(center_local), 1))
    center_world = local_to_world(np.concatenate((center_local, zero), axis=-1), pose0)[:, :2]
    return RoadGeometry(torch.tensor(center_world[None], dtype=torch.float32),
                        torch.tensor(half_width[None], dtype=torch.float32),
                        torch.tensor(half_width[None], dtype=torch.float32)), {
        'center_local': center_local.astype(np.float32),
        'half_width_m': half_width.astype(np.float32),
        'left_edge_local': left_xy.astype(np.float32),
        'right_edge_local': right_xy.astype(np.float32),
        'min_half_width_m': float(half_width.min()),
        'lane_count': len(same),
    }


@dataclass
class TrafficActor:
    pose: np.ndarray  # [H+1,3] world
    length_m: float
    width_m: float
    source_ego: int


def build_external_traffic(features_batch: dict, context: dict, horizon: int, dt: float,
                           *, max_actors: int, max_distance_m: float) -> list[TrafficActor]:
    """Collect non-controlled CV actors, deduplicated across the jointly planned egos."""
    all_actors = []
    for ego in range(context['koopman_pose0_si'].shape[0]):
        features = {key: value[ego] for key, value in features_batch.items()}
        agents = features['agent_states'].detach().cpu().numpy()
        valid = features['agent_valid_mask'].detach().cpu().numpy().astype(bool)
        pose0 = context['koopman_pose0_si'][ego].detach().cpu().numpy()
        hist = context['koopman_state_hist_si'][ego].detach().cpu().numpy()
        c, s = np.cos(pose0[2]), np.sin(pose0[2]); rotation = np.array([[c,-s],[s,c]])
        ego_vel_world = rotation @ hist[-1, :2]
        for idx in np.flatnonzero(valid):
            a = agents[idx]
            if a.shape[0] < 11 or not np.isfinite(a[:11]).all() or a[10] > .5:
                continue
            if np.linalg.norm(a[:2]) > max_distance_m:
                continue
            world_xy = pose0[:2] + rotation @ a[:2]
            world_vel = ego_vel_world + rotation @ a[2:4]
            yaw = pose0[2] + np.arctan2(a[6], a[7])
            length, width = max(float(a[8]), 1.), max(float(a[9]), 1.)
            times = np.arange(horizon + 1) * dt
            poses = np.column_stack((world_xy[0] + times * world_vel[0],
                                     world_xy[1] + times * world_vel[1],
                                     np.full(horizon+1, yaw)))
            duplicate = any(np.linalg.norm(poses[0, :2] - b.pose[0, :2]) < 1.0
                            and np.linalg.norm(poses[-1, :2] - b.pose[-1, :2]) < 2.0 for b in all_actors)
            if not duplicate:
                all_actors.append(TrafficActor(poses, length, width, ego))
    all_actors.sort(key=lambda a: min(np.linalg.norm(a.pose[0, :2] -
                  context['koopman_pose0_si'][i, :2].detach().cpu().numpy())
                  for i in range(context['koopman_pose0_si'].shape[0])))
    return all_actors[:max_actors]
