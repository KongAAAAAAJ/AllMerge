from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Mapping

import numpy as np
import torch
import yaml
from learning.config import load_yaml_config
from torch.utils.data import Subset

from highway_env.planner.anchor_scoring.safety import anchor_safety_cost
from highway_env.planner.diffusion.common.execution import K2_CONTEXT
from highway_env.planner.diffusion.config.experiment import resolve_repo_file

from .data import build_complete_frame_subset, make_loader, read_subset_metadata, select_figure_samples
from .figure_selection import sample_figure_candidates, select_safety_improvement_scenes
from .paper_safety import (
    drivable_area_compliance,
    external_collision_free,
    external_min_ttc_s,
    expert_local_poses,
    intertruck_collision_free,
    intertruck_group_min_ttc_s,
    local_to_world_poses,
    road_geometry_from_features,
)
from .paper_metrics import (
    compute_anchor_metrics,
    compute_planning_minade,
    geometry_valid_mask,
)
from .reporting import summarize, write_csv, write_json, write_latex_table
from .visualization import draw_anchor_ablation, draw_diffusion_planning


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _meta_int(meta: Mapping, key: str, default=-1) -> int:
    value = meta.get(key, default)
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    return int(value)


def _resolve_num_modes(diffusion_config: str | Path) -> tuple[int, Path]:
    """Resolve the single anchor-count source: control.model.num_modes."""
    path = resolve_repo_file(diffusion_config)
    if not path.is_file():
        raise FileNotFoundError(f"Diffusion configuration not found: {path}")
    raw = load_yaml_config(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Diffusion configuration must be a YAML mapping: {path}")
    control = raw.get("control") or {}
    model = control.get("model") or {}
    if not isinstance(model, dict):
        raise ValueError("control.model must be a YAML mapping")
    try:
        num_modes = int(model.get("num_modes", 10))
    except (TypeError, ValueError) as exc:
        raise ValueError("control.model.num_modes must be a positive integer") from exc
    if num_modes <= 0:
        raise ValueError(f"control.model.num_modes must be positive, got {num_modes}")
    return num_modes, path


def _features_to_device(features: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in features.items()
    }


def _context_to_device(targets: dict, device: torch.device) -> dict:
    missing = [key for key in K2_CONTEXT if key not in targets]
    if missing:
        raise KeyError(f"TRACE raw-K2 safety evaluation requires synchronized K2 context fields: {missing}")
    return {key: targets[key].to(device, non_blocking=True) for key in K2_CONTEXT}


def _strategy(model) -> str:
    return str((getattr(model, "anchors_config", None) or {}).get("strategy", "dynamic")).lower()


def _load_models(checkpoints: Mapping[str, Path], device: torch.device,
                 expected_num_modes: int | None = None):
    """Load requested methods and validate shared architecture, K, and K2 weights."""
    from highway_env.planner.diffusion.training.control import load_control_checkpoint

    expected = {"A0": ("fixed", "a0"), "A1": ("a1",), "A2": ("adaptive", "a2")}
    models, checkpoint_info = {}, {}
    reference_signature = None
    reference_k2_hash = None
    for method, checkpoint in checkpoints.items():
        path = resolve_repo_file(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"{method} diffusion checkpoint not found: {path}")
        model, payload = load_control_checkpoint(path, device="cpu")
        scene_config = getattr(model, "scene_config", None)
        checkpoint_num_modes = getattr(scene_config, "num_modes", None)
        if checkpoint_num_modes is None:
            model_config = payload.get("model_config") or {}
            checkpoint_num_modes = model_config.get("num_modes")
        if expected_num_modes is not None:
            if checkpoint_num_modes is None:
                raise ValueError(
                    f"{method} checkpoint does not expose model_config.num_modes; "
                    "cannot verify compatibility with control.model.num_modes"
                )
            if int(checkpoint_num_modes) != int(expected_num_modes):
                raise ValueError(
                    f"{method} checkpoint has num_modes={checkpoint_num_modes}, "
                    f"but diffusion config requires num_modes={expected_num_modes}. "
                    "Retrain this planner checkpoint with the configured anchor count."
                )
        strategy = _strategy(model)
        if strategy not in expected[method]:
            raise ValueError(
                f"{method} checkpoint uses strategy={strategy!r}; expected one of {expected[method]!r}"
            )
        if getattr(model.vehicle_model, "kind", None) != "koopman_k2":
            raise ValueError(f"{method} checkpoint must use the TRACE Koopman K2 execution model")
        if any(key.startswith("denoiser.mode_embedding.") for key in payload["model_state"]):
            raise RuntimeError(f"{method} checkpoint still contains deprecated semantic mode_embedding")

        signature = {
            "model_config": payload.get("model_config"),
            "control_config": payload.get("control_config"),
            "execution_config": payload.get("execution_config"),
            "loss_config": payload.get("loss_config"),
            "koopman_metadata": payload.get("koopman_metadata"),
        }
        signature_text = json.dumps(signature, sort_keys=True, default=_jsonable)
        if reference_signature is None:
            reference_signature = signature_text
        elif signature_text != reference_signature:
            raise ValueError("Selected diffusion checkpoints are not configuration-compatible")

        k2_state = {
            key: value for key, value in payload["model_state"].items()
            if key.startswith("vehicle_model.koopman.")
        }
        if not k2_state:
            raise ValueError(f"{method} checkpoint does not contain frozen K2 weights")
        digest = hashlib.sha256()
        for key in sorted(k2_state):
            digest.update(key.encode("utf-8"))
            digest.update(k2_state[key].detach().cpu().contiguous().numpy().tobytes())
        k2_hash = digest.hexdigest()
        if reference_k2_hash is None:
            reference_k2_hash = k2_hash
        elif k2_hash != reference_k2_hash:
            raise ValueError("Selected A0/A1/A2 checkpoints use different frozen K2 weights")

        model.to(device).eval()
        models[method] = model
        checkpoint_info[method] = {
            "path": str(path),
            "sha256": _sha256(path),
            "strategy": strategy,
            "epoch": payload.get("epoch"),
            "k2_state_sha256": k2_hash,
            "anchors_config": payload.get("anchors_config", {}),
            "num_modes": int(checkpoint_num_modes) if checkpoint_num_modes is not None else None,
        }
        # Inference needs the loaded model and provenance, not optimizer state
        # or a second copy of all weights for every requested method.
        del payload, k2_state
    return models, checkpoint_info


def _teacher_config_from_a2(model):
    scorer_path = (getattr(model, "anchors_config", None) or {}).get("scorer_checkpoint")
    if not scorer_path:
        raise ValueError("A2 diffusion checkpoint does not specify anchors.scorer_checkpoint")
    path = resolve_repo_file(scorer_path)
    if not path.is_file():
        raise FileNotFoundError(f"A2 safety scorer checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    teacher = payload.get("teacher_config")
    if not isinstance(teacher, dict) or not teacher:
        raise ValueError("A2 scorer checkpoint has no teacher_config")
    return teacher, path


def _extract_features_numpy(features: Mapping, index: int) -> dict:
    names = (
        "ego_state", "agent_states", "agent_valid_mask",
        "map_polylines", "map_valid_mask", "target_point",
    )
    missing = [key for key in names if key not in features]
    if missing:
        raise KeyError(f"Figure generation requires feature fields: {missing}")
    return {
        key: features[key][index].detach().cpu().numpy().copy()
        for key in names
    }


def _filter_a1_and_sample_frames(subset, *, expected_anchor_count: int,
                                 num_frames: int, seed: int, prepare_anchors,
                                 device="cpu", batch_size: int = 128, num_workers: int = 0):
    """Keep synchronized frames whose every ego has K valid *online* A1 anchors.

    A1 is regenerated in batches from raw road geometry. No prebuilt A1 tensor is
    expected in the raw dataset. Progress is reported during the scan.
    """
    groups = defaultdict(list)
    loader = make_loader(subset, batch_size=batch_size, num_workers=num_workers)
    local_cursor = 0
    total = len(subset)
    for batch_index, (features_cpu, _targets, metadata) in enumerate(loader, start=1):
        features = _features_to_device(features_cpu, device)
        with torch.inference_mode():
            prepared = prepare_anchors(features)
        if "coarse_trajectories" not in prepared:
            raise KeyError("Online A1 builder did not return coarse_trajectories")
        xy = prepared["coarse_trajectories"]
        if tuple(xy.shape[1:]) != (expected_anchor_count, 8, 2):
            raise ValueError(
                f"A1 online output must be [B,{expected_anchor_count},8,2], got {tuple(xy.shape)}"
            )
        finite = torch.isfinite(xy).all(dim=(-1, -2))
        path_length = torch.linalg.vector_norm(xy[:, :, 1:] - xy[:, :, :-1], dim=-1).sum(dim=-1)
        valid_count = (finite & (path_length > 0.5)).sum(dim=1)
        for offset, meta in enumerate(metadata):
            key = (_meta_int(meta, "episode_index"), _meta_int(meta, "frame_index"))
            groups[key].append((local_cursor + offset,
                                bool(valid_count[offset].item() == expected_anchor_count)))
        local_cursor += len(metadata)
        if batch_index == 1 or batch_index % 20 == 0 or local_cursor >= total:
            print(f"[A1 FILTER] batch={batch_index} samples={local_cursor}/{total}", flush=True)

    complete_keys = sorted(key for key, entries in groups.items() if all(ok for _, ok in entries))
    skipped_keys = sorted(key for key, entries in groups.items() if not all(ok for _, ok in entries))
    if not complete_keys:
        raise RuntimeError(
            f"No synchronized frames contain {expected_anchor_count} geometrically valid online A1 anchors for every ego."
        )
    if num_frames > 0 and num_frames < len(complete_keys):
        rng = random.Random(int(seed))
        selected_keys = sorted(rng.sample(complete_keys, int(num_frames)))
    else:
        selected_keys = complete_keys
    selected_local = [index for key in selected_keys for index, _ in groups[key]]
    original_indices = [subset.indices[index] for index in selected_local]
    filtered = Subset(subset.dataset, original_indices)
    return filtered, {
        "a1_full_anchor_frames_available": len(groups),
        "a1_full_anchor_frames_retained_before_sampling": len(complete_keys),
        "a1_full_anchor_frames_skipped": len(skipped_keys),
        "a1_selected_frames": len(selected_keys),
        "a1_common_subset_samples": len(filtered),
        "a1_skipped_frame_examples": skipped_keys[:20],
        "a1_mode_valid_mask_ignored_for_anchor_quality": True,
        "a1_generation": "online A1AnchorManager/A1AnchorBuilder from raw road geometry",
        "a1_geometry_rule": f"finite XY and travelled path length > 0.5 m; exactly {expected_anchor_count} anchors",
    }


def _select_raw_sample_subset(dataset, *, num_samples: int, seed: int, split: str):
    """Select controlled-ego samples directly; anchor quality does not need synchronized groups."""
    count = len(dataset)
    if count <= 0:
        raise RuntimeError("Selected raw expert split is empty")
    if num_samples <= 0 or num_samples >= count:
        indices = np.arange(count, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(count, size=int(num_samples), replace=False).astype(np.int64))
    subset = Subset(dataset, indices.tolist())
    return subset, {
        "split": str(split),
        "episode_split": getattr(dataset, "split_manifest", {}),
        "input_format": "current_anchor_independent_raw_expert_dataset",
        "dataset_samples_in_split": count,
        "selected_samples_before_a1_filter": int(len(indices)),
        "selected_samples": int(len(indices)),
        "sample_seed": int(seed),
        "sampling": "uniform without replacement over controlled ego samples",
    }


def _filter_a1_valid_samples(subset, *, expected_anchor_count: int, prepare_anchors,
                             device="cpu", batch_size: int = 128, num_workers: int = 0):
    """For anchor-only evaluation, retain individually valid samples without frame grouping."""
    loader = make_loader(subset, batch_size=batch_size, num_workers=num_workers)
    kept_local = []
    cursor = 0
    total = len(subset)
    for batch_index, (features_cpu, _targets, _metadata) in enumerate(loader, start=1):
        features = _features_to_device(features_cpu, device)
        with torch.inference_mode():
            prepared = prepare_anchors(features)
        xy = prepared["coarse_trajectories"]
        if tuple(xy.shape[1:]) != (expected_anchor_count, 8, 2):
            raise ValueError(f"A1 online output shape mismatch: {tuple(xy.shape)}")
        finite = torch.isfinite(xy).all(dim=(-1, -2))
        path_length = torch.linalg.vector_norm(xy[:, :, 1:] - xy[:, :, :-1], dim=-1).sum(dim=-1)
        good = (finite & (path_length > 0.5)).sum(dim=1) == expected_anchor_count
        kept_local.extend((cursor + i) for i, ok in enumerate(good.detach().cpu().tolist()) if ok)
        cursor += int(xy.shape[0])
        if batch_index == 1 or batch_index % 20 == 0 or cursor >= total:
            print(f"[A1 ONLINE FILTER] batch={batch_index} samples={cursor}/{total} valid={len(kept_local)}", flush=True)
    if not kept_local:
        raise RuntimeError("No raw samples have K geometrically valid online A1 anchors")
    original_indices = [subset.indices[i] for i in kept_local]
    return Subset(subset.dataset, original_indices), {
        "a1_filter_input_samples": total,
        "a1_valid_samples": len(kept_local),
        "a1_invalid_samples": total - len(kept_local),
        "a1_geometry_rule": f"finite XY and travelled path length > 0.5 m; exactly {expected_anchor_count} anchors",
    }


def _anchor_pass(method, model, loader, device, teacher_cfg, expected_count, figure_metas):
    rows, figure_payload = [], {}
    figure_keys = set(figure_metas)
    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        for batch_index, (features_cpu, targets_cpu, metadata) in enumerate(loader, start=1):
            features = _features_to_device(features_cpu, device)
            expert_xy = targets_cpu["trajectory"].to(device)
            prepared = model._prepare_anchors(features)
            xy = prepared["coarse_trajectories"]
            controls = prepared["coarse_controls"]
            valid = geometry_valid_mask(_strategy(model), prepared, targets_cpu, xy)
            safety = anchor_safety_cost(prepared, xy, valid=valid, **teacher_cfg)
            metrics = compute_anchor_metrics(
                xy=xy,
                controls=controls,
                expert_xy=expert_xy,
                valid=valid,
                safety=safety,
                expected_anchor_count=expected_count,
            )
            for index, meta in enumerate(metadata):
                key = (
                    _meta_int(meta, "episode_index"),
                    _meta_int(meta, "frame_index"),
                    _meta_int(meta, "ego_index"),
                )
                rows.append({
                    "method": method,
                    "episode_index": key[0],
                    "frame_index": key[1],
                    "ego_index": key[2],
                    "dataset_seed": _meta_int(meta, "seed", -1),
                    "anchor_minade_m": float(metrics["anchor_minade_m"][index].cpu()),
                    "safe_ratio": float(metrics["safe_ratio"][index].cpu()),
                    "safe_anchor_count": int(metrics["safe_anchor_count"][index].cpu()),
                    "valid_anchor_count": int(metrics["valid_anchor_count"][index].cpu()),
                    "trajectory_diversity_m": float(metrics["trajectory_diversity_m"][index].cpu()),
                    "control_diversity_normalized": float(metrics["control_diversity_normalized"][index].cpu()),
                })
                if key in figure_keys:
                    figure_payload[key] = {
                        "features": _extract_features_numpy(prepared, index),
                        "expert_xy": expert_xy[index].detach().cpu().numpy(),
                        "anchors_xy": xy[index].detach().cpu().numpy(),
                        "valid": valid[index].detach().cpu().numpy(),
                        "hard_safe": safety['hard_safe'][index].detach().cpu().numpy(),
                        "safety": {
                            "valid_count": int(metrics["valid_anchor_count"][index].cpu()),
                            "safe_count": int(metrics["safe_anchor_count"][index].cpu()),
                            "unsafe_count": int((valid[index] & ~safety['hard_safe'][index]).sum().cpu()),
                            "collision_unsafe_count": int(safety['collision_unsafe'][index].sum().cpu()),
                            "road_unsafe_count": int(safety['road_unsafe'][index].sum().cpu()),
                        },
                    }
            if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
                processed = min(batch_index * loader.batch_size, len(loader.dataset))
                print(f"[ANCHOR METRICS] method={method} batch={batch_index}/{len(loader)} samples={processed}/{len(loader.dataset)}", flush=True)
    if not rows:
        raise RuntimeError(f"{method} anchor evaluation produced no samples")
    return rows, figure_payload


def _trajectory_safety_record(method, seed, meta, features, local_poses, pose0,
                              planning_minade, vehicle_cfg, planning_safety_cfg):
    """Common geometry and safety protocol for model and expert trajectories."""
    world = local_to_world_poses(local_poses, pose0)
    road = road_geometry_from_features(features, pose0)
    external_ok = external_collision_free(local_poses, features, vehicle_cfg)
    external_ttc = external_min_ttc_s(
        local_poses, features, vehicle_cfg,
        closing_speed_epsilon_mps=float(planning_safety_cfg.get('closing_speed_epsilon_mps', 0.1)))
    dac = drivable_area_compliance(
        world, road, vehicle_cfg,
        tolerance_m=float(planning_safety_cfg.get('drivable_area_tolerance_m', 1e-6)))
    return {
        'method': method, 'seed': seed,
        **{key: _meta_int(meta, key) for key in ('episode_index', 'frame_index', 'ego_index')},
        'dataset_seed': _meta_int(meta, 'seed', -1),
        'planning_minade_m': float(planning_minade),
        'external_collision_free': bool(external_ok), 'external_min_ttc_s': float(external_ttc),
        'intertruck_collision_free': True, 'intertruck_min_ttc_s': float('inf'),
        'collision': not bool(external_ok), 'drivable_area_compliance_pct': float(dac),
        'min_ttc_s': float(external_ttc),
        'ttc_compliant': bool(external_ttc >= float(planning_safety_cfg.get('ttc_threshold_s', 1.0))),
        '_raw_world_poses': world,
    }


def _expert_pass(loader, vehicle_cfg, planning_safety_cfg):
    """Evaluate each synchronized expert scene once, independently of seeds."""
    rows = []
    for batch_index, (features, targets, metadata) in enumerate(loader, start=1):
        dense = targets['trajectory_dense'].detach().cpu().numpy()
        if dense.shape[1:] != (40, 2):
            raise ValueError('Expert reference requires 40 future XY nodes at 0.1 s')
        for index, meta in enumerate(metadata):
            poses = expert_local_poses(dense[index])
            pose0 = targets['koopman_pose0_si'][index].detach().cpu().numpy()
            rows.append(_trajectory_safety_record(
                'Expert', None, meta, _extract_features_numpy(features, index), poses, pose0,
                0.0, vehicle_cfg, planning_safety_cfg))
        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
            print(f'[EXPERT METRICS] batch={batch_index}/{len(loader)} '
                  f'samples={len(rows)}/{len(loader.dataset)}', flush=True)
    if not rows:
        raise RuntimeError('Expert reference evaluation produced no samples')
    _apply_synchronized_intertruck(rows, vehicle_cfg, planning_safety_cfg)
    return rows


def _planning_seed_pass(method, model, loader, device, seed,
                        vehicle_cfg, planning_safety_cfg, figure_meta):
    """Evaluate planning metrics on the selected raw K2 trajectory, before JTP."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    rows, figure_payload = [], None
    started = last_progress = time.perf_counter()
    print(f"[PLANNING METRICS] method={method} seed={seed} samples=0/{len(loader.dataset)}", flush=True)
    model.eval()
    with torch.no_grad():
        for batch_index, (features_cpu, targets_cpu, metadata) in enumerate(loader, start=1):
            features = _features_to_device(features_cpu, device)
            context = _context_to_device(targets_cpu, device)
            expert_xy = targets_cpu["trajectory"].to(device)
            prepared = model._prepare_anchors(features)
            controls_all = model._sample_controls(prepared, generator=generator)
            prediction = model._all_predictions(controls_all, prepared, context)
            poses_all = prediction["poses"]
            if poses_all.ndim != 4 or poses_all.shape[-1] != 3:
                raise ValueError(f"Expected diffusion K2 poses [B,M,H+1,3], got {tuple(poses_all.shape)}")

            selected, _ = model._select_goal_endpoint_mode(poses_all, prepared)
            sample_indices = torch.arange(controls_all.shape[0], device=device)
            selected_poses = poses_all[sample_indices, selected]
            candidate_xy = poses_all[:, :, 5::5, :2]
            if candidate_xy.shape[-2:] != (8, 2):
                raise ValueError(
                    f"Expected eight 0.5-s diffusion trajectory points, got {tuple(candidate_xy.shape)}"
                )
            valid_modes = geometry_valid_mask(
                _strategy(model), prepared, targets_cpu, prepared["coarse_trajectories"]
            )
            planning_minade = compute_planning_minade(candidate_xy, expert_xy, valid_modes)

            for index, meta in enumerate(metadata):
                key = (
                    _meta_int(meta, "episode_index"),
                    _meta_int(meta, "frame_index"),
                    _meta_int(meta, "ego_index"),
                )
                features_np = _extract_features_numpy(prepared, index)
                pose0 = context["koopman_pose0_si"][index].detach().cpu().numpy()
                raw_local = selected_poses[index].detach().cpu().numpy()
                record = _trajectory_safety_record(
                    method, int(seed), meta, features_np, raw_local, pose0,
                    float(planning_minade[index].cpu()), vehicle_cfg, planning_safety_cfg)
                rows.append(record)
                now = time.perf_counter()
                if len(rows) == 1 or now - last_progress >= 10 or len(rows) == len(loader.dataset):
                    elapsed = now - started
                    remaining = elapsed / len(rows) * (len(loader.dataset) - len(rows))
                    print(f"[PLANNING METRICS] method={method} seed={seed} "
                          f"batch={batch_index}/{len(loader)} samples={len(rows)}/{len(loader.dataset)} "
                          f"elapsed={elapsed:.1f}s eta={remaining:.1f}s", flush=True)
                    last_progress = now

                if figure_payload is None and key == figure_meta:
                    raw_local_sampled = raw_local[5::5, :2]
                    figure_payload = {
                        "features": features_np,
                        "expert_xy": expert_xy[index].detach().cpu().numpy(),
                        "anchors_xy": prepared["coarse_trajectories"][index].detach().cpu().numpy(),
                        "valid": valid_modes[index].detach().cpu().numpy(),
                        "diffusion_candidates": candidate_xy[index].detach().cpu().numpy(),
                        "diffusion_valid": valid_modes[index].detach().cpu().numpy(),
                        "selected_raw_xy": raw_local_sampled,
                    }
    if not rows:
        raise RuntimeError(f"{method} diffusion planning-safety evaluation produced no samples")
    return rows, figure_payload


def _apply_synchronized_intertruck(rows, vehicle_cfg, planning_safety_cfg):
    """Add synchronized controlled-truck collision/TTC for model or expert rows."""
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["method"], row["seed"], row["episode_index"], row["frame_index"], row["dataset_seed"]
        )
        groups[key].append(row)

    closing_speed_epsilon_mps = float(
        planning_safety_cfg.get("closing_speed_epsilon_mps", 0.1)
    )
    ttc_threshold_s = float(planning_safety_cfg.get("ttc_threshold_s", 1.0))
    for group in groups.values():
        records = [{"raw_world_poses": item["_raw_world_poses"]} for item in group]
        collision_free = intertruck_collision_free(records, vehicle_cfg)
        intertruck_ttc = intertruck_group_min_ttc_s(
            records, vehicle_cfg,
            closing_speed_epsilon_mps=closing_speed_epsilon_mps,
        )
        for index, item in enumerate(group):
            item["intertruck_collision_free"] = bool(collision_free[index])
            item["intertruck_min_ttc_s"] = float(intertruck_ttc[index])
            item["collision"] = bool(
                (not item["external_collision_free"]) or (not collision_free[index])
            )
            item["min_ttc_s"] = float(
                min(item["external_min_ttc_s"], item["intertruck_min_ttc_s"])
            )
            item["ttc_compliant"] = bool(item["min_ttc_s"] >= ttc_threshold_s)
            item.pop("_raw_world_poses", None)



def _parse_seeds(text: str) -> list[int]:
    values = [int(value.strip()) for value in str(text).split(",") if value.strip()]
    if not values:
        raise ValueError("At least one diffusion inference seed is required")
    return values


def run(args=None):
    parser = argparse.ArgumentParser(
        description="TRACE A0/A1/A2 Adaptive Anchor Ablation (anchors / planning / all)"
    )
    parser.add_argument("--config", default="configs/evaluation/anchor_ablation.yaml")
    parser.add_argument(
        "--diffusion-config", default=None,
        help="Shared diffusion YAML; control.model.num_modes is the sole anchor-count source.",
    )
    parser.add_argument("--checkpoint-a0")
    parser.add_argument("--checkpoint-a1")
    parser.add_argument("--checkpoint-a2")
    parser.add_argument("--stage", choices=("anchors", "feasibility", "all"), default=None,
                        help="feasibility is the legacy stage name for the four planning metrics")
    parser.add_argument("--methods", nargs="+", choices=("A0", "A1", "A2"), default=None)
    parser.add_argument("--a0-bank", default=None)
    parser.add_argument("--a2-scorer", default=None)
    parser.add_argument("--scorer-config", default="configs/train/a2_scorer.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Planning stage: number of complete synchronized frames; 0 means all")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Anchor stage: number of raw controlled-ego samples; 0 means all")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Data loading processes; default 0 avoids extra PyTorch/CUDA imports on Windows")
    parser.add_argument("--seeds", default=None, help="Comma-separated diffusion inference seeds")
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--figure-scenes", type=int, default=None, help="Number of distinct frames in the anchor figure (default: 3)")
    parser.add_argument("--figure-candidate-samples", type=int, default=None,
                        help="Observed ego scenes considered for A2-versus-A1 safety advantage figures")
    parser.add_argument("--figure-seed", type=int, default=None, help="Result-independent figure sampling seed")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--jtp-config", default=None,
                        help="Vehicle footprint and time-step config for raw-K2 CR/DAC/TTC; JTP is not invoked")
    parser.add_argument("--feasibility-config", default=None,
                        help="Deprecated alias for --jtp-config")
    parser.add_argument("--skip-feasibility", action="store_true",
                        help="Compatibility alias for --stage anchors")
    parsed = parser.parse_args(args)

    if parsed.jtp_config and parsed.feasibility_config:
        parser.error("Specify only one of --jtp-config and --feasibility-config")
    if parsed.feasibility_config:
        parsed.jtp_config = parsed.feasibility_config

    config_path = resolve_repo_file(parsed.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Anchor Ablation config must be a YAML mapping")

    diffusion_config = (
        parsed.diffusion_config
        or cfg.get("diffusion_config", "configs/train/diffusion.yaml")
    )
    num_modes, diffusion_config_path = _resolve_num_modes(diffusion_config)
    legacy_num_anchors = cfg.get("num_anchors")
    if legacy_num_anchors is not None and int(legacy_num_anchors) != num_modes:
        print(
            f"[CONFIG] Ignoring stale evaluation num_anchors={legacy_num_anchors}; "
            f"using control.model.num_modes={num_modes}.",
            flush=True,
        )
    # Keep the existing staged evaluator interface while enforcing one K source.
    cfg["num_anchors"] = num_modes

    stage = parsed.stage or cfg.get("stage", "all")
    if parsed.skip_feasibility:
        if parsed.stage not in (None, "anchors"):
            parser.error("--skip-feasibility conflicts with --stage; use --stage anchors")
        stage = "anchors"
    if stage not in ("anchors", "feasibility", "all"):
        parser.error("stage must be anchors, feasibility, or all")

    methods = parsed.methods or cfg.get("methods", ["A0", "A1", "A2"])
    if not methods or len(set(methods)) != len(methods):
        parser.error("--methods must contain one or more unique methods")
    if any(method not in ("A0", "A1", "A2") for method in methods):
        parser.error("methods must be selected from A0/A1/A2")

    checkpoint_paths = {}
    if stage != "anchors":
        for method in methods:
            value = getattr(parsed, f"checkpoint_{method.lower()}")
            if not value:
                parser.error(f"--checkpoint-{method.lower()} is required for stage={stage}")
            checkpoint_paths[method] = Path(value)

    dataset_root = resolve_repo_file(parsed.dataset_root or cfg["dataset_root"])
    split = parsed.split or cfg.get("split", "test")
    num_frames = int(parsed.num_frames if parsed.num_frames is not None else cfg.get("num_frames", 0))
    # Preserve the old --num-frames smoke usage for anchor-only stage, but use
    # explicit --num-samples in new commands.
    anchor_num_samples = int(
        parsed.num_samples if parsed.num_samples is not None
        else (parsed.num_frames if parsed.num_frames is not None else cfg.get("num_samples", 5000))
    )
    batch_size = int(parsed.batch_size if parsed.batch_size is not None else cfg.get("batch_size", 16))
    workers = int(parsed.num_workers if parsed.num_workers is not None else cfg.get("num_workers", 0))
    seeds = _parse_seeds(parsed.seeds if parsed.seeds is not None else cfg.get("seeds", "0,1,2"))
    sample_seed = int(parsed.sample_seed if parsed.sample_seed is not None else cfg.get("sample_seed", 2026))
    figure_count = int(parsed.figure_scenes if parsed.figure_scenes is not None else cfg.get('figure_scenes', 3))
    figure_seed = int(parsed.figure_seed if parsed.figure_seed is not None else cfg.get('figure_seed', 2027))
    figure_candidates = int(parsed.figure_candidate_samples if parsed.figure_candidate_samples is not None
                            else cfg.get('figure_candidate_samples', 100))
    if figure_count < 1:
        parser.error('--figure-scenes must be positive')
    if figure_candidates < figure_count:
        parser.error('--figure-candidate-samples must be at least --figure-scenes')
    device_name = parsed.device or cfg.get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )

    base_output = resolve_repo_file(parsed.output_dir or cfg.get(
        "output_dir", "outputs/anchor_ablation/evaluation"
    ))
    output_dir = base_output if parsed.output_dir else base_output / stage / "_".join(methods)

    jtp_path = None
    vehicle_cfg = None
    planning_safety_cfg = dict(cfg.get("planning_safety") or {})
    planning_safety_cfg.setdefault("ttc_threshold_s", 1.0)
    planning_safety_cfg.setdefault("closing_speed_epsilon_mps", 0.1)
    planning_safety_cfg.setdefault("drivable_area_tolerance_m", 1e-6)
    if float(planning_safety_cfg["ttc_threshold_s"]) <= 0:
        parser.error("planning_safety.ttc_threshold_s must be positive")
    if float(planning_safety_cfg["closing_speed_epsilon_mps"]) < 0:
        parser.error("planning_safety.closing_speed_epsilon_mps must be nonnegative")
    if float(planning_safety_cfg["drivable_area_tolerance_m"]) < 0:
        parser.error("planning_safety.drivable_area_tolerance_m must be nonnegative")
    if stage != "anchors":
        jtp_path = resolve_repo_file(
            parsed.jtp_config or cfg.get("jtp_config", "configs/projection/jtp.yaml")
        )
        if not jtp_path.is_file():
            raise FileNotFoundError(f"Vehicle geometry config not found: {jtp_path}")
        vehicle_cfg = yaml.safe_load(jtp_path.read_text(encoding="utf-8"))
        if not isinstance(vehicle_cfg, dict):
            raise ValueError("Vehicle geometry config must be a YAML mapping")

    if 'a1_dataset_root' in cfg:
        raise ValueError('a1_dataset_root is removed; A1 is generated online from dataset_root')
    checkpoint_info, source_info = {}, {}
    scorer_path = None
    if stage == "anchors":
        from .sources import build_sources
        models, teacher_cfg, source_info = build_sources(
            methods, cfg,
            bank_path=parsed.a0_bank,
            scorer_path=parsed.a2_scorer,
            scorer_config=parsed.scorer_config,
            diffusion_config=str(diffusion_config_path),
            device=device,
        )
        for info in source_info.values():
            if info.get("scorer_path"):
                info["scorer_sha256"] = _sha256(Path(info["scorer_path"]))
                scorer_path = info["scorer_path"]
    else:
        models, checkpoint_info = _load_models(
            checkpoint_paths, device, expected_num_modes=num_modes
        )
        if stage == "all" and "A2" in models:
            teacher_cfg, scorer_path = _teacher_config_from_a2(models["A2"])
        elif stage == "all":
            scorer_cfg_path = resolve_repo_file(parsed.scorer_config)
            from highway_env.planner.anchor_scoring.config import load_scorer_config
            scorer_cfg = load_scorer_config(scorer_cfg_path)
            teacher_cfg = scorer_cfg["teacher"]
        else:
            teacher_cfg = None
        for method, model in models.items():
            source_info[method] = {
                "strategy": _strategy(model),
                "anchors_config": getattr(model, "anchors_config", {}),
            }
            if method == "A0" and getattr(model, "anchor_manager", None) is not None:
                source_info[method]["fixed_anchor_metadata"] = getattr(
                    model.anchor_manager, "metadata", {}
                )

    if stage == "anchors":
        from data_pipeline.raw_expert_dataset import RawExpertDataset
        raw_dataset = RawExpertDataset(dataset_root, split=split, max_samples=cfg.get("max_samples"))
        subset, data_manifest = _select_raw_sample_subset(
            raw_dataset, num_samples=anchor_num_samples, seed=sample_seed, split=split
        )
        print(
            f"[ANCHOR DATA] raw controlled-ego samples={len(raw_dataset)} "
            f"selected={len(subset)} split={split}", flush=True
        )
        if "A1" in methods:
            subset, a1_filter_manifest = _filter_a1_valid_samples(
                subset, expected_anchor_count=num_modes,
                prepare_anchors=models["A1"]._prepare_anchors,
                device=device, batch_size=batch_size, num_workers=workers,
            )
            data_manifest.update(a1_filter_manifest)
        data_manifest["anchor_generation"] = "online A0/A1/A2 generation from current raw scene features"
    else:
        subset, data_manifest = build_complete_frame_subset(
            dataset_root, split, num_frames=0 if 'A1' in methods else num_frames,
            seed=sample_seed, max_samples=cfg.get('max_samples'))
        if 'A1' in methods:
            subset, a1_manifest = _filter_a1_and_sample_frames(
                subset, expected_anchor_count=num_modes, num_frames=num_frames, seed=sample_seed,
                prepare_anchors=models['A1']._prepare_anchors, device=device,
                batch_size=batch_size, num_workers=workers)
            data_manifest.update(a1_manifest)
        data_manifest['anchor_generation'] = 'raw scenes; shared online A1AnchorBuilder; synchronized frame filter'

    loader = make_loader(subset, batch_size=batch_size, num_workers=workers)
    all_metadata = read_subset_metadata(subset)
    first_meta = all_metadata[0]
    figure_meta = (
        _meta_int(first_meta, "episode_index"),
        _meta_int(first_meta, "frame_index"),
        _meta_int(first_meta, "ego_index"),
    )

    figure_metas = []
    rank_figure_safety = 'A1' in methods and 'A2' in methods
    figure_selection = {'strategy': 'random distinct frames', 'candidate_samples': 0}
    if stage != 'feasibility':
        if rank_figure_safety:
            figure_metas = sample_figure_candidates(
                all_metadata, count=figure_candidates, seed=figure_seed)
        else:
            figure_metas = select_figure_samples(
                all_metadata, count=figure_count, seed=figure_seed)
        figure_selection['candidate_samples'] = len(figure_metas)
        if len(figure_metas) < figure_count:
            print(f'[ANCHOR FIGURE] Only {len(figure_metas)} distinct frames available; plotting all.', flush=True)

    anchor_rows, planning_rows = [], []
    anchor_figures, planning_figures = {}, {}
    safety_info = {}

    for method in methods:
        model = models[method]
        if stage != "feasibility":
            print(f"[ANCHOR ABLATION] anchor quality: {method}", flush=True)
            method_anchor_rows, anchor_figure = _anchor_pass(
                method, model, loader, device, teacher_cfg,
                num_modes, figure_metas,
            )
            anchor_rows.extend(method_anchor_rows)
            if set(anchor_figure) != set(figure_metas):
                raise RuntimeError(f"{method} did not produce all common representative anchor scenes")
            anchor_figures[method] = anchor_figure

        if stage != "anchors":
            print(f"[ANCHOR ABLATION] raw-K2 planning safety metrics: {method}", flush=True)
            safety_info[method] = {
                "trajectory_source": "selected raw K2 prediction before JTP",
                "background_actor_prediction": "constant velocity",
                "controlled_vehicle_prediction": "synchronized raw K2 prediction",
                "jtp_projector_invoked": False,
                "vehicle_geometry_config": str(jtp_path),
            }
            for seed_index, seed in enumerate(seeds):
                print(f"[ANCHOR ABLATION] planning method={method}, seed={seed}", flush=True)
                method_rows, planning_figure = _planning_seed_pass(
                    method, model, loader, device, seed,
                    vehicle_cfg, planning_safety_cfg, figure_meta,
                )
                _apply_synchronized_intertruck(method_rows, vehicle_cfg, planning_safety_cfg)
                planning_rows.extend(method_rows)
                if seed_index == 0:
                    if planning_figure is None:
                        raise RuntimeError(
                            f"{method} did not produce the common representative planning scene"
                        )
                    planning_figures[method] = planning_figure

    summary_methods = list(methods)
    if stage != 'anchors':
        print('[ANCHOR ABLATION] expert trajectory reference', flush=True)
        planning_rows.extend(_expert_pass(loader, vehicle_cfg, planning_safety_cfg))
        summary_methods.append('Expert')
    summary = summarize(anchor_rows, planning_rows, methods=summary_methods)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stage != "feasibility":
        if rank_figure_safety:
            figure_metas, ranking = select_safety_improvement_scenes(anchor_figures, count=figure_count)
            figure_selection.update({
                'strategy': 'A2 advantage examples: descending A1-minus-A2 unsafe anchor count; '
                            'road then collision reduction break ties; prefer distinct episodes and unique frames',
                'candidate_samples_requested': figure_candidates,
                'candidate_seed': figure_seed,
                'quantitative_sample_selection_changed': False,
                'positive_gap_candidates': sum(row['unsafe_anchor_reduction'] > 0 for row in ranking),
                'ranking': ranking,
            })
            write_json(output_dir / 'anchor_figure_selection.json', figure_selection)
            for row in ranking:
                if row['selected']:
                    print(f"[ANCHOR FIGURE] episode={row['episode_index']} frame={row['frame_index']} "
                          f"ego={row['ego_index']} unsafe A1={row['A1']['unsafe_count']} "
                          f"A2={row['A2']['unsafe_count']} reduction={row['unsafe_anchor_reduction']}", flush=True)
            if len(figure_metas) < figure_count:
                print(f'[ANCHOR FIGURE] Only {len(figure_metas)} distinct frames available.', flush=True)
            if any(row['selected'] and row['unsafe_anchor_reduction'] <= 0 for row in ranking):
                print('[ANCHOR FIGURE] Some selected scenes do not show a positive A2 safety gap; '
                      'increase the candidate pool or evaluation sample count for more examples.', flush=True)
        write_csv(output_dir / "anchor_quality_per_sample.csv", anchor_rows)
        draw_anchor_ablation(
            output_dir / "anchor_ablation.pdf",
            [{method: anchor_figures[method][key] for method in methods} for key in figure_metas],
            scene_labels=[f"Scene {i+1}: episode {key[0]}, frame {key[1]}, ego {key[2]}"
                          for i, key in enumerate(figure_metas)],
        )
    if stage != "anchors":
        write_csv(output_dir / "planning_per_sample.csv", planning_rows)
        draw_diffusion_planning(
            output_dir / "anchor_planning_ablation.pdf",
            planning_figures,
            sample_label=f"episode {figure_meta[0]}, frame {figure_meta[1]}, ego {figure_meta[2]}",
        )

    write_csv(output_dir / "anchor_ablation_summary.csv", summary)
    write_latex_table(output_dir / "anchor_ablation_table.tex", summary)

    completed_metrics = []
    if stage != "feasibility":
        completed_metrics.extend([
            "anchor_minade_m", "safe_ratio_pct",
            "trajectory_diversity_m", "control_diversity_normalized",
        ])
    if stage != "anchors":
        completed_metrics.extend([
            "planning_minade_m", "collision_rate_pct",
            "drivable_area_compliance_pct", "ttc_compliance_pct",
        ])

    manifest = {
        "study": "TRACE Adaptive Anchor Ablation",
        "stage": stage,
        "selected_methods": methods,
        "expert_reference": {
            "evaluated": stage != 'anchors',
            "trajectory_source": "raw future_trajectory_dense; no K2 or diffusion inference",
            "heading_source": "path tangent; observed initial local yaw is zero",
            "planning_minade_definition": "zero by definition: expert reference compared with itself",
            "seed_independent": True,
            "sampling": "same synchronized scenes as model planning metrics; each scene evaluated once",
            "safety_protocol": "same vehicle footprint, road, CV background and synchronized truck checks",
        },
        "evaluation_protocol": (
            "staged: anchors computes four anchor-quality metrics on independent raw ego samples; "
            "feasibility computes planning minADE, collision rate, drivable-area compliance and TTC compliance "
            "on complete synchronized frames using the selected raw K2 trajectory before JTP; "
            "all computes all eight paper metrics; the JTP projector is never invoked"
        ),
        "methods": {
            "A0": "Fixed anchors generated from the configured fixed control-anchor bank and kinematic rollout",
            "A1": "Dynamic control samples paired with expert data and kinematic propagation",
            "A2": "Safety scoring, Top-R prescreening, and joint control/trajectory diversity selection",
        },
        "paper_metrics": [
            "anchor_minade_m", "safe_ratio_pct", "trajectory_diversity_m",
            "control_diversity_normalized", "planning_minade_m",
            "collision_rate_pct", "drivable_area_compliance_pct", "ttc_compliance_pct",
        ],
        "completed_metrics": completed_metrics,
        "definitions": {
            "anchor_minADE": f"Minimum average displacement error over {num_modes} geometrically instantiated anchors",
            "safe_ratio": f"Hard-safe anchors divided by {num_modes}, using the common A2 safety teacher's constant-velocity collision and drivable-road corridor labels",
            "trajectory_diversity": "Mean pairwise pointwise XY distance across anchor trajectories, in metres",
            "control_diversity": "Mean pairwise pointwise control distance after channel normalization by [2.5 m/s^2, 0.08 rad]",
            "planning_minADE": "Minimum ADE over valid diffusion-generated eight-point K2 execution trajectories",
            "collision_rate": "Percentage of selected raw K2 trajectories with at least one collision against uncontrolled actors under constant-velocity prediction or synchronized controlled trucks using their raw K2 predictions",
            "drivable_area_compliance": "Mean percentage of future time steps for which the full multi-circle ego footprint remains inside the union of recognized drivable-lane corridors",
            "ttc_compliance": "Percentage of selected raw K2 trajectories whose minimum predicted TTC to uncontrolled actors or synchronized controlled trucks is no less than the configured threshold",
        },
        "planning_safety": {
            "trajectory_source": "selected raw K2 prediction before JTP",
            "background_actor_prediction": "constant velocity",
            "controlled_vehicle_prediction": "synchronized raw K2 prediction",
            "ttc_threshold_s": float(planning_safety_cfg["ttc_threshold_s"]),
            "closing_speed_epsilon_mps": float(planning_safety_cfg["closing_speed_epsilon_mps"]),
            "drivable_area_tolerance_m": float(planning_safety_cfg["drivable_area_tolerance_m"]),
            "jtp_projector_invoked": False,
        },
        "jtp": {"invoked": False, "role": "not used by Anchor Ablation evaluation"},
        "jtp_config": str(jtp_path) if jtp_path else None,
        "dataset": {"root": str(dataset_root), **data_manifest},
        "split": split,
        "num_modes": num_modes,
        "diffusion_config": str(diffusion_config_path),
        "inference_seeds": seeds if stage != "anchors" else [],
        "scorer_checkpoint": str(scorer_path) if scorer_path else None,
        "scorer_sha256": _sha256(Path(scorer_path)) if scorer_path else None,
        "checkpoints": checkpoint_info,
        "anchor_sources": source_info,
        "a2_safety_teacher_config": teacher_cfg,
        "figure_sample": {
            "episode_index": figure_meta[0],
            "frame_index": figure_meta[1],
            "ego_index": figure_meta[2],
        },
        "anchor_figure_samples": [dict(episode_index=key[0], frame_index=key[1], ego_index=key[2])
                                  for key in figure_metas],
        "anchor_figure_seed": figure_seed,
        "anchor_figure_selection": figure_selection,
        "iterative_projection_enabled": False,
        "jtp_projector_invoked": False,
        "summary": summary,
        "device": str(device),
        "batch_size": batch_size,
        "sample_seed": sample_seed,
    }
    write_json(output_dir / "evaluation_manifest.json", manifest)
    write_json(output_dir / "evaluation_summary.json", {"summary": summary, "manifest": manifest})
    print(f"[ANCHOR ABLATION DONE] stage={stage} output={output_dir}", flush=True)
    for row in summary:
        print(f"[{row['method']}] {row}", flush=True)
    return {"summary": summary, "output_dir": str(output_dir), "manifest": manifest}


def main():
    run()


if __name__ == "__main__":
    main()
