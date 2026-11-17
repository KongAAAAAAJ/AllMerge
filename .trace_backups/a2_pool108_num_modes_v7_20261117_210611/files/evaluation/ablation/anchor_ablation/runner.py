from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from highway_env.planner.anchor_scoring.safety import anchor_safety_cost
from highway_env.planner.diffusion.common.execution import K2_CONTEXT
from highway_env.planner.diffusion.config.experiment import resolve_repo_file
from highway_env.planner.diffusion.projection import JointTrajectoryProjector

from .data import build_complete_frame_subset, make_loader, select_figure_samples
from .paper_feasibility import (
    external_collision_free,
    intertruck_group_collision_free,
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


def _features_to_device(features: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in features.items()
    }


def _context_to_device(targets: dict, device: torch.device) -> dict:
    missing = [key for key in K2_CONTEXT if key not in targets]
    if missing:
        raise KeyError(f"TRACE JTP evaluation requires synchronized K2 context fields: {missing}")
    return {key: targets[key].to(device, non_blocking=True) for key in K2_CONTEXT}


def _slice_context(context: Mapping[str, torch.Tensor], index: int) -> dict:
    return {key: value[index:index + 1] for key, value in context.items()}


def _strategy(model) -> str:
    return str((getattr(model, "anchors_config", None) or {}).get("strategy", "dynamic")).lower()


def _load_models(checkpoints: Mapping[str, Path], device: torch.device):
    """Load only requested methods, while checking shared architecture and K2 weights."""
    from highway_env.planner.diffusion.training.control import load_control_checkpoint

    expected = {"A0": "fixed", "A1": "dynamic", "A2": "adaptive"}
    models, payloads, checkpoint_info = {}, {}, {}
    reference_signature = None
    reference_k2_hash = None
    for method, checkpoint in checkpoints.items():
        path = resolve_repo_file(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"{method} diffusion checkpoint not found: {path}")
        model, payload = load_control_checkpoint(path, device="cpu")
        strategy = _strategy(model)
        if strategy != expected[method]:
            raise ValueError(f"{method} checkpoint uses strategy={strategy!r}; expected {expected[method]!r}")
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
        models[method], payloads[method] = model, payload
        checkpoint_info[method] = {
            "path": str(path),
            "sha256": _sha256(path),
            "strategy": strategy,
            "epoch": payload.get("epoch"),
            "k2_state_sha256": k2_hash,
            "anchors_config": payload.get("anchors_config", {}),
        }
    return models, payloads, checkpoint_info


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


def _world_to_local(world_poses: np.ndarray, pose0: np.ndarray) -> np.ndarray:
    pose0 = np.asarray(pose0, dtype=np.float64).reshape(3)
    world = np.asarray(world_poses, dtype=np.float64)
    dx, dy = world[:, 0] - pose0[0], world[:, 1] - pose0[1]
    c, s = np.cos(pose0[2]), np.sin(pose0[2])
    x = c * dx + s * dy
    y = -s * dx + c * dy
    yaw = world[:, 2] - pose0[2]
    return np.stack((x, y, yaw), axis=-1)


def _filter_a1_and_sample_frames(subset, *, expected_anchor_count: int,
                                 num_frames: int, seed: int):
    """Use only synchronized frames where every A1 ego has K geometric anchors."""
    groups = defaultdict(list)
    for local_index in range(len(subset)):
        features, _targets, meta = subset[local_index]
        if "coarse_trajectories" not in features:
            raise KeyError(
                "A1 anchor filtering requires dataset feature 'coarse_trajectories'; "
                "the current expert shard must contain paired dynamic anchors."
            )
        xy = features["coarse_trajectories"]
        if isinstance(xy, torch.Tensor):
            xy = xy.detach().cpu().numpy()
        xy = np.asarray(xy, dtype=np.float64)
        if xy.shape != (expected_anchor_count, 8, 2):
            raise ValueError(
                f"A1 stored anchors must be [{expected_anchor_count},8,2], got {xy.shape}"
            )
        path_length = np.linalg.norm(np.diff(xy, axis=1), axis=-1).sum(axis=-1)
        valid = np.isfinite(xy).all(axis=(-1, -2)) & (path_length > 0.5)
        key = (_meta_int(meta, "episode_index"), _meta_int(meta, "frame_index"))
        groups[key].append((local_index, bool(valid.sum() == expected_anchor_count)))

    complete_keys = sorted(key for key, entries in groups.items() if all(ok for _, ok in entries))
    skipped_keys = sorted(key for key, entries in groups.items() if not all(ok for _, ok in entries))
    if not complete_keys:
        raise RuntimeError(
            "No synchronized frames contain ten geometrically instantiated A1 anchors for every ego. "
            "Inspect/refill the paired dynamic-anchor data; unavailable zero-filled slots must not be marked valid."
        )
    if num_frames > 0 and num_frames < len(complete_keys):
        rng = random.Random(int(seed))
        selected_keys = sorted(rng.sample(complete_keys, int(num_frames)))
    else:
        selected_keys = complete_keys

    selected_local = [
        index for key in selected_keys for index, _ in groups[key]
    ]
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
        "a1_geometry_rule": f"finite XY and travelled path length > 0.5 m; exactly {expected_anchor_count} anchors",
    }


def _anchor_pass(method, model, loader, device, teacher_cfg, expected_count, figure_metas):
    rows, figure_payload = [], {}
    figure_keys = set(figure_metas)
    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        for features_cpu, targets_cpu, metadata in loader:
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
                    }
    if not rows:
        raise RuntimeError(f"{method} anchor evaluation produced no samples")
    return rows, figure_payload


def _planning_seed_pass(method, model, projector, loader, device, seed,
                        jtp_cfg, figure_meta):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    rows, figure_payload = [], None
    model.eval()
    with torch.no_grad():
        for features_cpu, targets_cpu, metadata in loader:
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
            batch_index = torch.arange(controls_all.shape[0], device=device)
            selected_controls = controls_all[batch_index, selected]
            selected_poses = poses_all[batch_index, selected]
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
                context_i = _slice_context(context, index)
                pose0 = context_i["koopman_pose0_si"][0].detach().cpu().numpy()
                road = road_geometry_from_features(features_np, pose0)
                nominal = selected_controls[index:index + 1].detach()
                jtp_pre = projector.evaluate_preprojection(nominal, context_i, road)
                jtp_result = projector.project(nominal, context_i, road)

                raw_world = jtp_pre["trajectories"][0].detach().cpu().numpy()
                projected_world = jtp_result.trajectories[0].detach().cpu().numpy()
                raw_local = selected_poses[index].detach().cpu().numpy()
                external_ok = external_collision_free(raw_local, features_np, jtp_cfg)
                projection_success = bool(jtp_result.projector_constraints_satisfied)
                correction = (
                    float(np.linalg.norm(
                        projected_world[1:, :2] - raw_world[1:, :2], axis=-1
                    ).mean())
                    if projection_success else float("nan")
                )
                jtp_metrics = jtp_pre["metrics"]
                record = {
                    "method": method,
                    "seed": int(seed),
                    "episode_index": key[0],
                    "frame_index": key[1],
                    "ego_index": key[2],
                    "dataset_seed": _meta_int(meta, "seed", -1),
                    "planning_minade_m": float(planning_minade[index].cpu()),
                    "jtp_constraints_feasible": bool(jtp_pre["projector_constraints_satisfied"]),
                    "external_collision_feasible": bool(external_ok),
                    "intertruck_collision_feasible": True,
                    "preprojection_feasible": bool(
                        jtp_pre["projector_constraints_satisfied"] and external_ok
                    ),
                    "jtp_max_violation": float(jtp_metrics.get("max_violation", float("nan"))),
                    "jtp_violation_count_by_group": jtp_metrics.get("violation_count_by_group", {}),
                    "projection_success": projection_success,
                    "projection_status": str(jtp_result.status),
                    "projection_correction_m": correction,
                    "projection_iterations": int(jtp_result.iterations),
                    "_raw_world_poses": raw_world,
                }
                rows.append(record)

                if figure_payload is None and key == figure_meta:
                    raw_local_sampled = raw_local[5::5, :2]
                    projected_local = _world_to_local(projected_world, pose0)[5::5, :2]
                    figure_payload = {
                        "features": features_np,
                        "expert_xy": expert_xy[index].detach().cpu().numpy(),
                        "anchors_xy": prepared["coarse_trajectories"][index].detach().cpu().numpy(),
                        "valid": valid_modes[index].detach().cpu().numpy(),
                        "diffusion_candidates": candidate_xy[index].detach().cpu().numpy(),
                        "diffusion_valid": valid_modes[index].detach().cpu().numpy(),
                        "selected_raw_xy": raw_local_sampled,
                        "selected_projected_xy": projected_local,
                    }
    if not rows:
        raise RuntimeError(f"{method} diffusion feasibility evaluation produced no samples")
    return rows, figure_payload


def _apply_synchronized_intertruck(rows, jtp_cfg):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["seed"], row["episode_index"], row["frame_index"], row["dataset_seed"]
        )
        groups[key].append(row)
    for key, group in groups.items():
        collision_free = (
            intertruck_group_collision_free(
                [{"raw_world_poses": item["_raw_world_poses"]} for item in group],
                jtp_cfg,
            )
            if len(group) > 1 else True
        )
        for item in group:
            item["intertruck_collision_feasible"] = bool(collision_free)
            item["preprojection_feasible"] = bool(
                item["jtp_constraints_feasible"]
                and item["external_collision_feasible"]
                and collision_free
            )
            item.pop("_raw_world_poses", None)


def _parse_seeds(text: str) -> list[int]:
    values = [int(value.strip()) for value in str(text).split(",") if value.strip()]
    if not values:
        raise ValueError("At least one diffusion inference seed is required")
    return values


def run(args=None):
    parser = argparse.ArgumentParser(
        description="TRACE A0/A1/A2 Adaptive Anchor Ablation (anchors / feasibility / all)"
    )
    parser.add_argument("--config", default="configs/evaluation/anchor_ablation.yaml")
    parser.add_argument("--checkpoint-a0")
    parser.add_argument("--checkpoint-a1")
    parser.add_argument("--checkpoint-a2")
    parser.add_argument("--stage", choices=("anchors", "feasibility", "all"), default=None)
    parser.add_argument("--methods", nargs="+", choices=("A0", "A1", "A2"), default=None)
    parser.add_argument("--a0-bank", default=None)
    parser.add_argument("--a2-scorer", default=None)
    parser.add_argument("--scorer-config", default="configs/train/adaptive_anchors.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--num-frames", type=int, default=None,
                        help="0 evaluates all complete synchronized frames")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated diffusion inference seeds")
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--figure-scenes", type=int, default=None, help="Number of distinct frames in the anchor figure (default: 3)")
    parser.add_argument("--figure-seed", type=int, default=None, help="Result-independent figure sampling seed")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--jtp-config", default=None,
                        help="JTP configuration used for pre-projection feasibility and projection")
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
    batch_size = int(parsed.batch_size if parsed.batch_size is not None else cfg.get("batch_size", 16))
    workers = int(parsed.num_workers if parsed.num_workers is not None else cfg.get("num_workers", 0))
    seeds = _parse_seeds(parsed.seeds if parsed.seeds is not None else cfg.get("seeds", "0,1,2"))
    sample_seed = int(parsed.sample_seed if parsed.sample_seed is not None else cfg.get("sample_seed", 2026))
    figure_count = int(parsed.figure_scenes if parsed.figure_scenes is not None else cfg.get('figure_scenes', 3))
    figure_seed = int(parsed.figure_seed if parsed.figure_seed is not None else cfg.get('figure_seed', 2027))
    if figure_count < 1:
        parser.error('--figure-scenes must be positive')
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
    jtp_cfg = None
    if stage != "anchors":
        jtp_path = resolve_repo_file(
            parsed.jtp_config or cfg.get("jtp_config", "configs/projection/jtp.yaml")
        )
        if not jtp_path.is_file():
            raise FileNotFoundError(f"JTP config not found: {jtp_path}")
        jtp_cfg = yaml.safe_load(jtp_path.read_text(encoding="utf-8"))

    # If A1 participates, first identify all complete frames, then remove frames
    # where any A1 ego has a geometrically unavailable anchor. Sampling occurs
    # after this common-subset filter so the requested frame count is retained.
    if "A1" in methods:
        subset, data_manifest = build_complete_frame_subset(
            dataset_root, split, num_frames=0, seed=sample_seed,
            max_samples=cfg.get("max_samples"),
        )
        subset, a1_manifest = _filter_a1_and_sample_frames(
            subset,
            expected_anchor_count=int(cfg.get("num_anchors", 10)),
            num_frames=num_frames,
            seed=sample_seed,
        )
        data_manifest.update(a1_manifest)
    else:
        subset, data_manifest = build_complete_frame_subset(
            dataset_root, split, num_frames=num_frames, seed=sample_seed,
            max_samples=cfg.get("max_samples"),
        )
    loader = make_loader(subset, batch_size=batch_size, num_workers=workers)
    first_batch = next(iter(loader))
    first_meta = first_batch[2][0]
    figure_meta = (
        _meta_int(first_meta, "episode_index"),
        _meta_int(first_meta, "frame_index"),
        _meta_int(first_meta, "ego_index"),
    )

    figure_metas = []
    if stage != 'feasibility':
        figure_metas = select_figure_samples(
            [meta for _, _, batch_metadata in loader for meta in batch_metadata],
            count=figure_count, seed=figure_seed)
        if len(figure_metas) < figure_count:
            print(f'[ANCHOR FIGURE] Only {len(figure_metas)} distinct frames available; plotting all.', flush=True)

    checkpoint_info, source_info = {}, {}
    scorer_path = None
    if stage == "anchors":
        from .sources import build_sources
        models, teacher_cfg, source_info = build_sources(
            methods, cfg,
            bank_path=parsed.a0_bank,
            scorer_path=parsed.a2_scorer,
            scorer_config=parsed.scorer_config,
            device=device,
        )
        for info in source_info.values():
            if info.get("scorer_path"):
                info["scorer_sha256"] = _sha256(Path(info["scorer_path"]))
                scorer_path = info["scorer_path"]
    else:
        models, _payloads, checkpoint_info = _load_models(checkpoint_paths, device)
        if stage == "all" and "A2" in models:
            teacher_cfg, scorer_path = _teacher_config_from_a2(models["A2"])
        elif stage == "all":
            scorer_cfg_path = resolve_repo_file(parsed.scorer_config)
            scorer_cfg = yaml.safe_load(scorer_cfg_path.read_text(encoding="utf-8"))
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

    anchor_rows, planning_rows = [], []
    anchor_figures, planning_figures = {}, {}
    projector_info = {}

    for method in methods:
        model = models[method]
        if stage != "feasibility":
            print(f"[ANCHOR ABLATION] anchor quality: {method}", flush=True)
            method_anchor_rows, anchor_figure = _anchor_pass(
                method, model, loader, device, teacher_cfg,
                int(cfg.get("num_anchors", 10)), figure_metas,
            )
            anchor_rows.extend(method_anchor_rows)
            if set(anchor_figure) != set(figure_metas):
                raise RuntimeError(f"{method} did not produce all common representative anchor scenes")
            anchor_figures[method] = anchor_figure

        if stage != "anchors":
            print(f"[ANCHOR ABLATION] diffusion feasibility: {method}", flush=True)
            projector = JointTrajectoryProjector(model.vehicle_model, config=jtp_path)
            projector_info[method] = {
                "config_path": str(jtp_path),
                "constraint_families": ["stability", "road", "curvature", "control"],
                "external_collision_evaluated_separately": True,
                "intertruck_collision_evaluated_separately": True,
            }
            for seed_index, seed in enumerate(seeds):
                print(f"[ANCHOR ABLATION] planning method={method}, seed={seed}", flush=True)
                method_rows, planning_figure = _planning_seed_pass(
                    method, model, projector, loader, device, seed,
                    jtp_cfg, figure_meta,
                )
                _apply_synchronized_intertruck(method_rows, jtp_cfg)
                planning_rows.extend(method_rows)
                if seed_index == 0:
                    if planning_figure is None:
                        raise RuntimeError(
                            f"{method} did not produce the common representative planning scene"
                        )
                    planning_figures[method] = planning_figure

    summary = summarize(anchor_rows, planning_rows, methods=methods)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stage != "feasibility":
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
            "planning_minade_m", "preprojection_feasibility_pct",
            "projection_correction_m",
        ])

    manifest = {
        "study": "TRACE Adaptive Anchor Ablation",
        "stage": stage,
        "selected_methods": methods,
        "evaluation_protocol": "staged; use --stage all for the complete seven-metric paper result",
        "methods": {
            "A0": "Fixed anchors generated from the configured fixed control-anchor bank and kinematic rollout",
            "A1": "Dynamic control samples paired with expert data and kinematic propagation",
            "A2": "Safety scoring, Top-R prescreening, and joint control/trajectory diversity selection",
        },
        "paper_metrics": [
            "anchor_minade_m", "safe_ratio_pct", "trajectory_diversity_m",
            "control_diversity_normalized", "planning_minade_m",
            "preprojection_feasibility_pct", "projection_correction_m",
        ],
        "completed_metrics": completed_metrics,
        "definitions": {
            "anchor_minADE": "Minimum average displacement error over ten geometrically instantiated anchors",
            "safe_ratio": "Hard-safe anchors divided by ten, using the common A2 safety teacher's constant-velocity collision and drivable-road corridor labels",
            "trajectory_diversity": "Mean pairwise pointwise XY distance across anchor trajectories, in metres",
            "control_diversity": "Mean pairwise pointwise control distance after channel normalization by [2.5 m/s^2, 0.08 rad]",
            "planning_minADE": "Minimum ADE over valid diffusion-generated eight-point execution trajectories",
            "preprojection_feasibility": "JTP nominal constraints plus independent constant-velocity external-actor and synchronized inter-truck collision checks, before projection",
            "projection_correction": "Mean pointwise XY displacement between raw and successfully JTP-projected execution trajectories; projection success/failure is reported separately",
        },
        "jtp": projector_info,
        "jtp_config": str(jtp_path) if jtp_path else None,
        "dataset": {"root": str(dataset_root), **data_manifest},
        "split": split,
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
        "anchor_figure_selection": "Random distinct frames, prioritizing distinct episodes; independent of method scores",
        "projection_failures_reported_separately": True,
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
