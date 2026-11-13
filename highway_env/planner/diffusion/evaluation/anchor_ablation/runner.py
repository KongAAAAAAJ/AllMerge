from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch
import yaml

from highway_env.planner.diffusion.common.execution import K2_CONTEXT
from highway_env.planner.diffusion.config.experiment import resolve_repo_file
from highway_env.planner.diffusion.training.control import load_control_checkpoint
from highway_env.planner.anchor_scoring.safety import anchor_safety_cost

from .data import build_complete_frame_subset, make_loader
from .feasibility import (
    apply_intertruck_group_constraints,
    evaluate_single_preprojection,
    validate_feasibility_config,
)
from .metrics import compute_anchor_metrics, geometry_valid_mask
from .reporting import summarize_anchor_rows, write_csv, write_json, write_latex_table
from .visualization import draw_anchor_ablation


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
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
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in features.items()}


def _targets_to_device_context(targets: dict, device: torch.device) -> dict:
    missing = [k for k in K2_CONTEXT if k not in targets]
    if missing:
        raise KeyError(f"TRACE K2 raw-feasibility evaluation requires synchronized context: {missing}")
    return {k: targets[k].to(device, non_blocking=True) for k in K2_CONTEXT}


def _strategy(model) -> str:
    return str((model.anchors_config or {}).get("strategy", "dynamic")).lower()


def _load_models(checkpoints: Mapping[str, Path], device: torch.device):
    expected = {"A0": "fixed", "A1": "dynamic", "A2": "adaptive"}
    models, payloads, checkpoint_info = {}, {}, {}
    reference_signature = None
    reference_k2_hash = None
    for method in ("A0", "A1", "A2"):
        path = checkpoints[method].expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{method} diffusion checkpoint not found: {path}")
        model, payload = load_control_checkpoint(path, device="cpu")
        strategy = _strategy(model)
        if strategy != expected[method]:
            raise ValueError(f"{method} checkpoint uses anchor strategy={strategy!r}; expected {expected[method]!r}")
        if model.vehicle_model.kind != "koopman_k2":
            raise ValueError(f"{method} checkpoint execution model must be koopman_k2 for TRACE feasibility")
        if any(k.startswith("denoiser.mode_embedding.") for k in payload["model_state"]):
            raise RuntimeError(f"{method} checkpoint still contains semantic mode_embedding")
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
            raise ValueError(f"{method} checkpoint is not configuration-compatible with the other ablation models")
        k2_state = {k: v for k, v in payload["model_state"].items()
                    if k.startswith("vehicle_model.koopman.")}
        if not k2_state:
            raise ValueError(f"{method} checkpoint does not contain frozen K2 weights in model_state")
        digest = hashlib.sha256()
        for key in sorted(k2_state):
            digest.update(key.encode("utf-8"))
            digest.update(k2_state[key].detach().cpu().contiguous().numpy().tobytes())
        k2_hash = digest.hexdigest()
        if reference_k2_hash is None:
            reference_k2_hash = k2_hash
        elif k2_hash != reference_k2_hash:
            raise ValueError(f"{method} checkpoint uses a different frozen K2 model; raw-feasibility comparison is invalid")
        model.to(device).eval()
        models[method], payloads[method] = model, payload
        checkpoint_info[method] = {
            "path": str(path), "sha256": _sha256(path), "strategy": strategy,
            "epoch": payload.get("epoch"), "k2_state_sha256": k2_hash,
            "anchors_config": payload.get("anchors_config", {}),
        }
    return models, payloads, checkpoint_info


def _teacher_config_from_a2(model) -> dict:
    scorer_path = (model.anchors_config or {}).get("scorer_checkpoint")
    if not scorer_path:
        raise ValueError("A2 checkpoint does not specify anchors.scorer_checkpoint")
    path = resolve_repo_file(scorer_path)
    if not path.is_file():
        raise FileNotFoundError(f"A2 scorer checkpoint not found: {path}")
    scorer_payload = torch.load(path, map_location="cpu", weights_only=True)
    teacher = scorer_payload.get("teacher_config")
    if not isinstance(teacher, dict) or not teacher:
        raise ValueError("A2 scorer checkpoint has no teacher_config; cannot reproduce common Safe Anchor Ratio labels")
    return teacher


def _anchor_batch_metrics(method: str, model, features: dict, targets: dict,
                          expert_xy: torch.Tensor, teacher_cfg: dict):
    prepared = model._prepare_anchors(features)
    xy = prepared["coarse_trajectories"]
    controls = prepared["coarse_controls"]
    if xy.shape[1] != 10 or controls.shape[1] != 10:
        raise ValueError(f"{method} must supply exactly 10 selected anchors; got {xy.shape[1]}")
    valid = geometry_valid_mask(_strategy(model), prepared, targets, xy)
    safety = anchor_safety_cost(prepared, xy, valid=valid, **teacher_cfg)
    metrics = compute_anchor_metrics(
        xy=xy, controls=controls, expert_xy=expert_xy,
        valid=valid, safety=safety,
        trajectory_weight=float(model.anchors_config.get("trajectory_weight", 1.0)),
        control_weight=float(model.anchors_config.get("control_weight", 1.0)),
    )
    return prepared, valid, metrics


def _extract_features_numpy(features: Mapping, index: int) -> dict:
    names = ("ego_state", "agent_states", "agent_valid_mask", "map_polylines", "map_valid_mask", "target_point")
    return {k: features[k][index].detach().cpu().numpy().copy() for k in names}


def _run_feasibility_seed(method: str, model, loader, device: torch.device, seed: int,
                          feasibility_cfg: dict, dt: float):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    pending = defaultdict(list)
    rows = []
    model.eval()
    with torch.no_grad():
        for features_cpu, targets_cpu, metadata in loader:
            features = _features_to_device(features_cpu, device)
            context = _targets_to_device_context(targets_cpu, device)
            prepared = model._prepare_anchors(features)
            controls_all = model._sample_controls(prepared, generator=generator)
            prediction = model._all_predictions(controls_all, prepared, context)
            if prediction.get("states") is None or prediction.get("ltr") is None:
                raise RuntimeError("K2 execution must return chassis states and LTR for full TRACE feasibility")
            selected, _ = model._select_goal_endpoint_mode(prediction["poses"], prepared)
            batch_index = torch.arange(controls_all.shape[0], device=device)
            controls = controls_all[batch_index, selected]
            poses = prediction["poses"][batch_index, selected]
            states = prediction["states"][batch_index, selected]
            ltr = prediction["ltr"][batch_index, selected]
            for i, meta in enumerate(metadata):
                key = (_meta_int(meta, "episode_index"), _meta_int(meta, "frame_index"),
                       _meta_int(meta, "seed", -1))
                plan_features = _extract_features_numpy(prepared, i)
                plan = {
                    "poses": poses[i].detach().cpu().numpy(),
                    "controls": controls[i].detach().cpu().numpy(),
                    "states": states[i].detach().cpu().numpy(),
                    "ltr": ltr[i].detach().cpu().numpy(),
                    "features": plan_features,
                    "road_mu": float(context["koopman_road_mu"][i].detach().cpu().reshape(-1)[0]),
                    "payload_kg": float(context["koopman_payload_kg"][i].detach().cpu().reshape(-1)[0]),
                    "pose0": context["koopman_pose0_si"][i].detach().cpu().numpy(),
                }
                feasibility = evaluate_single_preprojection(plan, feasibility_cfg, dt=dt)
                controlled_count = int((prepared["agent_valid_mask"][i].bool() &
                                         (prepared["agent_states"][i, :, 10] > 0.5)).sum().item())
                record = {
                    "method": method, "seed": int(seed),
                    "episode_index": key[0], "frame_index": key[1], "dataset_seed": key[2],
                    "ego_index": _meta_int(meta, "ego_index"),
                    "expected_ego_count": controlled_count + 1,
                    "feasibility": feasibility,
                    "poses": plan["poses"], "pose0": plan["pose0"],
                }
                pending[key].append(record)
    for key, group in pending.items():
        expected_values = {int(x["expected_ego_count"]) for x in group}
        expected = max(expected_values)
        ego_ids = [int(x["ego_index"]) for x in group]
        if len(expected_values) != 1 or len(group) != expected or len(set(ego_ids)) != expected:
            raise RuntimeError(
                f"Incomplete synchronized platoon frame for inter-truck feasibility: "
                f"frame={key}, observed={len(group)}, expected={expected}, ego_ids={ego_ids}"
            )
        plans = [{"poses": x["poses"], "pose0": x["pose0"], "feasibility": x["feasibility"]}
                 for x in group]
        apply_intertruck_group_constraints(plans, feasibility_cfg)
        for original, updated in zip(group, plans):
            original["feasibility"] = updated["feasibility"]
            f = original["feasibility"]
            rows.append({
                "method": method, "seed": int(seed),
                "episode_index": original["episode_index"],
                "frame_index": original["frame_index"],
                "dataset_seed": original["dataset_seed"],
                "ego_index": original["ego_index"],
                "feasible": bool(f["feasible"]),
                **{k: v for k, v in f.items() if isinstance(v, (bool, int, float, str))},
            })
    return rows


def _parse_seeds(text: str):
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("At least one inference seed is required")
    return values


def run(args=None):
    parser = argparse.ArgumentParser(description="TRACE A0/A1/A2 Adaptive Anchor Ablation Evaluator")
    parser.add_argument("--config", default="configs/evaluation/anchor_ablation.yaml")
    parser.add_argument("--checkpoint-a0", required=True)
    parser.add_argument("--checkpoint-a1", required=True)
    parser.add_argument("--checkpoint-a2", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--num-frames", type=int, default=None, help="0 evaluates all complete synchronized frames")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated diffusion inference seeds")
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--feasibility-config", default="configs/evaluation/trace_feasibility.yaml")
    parser.add_argument("--skip-feasibility", action="store_true",
                        help="Only calculate the three Anchor metrics. Feasibility is exported as --.")
    parsed = parser.parse_args(args)

    config_path = resolve_repo_file(parsed.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Anchor ablation config must be a YAML mapping")
    dataset_root = resolve_repo_file(parsed.dataset_root or cfg["dataset_root"])
    split = parsed.split or cfg.get("split", "test")
    num_frames = int(parsed.num_frames if parsed.num_frames is not None else cfg.get("num_frames", 0))
    batch_size = int(parsed.batch_size if parsed.batch_size is not None else cfg.get("batch_size", 16))
    workers = int(parsed.num_workers if parsed.num_workers is not None else cfg.get("num_workers", 0))
    seeds = _parse_seeds(parsed.seeds if parsed.seeds is not None else cfg.get("seeds", "0,1,2"))
    sample_seed = int(parsed.sample_seed if parsed.sample_seed is not None else cfg.get("sample_seed", 2026))
    output_dir = resolve_repo_file(parsed.output_dir or cfg.get("output_dir", "outputs/anchor_ablation/evaluation"))
    device_name = parsed.device or cfg.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)

    feasibility_cfg = None
    if not parsed.skip_feasibility:
        feasibility_path = resolve_repo_file(parsed.feasibility_config)
        feasibility_cfg = yaml.safe_load(feasibility_path.read_text(encoding="utf-8"))
        validate_feasibility_config(feasibility_cfg)

    subset, data_manifest = build_complete_frame_subset(
        dataset_root, split, num_frames=num_frames, seed=sample_seed,
        max_samples=cfg.get("max_samples"),
    )
    loader = make_loader(subset, batch_size=batch_size, num_workers=workers)
    checkpoint_paths = {
        "A0": Path(parsed.checkpoint_a0),
        "A1": Path(parsed.checkpoint_a1),
        "A2": Path(parsed.checkpoint_a2),
    }
    models, payloads, checkpoint_info = _load_models(checkpoint_paths, device)
    if feasibility_cfg is not None:
        control_cfg = feasibility_cfg.setdefault("control", {})
        if control_cfg.get("acceleration_min_mps2") is None:
            control_cfg["acceleration_min_mps2"] = -float(models["A0"].ctrl_cfg.max_accel_mps2)
        if control_cfg.get("acceleration_max_mps2") is None:
            control_cfg["acceleration_max_mps2"] = float(models["A0"].ctrl_cfg.max_accel_mps2)
        if control_cfg.get("steering_min_rad") is None:
            control_cfg["steering_min_rad"] = -float(models["A0"].ctrl_cfg.max_steer_rad)
        if control_cfg.get("steering_max_rad") is None:
            control_cfg["steering_max_rad"] = float(models["A0"].ctrl_cfg.max_steer_rad)
    teacher_cfg = _teacher_config_from_a2(models["A2"])
    anchor_rows = []
    feasibility_rows = []
    figure_payload = {}
    figure_meta = None
    dt = float(models["A0"].ctrl_cfg.dt)

    # Same selected scenes, batch order, and random seeds are used for all methods.
    for method in ("A0", "A1", "A2"):
        model = models[method]
        model.eval()
        seen_anchor = 0
        for features_cpu, targets_cpu, metadata in loader:
            features = _features_to_device(features_cpu, device)
            expert_xy = targets_cpu["trajectory"].to(device)
            prepared, valid, metrics = _anchor_batch_metrics(
                method, model, features, targets_cpu, expert_xy, teacher_cfg)
            B = expert_xy.shape[0]
            controls = prepared["coarse_controls"]
            xy = prepared["coarse_trajectories"]
            for i, meta in enumerate(metadata):
                episode = _meta_int(meta, "episode_index")
                frame = _meta_int(meta, "frame_index")
                ego_index = _meta_int(meta, "ego_index")
                row = {
                    "method": method,
                    "episode_index": episode,
                    "frame_index": frame,
                    "ego_index": ego_index,
                    "dataset_seed": _meta_int(meta, "seed", -1),
                    "minade_m": float(metrics["minade_m"][i].detach().cpu()),
                    "safe_anchor_count": int(metrics["safe_count"][i].detach().cpu()),
                    "valid_anchor_count": int(metrics["valid_anchor_count"][i].detach().cpu()),
                    "safe_ratio": float(metrics["safe_ratio"][i].detach().cpu()),
                    "diversity_m": float(metrics["diversity_m"][i].detach().cpu()),
                    "control_diversity_normalized": float(metrics["control_diversity_normalized"][i].detach().cpu()),
                    "joint_diversity_normalized": float(metrics["joint_diversity_normalized"][i].detach().cpu()),
                }
                anchor_rows.append(row)
                if figure_meta is None:
                    figure_meta = (episode, frame, ego_index)
                if (episode, frame, ego_index) == figure_meta:
                    figure_payload[method] = {
                        "features": _extract_features_numpy(prepared, i),
                        "expert_xy": expert_xy[i].detach().cpu().numpy(),
                        "anchors_xy": xy[i].detach().cpu().numpy(),
                        "valid": valid[i].detach().cpu().numpy(),
                    }
                seen_anchor += 1
        if seen_anchor != len(subset):
            raise RuntimeError(f"{method} anchor pass processed {seen_anchor} samples, expected {len(subset)}")

        if feasibility_cfg is not None:
            for seed in seeds:
                print(f"[ANCHOR ABLATION] feasibility method={method} seed={seed}", flush=True)
                feasibility_rows.extend(_run_feasibility_seed(
                    method, model, loader, device, seed, feasibility_cfg, dt))

    summary = summarize_anchor_rows(anchor_rows, feasibility_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "anchor_metrics_per_sample.csv", anchor_rows)
    write_csv(output_dir / "feasibility_per_sample.csv", feasibility_rows)
    write_csv(output_dir / "anchor_ablation_summary.csv", summary)
    write_latex_table(output_dir / "anchor_ablation_table.tex", summary)
    if len(figure_payload) == 3:
        draw_anchor_ablation(output_dir / "anchor_ablation.pdf", figure_payload,
                             sample_label=f"episode {figure_meta[0]}, frame {figure_meta[1]}, ego {figure_meta[2]}")
    else:
        raise RuntimeError("Could not collect the same representative scene for all A0/A1/A2 methods")

    manifest = {
        "study": "TRACE Adaptive Anchor Ablation",
        "methods": {"A0": "Fixed control-space K-means centroids",
                    "A1": "Dynamic control sampling and kinematic propagation",
                    "A2": "Safety scoring, Top-R prescreening and joint control/trajectory diversity selection"},
        "paper_metrics": ["minADE_m", "safe_anchor_ratio_pct", "diversity_m", "feasibility_pct"],
        "safe_anchor_definition": "Common A2 scorer teacher hard-safe label: constant-velocity collision plus drivable-lane corridor compliance; geometry-valid anchors form the denominator.",
        "diversity_definition": "Mean pairwise trajectory ADE in metres for the paper table; normalized control/trajectory joint distance is retained as an auxiliary diagnostic.",
        "feasibility_definition": "Selected diffusion execution trajectory, before MATP projection; full-horizon K2 yaw/sideslip/roll/LTR, road, curvature, external/inter-truck collision, and control/rate constraints.",
        "feasibility_enabled": feasibility_cfg is not None,
        "feasibility_config": str(resolve_repo_file(parsed.feasibility_config)) if feasibility_cfg is not None else None,
        "feasibility_note": "If --skip-feasibility is used, the table reports -- for feasibility; do not use that run as the complete paper result.",
        "dataset": {"root": str(dataset_root), **data_manifest},
        "seeds": seeds if feasibility_cfg is not None else [],
        "figure_sample": {"episode_index": figure_meta[0], "frame_index": figure_meta[1], "ego_index": figure_meta[2]},
        "checkpoints": checkpoint_info,
        "a2_safety_teacher_config": teacher_cfg,
        "device": str(device),
        "batch_size": batch_size,
        "sample_seed": sample_seed,
        "summary": summary,
    }
    write_json(output_dir / "evaluation_manifest.json", manifest)
    write_json(output_dir / "evaluation_summary.json", {"summary": summary, "manifest": manifest})
    print("[ANCHOR ABLATION DONE]", output_dir, flush=True)
    for row in summary:
        feas_text = "--" if row["feasibility_pct"] is None else f"{row['feasibility_pct']:.2f}%"
        print(f"[{row['method']}] minADE={row['minADE_m']:.3f} m "
              f"safe={row['safe_anchor_ratio_pct']:.2f}% "
              f"diversity={row['diversity_m']:.3f} m "
              f"feasibility={feas_text}", flush=True)
    return {"summary": summary, "output_dir": str(output_dir), "manifest": manifest}


def main():
    run()


if __name__ == "__main__":
    main()
