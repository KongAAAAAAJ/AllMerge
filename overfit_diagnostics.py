"""Diagnostics for W2 100-sample diffusion overfit.

This tool is intentionally read-only with respect to the planner/model.  It
compares W1 anchors, W2 inference candidates, repeated diffusion samples, mode
selection, and (optionally) the train-time forward path to isolate why an
apparently successful optimization still yields large open-loop ADE/FDE.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import torch

from highway_env.planner.diffusion.config import build_structured_diffusion_config
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter
from pretraining.checkpoint_io import load_checkpoint_file
from pretraining.config_io import load_config, resolve_pin_memory
from pretraining.contract import validate_model_schema
from pretraining.dataset_adapter import (
    build_w1_dataloader,
    metadata_item,
    unpack_w1_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose W2 diffusion 100-sample trajectory overfit."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/diffusion_pretrain.json")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--split", choices=("all", "train", "val", "test"), default="all"
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--inference-repeats",
        type=int,
        default=4,
        help="Repeat stochastic infer_multimodal calls with different seeds.",
    )
    parser.add_argument(
        "--train-probe-repeats",
        type=int,
        default=4,
        help="Repeat forward_train probes; use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", default="outputs/diffusion_pretrain/overfit_diagnostics"
    )
    parser.add_argument("--strict-checkpoint", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def l2_errors(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.linalg.norm(pred[..., :2] - target[..., :2], dim=-1)
    return dist.mean(dim=-1), dist[..., -1]


def gather_mode(candidates: torch.Tensor, mode_idx: torch.Tensor) -> torch.Tensor:
    batch_idx = torch.arange(candidates.shape[0], device=candidates.device)
    return candidates[batch_idx, mode_idx]


def candidate_metrics(
    candidates: torch.Tensor,
    target: torch.Tensor,
    expert_mode: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    expert_candidate = gather_mode(candidates, expert_mode)
    expert_ade, expert_fde = l2_errors(expert_candidate, target)
    all_dist = torch.linalg.norm(candidates[..., :2] - target[:, None, :, :2], dim=-1)
    candidate_ade = all_dist.mean(dim=-1)
    candidate_fde = all_dist[..., -1]
    return {
        "expert_ade": expert_ade,
        "expert_fde": expert_fde,
        "minade": candidate_ade.min(dim=-1).values,
        "minfde": candidate_fde.min(dim=-1).values,
        "candidate_ade": candidate_ade,
        "candidate_fde": candidate_fde,
        "expert_candidate": expert_candidate,
    }


def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu()


def tensor_mean(values: List[torch.Tensor]) -> float:
    if not values:
        return float("nan")
    return float(torch.cat(values).float().mean().item())


def tensor_std(values: List[torch.Tensor]) -> float:
    if not values:
        return float("nan")
    tensor = torch.cat(values).float()
    return float(tensor.std(unbiased=False).item())


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) < 1e-9:
        return None
    return numerator / denominator


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    return sum(values) / len(values) if values else None


def summarize_by_mode(records: List[dict], n_modes: int) -> Dict[str, dict]:
    grouped = defaultdict(list)
    for row in records:
        grouped[int(row["expert_mode"])].append(row)

    result = {}
    for mode in range(n_modes):
        rows = grouped.get(mode, [])
        if not rows:
            result[str(mode)] = {"count": 0}
            continue
        result[str(mode)] = {
            "count": len(rows),
            "selected_ADE": mean_or_none(row["selected_ade_mean"] for row in rows),
            "expert_mode_ADE": mean_or_none(row["expert_mode_ade_mean"] for row in rows),
            "minADE": mean_or_none(row["minade_mean"] for row in rows),
            "expert_anchor_ADE": mean_or_none(row["expert_anchor_ade"] for row in rows),
            "masked_mode_accuracy": mean_or_none(row["masked_mode_accuracy_over_repeats"] for row in rows),
        }
    return result


def build_heuristics(summary: dict) -> List[dict]:
    m = summary["metrics"]
    notes = []

    mode_acc = m.get("masked_mode_accuracy_over_repeats")
    selection_gap = m.get("selected_minus_expert_mode_ADE")
    oracle_gap = m.get("expert_mode_minus_oracle_ADE")
    anchor_gain = m.get("expert_mode_ADE_gain_vs_anchor")
    repeat_gain = m.get("best_RxM_ADE_gain_vs_single_repeat_oracle")
    train_infer_ratio = m.get("train_probe_vs_inference_expert_mode_ADE_ratio")

    if mode_acc is not None and mode_acc >= 0.8:
        notes.append({
            "signal": "mode_selection_not_primary",
            "evidence": f"masked mode accuracy across repeats is {mode_acc:.3f}",
        })
    if selection_gap is not None and selection_gap >= 2.0:
        notes.append({
            "signal": "mode_selection_adds_large_error",
            "evidence": f"selected ADE exceeds expert-mode ADE by {selection_gap:.3f} m",
        })
    if oracle_gap is not None and oracle_gap >= 2.0:
        notes.append({
            "signal": "expert_mode_candidate_not_best_geometry",
            "evidence": f"expert-mode ADE exceeds per-repeat oracle minADE by {oracle_gap:.3f} m",
        })
    if anchor_gain is not None:
        if anchor_gain <= 0.0:
            notes.append({
                "signal": "learned_candidate_not_improving_expert_anchor",
                "evidence": f"expert-mode candidate improves anchor ADE by only {anchor_gain:.3f} m",
            })
        elif anchor_gain < 1.0:
            notes.append({
                "signal": "weak_regression_gain_over_anchor",
                "evidence": f"expert-mode candidate improves anchor ADE by only {anchor_gain:.3f} m",
            })
    if repeat_gain is not None and repeat_gain >= 2.0:
        notes.append({
            "signal": "stochastic_inference_instability",
            "evidence": f"best over repeats+mode improves single-repeat oracle ADE by {repeat_gain:.3f} m",
        })
    if train_infer_ratio is not None and train_infer_ratio <= 0.7:
        notes.append({
            "signal": "train_inference_gap",
            "evidence": (
                "forward_train expert-mode ADE is much lower than inference; "
                f"ratio={train_infer_ratio:.3f}"
            ),
        })
    if not notes:
        notes.append({
            "signal": "no_single_dominant_failure",
            "evidence": "Inspect per-mode, per-timestep, anchor, and train-probe sections together.",
        })
    return notes


def main() -> int:
    args = parse_args()
    if args.inference_repeats < 1:
        raise ValueError("--inference-repeats must be >= 1")
    if args.train_probe_repeats < 0:
        raise ValueError("--train-probe-repeats must be >= 0")

    cfg = load_config(args.config)
    model_overrides = dict(cfg.get("model") or {})
    state_dict, stored_config = load_checkpoint_file(args.checkpoint)
    if stored_config:
        if model_overrides and model_overrides != stored_config:
            raise RuntimeError("Config model overrides disagree with checkpoint model config")
        model_overrides = stored_config

    device = resolve_device(args.device)
    model_config = build_structured_diffusion_config(**model_overrides)
    validate_model_schema(model_config)
    adapter = PlannerTensorAdapter(model_config, device)
    model = StructuredDiffusionPlanner(model_config, adapter).to(device)
    model.load_state_dict(state_dict, strict=bool(args.strict_checkpoint))
    model.eval()

    loader = build_w1_dataloader(
        args.dataset_root,
        args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=False,
        prefetch_factor=2,
        pin_memory=resolve_pin_memory("auto"),
        limit_samples=max(int(args.num_samples), 0),
        shuffle=False,
        include_metadata=True,
    )

    scalar_buffers: Dict[str, List[torch.Tensor]] = defaultdict(list)
    timestep_buffers: Dict[str, List[torch.Tensor]] = defaultdict(list)
    records: List[dict] = []
    mode_count = Counter()
    semantic_count = Counter()
    train_probe_available = False
    train_probe_error: Optional[str] = None
    n_modes_seen = 0

    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            batch = unpack_w1_batch(raw_batch)
            features = {key: value.to(device) for key, value in batch["features"].items()}
            target = batch["expert_trajectory"].to(device)
            expert_mode = batch["expert_mode"].to(device).long()
            expert_semantic = batch["expert_semantic"].to(device).long()
            coarse = features["coarse_trajectories"]
            n_modes_seen = int(coarse.shape[1])
            b = int(target.shape[0])

            for x in expert_mode.detach().cpu().tolist():
                mode_count[int(x)] += 1
            for x in expert_semantic.detach().cpu().tolist():
                semantic_count[int(x)] += 1

            anchor = candidate_metrics(coarse, target, expert_mode)
            scalar_buffers["expert_anchor_ade"].append(to_cpu(anchor["expert_ade"]))
            scalar_buffers["expert_anchor_fde"].append(to_cpu(anchor["expert_fde"]))
            scalar_buffers["anchor_minade"].append(to_cpu(anchor["minade"]))
            scalar_buffers["anchor_minfde"].append(to_cpu(anchor["minfde"]))

            expert_anchor = gather_mode(coarse, expert_mode)
            anchor_timestep_error = torch.linalg.norm(expert_anchor - target, dim=-1)
            timestep_buffers["expert_anchor"].append(to_cpu(anchor_timestep_error))

            repeat_selected_ade = []
            repeat_selected_fde = []
            repeat_expert_ade = []
            repeat_expert_fde = []
            repeat_minade = []
            repeat_minfde = []
            repeat_raw_idx = []
            repeat_masked_idx = []
            repeat_candidates = []
            repeat_expert_candidates = []
            repeat_selected_t = []
            repeat_expert_t = []
            repeat_oracle_ade_all = []
            repeat_oracle_fde_all = []

            for repeat in range(args.inference_repeats):
                seed_all(args.seed + batch_index * 1009 + repeat)
                output = model.infer_multimodal(features)
                candidates = output["trajectory_candidates"]
                selected = output["trajectory"]
                raw_logits = output["trajectory_mode_logits"]
                masked_idx = output["trajectory_mode_idx"].long()
                raw_idx = raw_logits.argmax(dim=-1)

                selected_ade, selected_fde = l2_errors(selected, target)
                cm = candidate_metrics(candidates, target, expert_mode)
                repeat_selected_ade.append(selected_ade)
                repeat_selected_fde.append(selected_fde)
                repeat_expert_ade.append(cm["expert_ade"])
                repeat_expert_fde.append(cm["expert_fde"])
                repeat_minade.append(cm["minade"])
                repeat_minfde.append(cm["minfde"])
                repeat_raw_idx.append(raw_idx)
                repeat_masked_idx.append(masked_idx)
                repeat_candidates.append(candidates)
                repeat_expert_candidates.append(cm["expert_candidate"])
                repeat_oracle_ade_all.append(cm["candidate_ade"])
                repeat_oracle_fde_all.append(cm["candidate_fde"])
                repeat_selected_t.append(torch.linalg.norm(selected - target, dim=-1))
                repeat_expert_t.append(torch.linalg.norm(cm["expert_candidate"] - target, dim=-1))

            selected_ade_r = torch.stack(repeat_selected_ade, dim=0)
            selected_fde_r = torch.stack(repeat_selected_fde, dim=0)
            expert_ade_r = torch.stack(repeat_expert_ade, dim=0)
            expert_fde_r = torch.stack(repeat_expert_fde, dim=0)
            minade_r = torch.stack(repeat_minade, dim=0)
            minfde_r = torch.stack(repeat_minfde, dim=0)
            raw_idx_r = torch.stack(repeat_raw_idx, dim=0)
            masked_idx_r = torch.stack(repeat_masked_idx, dim=0)
            candidates_r = torch.stack(repeat_candidates, dim=0)
            expert_candidates_r = torch.stack(repeat_expert_candidates, dim=0)
            oracle_ade_rm = torch.stack(repeat_oracle_ade_all, dim=0)
            oracle_fde_rm = torch.stack(repeat_oracle_fde_all, dim=0)
            selected_t_r = torch.stack(repeat_selected_t, dim=0)
            expert_t_r = torch.stack(repeat_expert_t, dim=0)

            scalar_buffers["selected_ade"].append(to_cpu(selected_ade_r.mean(dim=0)))
            scalar_buffers["selected_fde"].append(to_cpu(selected_fde_r.mean(dim=0)))
            scalar_buffers["expert_mode_ade"].append(to_cpu(expert_ade_r.mean(dim=0)))
            scalar_buffers["expert_mode_fde"].append(to_cpu(expert_fde_r.mean(dim=0)))
            scalar_buffers["minade"].append(to_cpu(minade_r.mean(dim=0)))
            scalar_buffers["minfde"].append(to_cpu(minfde_r.mean(dim=0)))
            scalar_buffers["best_rxm_ade"].append(to_cpu(oracle_ade_rm.amin(dim=(0, 2))))
            scalar_buffers["best_rxm_fde"].append(to_cpu(oracle_fde_rm.amin(dim=(0, 2))))
            scalar_buffers["raw_correct"].append(to_cpu((raw_idx_r == expert_mode[None]).float().mean(dim=0)))
            scalar_buffers["masked_correct"].append(to_cpu((masked_idx_r == expert_mode[None]).float().mean(dim=0)))

            # Fraction of repeats agreeing with each sample's most frequent masked mode.
            stability = []
            for i in range(b):
                counts = torch.bincount(masked_idx_r[:, i], minlength=n_modes_seen)
                stability.append(float(counts.max().item()) / args.inference_repeats)
            scalar_buffers["masked_mode_stability"].append(torch.tensor(stability))

            if args.inference_repeats > 1:
                repeat_std_xy = expert_candidates_r.float().std(dim=0, unbiased=False)
                repeat_std_m = torch.linalg.norm(repeat_std_xy, dim=-1).mean(dim=-1)
            else:
                repeat_std_m = torch.zeros(b, device=device)
            scalar_buffers["expert_candidate_repeat_std_m"].append(to_cpu(repeat_std_m))

            timestep_buffers["selected"].append(to_cpu(selected_t_r.mean(dim=0)))
            timestep_buffers["expert_mode"].append(to_cpu(expert_t_r.mean(dim=0)))

            # Optional train-time path probe.  Any API mismatch is reported, not hidden.
            train_probe_expert = []
            train_probe_min = []
            train_probe_target_match = []
            if args.train_probe_repeats > 0 and train_probe_error is None:
                try:
                    for repeat in range(args.train_probe_repeats):
                        seed_all(args.seed + 100000 + batch_index * 1009 + repeat)
                        train_out = model.forward_train(features, target, expert_semantic)
                        train_candidates = train_out.get("trajectory_candidates_train")
                        if train_candidates is None:
                            raise KeyError("forward_train output lacks trajectory_candidates_train")
                        tcm = candidate_metrics(train_candidates, target, expert_mode)
                        train_probe_expert.append(tcm["expert_ade"])
                        train_probe_min.append(tcm["minade"])
                        target_mode = train_out.get("target_mode")
                        if target_mode is not None:
                            train_probe_target_match.append((target_mode.long() == expert_mode).float())
                    train_probe_available = True
                except Exception as exc:
                    train_probe_error = f"{type(exc).__name__}: {exc}"

            if train_probe_expert:
                train_expert_r = torch.stack(train_probe_expert, dim=0)
                train_min_r = torch.stack(train_probe_min, dim=0)
                scalar_buffers["train_probe_expert_mode_ade"].append(to_cpu(train_expert_r.mean(dim=0)))
                scalar_buffers["train_probe_minade"].append(to_cpu(train_min_r.mean(dim=0)))
                if train_probe_target_match:
                    target_match_r = torch.stack(train_probe_target_match, dim=0)
                    scalar_buffers["train_probe_target_mode_match"].append(to_cpu(target_match_r.mean(dim=0)))

            mean_selected = selected_ade_r.mean(dim=0)
            mean_selected_fde = selected_fde_r.mean(dim=0)
            mean_expert = expert_ade_r.mean(dim=0)
            mean_expert_fde = expert_fde_r.mean(dim=0)
            mean_min = minade_r.mean(dim=0)
            mean_minfde = minfde_r.mean(dim=0)
            raw_correct_mean = (raw_idx_r == expert_mode[None]).float().mean(dim=0)
            masked_correct_mean = (masked_idx_r == expert_mode[None]).float().mean(dim=0)

            for i in range(b):
                records.append({
                    "scenario": metadata_item(batch["metadata"], "scenario", i, ""),
                    "seed": metadata_item(batch["metadata"], "seed", i, -1),
                    "episode_index": metadata_item(batch["metadata"], "episode_index", i, -1),
                    "frame_index": metadata_item(batch["metadata"], "frame_index", i, -1),
                    "ego_index": metadata_item(batch["metadata"], "ego_index", i, -1),
                    "expert_mode": int(expert_mode[i].item()),
                    "expert_semantic": int(expert_semantic[i].item()),
                    "selected_ade_mean": float(mean_selected[i].item()),
                    "selected_fde_mean": float(mean_selected_fde[i].item()),
                    "expert_mode_ade_mean": float(mean_expert[i].item()),
                    "expert_mode_fde_mean": float(mean_expert_fde[i].item()),
                    "minade_mean": float(mean_min[i].item()),
                    "minfde_mean": float(mean_minfde[i].item()),
                    "best_rxm_ade": float(oracle_ade_rm[:, i].min().item()),
                    "best_rxm_fde": float(oracle_fde_rm[:, i].min().item()),
                    "expert_anchor_ade": float(anchor["expert_ade"][i].item()),
                    "expert_anchor_fde": float(anchor["expert_fde"][i].item()),
                    "anchor_minade": float(anchor["minade"][i].item()),
                    "anchor_minfde": float(anchor["minfde"][i].item()),
                    "raw_mode_accuracy_over_repeats": float(raw_correct_mean[i].item()),
                    "masked_mode_accuracy_over_repeats": float(masked_correct_mean[i].item()),
                    "masked_mode_stability": float(stability[i]),
                    "expert_candidate_repeat_std_m": float(repeat_std_m[i].item()),
                })

    if not records:
        raise RuntimeError("No samples processed")

    metrics = {
        "selected_ADE": tensor_mean(scalar_buffers["selected_ade"]),
        "selected_FDE": tensor_mean(scalar_buffers["selected_fde"]),
        "expert_mode_ADE": tensor_mean(scalar_buffers["expert_mode_ade"]),
        "expert_mode_FDE": tensor_mean(scalar_buffers["expert_mode_fde"]),
        "minADE": tensor_mean(scalar_buffers["minade"]),
        "minFDE": tensor_mean(scalar_buffers["minfde"]),
        "best_RxM_ADE": tensor_mean(scalar_buffers["best_rxm_ade"]),
        "best_RxM_FDE": tensor_mean(scalar_buffers["best_rxm_fde"]),
        "expert_anchor_ADE": tensor_mean(scalar_buffers["expert_anchor_ade"]),
        "expert_anchor_FDE": tensor_mean(scalar_buffers["expert_anchor_fde"]),
        "anchor_minADE": tensor_mean(scalar_buffers["anchor_minade"]),
        "anchor_minFDE": tensor_mean(scalar_buffers["anchor_minfde"]),
        "raw_mode_accuracy_over_repeats": tensor_mean(scalar_buffers["raw_correct"]),
        "masked_mode_accuracy_over_repeats": tensor_mean(scalar_buffers["masked_correct"]),
        "masked_mode_stability": tensor_mean(scalar_buffers["masked_mode_stability"]),
        "expert_candidate_repeat_std_m": tensor_mean(scalar_buffers["expert_candidate_repeat_std_m"]),
    }
    metrics["selected_minus_expert_mode_ADE"] = metrics["selected_ADE"] - metrics["expert_mode_ADE"]
    metrics["expert_mode_minus_oracle_ADE"] = metrics["expert_mode_ADE"] - metrics["minADE"]
    metrics["expert_mode_ADE_gain_vs_anchor"] = metrics["expert_anchor_ADE"] - metrics["expert_mode_ADE"]
    metrics["oracle_ADE_gain_vs_anchor_oracle"] = metrics["anchor_minADE"] - metrics["minADE"]
    metrics["best_RxM_ADE_gain_vs_single_repeat_oracle"] = metrics["minADE"] - metrics["best_RxM_ADE"]

    if train_probe_available and scalar_buffers["train_probe_expert_mode_ade"]:
        metrics["train_probe_expert_mode_ADE"] = tensor_mean(scalar_buffers["train_probe_expert_mode_ade"])
        metrics["train_probe_minADE"] = tensor_mean(scalar_buffers["train_probe_minade"])
        if scalar_buffers["train_probe_target_mode_match"]:
            metrics["train_probe_target_mode_match"] = tensor_mean(scalar_buffers["train_probe_target_mode_match"])
        metrics["train_probe_vs_inference_expert_mode_ADE_ratio"] = safe_ratio(
            metrics["train_probe_expert_mode_ADE"], metrics["expert_mode_ADE"]
        )
    else:
        metrics["train_probe_expert_mode_ADE"] = None
        metrics["train_probe_minADE"] = None
        metrics["train_probe_target_mode_match"] = None
        metrics["train_probe_vs_inference_expert_mode_ADE_ratio"] = None

    # Mean point-wise error reveals late-horizon drift versus uniform offset.
    per_timestep = {}
    for name, chunks in timestep_buffers.items():
        if not chunks:
            continue
        values = torch.cat(chunks, dim=0)
        per_timestep[name] = [float(x) for x in values.mean(dim=0).tolist()]

    # Correct-vs-wrong mode split uses each sample's repeated masked accuracy.
    correct_rows = [r for r in records if r["masked_mode_accuracy_over_repeats"] >= 0.999]
    wrong_rows = [r for r in records if r["masked_mode_accuracy_over_repeats"] <= 0.001]
    mixed_rows = [r for r in records if r not in correct_rows and r not in wrong_rows]
    mode_conditioned = {
        "always_correct": {
            "count": len(correct_rows),
            "selected_ADE": mean_or_none(r["selected_ade_mean"] for r in correct_rows),
            "expert_mode_ADE": mean_or_none(r["expert_mode_ade_mean"] for r in correct_rows),
        },
        "always_wrong": {
            "count": len(wrong_rows),
            "selected_ADE": mean_or_none(r["selected_ade_mean"] for r in wrong_rows),
            "expert_mode_ADE": mean_or_none(r["expert_mode_ade_mean"] for r in wrong_rows),
        },
        "mixed_across_repeats": {
            "count": len(mixed_rows),
            "selected_ADE": mean_or_none(r["selected_ade_mean"] for r in mixed_rows),
            "expert_mode_ADE": mean_or_none(r["expert_mode_ade_mean"] for r in mixed_rows),
        },
    }

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "num_samples": len(records),
        "inference_repeats": args.inference_repeats,
        "train_probe_repeats": args.train_probe_repeats,
        "mode_counts": {str(k): int(v) for k, v in sorted(mode_count.items())},
        "semantic_counts": {str(k): int(v) for k, v in sorted(semantic_count.items())},
        "metrics": metrics,
        "per_timestep_L2_m": per_timestep,
        "mode_conditioned": mode_conditioned,
        "per_expert_mode": summarize_by_mode(records, n_modes_seen),
        "train_probe": {
            "available": train_probe_available,
            "error": train_probe_error,
        },
    }
    summary["heuristic_diagnosis"] = build_heuristics(summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "overfit_diagnostics_summary.json"
    samples_path = output_dir / "overfit_diagnostics_samples.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with samples_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps(summary, indent=2))
    print(f"[diagnostics] summary={summary_path}")
    print(f"[diagnostics] samples={samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
