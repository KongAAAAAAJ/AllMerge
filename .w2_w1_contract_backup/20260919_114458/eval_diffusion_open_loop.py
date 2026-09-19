"""Open-loop W2 evaluation on W1 expert shards.

Migrates the reusable trajectory metrics from Diffusion-metadrive and adapts
model/dataset I/O to StructuredDiffusionPlanner + ExpertDataset.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from data_pipeline.expert_dataset import ExpertDataset, collate_expert_samples
from highway_env.planner.diffusion.config import build_structured_diffusion_config
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter
from pretraining.checkpoint_io import load_checkpoint_file
from pretraining.contract import validate_model_schema
from train_diffusion_pretrain import build_dataloader, load_yaml, resolve_pin_memory


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def compute_ade_fde(pred: torch.Tensor, target: torch.Tensor):
    l2 = torch.linalg.norm(pred[..., :2] - target[..., :2], dim=-1)
    return l2.mean(dim=-1), l2[..., -1]


def parse_args():
    parser = argparse.ArgumentParser(description="Open-loop AllMerge diffusion evaluation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/diffusion_pretrain.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/diffusion_pretrain/open_loop_eval")
    parser.add_argument("--strict-checkpoint", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    model_overrides = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("training") or {})
    dataset_root = Path(args.dataset_root or train_cfg.get("dataset_root", "data/expert/allmerge_planner_v1"))
    device = resolve_device(args.device)

    state_dict, checkpoint_model_config = load_checkpoint_file(args.checkpoint)
    if checkpoint_model_config:
        if model_overrides and model_overrides != checkpoint_model_config:
            raise RuntimeError(
                "Config model overrides disagree with checkpoint allmerge_model_config. "
                "Use the training config or leave model: {}."
            )
        model_overrides = checkpoint_model_config

    model_config = build_structured_diffusion_config(**model_overrides)
    validate_model_schema(model_config)
    adapter = PlannerTensorAdapter(model_config, device)
    model = StructuredDiffusionPlanner(model_config, adapter).to(device)
    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=bool(args.strict_checkpoint),
    )
    if not args.strict_checkpoint and (missing or unexpected):
        print(f"[open_loop] non-strict checkpoint missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    loader = build_dataloader(
        dataset_root,
        args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=False,
        prefetch_factor=2,
        pin_memory=resolve_pin_memory("auto"),
        cache_shards=int(train_cfg.get("cache_shards", 2)),
        limit_samples=max(int(args.num_samples), 0),
        shuffle=False,
    )

    totals = {
        "selected_ade": 0.0,
        "selected_fde": 0.0,
        "minade": 0.0,
        "minfde": 0.0,
        "raw_mode_correct": 0,
        "masked_mode_correct": 0,
        "top3_mode_hit": 0,
        "expert_mode_traffic_valid": 0,
        "count": 0,
    }
    records: List[Dict] = []

    with torch.no_grad():
        for batch in loader:
            features = {key: value.to(device) for key, value in batch["features"].items()}
            target = batch["expert_trajectory"].to(device)
            expert_mode = batch["expert_mode"].to(device)
            output = model.infer_multimodal(features)

            selected = output["trajectory"]
            candidates = output["trajectory_candidates"]
            raw_logits = output["trajectory_mode_logits"]
            masked_idx = output["trajectory_mode_idx"]
            raw_idx = raw_logits.argmax(dim=-1)
            top3 = raw_logits.topk(k=min(3, raw_logits.shape[-1]), dim=-1).indices

            selected_ade, selected_fde = compute_ade_fde(selected, target)
            candidate_l2 = torch.linalg.norm(candidates - target[:, None, :, :], dim=-1)
            candidate_ade = candidate_l2.mean(dim=-1)
            candidate_fde = candidate_l2[..., -1]
            minade = candidate_ade.min(dim=-1).values
            minfde = candidate_fde.min(dim=-1).values

            traffic_valid = features["mode_valid_mask"].gather(1, expert_mode[:, None]).squeeze(1)
            top3_hit = (top3 == expert_mode[:, None]).any(dim=-1)

            b = int(target.shape[0])
            totals["selected_ade"] += float(selected_ade.sum().item())
            totals["selected_fde"] += float(selected_fde.sum().item())
            totals["minade"] += float(minade.sum().item())
            totals["minfde"] += float(minfde.sum().item())
            totals["raw_mode_correct"] += int((raw_idx == expert_mode).sum().item())
            totals["masked_mode_correct"] += int((masked_idx == expert_mode).sum().item())
            totals["top3_mode_hit"] += int(top3_hit.sum().item())
            totals["expert_mode_traffic_valid"] += int(traffic_valid.sum().item())
            totals["count"] += b

            metadata = batch["metadata"]
            for i in range(b):
                records.append({
                    "scenario": metadata[i]["scenario"],
                    "seed": metadata[i]["seed"],
                    "episode_id": metadata[i]["episode_id"],
                    "frame_id": metadata[i]["frame_id"],
                    "vehicle_role": metadata[i]["vehicle_role"],
                    "expert_mode": int(expert_mode[i].item()),
                    "raw_mode_idx": int(raw_idx[i].item()),
                    "masked_mode_idx": int(masked_idx[i].item()),
                    "expert_mode_traffic_valid": bool(traffic_valid[i].item()),
                    "selected_ade": float(selected_ade[i].item()),
                    "selected_fde": float(selected_fde[i].item()),
                    "minade": float(minade[i].item()),
                    "minfde": float(minfde[i].item()),
                })

    n = totals["count"]
    if n <= 0:
        raise RuntimeError("No evaluation samples were processed")
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "num_samples": n,
        "metrics": {
            "selected_ADE": totals["selected_ade"] / n,
            "selected_FDE": totals["selected_fde"] / n,
            "minADE": totals["minade"] / n,
            "minFDE": totals["minfde"] / n,
            "raw_mode_accuracy": totals["raw_mode_correct"] / n,
            "masked_mode_accuracy": totals["masked_mode_correct"] / n,
            "top3_raw_mode_hit_rate": totals["top3_mode_hit"] / n,
            "expert_mode_traffic_valid_fraction": totals["expert_mode_traffic_valid"] / n,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "open_loop_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    fieldnames = list(records[0].keys())
    with (output_dir / "open_loop_samples.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
