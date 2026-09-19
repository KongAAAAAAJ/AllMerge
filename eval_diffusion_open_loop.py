"""Open-loop ADE/FDE and mode evaluation for W2 checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from highway_env.planner.diffusion.config import build_structured_diffusion_config
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter
from pretraining.checkpoint_io import load_checkpoint_file
from pretraining.contract import validate_model_schema
from pretraining.dataset_adapter import build_w1_dataloader, metadata_item, unpack_w1_batch
from train_diffusion_pretrain import load_yaml, resolve_pin_memory


def resolve_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else (
        "cpu" if value == "auto" else value
    ))


def compute_ade_fde(pred: torch.Tensor, target: torch.Tensor):
    l2 = torch.linalg.norm(pred[..., :2] - target[..., :2], dim=-1)
    return l2.mean(dim=-1), l2[..., -1]


def parse_args():
    parser = argparse.ArgumentParser(description="Open-loop AllMerge diffusion evaluation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/diffusion_pretrain.yaml")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="val")
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
    dataset_root = Path(
        args.dataset_root or train_cfg.get("dataset_root", "outputs/expert_dataset/allmerge_expert")
    )
    device = resolve_device(args.device)

    state_dict, stored_config = load_checkpoint_file(args.checkpoint)
    if stored_config:
        if model_overrides and model_overrides != stored_config:
            raise RuntimeError("Config model overrides disagree with checkpoint model config")
        model_overrides = stored_config

    model_config = build_structured_diffusion_config(**model_overrides)
    validate_model_schema(model_config)
    adapter = PlannerTensorAdapter(model_config, device)
    model = StructuredDiffusionPlanner(model_config, adapter).to(device)
    model.load_state_dict(state_dict, strict=bool(args.strict_checkpoint))
    model.eval()

    loader = build_w1_dataloader(
        dataset_root,
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

    totals = {key: 0.0 for key in ("selected_ade", "selected_fde", "minade", "minfde")}
    totals.update({key: 0 for key in (
        "raw_mode_correct", "masked_mode_correct", "top3_mode_hit",
        "expert_mode_traffic_valid", "count"
    )})
    records = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = unpack_w1_batch(raw_batch)
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
            candidate_l2 = torch.linalg.norm(candidates - target[:, None], dim=-1)
            minade = candidate_l2.mean(dim=-1).min(dim=-1).values
            minfde = candidate_l2[..., -1].min(dim=-1).values
            traffic_valid = features["mode_valid_mask"].gather(1, expert_mode[:, None]).squeeze(1)
            top3_hit = (top3 == expert_mode[:, None]).any(dim=-1)

            b = int(target.shape[0])
            for key, value in (
                ("selected_ade", selected_ade), ("selected_fde", selected_fde),
                ("minade", minade), ("minfde", minfde),
            ):
                totals[key] += float(value.sum().item())
            totals["raw_mode_correct"] += int((raw_idx == expert_mode).sum().item())
            totals["masked_mode_correct"] += int((masked_idx == expert_mode).sum().item())
            totals["top3_mode_hit"] += int(top3_hit.sum().item())
            totals["expert_mode_traffic_valid"] += int(traffic_valid.sum().item())
            totals["count"] += b

            for i in range(b):
                records.append({
                    "scenario": metadata_item(batch["metadata"], "scenario", i, ""),
                    "seed": metadata_item(batch["metadata"], "seed", i, -1),
                    "episode_index": metadata_item(batch["metadata"], "episode_index", i, -1),
                    "frame_index": metadata_item(batch["metadata"], "frame_index", i, -1),
                    "ego_index": metadata_item(batch["metadata"], "ego_index", i, -1),
                    "expert_mode": int(expert_mode[i]),
                    "raw_mode_idx": int(raw_idx[i]),
                    "masked_mode_idx": int(masked_idx[i]),
                    "selected_ade": float(selected_ade[i]),
                    "selected_fde": float(selected_fde[i]),
                    "minade": float(minade[i]),
                    "minfde": float(minfde[i]),
                })

    n = int(totals["count"])
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
    (output_dir / "open_loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if records:
        with (output_dir / "open_loop_samples.csv").open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
