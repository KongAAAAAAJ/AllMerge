from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import torch

from evaluation.open_loop_metrics import (
    compute_open_loop_metrics,
    finite_metric_check,
)
from highway_env.planner.diffusion.mode_assignment import (
    assign_expert_mode,
)
from highway_env.planner.diffusion.runtime import (
    DiffusionPlannerRuntime,
)
from pretraining.dataset_adapter import (
    build_w1_dataloader,
    metadata_item,
    unpack_w1_batch,
)


CSV_FIELDS = [
    "sample_index",
    "seed",
    "episode_index",
    "episode_step",
    "frame_index",
    "ego_index",
    "shard_path",
    "shard_sample_index",
    "target_semantic",
    "dataset_target_mode",
    "eval_target_mode",
    "target_mode_matches_dataset",
    "target_assignment_ADE",
    "target_mode_traffic_valid",
    "raw_pred_mode",
    "selected_mode",
    "raw_mode_correct",
    "selected_mode_correct",
    "valid_mode_count",
    "selected_ADE",
    "selected_FDE",
    "minADE_at_M_all",
    "minFDE_at_M_all",
    "minADE_at_valid",
    "minFDE_at_valid",
    "raw_mode_entropy",
    "selected_mode_entropy",
]


def _move_features(
    features: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in features.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Feature {key!r} must be torch.Tensor, "
                f"got {type(value)!r}"
            )
        moved[key] = value.to(
            device=device,
            non_blocking=True,
        )
    return moved


def _build_generator(
    device: torch.device,
    seed: Optional[int],
) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _float(value: torch.Tensor, index: int) -> float:
    return float(
        value[index].detach().cpu().item()
    )


def _int(value: torch.Tensor, index: int) -> int:
    return int(
        value[index].detach().cpu().item()
    )


def _bool(value: torch.Tensor, index: int) -> bool:
    return bool(
        value[index].detach().cpu().item()
    )


def _metadata(
    metadata,
    key: str,
    index: int,
):
    value = metadata_item(
        metadata,
        key,
        index,
        default="",
    )
    return "" if value is None else value


def _write_csv(
    rows: List[dict],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def _mean(
    rows: Iterable[dict],
    key: str,
) -> float:
    values = [
        float(row[key])
        for row in rows
    ]
    return (
        float(sum(values) / len(values))
        if values
        else float("nan")
    )


def _rate(
    rows: Iterable[dict],
    key: str,
) -> float:
    values = [
        1.0 if bool(row[key]) else 0.0
        for row in rows
    ]
    return (
        float(sum(values) / len(values))
        if values
        else float("nan")
    )


def evaluate_open_loop(
    *,
    checkpoint: str | Path,
    dataset_root: str | Path,
    split: str = "val",
    output_csv: str | Path = (
        "outputs/evaluation/open_loop_results.csv"
    ),
    batch_size: int = 32,
    num_workers: int = 0,
    limit_samples: int = 0,
    device: Optional[str] = None,
    inference_seed: Optional[int] = 0,
    strict_checkpoint: bool = False,
) -> dict:
    checkpoint = Path(checkpoint).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_csv = Path(output_csv)

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint}"
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}"
        )

    runtime_config = {
        "checkpoint": str(checkpoint),
        "allow_random_weights": False,
        "strict_checkpoint": bool(strict_checkpoint),
        "deterministic_seed": inference_seed,
    }
    if device:
        runtime_config["device"] = device

    runtime = DiffusionPlannerRuntime(
        runtime_config
    )
    model = runtime.model
    model.eval()

    loader = build_w1_dataloader(
        dataset_root=dataset_root,
        split=split,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        shuffle=False,
        limit_samples=int(limit_samples),
        pin_memory=(
            runtime.device.type == "cuda"
        ),
        include_metadata=True,
    )

    # Use one generator for the full evaluation pass.
    # This gives a deterministic but non-repeated noise sequence across batches.
    generator = _build_generator(
        runtime.device,
        inference_seed,
    )

    rows: List[dict] = []
    batch_latencies_ms: List[float] = []
    sample_offset = 0

    with torch.no_grad():
        for batch in loader:
            unpacked = unpack_w1_batch(batch)

            features = _move_features(
                unpacked["features"],
                runtime.device,
            )
            expert_trajectory = (
                unpacked["expert_trajectory"]
                .to(
                    device=runtime.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            )
            expert_semantic = (
                unpacked["expert_semantic"]
                .to(
                    device=runtime.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
            )
            dataset_target_mode = (
                unpacked["expert_mode"]
                .to(
                    device=runtime.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
            )

            # Exactly mirror W2/F.2 training label construction:
            # semantic + geometry -> nearest anchor by ADE.
            # mode_valid_mask is intentionally NOT used for target assignment.
            assignment = assign_expert_mode(
                features=features,
                target_trajectory=expert_trajectory,
                target_semantic=expert_semantic,
            )
            eval_target_mode = assignment.target_mode

            target_assignment_ade = torch.gather(
                assignment.anchor_distance,
                dim=1,
                index=eval_target_mode[:, None],
            ).squeeze(1)

            if runtime.device.type == "cuda":
                torch.cuda.synchronize(
                    runtime.device
                )

            start = time.perf_counter()
            output = model.infer_multimodal(
                features,
                generator=generator,
            )

            if runtime.device.type == "cuda":
                torch.cuda.synchronize(
                    runtime.device
                )
            batch_latencies_ms.append(
                (time.perf_counter() - start)
                * 1000.0
            )

            metrics = compute_open_loop_metrics(
                selected_trajectory=output[
                    "trajectory"
                ],
                trajectory_candidates=output[
                    "trajectory_candidates"
                ],
                raw_logits=output[
                    "trajectory_mode_logits"
                ],
                masked_logits=output[
                    "trajectory_mode_logits_masked"
                ],
                selected_mode_idx=output[
                    "trajectory_mode_idx"
                ],
                target_trajectory=expert_trajectory,
                target_mode=eval_target_mode,
                target_assignment_ade=(
                    target_assignment_ade
                ),
                mode_valid_mask=features[
                    "mode_valid_mask"
                ],
            )
            finite_metric_check(metrics)

            metadata = unpacked["metadata"]
            batch_size_actual = int(
                expert_trajectory.shape[0]
            )

            for i in range(batch_size_actual):
                dataset_mode = _int(
                    dataset_target_mode,
                    i,
                )
                eval_mode = _int(
                    eval_target_mode,
                    i,
                )
                sample_index = _metadata(
                    metadata,
                    "sample_index",
                    i,
                )
                if sample_index == "":
                    sample_index = (
                        sample_offset + i
                    )

                rows.append(
                    {
                        "sample_index": sample_index,
                        "seed": _metadata(
                            metadata,
                            "seed",
                            i,
                        ),
                        "episode_index": _metadata(
                            metadata,
                            "episode_index",
                            i,
                        ),
                        "episode_step": _metadata(
                            metadata,
                            "episode_step",
                            i,
                        ),
                        "frame_index": _metadata(
                            metadata,
                            "frame_index",
                            i,
                        ),
                        "ego_index": _metadata(
                            metadata,
                            "ego_index",
                            i,
                        ),
                        "shard_path": _metadata(
                            metadata,
                            "shard_path",
                            i,
                        ),
                        "shard_sample_index": _metadata(
                            metadata,
                            "shard_sample_index",
                            i,
                        ),
                        "target_semantic": _int(
                            expert_semantic,
                            i,
                        ),
                        "dataset_target_mode": (
                            dataset_mode
                        ),
                        "eval_target_mode": (
                            eval_mode
                        ),
                        "target_mode_matches_dataset": (
                            dataset_mode == eval_mode
                        ),
                        "target_assignment_ADE": _float(
                            metrics.target_assignment_ade,
                            i,
                        ),
                        "target_mode_traffic_valid": _bool(
                            metrics.target_mode_traffic_valid,
                            i,
                        ),
                        "raw_pred_mode": _int(
                            metrics.raw_pred_mode,
                            i,
                        ),
                        "selected_mode": _int(
                            metrics.selected_pred_mode,
                            i,
                        ),
                        "raw_mode_correct": _bool(
                            metrics.raw_mode_correct,
                            i,
                        ),
                        "selected_mode_correct": _bool(
                            metrics.selected_mode_correct,
                            i,
                        ),
                        "valid_mode_count": _int(
                            metrics.valid_mode_count,
                            i,
                        ),
                        "selected_ADE": _float(
                            metrics.selected_ade,
                            i,
                        ),
                        "selected_FDE": _float(
                            metrics.selected_fde,
                            i,
                        ),
                        "minADE_at_M_all": _float(
                            metrics.min_ade_all,
                            i,
                        ),
                        "minFDE_at_M_all": _float(
                            metrics.min_fde_all,
                            i,
                        ),
                        "minADE_at_valid": _float(
                            metrics.min_ade_valid,
                            i,
                        ),
                        "minFDE_at_valid": _float(
                            metrics.min_fde_valid,
                            i,
                        ),
                        "raw_mode_entropy": _float(
                            metrics.raw_mode_entropy,
                            i,
                        ),
                        "selected_mode_entropy": _float(
                            metrics.selected_mode_entropy,
                            i,
                        ),
                    }
                )

            sample_offset += batch_size_actual

    if not rows:
        raise RuntimeError(
            "Open-loop evaluation produced zero rows"
        )

    _write_csv(
        rows,
        output_csv,
    )

    num_modes = int(
        runtime.config.num_modes
    )
    total_latency_ms = float(
        sum(batch_latencies_ms)
    )
    summary = {
        "samples": len(rows),
        "num_modes": num_modes,
        "mean_selected_ADE": _mean(
            rows,
            "selected_ADE",
        ),
        "mean_selected_FDE": _mean(
            rows,
            "selected_FDE",
        ),
        f"mean_minADE_at_{num_modes}_all": _mean(
            rows,
            "minADE_at_M_all",
        ),
        f"mean_minFDE_at_{num_modes}_all": _mean(
            rows,
            "minFDE_at_M_all",
        ),
        "mean_minADE_at_valid": _mean(
            rows,
            "minADE_at_valid",
        ),
        "mean_minFDE_at_valid": _mean(
            rows,
            "minFDE_at_valid",
        ),
        "raw_mode_accuracy": _rate(
            rows,
            "raw_mode_correct",
        ),
        "selected_mode_accuracy": _rate(
            rows,
            "selected_mode_correct",
        ),
        "target_mode_traffic_valid_ratio": _rate(
            rows,
            "target_mode_traffic_valid",
        ),
        "dataset_vs_eval_target_mode_agreement": _rate(
            rows,
            "target_mode_matches_dataset",
        ),
        "mean_target_assignment_ADE": _mean(
            rows,
            "target_assignment_ADE",
        ),
        "mean_raw_mode_entropy_nats": _mean(
            rows,
            "raw_mode_entropy",
        ),
        "mean_selected_mode_entropy_nats": _mean(
            rows,
            "selected_mode_entropy",
        ),
        "mean_batch_inference_latency_ms": (
            total_latency_ms
            / len(batch_latencies_ms)
        ),
        "mean_inference_latency_ms_per_sample": (
            total_latency_ms
            / len(rows)
        ),
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "split": split,
        "output_csv": str(
            output_csv.resolve()
        ),
    }
    return summary


def print_summary(
    summary: Mapping[str, object],
) -> None:
    print("=" * 96)
    print(
        "ALLMERGE PRETRAIN OPEN-LOOP EVALUATION"
    )
    print("=" * 96)
    for key, value in summary.items():
        if isinstance(value, float):
            shown = (
                "nan"
                if math.isnan(value)
                else f"{value:.6f}"
            )
        else:
            shown = value
        print(f"{key}: {shown}")
    print("=" * 96)
