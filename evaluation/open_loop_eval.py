from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import torch

from evaluation.open_loop_metrics import (
    compute_open_loop_metrics,
    finite_metric_check,
)
from evaluation.open_loop_visualization import (
    TrajectoryVisualizationSample,
    compute_open_loop_safety_proxies,
    plot_ade_fde_boxplot,
    plot_initial_vs_final_denoising,
    plot_metrics_table,
    plot_random_trajectory_gallery,
    plot_training_loss_curve,
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


# OPEN_LOOP_CHECKPOINT_MODEL_CONFIG_V1
def _load_checkpoint_model_config(checkpoint: Path) -> Dict[str, object]:
    """Load model construction config from either training or runtime checkpoints.

    Training checkpoints store ``allmerge_model_config`` while exported runtime
    checkpoints store the same dictionary under ``model_config``.  Open-loop
    evaluation must restore this config *before* constructing the planner;
    otherwise architecture/behavior parameters such as residual hard bounds
    silently fall back to local defaults.
    """
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping):
        return {}

    runtime_cfg = payload.get("model_config")
    training_cfg = payload.get("allmerge_model_config")

    if runtime_cfg:
        if not isinstance(runtime_cfg, Mapping):
            raise TypeError(
                "checkpoint['model_config'] must be a mapping, got "
                f"{type(runtime_cfg).__name__}"
            )
        return dict(runtime_cfg)

    if training_cfg:
        if not isinstance(training_cfg, Mapping):
            raise TypeError(
                "checkpoint['allmerge_model_config'] must be a mapping, got "
                f"{type(training_cfg).__name__}"
            )
        return dict(training_cfg)

    return {}


def _residual_bound_from_config(config: Mapping[str, object]) -> tuple[float, float]:
    return (
        float(config.get("dense_residual_max_x_m", 2.0)),
        float(config.get("dense_residual_max_y_m", 0.75)),
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
    # OPEN_LOOP_DENSE_RESIDUAL_EVAL_V1
    "execution_sparse_ADE",
    "execution_sparse_FDE",
    "base_dense_ADE",
    "base_dense_FDE",
    "execution_dense_ADE",
    "execution_dense_FDE",
    "dense_ADE_gain",
    "dense_FDE_gain",
    "residual_mean_abs_x_m",
    "residual_mean_abs_y_m",
    "residual_max_abs_x_m",
    "residual_max_abs_y_m",
    "residual_x_saturation_ratio",
    "residual_y_saturation_ratio",
    # BASE_DENSE_SAFETY_PROXY_V1
    "base_dense_collision_proxy",
    "base_dense_offroad_proxy",
    # DENSE_EXECUTION_SAFETY_PROXY_V1
    "execution_collision_proxy",
    "execution_offroad_proxy",
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


# OPEN_LOOP_DENSE_RESIDUAL_EVAL_V1
def _dense_execution_metrics(
    *,
    model,
    output: Mapping[str, torch.Tensor],
    features: Mapping[str, torch.Tensor],
    expert_sparse: torch.Tensor,
    expert_dense: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    # Compare raw/base spline and residual-corrected execution trajectories.
    raw_sparse = output["trajectory"].float()
    execution_sparse = output.get(
        "trajectory_execution_sparse", raw_sparse
    ).float()
    residual = output.get(
        "trajectory_execution_residual", execution_sparse - raw_sparse
    ).float()
    expert_sparse = expert_sparse.float()
    expert_dense = expert_dense.float()

    if expert_dense.ndim != 3 or expert_dense.shape[-1] != 2:
        raise ValueError(
            f"trajectory_dense must be [B,T,2], got {tuple(expert_dense.shape)}"
        )

    start_xy = torch.zeros_like(raw_sparse[:, 0, :])
    start_velocity_xy = features["ego_state"][:, 0:2].float()
    base_dense_all = model.dense_trajectory_spline(
        raw_sparse,
        start_xy=start_xy,
        start_velocity_xy=start_velocity_xy,
    )
    execution_dense_all = model.dense_trajectory_spline(
        execution_sparse,
        start_xy=start_xy,
        start_velocity_xy=start_velocity_xy,
    )
    base_dense = base_dense_all[:, 1:, :]
    execution_dense = execution_dense_all[:, 1:, :]
    if base_dense.shape != expert_dense.shape:
        raise ValueError(
            "dense prediction/GT shape mismatch: "
            f"pred={tuple(base_dense.shape)} gt={tuple(expert_dense.shape)}"
        )

    exec_sparse_dist = torch.linalg.vector_norm(
        execution_sparse - expert_sparse, dim=-1
    )
    base_dense_dist = torch.linalg.vector_norm(
        base_dense - expert_dense, dim=-1
    )
    execution_dense_dist = torch.linalg.vector_norm(
        execution_dense - expert_dense, dim=-1
    )

    abs_res = residual.abs()
    abs_x = abs_res[..., 0]
    abs_y = abs_res[..., 1]
    x_bound = float(model.config.dense_residual_max_x_m)
    y_bound = float(model.config.dense_residual_max_y_m)

    return {
        "execution_sparse_ADE": exec_sparse_dist.mean(dim=-1),
        "execution_sparse_FDE": exec_sparse_dist[:, -1],
        "base_dense_ADE": base_dense_dist.mean(dim=-1),
        "base_dense_FDE": base_dense_dist[:, -1],
        "execution_dense_ADE": execution_dense_dist.mean(dim=-1),
        "execution_dense_FDE": execution_dense_dist[:, -1],
        "dense_ADE_gain": (
            base_dense_dist.mean(dim=-1) - execution_dense_dist.mean(dim=-1)
        ),
        "dense_FDE_gain": (
            base_dense_dist[:, -1] - execution_dense_dist[:, -1]
        ),
        "residual_mean_abs_x_m": abs_x.mean(dim=-1),
        "residual_mean_abs_y_m": abs_y.mean(dim=-1),
        "residual_max_abs_x_m": abs_x.max(dim=-1).values,
        "residual_max_abs_y_m": abs_y.max(dim=-1).values,
        "residual_x_saturation_ratio": (
            abs_x >= (0.95 * x_bound)
        ).float().mean(dim=-1),
        "residual_y_saturation_ratio": (
            abs_y >= (0.95 * y_bound)
        ).float().mean(dim=-1),
    }


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
    visualize: bool = True,
    figure_dir: Optional[str | Path] = None,
    visualization_seed: Optional[int] = None,
    initial_checkpoint: Optional[str | Path] = None,
    miss_threshold_m: float = 2.0,
    collision_distance_m: float = 2.5,
    offroad_distance_m: float = 2.5,
) -> dict:
    checkpoint = Path(checkpoint).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_csv = Path(output_csv)
    if figure_dir is None:
        figure_dir = output_csv.parent / "figures"
    figure_dir = Path(figure_dir)

    # None intentionally means a fresh random 9-sample gallery each run.
    visualization_rng = random.Random(visualization_seed)

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint}"
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}"
        )

    # OPEN_LOOP_CHECKPOINT_MODEL_CONFIG_V1
    # Restore model construction/behavior config before constructing runtime.
    # In particular this preserves the residual hard bounds used in training.
    checkpoint_model_config = _load_checkpoint_model_config(checkpoint)

    runtime_config = {
        "checkpoint": str(checkpoint),
        "allow_random_weights": False,
        "strict_checkpoint": bool(strict_checkpoint),
        "deterministic_seed": inference_seed,
    }
    if checkpoint_model_config:
        runtime_config["model"] = checkpoint_model_config
    if device:
        runtime_config["device"] = device

    runtime = DiffusionPlannerRuntime(
        runtime_config
    )
    model = runtime.model
    model.eval()

    restored_bound = (
        float(runtime.config.dense_residual_max_x_m),
        float(runtime.config.dense_residual_max_y_m),
    )
    if checkpoint_model_config:
        expected_bound = _residual_bound_from_config(checkpoint_model_config)
        if restored_bound != expected_bound:
            raise RuntimeError(
                "Open-loop runtime residual-bound restore mismatch: "
                f"checkpoint={expected_bound}, runtime={restored_bound}"
            )
        print(
            "[open-loop] restored checkpoint model_config "
            f"residual_bound=({restored_bound[0]:g}, {restored_bound[1]:g}) "
            f"residual_hidden_dim="
            f"{int(runtime.config.dense_residual_hidden_dim)}"
        )
    else:
        print(
            "[open-loop] checkpoint contains no model_config; using local "
            f"defaults/overrides residual_bound="
            f"({restored_bound[0]:g}, {restored_bound[1]:g})"
        )

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
    collision_flags: List[bool] = []
    offroad_flags: List[bool] = []
    # BASE_DENSE_SAFETY_PROXY_V1
    base_dense_collision_flags: List[bool] = []
    base_dense_offroad_flags: List[bool] = []
    # DENSE_EXECUTION_SAFETY_PROXY_V1
    execution_collision_flags: List[bool] = []
    execution_offroad_flags: List[bool] = []
    gallery_samples: List[TrajectoryVisualizationSample] = []
    gallery_seen = 0
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
            # OPEN_LOOP_DENSE_RESIDUAL_EVAL_V1
            if unpacked.get("expert_trajectory_dense") is None:
                raise ValueError(
                    "Formal dense open-loop evaluation requires Stage-3 "
                    "trajectory_dense [40,2] targets."
                )
            expert_trajectory_dense = (
                unpacked["expert_trajectory_dense"]
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

            # OPEN_LOOP_DENSE_RESIDUAL_EVAL_V1
            dense_metrics = _dense_execution_metrics(
                model=model,
                output=output,
                features=features,
                expert_sparse=expert_trajectory,
                expert_dense=expert_trajectory_dense,
            )
            for metric_name, metric_value in dense_metrics.items():
                if not torch.isfinite(metric_value).all():
                    raise FloatingPointError(
                        f"Non-finite dense open-loop metric: {metric_name}"
                    )

            # DENSE_EXECUTION_SAFETY_PROXY_V1
            # Build the actual residual-corrected dense execution trajectory.
            # dense_trajectory_spline returns [B, 41, 2] including t=0;
            # geometry proxies use the future 40 samples at 10 Hz.
            raw_sparse = output["trajectory"].float()
            execution_sparse = output.get(
                "trajectory_execution_sparse",
                raw_sparse,
            ).float()
            start_xy = torch.zeros_like(execution_sparse[:, 0, :])
            start_velocity_xy = features["ego_state"][:, 0:2].float()
            # BASE_DENSE_SAFETY_PROXY_V1
            # Fair safety comparison: both trajectories use the same spline,
            # same start state and the same 40 future samples at 10 Hz.
            base_dense_all = model.dense_trajectory_spline(
                raw_sparse,
                start_xy=start_xy,
                start_velocity_xy=start_velocity_xy,
            )
            base_dense_trajectory = base_dense_all[:, 1:, :]

            execution_dense_all = model.dense_trajectory_spline(
                execution_sparse,
                start_xy=start_xy,
                start_velocity_xy=start_velocity_xy,
            )
            execution_dense_trajectory = execution_dense_all[:, 1:, :]

            for dense_name, dense_traj in (
                ("base", base_dense_trajectory),
                ("execution", execution_dense_trajectory),
            ):
                if dense_traj.ndim != 3 or dense_traj.shape[-1] != 2:
                    raise RuntimeError(
                        f"Dense {dense_name} trajectory must be [B,T,2], got "
                        f"{tuple(dense_traj.shape)}"
                    )
                if not torch.isfinite(dense_traj).all():
                    raise FloatingPointError(
                        f"Dense {dense_name} trajectory contains NaN/Inf"
                    )

            if base_dense_trajectory.shape != execution_dense_trajectory.shape:
                raise RuntimeError(
                    "Base/execution dense safety trajectories must have the "
                    "same shape, got "
                    f"{tuple(base_dense_trajectory.shape)} vs "
                    f"{tuple(execution_dense_trajectory.shape)}"
                )

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

                # EVAL_VIZ_V2_SAMPLE_COLLECTION
                # Keep raw planner proxies for backward compatibility.
                collision_flag, offroad_flag = (
                    compute_open_loop_safety_proxies(
                        output["trajectory"][i],
                        features,
                        i,
                        collision_distance_m=collision_distance_m,
                        offroad_distance_m=offroad_distance_m,
                    )
                )
                if collision_flag is not None:
                    collision_flags.append(bool(collision_flag))
                if offroad_flag is not None:
                    offroad_flags.append(bool(offroad_flag))

                # BASE_DENSE_SAFETY_PROXY_V1
                base_dense_collision_flag, base_dense_offroad_flag = (
                    compute_open_loop_safety_proxies(
                        base_dense_trajectory[i],
                        features,
                        i,
                        collision_distance_m=collision_distance_m,
                        offroad_distance_m=offroad_distance_m,
                    )
                )
                if base_dense_collision_flag is not None:
                    base_dense_collision_flags.append(
                        bool(base_dense_collision_flag)
                    )
                if base_dense_offroad_flag is not None:
                    base_dense_offroad_flags.append(
                        bool(base_dense_offroad_flag)
                    )

                # DENSE_EXECUTION_SAFETY_PROXY_V1
                execution_collision_flag, execution_offroad_flag = (
                    compute_open_loop_safety_proxies(
                        execution_dense_trajectory[i],
                        features,
                        i,
                        collision_distance_m=collision_distance_m,
                        offroad_distance_m=offroad_distance_m,
                    )
                )
                if execution_collision_flag is not None:
                    execution_collision_flags.append(
                        bool(execution_collision_flag)
                    )
                if execution_offroad_flag is not None:
                    execution_offroad_flags.append(
                        bool(execution_offroad_flag)
                    )

                if visualize:
                    # Uniform reservoir sampling across all evaluated samples.
                    gallery_seen += 1
                    if len(gallery_samples) < 9:
                        gallery_slot = len(gallery_samples)
                    else:
                        candidate_slot = visualization_rng.randrange(gallery_seen)
                        gallery_slot = (
                            candidate_slot if candidate_slot < 9 else None
                        )
                    if gallery_slot is not None:
                        # OPEN_LOOP_DENSE_VIZ_V1: eval
                        visual_sample = TrajectoryVisualizationSample(
                            sample_index=sample_index,
                            expert_trajectory=(
                                expert_trajectory[i].detach().cpu().numpy()
                            ),
                            predicted_trajectory=(
                                output["trajectory"][i].detach().cpu().numpy()
                            ),
                            selected_ade=_float(metrics.selected_ade, i),
                            selected_fde=_float(metrics.selected_fde, i),
                            target_mode=eval_mode,
                            selected_mode=_int(
                                metrics.selected_pred_mode,
                                i,
                            ),
                            trajectory_time_s=metadata_item(
                                metadata,
                                "trajectory_time_s",
                                i,
                                default=None,
                            ),
                            expert_trajectory_10hz=metadata_item(
                                metadata,
                                "expert_trajectory_10hz_xy",
                                i,
                                default=None,
                            ),
                            expert_trajectory_10hz_time_s=metadata_item(
                                metadata,
                                "expert_trajectory_10hz_time_s",
                                i,
                                default=None,
                            ),
                            features_cpu={
                                key: value[i : i + 1].detach().cpu()
                                for key, value in features.items()
                                if hasattr(value, "detach")
                            },
                        )
                        if gallery_slot == len(gallery_samples):
                            gallery_samples.append(visual_sample)
                        else:
                            gallery_samples[gallery_slot] = visual_sample

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
                        # OPEN_LOOP_DENSE_RESIDUAL_EVAL_V1
                        **{
                            name: _float(value, i)
                            for name, value in dense_metrics.items()
                        },
                        # BASE_DENSE_SAFETY_PROXY_V1
                        "base_dense_collision_proxy": (
                            ""
                            if base_dense_collision_flag is None
                            else bool(base_dense_collision_flag)
                        ),
                        "base_dense_offroad_proxy": (
                            ""
                            if base_dense_offroad_flag is None
                            else bool(base_dense_offroad_flag)
                        ),
                        # DENSE_EXECUTION_SAFETY_PROXY_V1
                        "execution_collision_proxy": (
                            ""
                            if execution_collision_flag is None
                            else bool(execution_collision_flag)
                        ),
                        "execution_offroad_proxy": (
                            ""
                            if execution_offroad_flag is None
                            else bool(execution_offroad_flag)
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
        # OPEN_LOOP_DENSE_RESIDUAL_EVAL_V1
        "mean_execution_sparse_ADE": _mean(rows, "execution_sparse_ADE"),
        "mean_execution_sparse_FDE": _mean(rows, "execution_sparse_FDE"),
        "mean_base_dense_ADE": _mean(rows, "base_dense_ADE"),
        "mean_base_dense_FDE": _mean(rows, "base_dense_FDE"),
        "mean_execution_dense_ADE": _mean(rows, "execution_dense_ADE"),
        "mean_execution_dense_FDE": _mean(rows, "execution_dense_FDE"),
        "mean_dense_ADE_gain": _mean(rows, "dense_ADE_gain"),
        "mean_dense_FDE_gain": _mean(rows, "dense_FDE_gain"),
        "mean_residual_abs_x_m": _mean(rows, "residual_mean_abs_x_m"),
        "mean_residual_abs_y_m": _mean(rows, "residual_mean_abs_y_m"),
        "mean_residual_max_abs_x_m": _mean(rows, "residual_max_abs_x_m"),
        "mean_residual_max_abs_y_m": _mean(rows, "residual_max_abs_y_m"),
        "residual_x_saturation_ratio": _mean(rows, "residual_x_saturation_ratio"),
        "residual_y_saturation_ratio": _mean(rows, "residual_y_saturation_ratio"),
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
        # OPEN_LOOP_CHECKPOINT_MODEL_CONFIG_V1
        "residual_bound_x_m": float(runtime.config.dense_residual_max_x_m),
        "residual_bound_y_m": float(runtime.config.dense_residual_max_y_m),
        "residual_hidden_dim": int(runtime.config.dense_residual_hidden_dim),
        "checkpoint_model_config_restored": bool(checkpoint_model_config),
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "split": split,
        "output_csv": str(
            output_csv.resolve()
        ),
    }
    # EVAL_VIZ_V2_SUMMARY
    miss_rate = float(
        sum(
            1
            for row in rows
            if float(row["minFDE_at_M_all"]) > float(miss_threshold_m)
        )
        / len(rows)
    )
    collision_rate = (
        float(sum(collision_flags) / len(collision_flags))
        if collision_flags
        else float("nan")
    )
    offroad_rate = (
        float(sum(offroad_flags) / len(offroad_flags))
        if offroad_flags
        else float("nan")
    )
    summary["miss_rate"] = miss_rate
    summary["collision_rate_open_loop_proxy"] = collision_rate
    summary["offroad_rate_open_loop_proxy"] = offroad_rate

    # BASE_DENSE_SAFETY_PROXY_V1
    base_dense_collision_rate = (
        float(
            sum(base_dense_collision_flags)
            / len(base_dense_collision_flags)
        )
        if base_dense_collision_flags
        else float("nan")
    )
    base_dense_offroad_rate = (
        float(
            sum(base_dense_offroad_flags)
            / len(base_dense_offroad_flags)
        )
        if base_dense_offroad_flags
        else float("nan")
    )
    summary["base_dense_collision_rate_open_loop_proxy"] = (
        base_dense_collision_rate
    )
    summary["base_dense_offroad_rate_open_loop_proxy"] = (
        base_dense_offroad_rate
    )

    # DENSE_EXECUTION_SAFETY_PROXY_V1
    execution_collision_rate = (
        float(
            sum(execution_collision_flags)
            / len(execution_collision_flags)
        )
        if execution_collision_flags
        else float("nan")
    )
    execution_offroad_rate = (
        float(
            sum(execution_offroad_flags)
            / len(execution_offroad_flags)
        )
        if execution_offroad_flags
        else float("nan")
    )
    summary["execution_collision_rate_open_loop_proxy"] = (
        execution_collision_rate
    )
    summary["execution_offroad_rate_open_loop_proxy"] = (
        execution_offroad_rate
    )

    # BASE_DENSE_SAFETY_PROXY_V1
    # Positive delta: execution triggers the proxy more often than base dense.
    # Negative delta: execution triggers the proxy less often.
    summary["collision_proxy_delta"] = (
        execution_collision_rate - base_dense_collision_rate
        if (
            math.isfinite(execution_collision_rate)
            and math.isfinite(base_dense_collision_rate)
        )
        else float("nan")
    )
    summary["offroad_proxy_delta"] = (
        execution_offroad_rate - base_dense_offroad_rate
        if (
            math.isfinite(execution_offroad_rate)
            and math.isfinite(base_dense_offroad_rate)
        )
        else float("nan")
    )

    if visualize:
        figure_dir.mkdir(parents=True, exist_ok=True)

        # TRAINING_LOSS_CURVE_V1
        # checkpoints/<name>.pt and runtime/<name>.pt both live one directory
        # below the run root, whose TensorBoard history is stored in run/tb.
        run_dir = checkpoint.parent.parent
        training_log_dir = run_dir / "tb"
        training_loss_path = plot_training_loss_curve(
            log_dir=training_log_dir,
            output_path=figure_dir / "05_training_loss_curve.png",
            output_csv=figure_dir / "training_loss_curve.csv",
        )
        if training_loss_path is not None:
            summary["training_loss_curve"] = str(
                Path(training_loss_path).resolve()
            )
            summary["training_loss_curve_csv"] = str(
                (figure_dir / "training_loss_curve.csv").resolve()
            )
        else:
            summary["training_loss_curve"] = (
                "not generated; no scalar history found under "
                f"{training_log_dir}"
            )

        plot_ade_fde_boxplot(
            rows,
            figure_dir / "01_ade_fde_boxplot.png",
        )
        plot_random_trajectory_gallery(
            gallery_samples,
            figure_dir / "02_random_trajectory_gallery_3x3.png",
        )
        plot_metrics_table(
            min_ade=_mean(rows, "minADE_at_M_all"),
            min_fde=_mean(rows, "minFDE_at_M_all"),
            miss_rate=miss_rate,
            # DENSE_EXECUTION_SAFETY_PROXY_V1
            # The rendered table reports final execution-trajectory proxies.
            collision_rate=(
                execution_collision_rate
                if math.isfinite(execution_collision_rate)
                else None
            ),
            offroad_rate=(
                execution_offroad_rate
                if math.isfinite(execution_offroad_rate)
                else None
            ),
            miss_threshold_m=miss_threshold_m,
            collision_distance_m=collision_distance_m,
            offroad_distance_m=offroad_distance_m,
            output_path=figure_dir / "04_metrics_table.png",
            output_csv=figure_dir / "metrics_table.csv",
        )

        resolved_initial = (
            Path(initial_checkpoint).expanduser().resolve()
            if initial_checkpoint is not None
            else checkpoint.parent / "initial.pt"
        )
        if gallery_samples and resolved_initial.is_file():
            comparison_seed = (
                int(inference_seed) if inference_seed is not None else 0
            )
            plot_initial_vs_final_denoising(
                final_model=model,
                initial_checkpoint=resolved_initial,
                sample=gallery_samples[0],
                output_path=(
                    figure_dir / "03_denoising_initial_vs_final.png"
                ),
                seed=comparison_seed,
                strict_checkpoint=strict_checkpoint,
            )
            summary["initial_checkpoint_for_visualization"] = str(
                resolved_initial
            )
        else:
            summary["initial_checkpoint_for_visualization"] = (
                "not found; pass --initial-checkpoint or retrain once "
                "to create checkpoints/initial.pt"
            )
        summary["figure_dir"] = str(figure_dir.resolve())

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
