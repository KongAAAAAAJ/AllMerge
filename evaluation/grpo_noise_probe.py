from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


DEFAULT_IMPROVEMENT_EPS = 1e-6


@dataclass(frozen=True)
class ProbeSampleSummary:
    checkpoint: str
    scenario: str
    sample_index: int
    env_seed: int
    noise_seed: int
    valid_pair_count: int
    group_size: int
    group0_reward: float
    best_group_reward: float
    worst_group_reward: float
    best_group_id: int
    best_group_gain: float
    worst_group_delta: float
    group_reward_mean: float
    group_reward_median: float
    group_reward_range: float
    group_reward_std: float
    best_minus_group_mean: float
    best_minus_group_median: float
    best_minus_nonbest_mean: float
    group0_minus_group_mean: float
    better_group_count: int
    better_group_fraction: float
    group0_rank: int
    pair_better_fraction: float
    improvable_pair_fraction: float
    element_best_gain_mean: float
    element_reward_range_mean: float
    element_reward_std_mean: float
    element_best_minus_mean_mean: float
    element_best_minus_median_mean: float
    element_best_minus_nonbest_mean_mean: float
    element_sparse_pairwise_ade_mean: float
    element_sparse_pairwise_ade_p90_mean: float
    element_sparse_endpoint_dist_mean: float
    element_sparse_endpoint_dist_max_mean: float
    element_ade_reward_std_corr: float
    element_ade_reward_range_corr: float


@dataclass(frozen=True)
class ProbeGroupSummary:
    checkpoint: str
    scenario: str
    sample_index: int
    env_seed: int
    noise_seed: int
    group_id: int
    group_reward: float
    delta_vs_group0: float
    better_than_group0: bool


@dataclass(frozen=True)
class ProbeVehicleModeSummary:
    checkpoint: str
    scenario: str
    sample_index: int
    env_seed: int
    noise_seed: int
    vehicle_role: int
    mode: int
    group0_reward: float
    best_reward: float
    worst_reward: float
    reward_mean: float
    reward_median: float
    best_group_id: int
    best_gain: float
    reward_range: float
    reward_std: float
    best_minus_mean: float
    best_minus_median: float
    best_minus_nonbest_mean: float
    better_count: int
    better_fraction: float
    sparse_pairwise_ade_mean: float
    sparse_pairwise_ade_p90: float
    sparse_endpoint_dist_mean: float
    sparse_endpoint_dist_max: float


def _as_numpy(value: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _pairwise_trajectory_stats(trajectories: np.ndarray) -> tuple[float, float, float, float]:
    """Return pairwise ADE mean/p90 and endpoint distance mean/max for [G,H,2]."""
    traj = np.asarray(trajectories, dtype=np.float64)
    if traj.ndim != 3 or traj.shape[-1] != 2 or traj.shape[0] < 2:
        raise ValueError(f"trajectories must be [G>=2,H,2], got {traj.shape}")
    delta = traj[:, None, :, :] - traj[None, :, :, :]
    point_dist = np.linalg.norm(delta, axis=-1)  # [G,G,H]
    pair_ade = point_dist.mean(axis=-1)
    endpoint = np.linalg.norm(
        traj[:, None, -1, :] - traj[None, :, -1, :], axis=-1
    )
    iu = np.triu_indices(traj.shape[0], k=1)
    ade_values = pair_ade[iu]
    endpoint_values = endpoint[iu]
    return (
        float(np.mean(ade_values)),
        float(np.quantile(ade_values, 0.90)),
        float(np.mean(endpoint_values)),
        float(np.max(endpoint_values)),
    )


def _safe_corr(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[finite], ya[finite]
    if xa.size < 3 or float(np.std(xa)) <= 1e-12 or float(np.std(ya)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def analyze_group_rewards(
    rewards: torch.Tensor | np.ndarray,
    valid_mode_mask: torch.Tensor | np.ndarray,
    *,
    candidates: torch.Tensor | np.ndarray | None = None,
    checkpoint: str,
    scenario: str,
    sample_index: int,
    env_seed: int,
    noise_seed: int,
    improvement_eps: float = DEFAULT_IMPROVEMENT_EPS,
) -> tuple[ProbeSampleSummary, list[ProbeGroupSummary], list[ProbeVehicleModeSummary]]:
    """Analyze one state using exact GRPO [vehicle, group, mode] reward layout.

    ``candidates`` should be [vehicle, group, mode, horizon, 2]. When supplied,
    v2 also measures how much the trajectory geometry actually changes across
    diffusion-noise groups.
    """
    values = np.asarray(_as_numpy(rewards), dtype=np.float64)
    valid = np.asarray(_as_numpy(valid_mode_mask), dtype=np.bool_)

    if values.ndim != 3:
        raise ValueError(f"rewards must be [vehicle,group,mode], got {values.shape}")
    if valid.shape != (values.shape[0], values.shape[2]):
        raise ValueError(
            "valid_mode_mask must be [vehicle,mode], got "
            f"{valid.shape} for rewards {values.shape}"
        )
    if values.shape[1] < 2:
        raise ValueError("group_size must be >= 2")
    if not np.isfinite(values).all():
        raise ValueError("rewards contain NaN/Inf")
    if not valid.any():
        raise ValueError("valid_mode_mask contains no valid vehicle-mode pair")

    trajectory_values: np.ndarray | None = None
    if candidates is not None:
        trajectory_values = np.asarray(_as_numpy(candidates), dtype=np.float64)
        expected = (values.shape[0], values.shape[1], values.shape[2])
        if trajectory_values.ndim != 5 or trajectory_values.shape[:3] != expected or trajectory_values.shape[-1] != 2:
            raise ValueError(
                "candidates must be [vehicle,group,mode,horizon,2], got "
                f"{trajectory_values.shape} for rewards {values.shape}"
            )
        if not np.isfinite(trajectory_values).all():
            raise ValueError("candidates contain NaN/Inf")

    group_size = int(values.shape[1])
    valid_pair_count = int(valid.sum())

    group_rewards = np.asarray(
        [values[:, g, :][valid].mean() for g in range(group_size)],
        dtype=np.float64,
    )
    group0_reward = float(group_rewards[0])
    group_delta = group_rewards - group0_reward
    better_group = group_delta > float(improvement_eps)
    better_group[0] = False
    best_group_id = int(np.argmax(group_rewards))
    best_group_reward = float(group_rewards[best_group_id])
    worst_group_reward = float(np.min(group_rewards))
    group_reward_mean = float(np.mean(group_rewards))
    group_reward_median = float(np.median(group_rewards))
    nonbest_group_rewards = np.delete(group_rewards, best_group_id)

    nonzero_group_count = group_size - 1
    better_group_count = int(better_group[1:].sum())
    better_group_fraction = float(better_group_count / nonzero_group_count)
    group0_rank = int(1 + np.sum(group_rewards[1:] > group0_reward + improvement_eps))

    baseline = values[:, 0, :]
    deltas = values[:, 1:, :] - baseline[:, None, :]
    valid_expanded = np.broadcast_to(valid[:, None, :], deltas.shape)
    valid_deltas = deltas[valid_expanded]
    pair_better_fraction = float(np.mean(valid_deltas > improvement_eps))

    pair_best = np.max(values, axis=1)
    pair_worst = np.min(values, axis=1)
    pair_mean = np.mean(values, axis=1)
    pair_median = np.median(values, axis=1)
    pair_std = np.std(values, axis=1)
    pair_best_gain = pair_best - baseline
    pair_improvable = pair_best_gain > improvement_eps

    vm_rows: list[ProbeVehicleModeSummary] = []
    pairwise_ade_values: list[float] = []
    pairwise_ade_p90_values: list[float] = []
    endpoint_mean_values: list[float] = []
    endpoint_max_values: list[float] = []
    vm_reward_std_values: list[float] = []
    vm_reward_range_values: list[float] = []
    vm_best_minus_mean_values: list[float] = []
    vm_best_minus_median_values: list[float] = []
    vm_best_minus_nonbest_values: list[float] = []

    for role in range(values.shape[0]):
        for mode in range(values.shape[2]):
            if not valid[role, mode]:
                continue
            row = values[role, :, mode]
            r0 = float(row[0])
            delta = row[1:] - r0
            best_id = int(np.argmax(row))
            best_reward = float(row[best_id])
            mean_reward = float(np.mean(row))
            median_reward = float(np.median(row))
            nonbest = np.delete(row, best_id)
            reward_range = float(np.max(row) - np.min(row))
            reward_std = float(np.std(row))

            if trajectory_values is not None:
                ade_mean, ade_p90, endpoint_mean, endpoint_max = _pairwise_trajectory_stats(
                    trajectory_values[role, :, mode]
                )
            else:
                ade_mean = ade_p90 = endpoint_mean = endpoint_max = float("nan")

            pairwise_ade_values.append(ade_mean)
            pairwise_ade_p90_values.append(ade_p90)
            endpoint_mean_values.append(endpoint_mean)
            endpoint_max_values.append(endpoint_max)
            vm_reward_std_values.append(reward_std)
            vm_reward_range_values.append(reward_range)
            vm_best_minus_mean_values.append(best_reward - mean_reward)
            vm_best_minus_median_values.append(best_reward - median_reward)
            vm_best_minus_nonbest_values.append(best_reward - float(np.mean(nonbest)))

            vm_rows.append(
                ProbeVehicleModeSummary(
                    checkpoint=checkpoint,
                    scenario=scenario,
                    sample_index=int(sample_index),
                    env_seed=int(env_seed),
                    noise_seed=int(noise_seed),
                    vehicle_role=int(role),
                    mode=int(mode),
                    group0_reward=r0,
                    best_reward=best_reward,
                    worst_reward=float(np.min(row)),
                    reward_mean=mean_reward,
                    reward_median=median_reward,
                    best_group_id=best_id,
                    best_gain=float(best_reward - r0),
                    reward_range=reward_range,
                    reward_std=reward_std,
                    best_minus_mean=float(best_reward - mean_reward),
                    best_minus_median=float(best_reward - median_reward),
                    best_minus_nonbest_mean=float(best_reward - np.mean(nonbest)),
                    better_count=int(np.sum(delta > improvement_eps)),
                    better_fraction=float(np.mean(delta > improvement_eps)),
                    sparse_pairwise_ade_mean=ade_mean,
                    sparse_pairwise_ade_p90=ade_p90,
                    sparse_endpoint_dist_mean=endpoint_mean,
                    sparse_endpoint_dist_max=endpoint_max,
                )
            )

    finite_ade = np.asarray(pairwise_ade_values, dtype=np.float64)
    sample_summary = ProbeSampleSummary(
        checkpoint=checkpoint,
        scenario=scenario,
        sample_index=int(sample_index),
        env_seed=int(env_seed),
        noise_seed=int(noise_seed),
        valid_pair_count=valid_pair_count,
        group_size=group_size,
        group0_reward=group0_reward,
        best_group_reward=best_group_reward,
        worst_group_reward=worst_group_reward,
        best_group_id=best_group_id,
        best_group_gain=float(best_group_reward - group0_reward),
        worst_group_delta=float(worst_group_reward - group0_reward),
        group_reward_mean=group_reward_mean,
        group_reward_median=group_reward_median,
        group_reward_range=float(group_rewards.max() - group_rewards.min()),
        group_reward_std=float(group_rewards.std()),
        best_minus_group_mean=float(best_group_reward - group_reward_mean),
        best_minus_group_median=float(best_group_reward - group_reward_median),
        best_minus_nonbest_mean=float(best_group_reward - np.mean(nonbest_group_rewards)),
        group0_minus_group_mean=float(group0_reward - group_reward_mean),
        better_group_count=better_group_count,
        better_group_fraction=better_group_fraction,
        group0_rank=group0_rank,
        pair_better_fraction=pair_better_fraction,
        improvable_pair_fraction=float(pair_improvable[valid].mean()),
        element_best_gain_mean=float(pair_best_gain[valid].mean()),
        element_reward_range_mean=float((pair_best - pair_worst)[valid].mean()),
        element_reward_std_mean=float(pair_std[valid].mean()),
        element_best_minus_mean_mean=float(np.mean(vm_best_minus_mean_values)),
        element_best_minus_median_mean=float(np.mean(vm_best_minus_median_values)),
        element_best_minus_nonbest_mean_mean=float(np.mean(vm_best_minus_nonbest_values)),
        element_sparse_pairwise_ade_mean=float(np.nanmean(finite_ade)),
        element_sparse_pairwise_ade_p90_mean=float(np.nanmean(pairwise_ade_p90_values)),
        element_sparse_endpoint_dist_mean=float(np.nanmean(endpoint_mean_values)),
        element_sparse_endpoint_dist_max_mean=float(np.nanmean(endpoint_max_values)),
        element_ade_reward_std_corr=_safe_corr(pairwise_ade_values, vm_reward_std_values),
        element_ade_reward_range_corr=_safe_corr(pairwise_ade_values, vm_reward_range_values),
    )

    group_rows = [
        ProbeGroupSummary(
            checkpoint=checkpoint,
            scenario=scenario,
            sample_index=int(sample_index),
            env_seed=int(env_seed),
            noise_seed=int(noise_seed),
            group_id=int(group_id),
            group_reward=float(group_rewards[group_id]),
            delta_vs_group0=float(group_delta[group_id]),
            better_than_group0=bool(group_id > 0 and better_group[group_id]),
        )
        for group_id in range(group_size)
    ]
    return sample_summary, group_rows, vm_rows


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def _finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def aggregate_sample_summaries(
    rows: Sequence[ProbeSampleSummary],
    *,
    checkpoint: str,
    scenario: str,
) -> dict[str, float | str | int]:
    subset = [
        row for row in rows
        if row.checkpoint == checkpoint and (scenario == "all" or row.scenario == scenario)
    ]
    if not subset:
        raise ValueError(f"no probe samples for checkpoint={checkpoint} scenario={scenario}")

    def arr(name: str) -> np.ndarray:
        return np.asarray([float(getattr(row, name)) for row in subset], dtype=np.float64)

    best_gain = arr("best_group_gain")
    better_fraction = arr("better_group_fraction")
    pair_better_fraction = arr("pair_better_fraction")
    group0_rank = arr("group0_rank")

    return {
        "checkpoint": checkpoint,
        "scenario": scenario,
        "sample_count": len(subset),
        "group_size": int(subset[0].group_size),
        "mean_group0_reward": float(arr("group0_reward").mean()),
        "mean_group_reward": float(arr("group_reward_mean").mean()),
        "mean_best_group_reward": float(arr("best_group_reward").mean()),
        "mean_best_group_gain": float(best_gain.mean()),
        "median_best_group_gain": float(np.median(best_gain)),
        "p25_best_group_gain": _quantile(best_gain, 0.25),
        "p75_best_group_gain": _quantile(best_gain, 0.75),
        "p90_best_group_gain": _quantile(best_gain, 0.90),
        "mean_group_reward_range": float(arr("group_reward_range").mean()),
        "median_group_reward_range": float(np.median(arr("group_reward_range"))),
        "mean_group_reward_std": float(arr("group_reward_std").mean()),
        "mean_best_minus_group_mean": float(arr("best_minus_group_mean").mean()),
        "mean_best_minus_group_median": float(arr("best_minus_group_median").mean()),
        "mean_best_minus_nonbest_mean": float(arr("best_minus_nonbest_mean").mean()),
        "mean_group0_minus_group_mean": float(arr("group0_minus_group_mean").mean()),
        "mean_better_group_fraction": float(better_fraction.mean()),
        "median_better_group_fraction": float(np.median(better_fraction)),
        "mean_pair_better_fraction": float(pair_better_fraction.mean()),
        "improvable_state_fraction": float(np.mean(best_gain > DEFAULT_IMPROVEMENT_EPS)),
        "mean_improvable_pair_fraction": float(arr("improvable_pair_fraction").mean()),
        "mean_element_best_gain": float(arr("element_best_gain_mean").mean()),
        "mean_element_reward_range": float(arr("element_reward_range_mean").mean()),
        "mean_element_reward_std": float(arr("element_reward_std_mean").mean()),
        "mean_element_best_minus_mean": float(arr("element_best_minus_mean_mean").mean()),
        "mean_element_best_minus_median": float(arr("element_best_minus_median_mean").mean()),
        "mean_element_best_minus_nonbest_mean": float(arr("element_best_minus_nonbest_mean_mean").mean()),
        "mean_element_sparse_pairwise_ADE": float(arr("element_sparse_pairwise_ade_mean").mean()),
        "mean_element_sparse_pairwise_ADE_p90": float(arr("element_sparse_pairwise_ade_p90_mean").mean()),
        "mean_element_sparse_endpoint_dist": float(arr("element_sparse_endpoint_dist_mean").mean()),
        "mean_element_sparse_endpoint_dist_max": float(arr("element_sparse_endpoint_dist_max_mean").mean()),
        "mean_element_ADE_reward_std_corr": _finite_mean(arr("element_ade_reward_std_corr")),
        "mean_element_ADE_reward_range_corr": _finite_mean(arr("element_ade_reward_range_corr")),
        "mean_group0_rank": float(group0_rank.mean()),
        "median_group0_rank": float(np.median(group0_rank)),
        "group0_top1_fraction": float(np.mean(group0_rank <= 1.0)),
        "group0_top5_fraction": float(np.mean(group0_rank <= 5.0)),
        "group0_top10_fraction": float(np.mean(group0_rank <= 10.0)),
    }


def write_dataclass_csv(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    records = [asdict(row) for row in rows]
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_dict_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(list(rows), file, ensure_ascii=False, indent=2)


def maybe_plot_summary(output_dir: Path, rows: Sequence[ProbeSampleSummary]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot] skipped: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted({row.checkpoint for row in rows})
    specs = (
        ("better_group_fraction", "Better-group fraction vs group 0", "better_group_fraction_hist.png"),
        ("best_group_gain", "Best group reward gain vs group 0", "best_group_gain_hist.png"),
        ("group_reward_range", "Per-state group reward range", "group_reward_range_hist.png"),
        ("element_reward_range_mean", "Vehicle-mode reward range", "element_reward_range_hist.png"),
        ("element_sparse_pairwise_ade_mean", "Vehicle-mode trajectory pairwise ADE", "trajectory_pairwise_ADE_hist.png"),
        ("best_minus_group_mean", "Best-of-48 minus ordinary group mean", "best_minus_group_mean_hist.png"),
    )
    for field, title, filename in specs:
        plt.figure(figsize=(8, 5))
        for checkpoint in checkpoints:
            values = [float(getattr(row, field)) for row in rows if row.checkpoint == checkpoint]
            plt.hist(values, bins=20, alpha=0.45, label=checkpoint)
        plt.xlabel(field)
        plt.ylabel("sample count")
        plt.title(title)
        plt.grid(alpha=0.2)
        if len(checkpoints) > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()
