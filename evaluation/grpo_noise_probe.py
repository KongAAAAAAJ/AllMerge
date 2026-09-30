from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

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
    group_reward_range: float
    group_reward_std: float
    better_group_count: int
    better_group_fraction: float
    group0_rank: int
    pair_better_fraction: float
    improvable_pair_fraction: float
    element_best_gain_mean: float
    element_reward_range_mean: float
    element_reward_std_mean: float


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
    best_group_id: int
    best_gain: float
    reward_range: float
    reward_std: float
    better_count: int
    better_fraction: float


def _as_numpy(value: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def analyze_group_rewards(
    rewards: torch.Tensor | np.ndarray,
    valid_mode_mask: torch.Tensor | np.ndarray,
    *,
    checkpoint: str,
    scenario: str,
    sample_index: int,
    env_seed: int,
    noise_seed: int,
    improvement_eps: float = DEFAULT_IMPROVEMENT_EPS,
) -> tuple[ProbeSampleSummary, list[ProbeGroupSummary], list[ProbeVehicleModeSummary]]:
    """Analyze one GRPO reward tensor using group 0 as the fixed reference.

    Parameters
    ----------
    rewards:
        [vehicle=3, group=G, mode=10]. This is the exact trainer-side W4 layout.
    valid_mode_mask:
        [vehicle=3, mode=10]. Invalid vehicle-mode pairs are excluded from every
        aggregate statistic.
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

    group_size = int(values.shape[1])
    valid_pair_count = int(valid.sum())

    # Group-level reward: mean across the exact GRPO learning units
    # (vehicle x valid mode) for the same group id.
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

    nonzero_group_count = group_size - 1
    better_group_count = int(better_group[1:].sum())
    better_group_fraction = float(better_group_count / nonzero_group_count)
    group0_rank = int(1 + np.sum(group_rewards[1:] > group0_reward + improvement_eps))

    # Exact GRPO element-level diagnostics. Each (vehicle, mode) has G candidates.
    # Compare g=1..G-1 with the corresponding g=0 reward.
    baseline = values[:, 0, :]  # [V,M]
    deltas = values[:, 1:, :] - baseline[:, None, :]  # [V,G-1,M]
    valid_expanded = np.broadcast_to(valid[:, None, :], deltas.shape)
    valid_deltas = deltas[valid_expanded]
    pair_better_fraction = float(np.mean(valid_deltas > improvement_eps))

    pair_best = np.max(values, axis=1)  # [V,M]
    pair_worst = np.min(values, axis=1)
    pair_std = np.std(values, axis=1)
    pair_best_gain = pair_best - baseline
    pair_improvable = pair_best_gain > improvement_eps

    improvable_pair_fraction = float(pair_improvable[valid].mean())
    element_best_gain_mean = float(pair_best_gain[valid].mean())
    element_reward_range_mean = float((pair_best - pair_worst)[valid].mean())
    element_reward_std_mean = float(pair_std[valid].mean())

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
        group_reward_range=float(group_rewards.max() - group_rewards.min()),
        group_reward_std=float(group_rewards.std()),
        better_group_count=better_group_count,
        better_group_fraction=better_group_fraction,
        group0_rank=group0_rank,
        pair_better_fraction=pair_better_fraction,
        improvable_pair_fraction=improvable_pair_fraction,
        element_best_gain_mean=element_best_gain_mean,
        element_reward_range_mean=element_reward_range_mean,
        element_reward_std_mean=element_reward_std_mean,
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

    vm_rows: list[ProbeVehicleModeSummary] = []
    for role in range(values.shape[0]):
        for mode in range(values.shape[2]):
            if not valid[role, mode]:
                continue
            row = values[role, :, mode]
            r0 = float(row[0])
            delta = row[1:] - r0
            best_id = int(np.argmax(row))
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
                    best_reward=float(row[best_id]),
                    worst_reward=float(row.min()),
                    best_group_id=best_id,
                    best_gain=float(row[best_id] - r0),
                    reward_range=float(row.max() - row.min()),
                    reward_std=float(row.std()),
                    better_count=int(np.sum(delta > improvement_eps)),
                    better_fraction=float(np.mean(delta > improvement_eps)),
                )
            )

    return sample_summary, group_rows, vm_rows


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def aggregate_sample_summaries(
    rows: Sequence[ProbeSampleSummary],
    *,
    checkpoint: str,
    scenario: str,
) -> dict[str, float | str | int]:
    subset = [row for row in rows if row.checkpoint == checkpoint and (scenario == "all" or row.scenario == scenario)]
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
        "mean_best_group_reward": float(arr("best_group_reward").mean()),
        "mean_best_group_gain": float(best_gain.mean()),
        "median_best_group_gain": float(np.median(best_gain)),
        "p25_best_group_gain": _quantile(best_gain, 0.25),
        "p75_best_group_gain": _quantile(best_gain, 0.75),
        "p90_best_group_gain": _quantile(best_gain, 0.90),
        "mean_group_reward_range": float(arr("group_reward_range").mean()),
        "median_group_reward_range": float(np.median(arr("group_reward_range"))),
        "mean_group_reward_std": float(arr("group_reward_std").mean()),
        "mean_better_group_fraction": float(better_fraction.mean()),
        "median_better_group_fraction": float(np.median(better_fraction)),
        "mean_pair_better_fraction": float(pair_better_fraction.mean()),
        "improvable_state_fraction": float(np.mean(best_gain > DEFAULT_IMPROVEMENT_EPS)),
        "mean_improvable_pair_fraction": float(arr("improvable_pair_fraction").mean()),
        "mean_element_best_gain": float(arr("element_best_gain_mean").mean()),
        "mean_element_reward_range": float(arr("element_reward_range_mean").mean()),
        "mean_element_reward_std": float(arr("element_reward_std_mean").mean()),
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
    """Create lightweight diagnostic plots; skip cleanly if matplotlib is unavailable."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[plot] skipped: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted({row.checkpoint for row in rows})

    specs = (
        ("better_group_fraction", "Better-group fraction vs group 0", "better_group_fraction_hist.png"),
        ("best_group_gain", "Best group reward gain vs group 0", "best_group_gain_hist.png"),
        ("group_reward_range", "Per-state group reward range", "group_reward_range_hist.png"),
        ("pair_better_fraction", "Vehicle-mode candidate better fraction", "pair_better_fraction_hist.png"),
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
