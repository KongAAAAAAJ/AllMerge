"""Reusable GRPO / Projection / Lagrangian / SafeMPO-Diff evaluation.

Offline curve plotting never imports the simulator. Model evaluation imports the
project runtime lazily, so saved train logs can be plotted on CPU-only machines.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

_STEP = re.compile(r"^\[step\s+(\d+)\s*/\s*\d+\](.*)$")
_PAIR = re.compile(r"(?<!\S)([\w./:-]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
CONSTRAINTS = ("collision", "road", "ttc", "background_gap", "teammate_gap")
EVALUATION_GALLERY_V2 = "execution_dense_only"


def parse_train_log(path: str | Path) -> list[dict[str, float]]:
    """Parse compact one-line step logs without assuming a specific trainer."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"train.log not found: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _STEP.match(line)
        if not match:
            continue
        row = {key: float(value) for key, value in _PAIR.findall(match.group(2))}
        row["step"] = float(match.group(1))
        rows.append(row)
    if not rows:
        raise ValueError(f"No [step 001/500] metric lines found in {path}")
    return rows


def parse_fixed_validation(path: str | Path) -> list[dict[str, float]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"fixed-validation CSV not found: {path}")
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                if value is None or not value.strip():
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    continue
            if "step" in row:
                rows.append(row)
    return rows


def _series(rows: list[dict[str, float]], key: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([r["step"] for r in rows if key in r and np.isfinite(r[key])])
    y = np.asarray([r[key] for r in rows if key in r and np.isfinite(r[key])])
    return x, y


def _first_key(rows: list[dict[str, float]], *keys: str) -> str | None:
    for key in keys:
        if any(key in row for row in rows):
            return key
    return None


def _smooth_nan(y: np.ndarray, window: int = 15) -> np.ndarray:
    if len(y) < 2 or window <= 1:
        return y
    k = min(window, len(y))
    if k < 2:
        return y
    weighted = np.convolve(y, np.ones(k), mode="same")
    denominator = np.convolve(np.ones_like(y), np.ones(k), mode="same")
    return weighted / denominator


def plot_training_curves(
    train_rows: list[dict[str, float]],
    fixed_rows: list[dict[str, float]],
    output_dir: str | Path,
) -> list[str]:
    """Plot training curves; never relabel mean violation as violation rate."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    def save(fig, name):
        p = out / name
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(p))

    reward_key = _first_key(
        train_rows,
        "reward/task_reward_mean", "reward_mean", "reward/effective_reward_mean",
    )
    fig, ax = plt.subplots(figsize=(10.3, 5.1))
    if reward_key:
        x, y = _series(train_rows, reward_key)
        ax.plot(x, y, alpha=0.32, linewidth=0.8, label="Train reward")
        ax.plot(x, _smooth_nan(y), linewidth=2, label="Train reward (moving avg)")
    fixed_current = _first_key(fixed_rows, "fixed_validation/current_reward_mean")
    fixed_pre = _first_key(fixed_rows, "fixed_validation/frozen_reward_mean")
    if fixed_current:
        x, y = _series(fixed_rows, fixed_current)
        ax.plot(x, y, marker="o", markersize=2.4, linewidth=1.5, label="Fixed validation RL")
    if fixed_pre:
        x, y = _series(fixed_rows, fixed_pre)
        ax.plot(x, y, linestyle="--", linewidth=1.1, label="Fixed validation pretrain")
    ax.set(title="Reward vs. training step", xlabel="Training step", ylabel="Reward")
    ax.grid(alpha=0.22)
    if ax.lines: ax.legend(fontsize=9)
    save(fig, "01_reward_curve.png")

    fig, ax = plt.subplots(figsize=(10.3, 5.1))
    losses = [
        ("loss", "Total loss"), ("policy_loss", "Policy loss"),
        ("safempo/distill_kl", "SafeMPO distillation KL"),
    ]
    for key, label in losses:
        if not any(key in r for r in train_rows):
            continue
        x, y = _series(train_rows, key)
        ax.plot(x, y, alpha=0.35, linewidth=0.9)
        ax.plot(x, _smooth_nan(y), linewidth=1.9, label=label)
    ax.set(title="Training loss", xlabel="Training step", ylabel="Loss / distillation KL")
    ax.grid(alpha=0.22)
    if ax.lines: ax.legend(fontsize=9)
    save(fig, "02_loss_curve.png")

    # Primary diagnostic: mean violation severity. These are NOT rates.
    fig, axes = plt.subplots(2, 1, figsize=(10.3, 8.2), sharex=True)
    for name in CONSTRAINTS:
        key = f"constraint/{name}_violation_mean"
        if any(key in row for row in train_rows):
            x, y = _series(train_rows, key)
            axes[0].plot(x, _smooth_nan(y), linewidth=1.65, label=name)
    max_key = _first_key(train_rows, "constraint/max_violation_mean")
    if max_key:
        x, y = _series(train_rows, max_key)
        axes[1].plot(x, y, alpha=.25, linewidth=.8)
        axes[1].plot(x, _smooth_nan(y), linewidth=2,
                     label="Maximum constraint violation (train)")
    fixed_max_key = _first_key(fixed_rows,
        "fixed_validation/current_constraint_max_violation_mean_mean")
    if fixed_max_key:
        x, y = _series(fixed_rows, fixed_max_key)
        axes[1].plot(x, y, marker="o", markersize=2.5,
                     label="Maximum violation (fixed validation)")
    axes[0].set_title("Training: per-constraint mean violation magnitude (moving average)")
    axes[1].set_title("Maximum violation magnitude (train vs fixed validation)")
    axes[1].set_xlabel("Training step")
    for ax in axes:
        ax.set_ylabel("Normalized violation magnitude")
        ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    save(fig, "03_constraint_violation_magnitude.png")

    # Fixed validation has continuous mean violation magnitudes, not rates.
    if fixed_rows:
        fig, axes = plt.subplots(2, 1, figsize=(10.3, 8.1), sharex=True)
        for name in CONSTRAINTS:
            current = f"fixed_validation/current_constraint_{name}_violation_mean_mean"
            delta = f"fixed_validation/constraint_{name}_violation_change_mean"
            if any(current in r for r in fixed_rows):
                x, y = _series(fixed_rows, current)
                axes[0].plot(x, y, linewidth=1.5, label=name)
            if any(delta in r for r in fixed_rows):
                x, y = _series(fixed_rows, delta)
                axes[1].plot(x, y, linewidth=1.5, label=name)
        axes[1].axhline(0, color="grey", linewidth=0.8)
        axes[0].set_title("Fixed validation: mean constraint violation MAGNITUDE")
        axes[1].set_title("Fixed validation: change vs pretrain (negative = improvement)")
        axes[1].set_xlabel("Training step")
        axes[0].set_ylabel("Normalized violation magnitude")
        axes[1].set_ylabel("RL - pretrain")
        for a in axes:
            a.grid(alpha=0.2)
            if a.lines: a.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        save(fig, "04_fixed_validation_violation_magnitude.png")

    # Optional rate plot only if genuine violation fraction metrics exist.
    if any(f"constraint/{name}_violation_fraction" in row
           for row in train_rows for name in CONSTRAINTS):
        fig, axes = plt.subplots(2, 1, figsize=(10.3, 8), sharex=True)
        for name in CONSTRAINTS:
            key = f"constraint/{name}_violation_fraction"
            if any(key in row for row in train_rows):
                x, y = _series(train_rows, key)
                axes[0].plot(x, _smooth_nan(y), linewidth=1.5, label=name)
        for rows, key, label in (
            (train_rows, "constraint/feasible_fraction", "Training"),
            (fixed_rows, "fixed_validation/current_constraint_feasible_fraction_mean", "Fixed validation"),
        ):
            if any(key in row for row in rows):
                x, y = _series(rows, key)
                axes[1].plot(x, y, label=label, linewidth=1.6)
        axes[0].set_title("Training violation fraction (supplementary)")
        axes[1].set_title("Joint feasible fraction")
        axes[1].set_xlabel("Training step")
        for ax in axes:
            ax.set_ylabel("Fraction [0, 1]")
            ax.set_ylim(-0.025, 1.025)
            ax.grid(alpha=.2)
            if ax.lines: ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, "05_constraint_violation_rates_optional.png")

    if any("reference_kl" in r or "safempo/teacher_kl" in r for r in train_rows):
        fig, ax = plt.subplots(figsize=(10.3, 4.7))
        for key, label in (
            ("reference_kl", "Global reference KL (training state)"),
            ("approx_kl", "Local approx KL"),
            ("safempo/teacher_kl", "SafeMPO teacher KL"),
        ):
            if any(key in r for r in train_rows):
                x, y = _series(train_rows, key)
                ax.plot(x, y, linewidth=1.1, alpha=0.85, label=label)
        ax.set(title="Policy KL diagnostics (state distribution may vary)",
               xlabel="Training step", ylabel="KL estimate")
        ax.grid(alpha=0.2)
        if ax.lines: ax.legend(fontsize=8)
        save(fig, "06_policy_kl.png")

    return generated


def _model(checkpoint: str | Path, device):
    import torch
    from pretraining.checkpoint_io import load_checkpoint_file
    from highway_env.planner.diffusion.config import build_structured_diffusion_config
    from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
    from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter
    state, stored = load_checkpoint_file(checkpoint, map_location="cpu")
    config = build_structured_diffusion_config(**stored)
    adapter = PlannerTensorAdapter(config, device)
    planner = StructuredDiffusionPlanner(config, adapter).to(device)
    planner.load_state_dict(state, strict=True)
    planner.eval()
    planner.requires_grad_(False)
    return planner, adapter


def _scenario_names(scenario: str) -> list[str]:
    if scenario == "all":
        return ["straight", "curved", "merge_in", "merge_out"]
    if scenario not in {"straight", "curved", "merge_in", "merge_out"}:
        raise ValueError(f"Unknown scenario: {scenario}")
    return [scenario]


# PAIRED_BASE_POINTS_EXECUTION_DENSE_V1_20261015
def _execution_dense_from_group_trace(model, trace, features, role: int, group_idx: int, mode: int):
    """Get the REAL residual-head+spline path for one paired G48 candidate.

    trace.candidates are metric-space BASE diffusion 8-point trajectories.
    Reuse trace.next_states[-1] (the exact state of the FINAL denoiser pass),
    re-evaluate its final mode tokens without randomness, then calculate
    residual-head sparse waypoints and the model's 41-knot execution spline.
    """
    import torch

    if not trace.next_states:
        raise RuntimeError("Cannot replay execution head: diffusion trace has no final state")
    final_state = trace.next_states[-1][role, group_idx].unsqueeze(0)
    one = {key: value[role:role+1] for key, value in features.items()
           if torch.is_tensor(value)}
    scene = model.scene_encoder(one)
    anchor = model.adapter.normalize_trajectory(one["coarse_trajectories"])
    last_t = torch.full((1,), int(trace.timesteps[-1]),
                        dtype=torch.long, device=final_state.device)
    predicted_x0, _, mode_tokens = model.denoiser(
        final_state, last_t, scene,
        navigation_point_norm=anchor[:, :, -1, :],
        return_mode_tokens=True,
    )
    replayed_base = model.adapter.denormalize_trajectory(predicted_x0)[0, mode]
    sampled_base = trace.candidates[role, group_idx, mode]
    if tuple(sampled_base.shape) != (8, 2):
        raise RuntimeError(f"Expected 8 base points, found {tuple(sampled_base.shape)}")
    deviation = float((replayed_base - sampled_base).abs().max().item())
    if not torch.isfinite(replayed_base).all() or deviation > 0.01:
        raise RuntimeError(
            f"Final-state replay mismatch: max base difference={deviation:.6g}m; "
            "cannot safely pair residual head with G48 sampled candidate"
        )

    x0_mode_norm = predicted_x0[:, mode]
    mode_feature = mode_tokens[:, mode]
    execution_residual = model.dense_residual_head(mode_feature, x0_mode_norm)
    execution_sparse = sampled_base.unsqueeze(0) + execution_residual
    start_xy = torch.zeros_like(execution_sparse[:, 0, :])
    start_velocity_xy = one["ego_state"][:, :2].to(dtype=execution_sparse.dtype)
    dense = model.dense_trajectory_spline(
        execution_sparse, start_xy=start_xy,
        start_velocity_xy=start_velocity_xy,
    )[0]
    if tuple(dense.shape) != (41, 2):
        raise RuntimeError(f"Expected 41-point 10Hz execution, got {tuple(dense.shape)}")
    if not torch.isfinite(dense).all():
        raise RuntimeError("Execution dense trajectory has NaN/Inf")
    knot_err = float((dense[5::5] - execution_sparse[0]).abs().max().item())
    if knot_err > 0.001:
        raise RuntimeError(f"Execution spline knot mismatch: {knot_err:.6g}m")
    return (dense.detach().cpu().numpy(),
            execution_sparse[0].detach().cpu().numpy())


def render_paired_gallery(gallery: list[dict[str, Any]], output_dir: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    """Plot paired diffusion base 8-point scatter + true execution 41-point line."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if len(gallery) != 9:
        raise ValueError(f"Expected exactly nine paired trajectory records, got {len(gallery)}")
    gallery = sorted(gallery, key=lambda e: e["state_index"])
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5), squeeze=False)
    manifest = []
    for ax, record in zip(axes.flat, gallery):
        base_delta = np.asarray(record["rl"], dtype=np.float64) - np.asarray(record["pre"], dtype=np.float64)
        dense_delta = np.asarray(record["rl_dense"], dtype=np.float64) - np.asarray(record["pre_dense"], dtype=np.float64)
        for policy, color, line_style, title in (
            ("pre", "#2473B8", "--", "Pretrain"),
            ("rl", "#E47E32", "-", "Trained"),
        ):
            base = np.asarray(record[policy], dtype=np.float64)
            dense = np.asarray(record[policy + "_dense"], dtype=np.float64)
            if base.shape != (8, 2) or dense.shape != (41, 2):
                raise ValueError(f"{title}: expected base [8,2] and execution dense [41,2]; "
                                 f"got {base.shape}, {dense.shape}")
            if not np.isfinite(base).all() or not np.isfinite(dense).all():
                raise ValueError("Non-finite trajectory data in paired gallery")
            # Dense line comes from the model's ACTUAL execution residual head + spline.
            ax.plot(dense[:, 0], dense[:, 1], color=color, linestyle=line_style,
                    lw=2.05, alpha=0.96, label=f"{title}: execution dense (10Hz)")
            # Eight visible markers are RAW diffusion coordinates, NOT dense knots.
            ax.scatter(base[:, 0], base[:, 1], color=color, marker="o" if policy == "pre" else "x",
                       s=28 if policy == "pre" else 34, zorder=5, alpha=0.92,
                       label=f"{title}: diffusion base (8 pts)")
        ax.scatter([0], [0], s=38, marker="+", color="black", label="Ego")
        ax.set(title=(f"{record['scenario']} | state {record['state_index']:02d} | "
                      f"veh {record['role']} | mode {record['mode']} | g{record['group_idx']} "
                      f"| base maxΔ {np.linalg.norm(base_delta,axis=-1).max():.2f}m"),
               xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6.5, loc="best")
        manifest.append({
            **{k: record[k] for k in ("state_index", "scenario", "env_seed", "noise_seed",
                                    "role", "mode", "group_idx")},
            "mean_point_displacement_m": float(np.linalg.norm(base_delta, axis=-1).mean()),
            "max_point_displacement_m": float(np.linalg.norm(base_delta, axis=-1).max()),
            "mean_dense_displacement_m": float(np.linalg.norm(dense_delta, axis=-1).mean()),
            "max_dense_displacement_m": float(np.linalg.norm(dense_delta, axis=-1).max()),
            "delta_base_end_x_m": float(base_delta[-1, 0]),
            "delta_base_end_y_m": float(base_delta[-1, 1]),
            "delta_execution_end_x_m": float(dense_delta[-1, 0]),
            "delta_execution_end_y_m": float(dense_delta[-1, 1]),
        })
    fig.suptitle("Pretrain vs trained checkpoint: base diffusion points and TRUE execution dense trajectories",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    figure = out / "07_pretrain_vs_rl_trajectories_3x3.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    with (out / "gallery_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return figure, manifest


def evaluate_paired_trajectories(
    checkpoint: str | Path,
    pretrained_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    scenario: str = "curved",
    num_states: int = 16,
    gallery_size: int = 9,
    group_size: int = 48,
    group_action: int = 3,
    seed: int = 7,
    validation_seed_offset: int = 10000,
    visualization_seed: int = 0,
    eta: float = 0.02,
    device_name: str = "auto",
    constraint_names: Iterable[str] = CONSTRAINTS,
    constraint_residual_cap: float = 5.0,
    task_reward_type: str = "unified_v6",
) -> dict[str, Any]:
    """Fresh paired-state evaluation with matching noise and frozen pretrain modes.

    Draws one predeclared random group per displayed state instead of selecting
    a post-hoc reward maximum; rates and magnitudes are measured over the full
    sampled valid-mode candidate support.
    """
    # NATIVE_EVAL_V6_ONLY_20261019: V6 is the only available evaluation reward.
    if task_reward_type != "unified_v6":
        raise ValueError(f"Only V6 reward is supported in evaluation, got {task_reward_type!r}")
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import torch
    from highway_env.envs.scenarios.curved_lane_change_env import CurvedLaneChangeEnv
    from highway_env.envs.scenarios.merge_in_env import MergeInEnv
    from highway_env.envs.scenarios.merge_out_env import MergeOutEnv
    from highway_env.envs.scenarios.straight_lane_change_env import StraightLaneChangeEnv
    from highway_env.planner.diffusion.grpo import (
        CandidateRewardAdapter, GroupDiffusionSampler,
        evaluate_w4_constraints, resolve_reward_evaluator,
        task_reward_from_result,
    )
    from highway_env.planner.diffusion.grpo.reward_adapter import GRPORewardContext
    from highway_env.planner.diffusion.tensor_adapter import BOOL_KEYS, FLOAT_KEYS

    if num_states < 9 or gallery_size != 9:
        raise ValueError("3x3 gallery requires --num-states >= 9 and --gallery-size 9")
    if group_size < 2:
        raise ValueError("GRPO group sampler requires group size >= 2")
    names = tuple(constraint_names)
    unknown = set(names) - set(CONSTRAINTS)
    if unknown:
        raise ValueError(f"unknown constraint names: {unknown}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    pretrained_model, adapter = _model(pretrained_checkpoint, device)
    trained_model, _ = _model(checkpoint, device)
    sampler_rl = GroupDiffusionSampler(trained_model, group_size=group_size, eta=eta)
    sampler_pre = GroupDiffusionSampler(pretrained_model, group_size=group_size, eta=eta)
    reward_adapter = CandidateRewardAdapter(resolve_reward_evaluator("auto"))
    scenario_classes = {
        "straight": StraightLaneChangeEnv,
        "curved": CurvedLaneChangeEnv,
        "merge_in": MergeInEnv,
        "merge_out": MergeOutEnv,
    }
    scenario_list = _scenario_names(scenario)
    feature_keys = (*FLOAT_KEYS, *BOOL_KEYS)
    rng = np.random.default_rng(visualization_seed)
    # Randomly select 9 distinct states. Within each state draw vehicle role
    # and group candidate; use the SAME (role, mode, group) in both policies.
    gallery_states = set(rng.choice(num_states, size=9, replace=False).tolist())
    gallery: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    reward_pair = {"pretrain": [], "rl": []}
    constraint_counts = {policy: {name: [] for name in names} for policy in ("pretrain", "rl")}
    constraint_magnitudes = {policy: {name: [] for name in names} for policy in ("pretrain", "rl")}
    max_magnitudes = {policy: [] for policy in ("pretrain", "rl")}
    feasible_values = {"pretrain": [], "rl": []}
    for index in range(num_states):
        env_name = scenario_list[index % len(scenario_list)]
        env_seed = seed + validation_seed_offset + index
        noise_seed = seed + 2 * validation_seed_offset + index
        env = scenario_classes[env_name](
            config={"show_trajectories": False, "show_future_trajectories": False},
            render_mode=None,
        )
        try:
            env.reset(seed=env_seed)
            _, _, terminated, truncated, _ = env.step(group_action)
            if terminated or truncated:
                raise RuntimeError(f"Scenario ended during state collection: {env_name}, seed={env_seed}")
            values = getattr(env, "latest_planner_features", None)
            if values is None:
                raise RuntimeError("scenario has no latest_planner_features after env.step")
            features = adapter.to_torch({key: values[key] for key in feature_keys})
            with torch.no_grad():
                frozen_generator = torch.Generator(device=device.type).manual_seed(noise_seed + 314159)
                frozen = pretrained_model.infer_multimodal(features, generator=frozen_generator)
                mode_idx = frozen["trajectory_mode_idx"].detach().long().flatten()
                context = GRPORewardContext(
                    env=env,
                    frozen_all_mode_trajectories=frozen["trajectory_candidates"],
                    frozen_argmax_joint_trajectories=frozen["trajectory"],
                    valid_mode_mask=features["mode_valid_mask"],
                )
                generator_pre = torch.Generator(device=device.type).manual_seed(noise_seed)
                generator_rl = torch.Generator(device=device.type).manual_seed(noise_seed)
                trace_pre = sampler_pre.sample(features, generator=generator_pre)
                trace_rl = sampler_rl.sample(features, generator=generator_rl)
                results = {}
                for policy, trace in (("pretrain", trace_pre), ("rl", trace_rl)):
                    result = reward_adapter.evaluate_result(
                        trace.candidates, features=features, context=context,
                    )
                    cb = evaluate_w4_constraints(
                        result, context=context,
                        device=trace.candidates.device, dtype=trace.candidates.dtype,
                        names=names, residual_cap=constraint_residual_cap,
                    )
                    valid = features["mode_valid_mask"].bool()
                    if valid.ndim != 2:
                        raise RuntimeError(f"expected [vehicle,mode] valid mask, got {valid.shape}")
                    mask = valid[:, None, :].expand(-1, group_size, -1)
                    for j, name in enumerate(names):
                        values = cb.violation[..., j][mask].detach().cpu().numpy()
                        constraint_counts[policy][name].append(float(np.mean(values > 1e-6)))
                        constraint_magnitudes[policy][name].append(float(np.mean(values)))
                    max_magnitudes[policy].append(float(cb.violation.max(dim=-1).values[mask].mean().item()))
                    feasible_values[policy].append(float(cb.feasible_mask[mask].float().mean().item()))
                    task = task_reward_from_result(
                        result, context=context, device=trace.candidates.device,
                        dtype=trace.candidates.dtype, reward_type=task_reward_type,
                    )
                    if task.shape != mask.shape:
                        raise RuntimeError(f"Unexpected reward/mask shapes: {task.shape} vs {mask.shape}")
                    reward_pair[policy].append(float(task[mask].mean().item()))
                    results[policy] = trace

                row = {"state_index": index, "scenario": env_name,
                       "env_seed": env_seed, "noise_seed": noise_seed}
                for policy in ("pretrain", "rl"):
                    row[f"{policy}_reward"] = reward_pair[policy][-1]
                    row[f"{policy}_feasible_fraction"] = feasible_values[policy][-1]
                    row[f"{policy}_max_violation_mean"] = max_magnitudes[policy][-1]
                    for name in names:
                        row[f"{policy}_{name}_violation_fraction"] = constraint_counts[policy][name][-1]
                        row[f"{policy}_{name}_violation_mean"] = constraint_magnitudes[policy][name][-1]
                rows.append(row)

                if index in gallery_states:
                    # Fair pairing: randomly select role/group but fix pretrained mode.
                    role = int(rng.integers(0, int(valid.shape[0])))
                    group_idx = int(rng.integers(0, group_size))
                    selected_mode = int(mode_idx[role].item())
                    if not bool(features["mode_valid_mask"][role, selected_mode]):
                        raise RuntimeError("pretrain-selected mode is invalid")
                    pre = results["pretrain"].candidates[role, group_idx, selected_mode].cpu().numpy()
                    rl = results["rl"].candidates[role, group_idx, selected_mode].cpu().numpy()
                    # PAIRED_BASE_POINTS_EXECUTION_DENSE_V1_20261015
                    pre_dense, pre_execution_sparse = _execution_dense_from_group_trace(
                        pretrained_model, results["pretrain"], features, role, group_idx, selected_mode)
                    rl_dense, rl_execution_sparse = _execution_dense_from_group_trace(
                        trained_model, results["rl"], features, role, group_idx, selected_mode)
                    gallery.append({**row, "role": role, "mode": selected_mode,
                                    "group_idx": group_idx, "pre": pre, "rl": rl,
                                    "pre_dense": pre_dense, "rl_dense": rl_dense,
                                    "pre_execution_sparse": pre_execution_sparse,
                                    "rl_execution_sparse": rl_execution_sparse})
        finally:
            env.close()

    with (out / "paired_eval_per_state.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, manifest = render_paired_gallery(gallery, out)

    summary = {"model_checkpoint": str(checkpoint),
               "pretrain_checkpoint": str(pretrained_checkpoint),
               "scenario": scenario, "seed": seed, "group_size": group_size,
               "task_reward_type": task_reward_type,
               "state_count": len(rows),
               "gallery": str(figure),
               "reward": {p: float(np.mean(v)) for p, v in reward_pair.items()},
               "joint_feasible_fraction": {p: float(np.mean(v)) for p, v in feasible_values.items()},
               "constraint_violation_fraction": {
                   p: {n: float(np.mean(constraint_counts[p][n])) for n in names}
                   for p in ("pretrain", "rl")},
               "constraint_violation_mean": {
                   p: {n: float(np.mean(constraint_magnitudes[p][n])) for n in names}
                   for p in ("pretrain", "rl")},
               "max_violation_mean": {
                   p: float(np.mean(max_magnitudes[p])) for p in ("pretrain", "rl")},
               "note": "Rates reflect sampled valid-mode G48 candidates on paired states; "
                       "not online deployment collision statistics."}
    (out / "paired_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(10.3, 5.2))
    xs = np.arange(len(names))
    width = 0.36
    for offset, policy, label in ((-width/2, "pretrain", "Pretrain"),
                                  (width/2, "rl", "RL")):
        ax.bar(xs + offset,
               [summary["constraint_violation_mean"][policy][n] for n in names],
               width, label=label)
    ax.set_xticks(xs, names, rotation=12)
    ax.set_ylabel("Mean normalized violation magnitude")
    ax.set_title("Paired checkpoint: constraint violation MAGNITUDE")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "08_paired_constraint_violation_mean.png", dpi=180)
    plt.close(fig)

    # ALLMERGE_UNIFIED_EXECUTION_DENSE_EVAL_V2_20261015
    # Plot only the true execution-dense trajectory displacement (41 x 0.1s),
    # consistent with the execution curves in 07. No legacy base-point plot.
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), squeeze=False)
    for ax, record in zip(axes.flat, gallery):
        displacement = np.asarray(record["rl_dense"]) - np.asarray(record["pre_dense"])
        if displacement.shape != (41, 2):
            raise ValueError("Unified execution evaluation requires paired 41-point dense trajectories")
        time_steps = np.arange(41, dtype=np.float64) * 0.1
        ax.plot(time_steps, displacement[:, 0], label="Δ execution x")
        ax.plot(time_steps, displacement[:, 1], label="Δ execution y")
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set(title=f"state {record['state_index']} | veh {record['role']} | g{record['group_idx']}",
               xlabel="Time (s)", ylabel="Trained − Pretrain execution (m)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Pretrain vs trained: execution-dense displacement (41 points, 10Hz)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "09_paired_trajectory_displacement_3x3.png", dpi=180)
    plt.close(fig)
    return summary
