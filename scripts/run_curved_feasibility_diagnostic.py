from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_grpo_noise_probe import (
    CheckpointSpec,
    _build_model,
    _frozen_reward_context,
    _parse_checkpoint,
    _reward_result_arrays,
    _sample_noise_seed,
    _scenario_features,
    _state_seed,
)
from evaluation.grpo_probe_sampling import (
    HybridProbeGroupDiffusionSampler,
    MultiplicativeProbeGroupDiffusionSampler,
)
from highway_env.envs.scenarios.curved_lane_change_env import CurvedLaneChangeEnv
from highway_env.planner.geometry import world_to_ego_point
from highway_env.planner.diffusion.grpo.reward_adapter import (
    CandidateRewardAdapter,
    resolve_reward_evaluator,
)
from highway_env.planner.diffusion.grpo.sampling import GroupDiffusionSampler
from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
from highway_env.planner.diffusion.trajectory_mode_reward.geometry import (
    _dense_local_trajectories,
    local_to_world,
    road_margin_series,
)


@dataclass(frozen=True)
class StateRoleSummary:
    checkpoint: str
    sample_index: int
    env_seed: int
    noise_seed: int
    vehicle_role: int
    selected_mode: int
    group_size: int
    feasible_group_count: int
    feasible_group_fraction: float
    all_groups_offroad: int
    expert_min_margin_m: float
    expert_offroad: int
    group0_reward: float
    group0_min_margin_m: float
    group0_offroad: int
    group0_first_offroad_time_s: float
    reward_best_group_id: int
    reward_best_reward: float
    reward_best_min_margin_m: float
    reward_best_offroad: int
    reward_best_first_offroad_time_s: float
    margin_best_group_id: int
    margin_best_reward: float
    margin_best_min_margin_m: float
    margin_best_offroad: int
    candidate_min_margin_mean_m: float
    candidate_min_margin_p10_m: float
    candidate_min_margin_p50_m: float
    candidate_min_margin_p90_m: float
    candidate_endpoint_dy_vs_expert_mean_m: float
    candidate_endpoint_dy_vs_expert_std_m: float
    candidate_mean_dy_vs_expert_mean_m: float
    candidate_mean_abs_dy_vs_expert_mean_m: float
    margin_endpoint_dy_corr: float
    margin_mean_dy_corr: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Curved-only feasibility diagnostic for selected diffusion modes. "
            "Measures exact production road-margin profiles for expert, Group0, "
            "reward-best, margin-best and all 48 groups."
        )
    )
    p.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True)
    p.add_argument("--samples", type=int, default=25)
    p.add_argument("--states-per-gallery", type=int, default=9)
    p.add_argument("--group-size", type=int, default=48)
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--noise-type", choices=("additive", "multiplicative", "hybrid"), default="hybrid")
    p.add_argument("--multiplicative-std", type=float, default=0.04)
    p.add_argument("--hybrid-multiplicative-x-std", type=float, default=0.12)
    p.add_argument("--hybrid-additive-y-std-m", type=float, default=1.00)
    p.add_argument("--group-action", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--noise-seed", type=int, default=700000)
    p.add_argument("--visualization-seed", type=int, default=0)
    p.add_argument("--max-seed-attempts", type=int, default=1000)
    p.add_argument("--reward-fn", default="auto")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/curved_feasibility_diagnostic"))
    return p.parse_args()


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _build_sampler(args: argparse.Namespace, model: Any):
    if args.noise_type == "additive":
        return GroupDiffusionSampler(model, group_size=args.group_size, eta=args.eta)
    if args.noise_type == "multiplicative":
        return MultiplicativeProbeGroupDiffusionSampler(
            model,
            group_size=args.group_size,
            eta=args.eta,
            multiplicative_std=args.multiplicative_std,
        )
    return HybridProbeGroupDiffusionSampler(
        model,
        group_size=args.group_size,
        eta=args.eta,
        multiplicative_x_std=args.hybrid_multiplicative_x_std,
        additive_y_std_m=args.hybrid_additive_y_std_m,
    )


def _planning_poses(env: Any) -> np.ndarray:
    snapshot = getattr(env, "latest_planner_reward_state_snapshot", None)
    if not isinstance(snapshot, dict):
        raise RuntimeError(
            "planning-time reward snapshot missing; install the reward-frame snapshot fix first"
        )
    poses = np.asarray(snapshot.get("controlled_poses"), dtype=np.float64)
    if poses.shape != (3, 3) or not np.isfinite(poses).all():
        raise RuntimeError(f"invalid planning snapshot controlled_poses: {poses.shape}")
    return poses


def _expert_sparse(env: Any) -> np.ndarray:
    alignment = getattr(env, "latest_expert_alignment", None)
    if not isinstance(alignment, dict) or "expert_trajectory_xy" not in alignment:
        raise RuntimeError("latest_expert_alignment/expert_trajectory_xy missing")
    expert = np.asarray(alignment["expert_trajectory_xy"], dtype=np.float64)
    if expert.shape != (3, 8, 2):
        raise RuntimeError(f"unexpected expert trajectory shape: {expert.shape}")
    return expert


def _lane_lines_local(env: Any, pose: np.ndarray) -> list[np.ndarray]:
    graph = getattr(env.road.network, "graph", {})
    lines: list[np.ndarray] = []
    seen: set[int] = set()
    for outgoing in graph.values():
        if not isinstance(outgoing, dict):
            continue
        for lanes in outgoing.values():
            if not isinstance(lanes, (list, tuple)):
                continue
            for lane in lanes:
                if id(lane) in seen:
                    continue
                seen.add(id(lane))
                try:
                    ss = np.linspace(0.0, float(lane.length), 160)
                    world = np.asarray([lane.position(float(s), 0.0) for s in ss], dtype=np.float32)
                    local = world_to_ego_point(world, pose[:2], float(pose[2]))
                    mask = (
                        (local[:, 0] >= -20.0)
                        & (local[:, 0] <= 145.0)
                        & (np.abs(local[:, 1]) <= 30.0)
                    )
                    if int(mask.sum()) >= 2:
                        lines.append(local[mask])
                except (AttributeError, TypeError, ValueError):
                    continue
    return lines


def _dense_profiles(
    sparse_candidates: np.ndarray,
    *,
    role: int,
    frozen_argmax: np.ndarray,
    pose: np.ndarray,
    road: Any,
    config: TrajectoryModeRewardConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.asarray(sparse_candidates, dtype=np.float64)
    group_size = int(candidates.shape[0])
    joint = np.broadcast_to(frozen_argmax[None], (group_size, 3, 8, 2)).copy()
    joint[:, role] = candidates
    dense, times = _dense_local_trajectories(joint, config)
    local = dense[:, role]
    world = np.asarray([local_to_world(local[g], pose) for g in range(group_size)])
    margins = np.asarray(
        [road_margin_series(world[g], road, config, tracking_aware=True) for g in range(group_size)],
        dtype=np.float64,
    )
    return local, margins, times


def _expert_profile(
    expert: np.ndarray,
    *,
    role: int,
    pose: np.ndarray,
    road: Any,
    config: TrajectoryModeRewardConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense, times = _dense_local_trajectories(expert[None], config)
    local = dense[0, role]
    world = local_to_world(local, pose)
    margin = road_margin_series(world, road, config, tracking_aware=True)
    return local, margin, times


def _offroad(profile: np.ndarray) -> bool:
    return bool(np.any(np.asarray(profile)[1:] < 0.0))


def _first_offroad_time(profile: np.ndarray, times: np.ndarray) -> float:
    indices = np.flatnonzero(np.asarray(profile)[1:] < 0.0)
    if len(indices) == 0:
        return float("nan")
    return float(times[int(indices[0]) + 1])


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(checkpoint: str, state_rows: list[StateRoleSummary]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for role in range(3):
        rows = [r for r in state_rows if r.vehicle_role == role]
        if not rows:
            continue
        arr = lambda name: np.asarray([getattr(r, name) for r in rows], dtype=np.float64)
        item: dict[str, Any] = {
            "checkpoint": checkpoint,
            "vehicle_role": role,
            "state_count": len(rows),
            "mean_feasible_group_fraction": float(np.mean(arr("feasible_group_fraction"))),
            "state_has_feasible_candidate_fraction": float(np.mean(arr("feasible_group_count") > 0)),
            "state_all_groups_offroad_fraction": float(np.mean(arr("all_groups_offroad"))),
            "expert_offroad_fraction": float(np.mean(arr("expert_offroad"))),
            "group0_offroad_fraction": float(np.mean(arr("group0_offroad"))),
            "reward_best_offroad_fraction": float(np.mean(arr("reward_best_offroad"))),
            "margin_best_offroad_fraction": float(np.mean(arr("margin_best_offroad"))),
            "mean_expert_min_margin_m": float(np.mean(arr("expert_min_margin_m"))),
            "mean_group0_min_margin_m": float(np.mean(arr("group0_min_margin_m"))),
            "mean_reward_best_min_margin_m": float(np.mean(arr("reward_best_min_margin_m"))),
            "mean_margin_best_min_margin_m": float(np.mean(arr("margin_best_min_margin_m"))),
            "mean_candidate_endpoint_dy_vs_expert_m": float(np.mean(arr("candidate_endpoint_dy_vs_expert_mean_m"))),
            "mean_candidate_abs_dy_vs_expert_m": float(np.mean(arr("candidate_mean_abs_dy_vs_expert_mean_m"))),
        }
        modes = [r.selected_mode for r in rows]
        for mode in range(10):
            item[f"selected_mode_{mode}_fraction"] = float(np.mean(np.asarray(modes) == mode))
        out.append(item)
    return out


def _plot_galleries(
    args: argparse.Namespace,
    checkpoint: CheckpointSpec,
    plot_records: dict[int, list[dict[str, Any]]],
) -> None:
    rng = np.random.default_rng(args.visualization_seed)
    plot_dir = args.output_dir / checkpoint.name / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for role in range(3):
        records = plot_records[role]
        if not records:
            continue
        count = min(args.states_per_gallery, len(records))
        indices = sorted(rng.choice(len(records), size=count, replace=False).tolist())
        chosen = [records[i] for i in indices]

        fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
        for ax, rec in zip(axes.ravel(), chosen):
            for line in rec["lane_lines"]:
                ax.plot(line[:, 0], line[:, 1], linewidth=0.8, alpha=0.45)
            all_local = rec["candidate_local"]
            for g in range(all_local.shape[0]):
                ax.plot(all_local[g, :, 0], all_local[g, :, 1], linewidth=0.6, alpha=0.12)
            ax.plot(rec["expert_local"][:, 0], rec["expert_local"][:, 1], linewidth=2.4, label="Expert")
            ax.plot(all_local[0, :, 0], all_local[0, :, 1], linewidth=1.8, label="Group0")
            rb = rec["reward_best_group"]
            mb = rec["margin_best_group"]
            ax.plot(all_local[rb, :, 0], all_local[rb, :, 1], linewidth=2.0, label="Reward best")
            ax.plot(all_local[mb, :, 0], all_local[mb, :, 1], linewidth=1.7, linestyle="--", label="Margin best")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.18)
            ax.set_title(
                f"state={rec['sample_index']:02d} mode={rec['mode']} feasible={rec['feasible_count']}/{args.group_size}\n"
                f"m: exp={rec['expert_min']:+.2f} G0={rec['group0_min']:+.2f} "
                f"Rbest={rec['reward_best_min']:+.2f} Mbest={rec['margin_best_min']:+.2f}",
                fontsize=8,
            )
            ax.tick_params(labelsize=7)
        axes.ravel()[0].legend(fontsize=7)
        fig.suptitle(f"Curved Feasibility Geometry | {checkpoint.name} | role {role}")
        path = plot_dir / f"curved_feasibility_geometry_role{role}_3x3_{checkpoint.name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"[write] {path}")

        fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
        for ax, rec in zip(axes.ravel(), chosen):
            profiles = rec["candidate_margin"]
            times = rec["times"]
            for g in range(profiles.shape[0]):
                ax.plot(times, profiles[g], linewidth=0.6, alpha=0.12)
            ax.axhline(0.0, linewidth=1.0)
            ax.plot(times, rec["expert_margin"], linewidth=2.4, label="Expert")
            ax.plot(times, profiles[0], linewidth=1.8, label="Group0")
            rb = rec["reward_best_group"]
            mb = rec["margin_best_group"]
            ax.plot(times, profiles[rb], linewidth=2.0, label="Reward best")
            ax.plot(times, profiles[mb], linewidth=1.7, linestyle="--", label="Margin best")
            ax.set_xlabel("time [s]", fontsize=7)
            ax.set_ylabel("road margin [m]", fontsize=7)
            ax.grid(True, alpha=0.18)
            ax.set_title(
                f"state={rec['sample_index']:02d} feasible={rec['feasible_count']}/{args.group_size}\n"
                f"first off: G0={rec['group0_first']:.2f}s Rbest={rec['reward_best_first']:.2f}s",
                fontsize=8,
            )
            ax.tick_params(labelsize=7)
        axes.ravel()[0].legend(fontsize=7)
        fig.suptitle(f"Curved Road-Margin Profiles | {checkpoint.name} | role {role}")
        path = plot_dir / f"curved_margin_profile_role{role}_3x3_{checkpoint.name}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"[write] {path}")


def _run_checkpoint(args: argparse.Namespace, checkpoint: CheckpointSpec) -> None:
    device = torch.device(args.device)
    adapter, model = _build_model(checkpoint, device)
    sampler = _build_sampler(args, model)
    reward_adapter = CandidateRewardAdapter(resolve_reward_evaluator(args.reward_fn))
    config = TrajectoryModeRewardConfig()

    state_rows: list[StateRoleSummary] = []
    candidate_rows: list[dict[str, Any]] = []
    plot_records: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}

    collected = 0
    attempts = 0
    while collected < args.samples:
        if attempts >= args.max_seed_attempts:
            raise RuntimeError("could not collect enough valid curved states")
        env_seed = _state_seed(args.seed, 0, attempts)
        attempts += 1
        env = CurvedLaneChangeEnv(
            config={"show_trajectories": False, "show_future_trajectories": False},
            render_mode=None,
        )
        try:
            env.reset(seed=env_seed)
            _, _, terminated, truncated, _ = env.step(args.group_action)
            if bool(terminated) or bool(truncated):
                continue
            features = _scenario_features(env, adapter)
            context, selected_mode_idx = _frozen_reward_context(model, features, env)
            noise_seed = _sample_noise_seed(args.noise_seed, 0, collected)
            generator = torch.Generator(device=device.type).manual_seed(noise_seed)
            with torch.no_grad():
                trace = sampler.sample(features, generator=generator)
                reward_result = reward_adapter.evaluate_result(
                    trace.candidates, features=features, context=context
                )
                arrays = _reward_result_arrays(reward_result, group_size=args.group_size)

            poses = _planning_poses(env)
            expert = _expert_sparse(env)
            frozen_argmax = _as_numpy(context.frozen_argmax_joint_trajectories).astype(np.float64)
            candidates = trace.candidates.detach().cpu().numpy().astype(np.float64)
            lane_cache = {role: _lane_lines_local(env, poses[role]) for role in range(3)}

            for role in range(3):
                mode = int(selected_mode_idx[role].detach().cpu().item())
                sparse = candidates[role, :, mode]
                dense_local, margins, times = _dense_profiles(
                    sparse,
                    role=role,
                    frozen_argmax=frozen_argmax,
                    pose=poses[role],
                    road=env.road,
                    config=config,
                )
                expert_local, expert_margin, expert_times = _expert_profile(
                    expert,
                    role=role,
                    pose=poses[role],
                    road=env.road,
                    config=config,
                )
                if not np.allclose(times, expert_times):
                    raise RuntimeError("candidate/expert dense time grids differ")

                min_margin = np.min(margins, axis=1)
                offroad = np.any(margins[:, 1:] < 0.0, axis=1)
                feasible_count = int(np.sum(~offroad))
                rewards = arrays["reward"][role, :, mode]
                reward_best = int(np.argmax(rewards))
                margin_best = int(np.argmax(min_margin))
                endpoint_dy = sparse[:, -1, 1] - expert[role, -1, 1]
                mean_dy = np.mean(sparse[:, :, 1] - expert[role, None, :, 1], axis=1)
                mean_abs_dy = np.mean(np.abs(sparse[:, :, 1] - expert[role, None, :, 1]), axis=1)

                summary = StateRoleSummary(
                    checkpoint=checkpoint.name,
                    sample_index=collected,
                    env_seed=env_seed,
                    noise_seed=noise_seed,
                    vehicle_role=role,
                    selected_mode=mode,
                    group_size=args.group_size,
                    feasible_group_count=feasible_count,
                    feasible_group_fraction=float(feasible_count / args.group_size),
                    all_groups_offroad=int(feasible_count == 0),
                    expert_min_margin_m=float(np.min(expert_margin)),
                    expert_offroad=int(_offroad(expert_margin)),
                    group0_reward=float(rewards[0]),
                    group0_min_margin_m=float(min_margin[0]),
                    group0_offroad=int(offroad[0]),
                    group0_first_offroad_time_s=_first_offroad_time(margins[0], times),
                    reward_best_group_id=reward_best,
                    reward_best_reward=float(rewards[reward_best]),
                    reward_best_min_margin_m=float(min_margin[reward_best]),
                    reward_best_offroad=int(offroad[reward_best]),
                    reward_best_first_offroad_time_s=_first_offroad_time(margins[reward_best], times),
                    margin_best_group_id=margin_best,
                    margin_best_reward=float(rewards[margin_best]),
                    margin_best_min_margin_m=float(min_margin[margin_best]),
                    margin_best_offroad=int(offroad[margin_best]),
                    candidate_min_margin_mean_m=float(np.mean(min_margin)),
                    candidate_min_margin_p10_m=float(np.quantile(min_margin, 0.10)),
                    candidate_min_margin_p50_m=float(np.quantile(min_margin, 0.50)),
                    candidate_min_margin_p90_m=float(np.quantile(min_margin, 0.90)),
                    candidate_endpoint_dy_vs_expert_mean_m=float(np.mean(endpoint_dy)),
                    candidate_endpoint_dy_vs_expert_std_m=float(np.std(endpoint_dy)),
                    candidate_mean_dy_vs_expert_mean_m=float(np.mean(mean_dy)),
                    candidate_mean_abs_dy_vs_expert_mean_m=float(np.mean(mean_abs_dy)),
                    margin_endpoint_dy_corr=_safe_corr(min_margin, endpoint_dy),
                    margin_mean_dy_corr=_safe_corr(min_margin, mean_dy),
                )
                state_rows.append(summary)

                for group in range(args.group_size):
                    candidate_rows.append({
                        "checkpoint": checkpoint.name,
                        "sample_index": collected,
                        "env_seed": env_seed,
                        "noise_seed": noise_seed,
                        "vehicle_role": role,
                        "selected_mode": mode,
                        "group_id": group,
                        "reward": float(rewards[group]),
                        "minimum_road_margin_m": float(min_margin[group]),
                        "offroad": int(offroad[group]),
                        "first_offroad_time_s": _first_offroad_time(margins[group], times),
                        "endpoint_dx_vs_expert_m": float(sparse[group, -1, 0] - expert[role, -1, 0]),
                        "endpoint_dy_vs_expert_m": float(endpoint_dy[group]),
                        "mean_dy_vs_expert_m": float(mean_dy[group]),
                        "mean_abs_dy_vs_expert_m": float(mean_abs_dy[group]),
                        "is_group0": int(group == 0),
                        "is_reward_best": int(group == reward_best),
                        "is_margin_best": int(group == margin_best),
                    })

                plot_records[role].append({
                    "sample_index": collected,
                    "mode": mode,
                    "feasible_count": feasible_count,
                    "lane_lines": lane_cache[role],
                    "candidate_local": dense_local,
                    "candidate_margin": margins,
                    "expert_local": expert_local,
                    "expert_margin": expert_margin,
                    "times": times,
                    "reward_best_group": reward_best,
                    "margin_best_group": margin_best,
                    "expert_min": float(np.min(expert_margin)),
                    "group0_min": float(min_margin[0]),
                    "reward_best_min": float(min_margin[reward_best]),
                    "margin_best_min": float(min_margin[margin_best]),
                    "group0_first": _first_offroad_time(margins[0], times),
                    "reward_best_first": _first_offroad_time(margins[reward_best], times),
                })

            print(
                f"[curved] ckpt={checkpoint.name} sample={collected + 1:03d}/{args.samples} "
                + " ".join(
                    f"r{role}: feasible={state_rows[-3 + role].feasible_group_count}/{args.group_size} "
                    f"exp={state_rows[-3 + role].expert_min_margin_m:+.2f} "
                    f"G0={state_rows[-3 + role].group0_min_margin_m:+.2f} "
                    f"Rbest={state_rows[-3 + role].reward_best_min_margin_m:+.2f}"
                    for role in range(3)
                )
            )
            collected += 1
        finally:
            env.close()

    out = args.output_dir / checkpoint.name
    _write_rows(out / "curved_feasibility_state_summary.csv", [asdict(r) for r in state_rows])
    _write_rows(out / "curved_feasibility_candidate_summary.csv", candidate_rows)
    overall = _aggregate(checkpoint.name, state_rows)
    _write_rows(out / "curved_feasibility_overall_summary.csv", overall)
    _plot_galleries(args, checkpoint, plot_records)
    with (out / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args) | {"checkpoint": checkpoint.name, "checkpoint_path": str(checkpoint.path)}, fh, indent=2, default=str)

    print("=== CURVED FEASIBILITY SUMMARY ===")
    for row in overall:
        print(
            f"role={row['vehicle_role']} feasible_group_fraction={row['mean_feasible_group_fraction']:.3f} "
            f"has_feasible_state={row['state_has_feasible_candidate_fraction']:.3f} "
            f"all48_offroad={row['state_all_groups_offroad_fraction']:.3f} "
            f"expert_off={row['expert_offroad_fraction']:.3f} "
            f"G0_off={row['group0_offroad_fraction']:.3f} "
            f"Rbest_off={row['reward_best_offroad_fraction']:.3f} "
            f"margin_best={row['mean_margin_best_min_margin_m']:+.3f}m"
        )

    del sampler
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in args.checkpoint:
        _run_checkpoint(args, checkpoint)
    print("[OK] Curved feasibility diagnostic completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
