
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
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
    SCENARIOS,
    _build_model,
    _frozen_reward_context,
    _parse_checkpoint,
    _sample_noise_seed,
    _scenario_features,
    _scenario_order,
    _state_seed,
    _reward_result_arrays,
)
from highway_env.planner.diffusion.grpo.reward_adapter import (
    CandidateRewardAdapter,
    resolve_reward_evaluator,
)
from highway_env.planner.diffusion.grpo.sampling import GroupDiffusionSampler
from evaluation.grpo_probe_sampling import (
    HybridProbeGroupDiffusionSampler,
    MultiplicativeProbeGroupDiffusionSampler,
)
from highway_env.planner.diffusion.trajectory_mode_reward.config import (
    TrajectoryModeRewardConfig,
)
from highway_env.planner.diffusion.trajectory_mode_reward.counterfactual import (
    TrajectoryModeCounterfactualReward,
)
from highway_env.planner.diffusion.trajectory_mode_reward.geometry import (
    _dense_local_trajectories,
    _footprint_corners,
    _iter_lanes,
    _lane_width_at,
    local_to_world,
    road_margin_series,
    tracking_aware_dimensions,
)


@dataclass
class ExactRoadDiagnostic:
    local_traj_dense: np.ndarray
    world_traj_dense: np.ndarray
    margins: np.ndarray
    min_time_index: int
    first_negative_time_index: int
    min_corner_world: np.ndarray
    min_corner_local: np.ndarray
    footprint_world: np.ndarray
    footprint_local: np.ndarray
    best_lane_index: int
    best_lane_margin: float
    best_lane_s: float
    best_lane_length: float
    best_lane_lateral: float
    best_lane_width: float
    best_lane_lateral_margin: float
    best_lane_longitudinal_margin: float


@dataclass
class DiagnosticState:
    checkpoint: str
    scenario: str
    sample_index: int
    env_seed: int
    noise_seed: int
    vehicle_role: int
    mode: int
    group0_reward: float
    best_reward: float
    best_group_id: int
    group0_offroad: int
    best_offroad: int
    group0_collision: int
    best_collision: int
    group0_reported_min_margin_m: float
    best_reported_min_margin_m: float
    group0: ExactRoadDiagnostic
    best: ExactRoadDiagnostic
    lane_ribbons_local: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    pose_world: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact Reward Geometry Diagnostic for the current GRPO branch. Uses the same "
            "env.road lane geometry, local->world transform, tracking-aware footprint, "
            "and road_margin_series() as the production trajectory-mode reward."
        )
    )
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS.keys()), default="all")
    parser.add_argument("--samples-per-scenario", type=int, default=25)
    parser.add_argument("--states-per-gallery", type=int, default=9)
    parser.add_argument("--group-size", type=int, default=48)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--noise-type", choices=("additive", "multiplicative", "hybrid"), default="hybrid")
    parser.add_argument("--multiplicative-std", type=float, default=0.04)
    parser.add_argument("--hybrid-multiplicative-x-std", type=float, default=0.12)
    parser.add_argument("--hybrid-additive-y-std-m", type=float, default=1.00)
    parser.add_argument("--group-action", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-seed", type=int, default=700000)
    parser.add_argument("--max-seed-attempts", type=int, default=1000)
    parser.add_argument("--reward-fn", default="auto")
    parser.add_argument("--vehicle-role", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--visualization-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reward_geometry_diagnostic_v2"))
    return parser.parse_args()


def _build_sampler(args: argparse.Namespace, model: Any):
    if args.noise_type == "additive":
        return GroupDiffusionSampler(model, group_size=int(args.group_size), eta=float(args.eta))
    if args.noise_type == "multiplicative":
        return MultiplicativeProbeGroupDiffusionSampler(
            model,
            group_size=int(args.group_size),
            eta=float(args.eta),
            multiplicative_std=float(args.multiplicative_std),
        )
    return HybridProbeGroupDiffusionSampler(
        model,
        group_size=int(args.group_size),
        eta=float(args.eta),
        multiplicative_x_std=float(args.hybrid_multiplicative_x_std),
        additive_y_std_m=float(args.hybrid_additive_y_std_m),
    )


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _world_xy_to_local(points: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    pose = np.asarray(pose_world, dtype=np.float64).reshape(3)
    shifted = pts - pose[None, :2]
    c = math.cos(float(pose[2]))
    s = math.sin(float(pose[2]))
    # inverse rotation R(-heading)
    x = c * shifted[:, 0] + s * shifted[:, 1]
    y = -s * shifted[:, 0] + c * shifted[:, 1]
    return np.column_stack((x, y))


def _lane_diag_for_point(point_world: np.ndarray, lanes: list[object]) -> dict[str, float | int]:
    best = None
    for lane_index, lane in enumerate(lanes):
        try:
            s, lateral = lane.local_coordinates(point_world)
            s = float(s)
            lateral = float(lateral)
            length = float(lane.length)
            width = float(_lane_width_at(lane, float(np.clip(s, 0.0, length))))
            lateral_margin = 0.5 * width - abs(lateral)
            longitudinal_margin = min(s, length - s)
            margin = min(lateral_margin, longitudinal_margin)
            row = {
                "lane_index": int(lane_index),
                "margin": float(margin),
                "s": s,
                "length": length,
                "lateral": lateral,
                "width": width,
                "lateral_margin": float(lateral_margin),
                "longitudinal_margin": float(longitudinal_margin),
            }
            if best is None or float(row["margin"]) > float(best["margin"]):
                best = row
        except (AttributeError, TypeError, ValueError):
            continue
    if best is None:
        return {
            "lane_index": -1,
            "margin": -1.0e6,
            "s": float("nan"),
            "length": float("nan"),
            "lateral": float("nan"),
            "width": float("nan"),
            "lateral_margin": float("nan"),
            "longitudinal_margin": float("nan"),
        }
    return best


def _exact_road_diag(
    *,
    target_trajectory: np.ndarray,
    role: int,
    frozen_argmax: np.ndarray,
    geometry_context: Any,
    config: TrajectoryModeRewardConfig,
) -> ExactRoadDiagnostic:
    frozen = np.asarray(frozen_argmax, dtype=np.float64)
    joint = frozen.copy()
    joint[role] = np.asarray(target_trajectory, dtype=np.float64)
    dense_local, _ = _dense_local_trajectories(joint[None], config)
    local = dense_local[0, role]
    world = local_to_world(local, np.asarray(geometry_context.poses[role], dtype=np.float64))
    margins = road_margin_series(world, geometry_context.road, config, tracking_aware=True)
    min_idx = int(np.argmin(margins))
    neg = np.flatnonzero(margins[1:] < 0.0)
    first_neg_idx = int(neg[0] + 1) if neg.size else -1

    dims = tracking_aware_dimensions(config)
    footprints = _footprint_corners(world, dims)
    footprint_world = footprints[min_idx]
    lanes = list(_iter_lanes(geometry_context.road))
    corner_margins = []
    corner_diags = []
    for corner in footprint_world:
        diag = _lane_diag_for_point(corner, lanes)
        corner_margins.append(float(diag["margin"]))
        corner_diags.append(diag)
    worst_corner_idx = int(np.argmin(np.asarray(corner_margins)))
    worst_corner_world = footprint_world[worst_corner_idx]
    lane_diag = corner_diags[worst_corner_idx]
    pose = np.asarray(geometry_context.poses[role], dtype=np.float64)
    footprint_local = _world_xy_to_local(footprint_world, pose)
    corner_local = _world_xy_to_local(worst_corner_world[None, :], pose)[0]

    return ExactRoadDiagnostic(
        local_traj_dense=local[:, :2].astype(np.float32),
        world_traj_dense=world[:, :2].astype(np.float64),
        margins=margins.astype(np.float64),
        min_time_index=min_idx,
        first_negative_time_index=first_neg_idx,
        min_corner_world=worst_corner_world.astype(np.float64),
        min_corner_local=corner_local.astype(np.float64),
        footprint_world=footprint_world.astype(np.float64),
        footprint_local=footprint_local.astype(np.float64),
        best_lane_index=int(lane_diag["lane_index"]),
        best_lane_margin=float(lane_diag["margin"]),
        best_lane_s=float(lane_diag["s"]),
        best_lane_length=float(lane_diag["length"]),
        best_lane_lateral=float(lane_diag["lateral"]),
        best_lane_width=float(lane_diag["width"]),
        best_lane_lateral_margin=float(lane_diag["lateral_margin"]),
        best_lane_longitudinal_margin=float(lane_diag["longitudinal_margin"]),
    )


def _lane_ribbons_local(road: object, pose_world: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ribbons: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for lane in _iter_lanes(road):
        try:
            length = float(lane.length)
            if not np.isfinite(length) or length <= 0.0:
                continue
            sample_count = max(25, min(160, int(length / 1.5) + 2))
            ss = np.linspace(0.0, length, sample_count)
            center_world = []
            left_world = []
            right_world = []
            for s in ss:
                width = float(_lane_width_at(lane, float(s)))
                center_world.append(np.asarray(lane.position(float(s), 0.0), dtype=np.float64)[:2])
                left_world.append(np.asarray(lane.position(float(s), +0.5 * width), dtype=np.float64)[:2])
                right_world.append(np.asarray(lane.position(float(s), -0.5 * width), dtype=np.float64)[:2])
            center = _world_xy_to_local(np.asarray(center_world), pose_world)
            left = _world_xy_to_local(np.asarray(left_world), pose_world)
            right = _world_xy_to_local(np.asarray(right_world), pose_world)
            ribbons.append((center.astype(np.float32), left.astype(np.float32), right.astype(np.float32)))
        except (AttributeError, TypeError, ValueError):
            continue
    return ribbons


def _collect_states_for_checkpoint(args: argparse.Namespace, checkpoint: CheckpointSpec) -> dict[str, list[DiagnosticState]]:
    device = torch.device(args.device)
    adapter, model = _build_model(checkpoint, device)
    sampler = _build_sampler(args, model)
    reward_adapter = CandidateRewardAdapter(resolve_reward_evaluator(args.reward_fn))
    reward_config = TrajectoryModeRewardConfig(trajectories_per_mode=int(args.group_size))
    exact_scorer = TrajectoryModeCounterfactualReward(reward_config)

    output = {name: [] for name in _scenario_order(args)}
    for scenario_index, scenario_name in enumerate(_scenario_order(args)):
        env_cls = SCENARIOS[scenario_name]
        collected = 0
        attempts = 0
        while collected < int(args.samples_per_scenario):
            if attempts >= int(args.max_seed_attempts):
                raise RuntimeError(
                    f"could not collect {args.samples_per_scenario} valid states for scenario={scenario_name}"
                )
            env_seed = _state_seed(args.seed, scenario_index, attempts)
            attempts += 1
            env = env_cls(config={"show_trajectories": False, "show_future_trajectories": False}, render_mode=None)
            try:
                env.reset(seed=env_seed)
                _, _, terminated, truncated, _ = env.step(int(args.group_action))
                if bool(terminated) or bool(truncated):
                    continue
                features = _scenario_features(env, adapter)
                context, selected_mode_idx = _frozen_reward_context(model, features, env)
                noise_seed = _sample_noise_seed(args.noise_seed, scenario_index, collected)
                generator = torch.Generator(device=device.type).manual_seed(noise_seed)
                with torch.no_grad():
                    trace = sampler.sample(features, generator=generator)
                    reward_result = reward_adapter.evaluate_result(trace.candidates, features=features, context=context)
                    arrays = _reward_result_arrays(reward_result, group_size=int(args.group_size))

                role = int(args.vehicle_role)
                mode = int(selected_mode_idx[role].detach().cpu().item())
                rewards = arrays["reward"][role, :, mode]
                best_group = int(np.argmax(rewards))
                frozen_argmax = _to_numpy(context.frozen_argmax_joint_trajectories)
                geometry_context = exact_scorer.build_geometry_context(env, frozen_argmax)
                group0_traj = _to_numpy(trace.candidates[role, 0, mode]).astype(np.float64)
                best_traj = _to_numpy(trace.candidates[role, best_group, mode]).astype(np.float64)
                group0_diag = _exact_road_diag(
                    target_trajectory=group0_traj,
                    role=role,
                    frozen_argmax=frozen_argmax,
                    geometry_context=geometry_context,
                    config=reward_config,
                )
                best_diag = _exact_road_diag(
                    target_trajectory=best_traj,
                    role=role,
                    frozen_argmax=frozen_argmax,
                    geometry_context=geometry_context,
                    config=reward_config,
                )
                reported_g0 = float(arrays["minimum_road_margin_m"][role, 0, mode])
                reported_best = float(arrays["minimum_road_margin_m"][role, best_group, mode])
                if abs(float(np.min(group0_diag.margins)) - reported_g0) > 1e-4:
                    raise RuntimeError(
                        f"exact Group0 margin mismatch: recomputed={np.min(group0_diag.margins):.6f} reported={reported_g0:.6f}"
                    )
                if abs(float(np.min(best_diag.margins)) - reported_best) > 1e-4:
                    raise RuntimeError(
                        f"exact Best margin mismatch: recomputed={np.min(best_diag.margins):.6f} reported={reported_best:.6f}"
                    )
                pose_world = np.asarray(geometry_context.poses[role], dtype=np.float64)
                ribbons = _lane_ribbons_local(geometry_context.road, pose_world)
                state = DiagnosticState(
                    checkpoint=checkpoint.name,
                    scenario=scenario_name,
                    sample_index=collected,
                    env_seed=env_seed,
                    noise_seed=noise_seed,
                    vehicle_role=role,
                    mode=mode,
                    group0_reward=float(rewards[0]),
                    best_reward=float(rewards[best_group]),
                    best_group_id=best_group,
                    group0_offroad=int(arrays["out_of_drivable"][role, 0, mode]),
                    best_offroad=int(arrays["out_of_drivable"][role, best_group, mode]),
                    group0_collision=int(arrays["collision"][role, 0, mode]),
                    best_collision=int(arrays["collision"][role, best_group, mode]),
                    group0_reported_min_margin_m=reported_g0,
                    best_reported_min_margin_m=reported_best,
                    group0=group0_diag,
                    best=best_diag,
                    lane_ribbons_local=ribbons,
                    pose_world=pose_world,
                )
                output[scenario_name].append(state)
                print(
                    f"[diag-v2] ckpt={checkpoint.name} scenario={scenario_name} sample={collected+1:03d}/{args.samples_per_scenario} "
                    f"role={role} mode={mode} g0R={state.group0_reward:.3f} bestR={state.best_reward:.3f} "
                    f"g0_off={state.group0_offroad} g0_margin={reported_g0:.3f} "
                    f"g0_latM={group0_diag.best_lane_lateral_margin:.3f} g0_lonM={group0_diag.best_lane_longitudinal_margin:.3f} "
                    f"best_off={state.best_offroad} best_margin={reported_best:.3f}"
                )
                collected += 1
            finally:
                env.close()
    del sampler
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _state_limits(st: DiagnosticState) -> tuple[float, float, float, float]:
    arrays = [st.group0.local_traj_dense, st.best.local_traj_dense]
    for center, left, right in st.lane_ribbons_local:
        arrays.extend([center, left, right])
    pts = np.concatenate([a.reshape(-1, 2) for a in arrays if a.size], axis=0)
    # Keep plots focused near the planned horizon, not the entire road network.
    traj = np.concatenate((st.group0.local_traj_dense, st.best.local_traj_dense), axis=0)
    tx0, tx1 = float(np.min(traj[:, 0])), float(np.max(traj[:, 0]))
    ty0, ty1 = float(np.min(traj[:, 1])), float(np.max(traj[:, 1]))
    xmin = min(-10.0, tx0 - 8.0)
    xmax = tx1 + 12.0
    ymin = min(-12.0, ty0 - 8.0)
    ymax = max(12.0, ty1 + 8.0)
    return xmin, xmax, ymin, ymax


def _draw_exact(ax: Any, st: DiagnosticState) -> None:
    x0, x1, y0, y1 = _state_limits(st)
    for center, left, right in st.lane_ribbons_local:
        mask = (
            ((center[:, 0] >= x0 - 20) & (center[:, 0] <= x1 + 20))
            | ((left[:, 0] >= x0 - 20) & (left[:, 0] <= x1 + 20))
            | ((right[:, 0] >= x0 - 20) & (right[:, 0] <= x1 + 20))
        )
        if not bool(np.any(mask)):
            continue
        ax.plot(center[:, 0], center[:, 1], linewidth=0.7, alpha=0.35)
        ax.plot(left[:, 0], left[:, 1], linewidth=0.9, alpha=0.60)
        ax.plot(right[:, 0], right[:, 1], linewidth=0.9, alpha=0.60)
        try:
            ax.fill(
                np.concatenate((left[:, 0], right[::-1, 0])),
                np.concatenate((left[:, 1], right[::-1, 1])),
                alpha=0.06,
            )
        except Exception:
            pass

    # Group0 and Best dense trajectories (same interpolated geometry used by reward).
    ax.plot(st.group0.local_traj_dense[:, 0], st.group0.local_traj_dense[:, 1], linewidth=2.0, label="Group0")
    ax.plot(st.best.local_traj_dense[:, 0], st.best.local_traj_dense[:, 1], linewidth=2.0, label="Best")

    # Exact minimum-margin footprint and exact worst corner.
    gfp = np.vstack((st.group0.footprint_local, st.group0.footprint_local[0]))
    bfp = np.vstack((st.best.footprint_local, st.best.footprint_local[0]))
    ax.plot(gfp[:, 0], gfp[:, 1], linewidth=1.4)
    ax.plot(bfp[:, 0], bfp[:, 1], linewidth=1.4)
    ax.scatter([st.group0.min_corner_local[0]], [st.group0.min_corner_local[1]], marker="x", s=45)
    ax.scatter([st.best.min_corner_local[0]], [st.best.min_corner_local[1]], marker="x", s=45)

    # Mark initial ego frame.
    ax.scatter([0.0], [0.0], marker="o", s=18)
    ax.axhline(0.0, linewidth=0.5, alpha=0.2)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.16)
    ax.set_title(
        f"id={st.sample_index:02d} seed={st.env_seed} role={st.vehicle_role} m={st.mode}\n"
        f"G0 R={st.group0_reward:.2f} off={st.group0_offroad} min={st.group0_reported_min_margin_m:.2f} "
        f"latM={st.group0.best_lane_lateral_margin:.2f} lonM={st.group0.best_lane_longitudinal_margin:.2f}\n"
        f"Best#{st.best_group_id} R={st.best_reward:.2f} off={st.best_offroad} min={st.best_reported_min_margin_m:.2f}",
        fontsize=7.5,
    )
    ax.tick_params(labelsize=7)


def _diag_row(st: DiagnosticState, label: str, diag: ExactRoadDiagnostic, reward: float, offroad: int, collision: int) -> dict[str, Any]:
    return {
        "checkpoint": st.checkpoint,
        "scenario": st.scenario,
        "sample_index": st.sample_index,
        "env_seed": st.env_seed,
        "noise_seed": st.noise_seed,
        "vehicle_role": st.vehicle_role,
        "mode": st.mode,
        "candidate": label,
        "reward": reward,
        "offroad": offroad,
        "collision": collision,
        "reported_minimum_road_margin_m": st.group0_reported_min_margin_m if label == "group0" else st.best_reported_min_margin_m,
        "recomputed_minimum_road_margin_m": float(np.min(diag.margins)),
        "min_time_index": diag.min_time_index,
        "min_time_s": float(diag.min_time_index * 0.1),
        "first_negative_time_index": diag.first_negative_time_index,
        "first_negative_time_s": float(diag.first_negative_time_index * 0.1) if diag.first_negative_time_index >= 0 else float("nan"),
        "worst_corner_local_x": float(diag.min_corner_local[0]),
        "worst_corner_local_y": float(diag.min_corner_local[1]),
        "best_lane_index": diag.best_lane_index,
        "best_lane_margin_m": diag.best_lane_margin,
        "best_lane_s_m": diag.best_lane_s,
        "best_lane_length_m": diag.best_lane_length,
        "best_lane_lateral_m": diag.best_lane_lateral,
        "best_lane_width_m": diag.best_lane_width,
        "best_lane_lateral_margin_m": diag.best_lane_lateral_margin,
        "best_lane_longitudinal_margin_m": diag.best_lane_longitudinal_margin,
        "negative_due_to_lateral": int(diag.best_lane_lateral_margin < 0.0),
        "negative_due_to_longitudinal_endpoint": int(diag.best_lane_longitudinal_margin < 0.0),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def _plot_checkpoint(args: argparse.Namespace, checkpoint: CheckpointSpec, states_by_scenario: dict[str, list[DiagnosticState]]) -> None:
    root = args.output_dir / checkpoint.name
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.visualization_seed))
    rows: list[dict[str, Any]] = []
    for scenario_name, states in states_by_scenario.items():
        if not states:
            continue
        count = min(int(args.states_per_gallery), len(states))
        ids = sorted(rng.choice(len(states), size=count, replace=False).tolist())
        chosen = [states[i] for i in ids]
        fig, axes = plt.subplots(3, 3, figsize=(15, 11.5), constrained_layout=True)
        for idx, ax in enumerate(axes.ravel()):
            if idx < len(chosen):
                st = chosen[idx]
                _draw_exact(ax, st)
                rows.append(_diag_row(st, "group0", st.group0, st.group0_reward, st.group0_offroad, st.group0_collision))
                rows.append(_diag_row(st, "best", st.best, st.best_reward, st.best_offroad, st.best_collision))
            else:
                ax.axis("off")
        fig.suptitle(
            f"Exact Reward Geometry Diagnostic | {checkpoint.name} | {scenario_name}\n"
            "Lane ribbons come from env.road.network.graph used by production reward; trajectories/footprints use the exact reward transform. "
            "X = exact worst footprint corner at minimum road margin.",
            fontsize=10.5,
        )
        out = plots / f"reward_geometry_exact_{scenario_name}_3x3_{checkpoint.name}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"[write] {out}")
    _write_csv(root / "reward_geometry_exact_manifest.csv", rows)
    print(f"[write] {root / 'reward_geometry_exact_manifest.csv'}")


def main() -> int:
    args = parse_args()
    for checkpoint in args.checkpoint:
        states = _collect_states_for_checkpoint(args, checkpoint)
        _plot_checkpoint(args, checkpoint, states)
    print("[OK] Exact Reward Geometry Diagnostic completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
