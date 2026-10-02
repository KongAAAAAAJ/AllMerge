
from __future__ import annotations

import argparse
import csv
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
    group0_min_margin_m: float
    best_min_margin_m: float
    group0_traj: np.ndarray
    best_traj: np.ndarray
    map_polylines: list[np.ndarray]
    target_lane: np.ndarray | None
    group0_margin_proxy_idx: int
    best_margin_proxy_idx: int
    vehicle_length_m: float
    vehicle_width_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reward Geometry Diagnostic: sample random states per scenario and plot "
            "road geometry (BEV-style map polylines), selected-mode candidate trajectories, "
            "vehicle footprints, and proxy minimum road-margin points in 3x3 galleries."
        )
    )
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS.keys()), default="all")
    parser.add_argument("--samples-per-scenario", type=int, default=25,
                        help="collect this many valid states per scenario before random selection")
    parser.add_argument("--states-per-gallery", type=int, default=9,
                        help="number of random states shown per scenario gallery")
    parser.add_argument("--group-size", type=int, default=48)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument(
        "--noise-type",
        choices=("additive", "multiplicative", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--multiplicative-std", type=float, default=0.04)
    parser.add_argument("--hybrid-multiplicative-x-std", type=float, default=0.12)
    parser.add_argument("--hybrid-additive-y-std-m", type=float, default=1.00)
    parser.add_argument("--group-action", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-seed", type=int, default=700000)
    parser.add_argument("--max-seed-attempts", type=int, default=1000)
    parser.add_argument("--reward-fn", default="auto")
    parser.add_argument("--vehicle-role", type=int, default=0, choices=(0, 1, 2),
                        help="controlled vehicle role to visualize")
    parser.add_argument("--visualization-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/reward_geometry_diagnostic"))
    return parser.parse_args()


def _build_sampler(args: argparse.Namespace, model: Any):
    if args.noise_type == "additive":
        return GroupDiffusionSampler(
            model,
            group_size=int(args.group_size),
            eta=float(args.eta),
        )
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


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_map_polylines(features: dict[str, Any]) -> tuple[list[np.ndarray], np.ndarray | None]:
    polys = _as_numpy(features.get("map_polylines"))
    mask = _as_numpy(features.get("map_polylines_mask"))
    target_lane = _as_numpy(features.get("target_lane_polyline"))

    if polys is None:
        return [], target_lane

    polys = np.squeeze(polys)
    if polys.ndim == 2:
        polys = polys[None, ...]
    if polys.ndim != 3 or polys.shape[-1] < 2:
        return [], target_lane

    if mask is not None:
        mask = np.squeeze(mask)
        if mask.ndim == 1:
            mask = np.broadcast_to(mask[:, None], polys.shape[:2])
        elif mask.ndim == 2:
            pass
        else:
            mask = None
    out: list[np.ndarray] = []
    for i in range(polys.shape[0]):
        pts = polys[i, :, :2]
        if mask is None:
            valid = np.isfinite(pts).all(axis=1)
        else:
            valid = mask[i].astype(bool) & np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if pts.shape[0] >= 2:
            out.append(pts.astype(np.float32))
    if target_lane is not None:
        target_lane = np.squeeze(target_lane)
        if target_lane.ndim == 2 and target_lane.shape[-1] >= 2:
            target_lane = target_lane[:, :2].astype(np.float32)
        else:
            target_lane = None
    return out, target_lane


def _vehicle_size(env: Any, vehicle_role: int) -> tuple[float, float]:
    vehicle = None
    if hasattr(env, "controlled_vehicles"):
        cvs = getattr(env, "controlled_vehicles")
        if isinstance(cvs, (list, tuple)) and len(cvs) > vehicle_role:
            vehicle = cvs[vehicle_role]
    if vehicle is None and hasattr(env, "vehicle"):
        vehicle = getattr(env, "vehicle")
    length = float(getattr(vehicle, "LENGTH", getattr(vehicle, "length", 5.0))) if vehicle is not None else 5.0
    width = float(getattr(vehicle, "WIDTH", getattr(vehicle, "width", 2.0))) if vehicle is not None else 2.0
    return length, width


def _trajectory_heading(traj: np.ndarray, idx: int) -> float:
    if traj.shape[0] == 1:
        return 0.0
    if idx <= 0:
        delta = traj[1] - traj[0]
    elif idx >= traj.shape[0] - 1:
        delta = traj[-1] - traj[-2]
    else:
        delta = traj[idx + 1] - traj[idx - 1]
    return float(np.arctan2(delta[1], delta[0] + 1e-9))


def _footprint_corners(center_xy: np.ndarray, heading: float, length: float, width: float) -> np.ndarray:
    c = np.cos(heading)
    s = np.sin(heading)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    half = np.asarray(
        [[ length / 2,  width / 2],
         [ length / 2, -width / 2],
         [-length / 2, -width / 2],
         [-length / 2,  width / 2],
         [ length / 2,  width / 2]],
        dtype=np.float32,
    )
    return center_xy[None, :] + half @ rot.T


def _proxy_min_margin_index(traj: np.ndarray, polylines: list[np.ndarray]) -> int:
    if traj.size == 0:
        return 0
    if not polylines:
        return int(traj.shape[0] - 1)
    cloud = np.concatenate(polylines, axis=0)
    d2 = np.sum((traj[:, None, :2] - cloud[None, :, :2]) ** 2, axis=2)
    # smaller road margin => trajectory gets closer to any road polyline / boundary proxy
    nearest_d2 = np.min(d2, axis=1)
    return int(np.argmin(nearest_d2))


def _collect_states_for_checkpoint(args: argparse.Namespace, checkpoint: CheckpointSpec) -> dict[str, list[DiagnosticState]]:
    device = torch.device(args.device)
    adapter, model = _build_model(checkpoint, device)
    sampler = _build_sampler(args, model)
    reward_adapter = CandidateRewardAdapter(resolve_reward_evaluator(args.reward_fn))

    output: dict[str, list[DiagnosticState]] = {name: [] for name in _scenario_order(args)}
    for scenario_index, scenario_name in enumerate(_scenario_order(args)):
        env_cls = SCENARIOS[scenario_name]
        collected = 0
        attempts = 0
        while collected < int(args.samples_per_scenario):
            if attempts >= int(args.max_seed_attempts):
                raise RuntimeError(
                    f"could not collect {args.samples_per_scenario} valid states for "
                    f"scenario={scenario_name} within {args.max_seed_attempts} attempts"
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
                group0_traj = trace.candidates[role, 0, mode].detach().cpu().numpy().astype(np.float32)
                best_traj = trace.candidates[role, best_group, mode].detach().cpu().numpy().astype(np.float32)
                polylines, target_lane = _extract_map_polylines(features)
                vehicle_length_m, vehicle_width_m = _vehicle_size(env, role)
                state = DiagnosticState(
                    checkpoint=checkpoint.name,
                    scenario=scenario_name,
                    sample_index=int(collected),
                    env_seed=int(env_seed),
                    noise_seed=int(noise_seed),
                    vehicle_role=role,
                    mode=mode,
                    group0_reward=float(rewards[0]),
                    best_reward=float(rewards[best_group]),
                    best_group_id=best_group,
                    group0_offroad=int(arrays["out_of_drivable"][role, 0, mode]),
                    best_offroad=int(arrays["out_of_drivable"][role, best_group, mode]),
                    group0_collision=int(arrays["collision"][role, 0, mode]),
                    best_collision=int(arrays["collision"][role, best_group, mode]),
                    group0_min_margin_m=float(arrays["minimum_road_margin_m"][role, 0, mode]),
                    best_min_margin_m=float(arrays["minimum_road_margin_m"][role, best_group, mode]),
                    group0_traj=group0_traj,
                    best_traj=best_traj,
                    map_polylines=polylines,
                    target_lane=target_lane,
                    group0_margin_proxy_idx=_proxy_min_margin_index(group0_traj, polylines),
                    best_margin_proxy_idx=_proxy_min_margin_index(best_traj, polylines),
                    vehicle_length_m=vehicle_length_m,
                    vehicle_width_m=vehicle_width_m,
                )
                output[scenario_name].append(state)
                print(
                    f"[diag] ckpt={checkpoint.name} scenario={scenario_name} sample={collected + 1:03d}/{args.samples_per_scenario} "
                    f"role={role} mode={mode} g0R={state.group0_reward:.3f} bestR={state.best_reward:.3f} "
                    f"g0_off={state.group0_offroad} best_off={state.best_offroad} "
                    f"g0_margin={state.group0_min_margin_m:.3f} best_margin={state.best_min_margin_m:.3f}"
                )
                collected += 1
            finally:
                env.close()
    del sampler
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _subplot_limits(states: list[DiagnosticState]) -> tuple[float, float, float, float]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for st in states:
        xs.append(st.group0_traj[:, 0])
        xs.append(st.best_traj[:, 0])
        ys.append(st.group0_traj[:, 1])
        ys.append(st.best_traj[:, 1])
        if st.target_lane is not None:
            xs.append(st.target_lane[:, 0]); ys.append(st.target_lane[:, 1])
        for poly in st.map_polylines:
            xs.append(poly[:, 0]); ys.append(poly[:, 1])
    x = np.concatenate(xs) if xs else np.asarray([0.0])
    y = np.concatenate(ys) if ys else np.asarray([0.0])
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xpad = max(5.0, 0.08 * max(1.0, xmax - xmin))
    ypad = max(3.0, 0.12 * max(1.0, ymax - ymin))
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def _draw_state(ax: Any, st: DiagnosticState, *, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_facecolor("white")
    for poly in st.map_polylines:
        ax.plot(poly[:, 0], poly[:, 1], linewidth=0.8, alpha=0.55)
    if st.target_lane is not None:
        ax.plot(st.target_lane[:, 0], st.target_lane[:, 1], linewidth=1.5, alpha=0.9)

    # Group 0 trajectory and proxy minimum-road-margin footprint.
    g0 = st.group0_traj
    ax.plot(g0[:, 0], g0[:, 1], linewidth=2.0, alpha=0.95)
    idx = st.group0_margin_proxy_idx
    ax.scatter([g0[idx, 0]], [g0[idx, 1]], marker="x", s=36)
    heading = _trajectory_heading(g0, idx)
    corners = _footprint_corners(g0[idx, :2], heading, st.vehicle_length_m, st.vehicle_width_m)
    ax.plot(corners[:, 0], corners[:, 1], linewidth=1.1, alpha=0.95)

    # Best-of-48 trajectory and proxy minimum-road-margin footprint.
    best = st.best_traj
    ax.plot(best[:, 0], best[:, 1], linewidth=2.0, alpha=0.95)
    idxb = st.best_margin_proxy_idx
    ax.scatter([best[idxb, 0]], [best[idxb, 1]], marker="x", s=36)
    headingb = _trajectory_heading(best, idxb)
    cornersb = _footprint_corners(best[idxb, :2], headingb, st.vehicle_length_m, st.vehicle_width_m)
    ax.plot(cornersb[:, 0], cornersb[:, 1], linewidth=1.1, alpha=0.95)

    # start footprint using group 0 pose for context
    start_heading = _trajectory_heading(g0, 0)
    start_box = _footprint_corners(g0[0, :2], start_heading, st.vehicle_length_m, st.vehicle_width_m)
    ax.plot(start_box[:, 0], start_box[:, 1], linewidth=0.9, alpha=0.7)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18)
    ax.set_title(
        f"id={st.sample_index:02d} seed={st.env_seed} role={st.vehicle_role} m={st.mode}\n"
        f"G0 R={st.group0_reward:.2f} off={st.group0_offroad} col={st.group0_collision} mr={st.group0_min_margin_m:.2f}\n"
        f"Best#{st.best_group_id} R={st.best_reward:.2f} off={st.best_offroad} col={st.best_collision} mr={st.best_min_margin_m:.2f}",
        fontsize=8,
    )
    ax.tick_params(labelsize=7)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_checkpoint(args: argparse.Namespace, checkpoint: CheckpointSpec, states_by_scenario: dict[str, list[DiagnosticState]]) -> None:
    out_dir = args.output_dir / checkpoint.name / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.visualization_seed))
    manifest_rows: list[dict[str, Any]] = []
    for scenario_name, states in states_by_scenario.items():
        if not states:
            continue
        count = min(int(args.states_per_gallery), len(states))
        select_ids = sorted(rng.choice(len(states), size=count, replace=False).tolist())
        chosen = [states[i] for i in select_ids]
        while len(chosen) < 9:
            chosen.append(chosen[-1])
        xlim = _subplot_limits(chosen)[:2]
        ylim = _subplot_limits(chosen)[2:]
        fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
        for ax, st in zip(axes.ravel(), chosen):
            _draw_state(ax, st, xlim=xlim, ylim=ylim)
            manifest_rows.append({
                "checkpoint": checkpoint.name,
                "scenario": scenario_name,
                "sample_index": st.sample_index,
                "env_seed": st.env_seed,
                "noise_seed": st.noise_seed,
                "vehicle_role": st.vehicle_role,
                "mode": st.mode,
                "group0_reward": st.group0_reward,
                "best_reward": st.best_reward,
                "group0_offroad": st.group0_offroad,
                "best_offroad": st.best_offroad,
                "group0_collision": st.group0_collision,
                "best_collision": st.best_collision,
                "group0_min_margin_m": st.group0_min_margin_m,
                "best_min_margin_m": st.best_min_margin_m,
            })
        fig.suptitle(
            f"Reward Geometry Diagnostic | {checkpoint.name} | {scenario_name} | "
            f"noise={args.noise_type} | role={args.vehicle_role}\n"
            "Background: drivable/road geometry proxy from valid map polylines; "
            "blue=Group0, red=Best-of-48, X=proxy min-road-margin point, boxes=vehicle footprint",
            fontsize=11,
        )
        out_path = out_dir / f"reward_geometry_diagnostic_{scenario_name}_3x3_{checkpoint.name}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"[write] {out_path}")
    _write_manifest(args.output_dir / checkpoint.name / "reward_geometry_diagnostic_manifest.csv", manifest_rows)


def main() -> int:
    args = parse_args()
    for checkpoint in args.checkpoint:
        states = _collect_states_for_checkpoint(args, checkpoint)
        _plot_checkpoint(args, checkpoint, states)
    print("[OK] Reward Geometry Diagnostic completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
