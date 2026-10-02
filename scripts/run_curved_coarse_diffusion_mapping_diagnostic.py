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
    _sample_noise_seed,
    _scenario_features,
    _state_seed,
)
from scripts.run_curved_feasibility_diagnostic import (
    _as_numpy,
    _build_sampler,
    _dense_profiles,
    _expert_profile,
    _first_offroad_time,
    _lane_lines_local,
    _offroad,
    _planning_poses,
    _expert_sparse,
)
from highway_env.envs.scenarios.curved_lane_change_env import CurvedLaneChangeEnv
from highway_env.planner.diffusion.trajectory_mode_reward.config import TrajectoryModeRewardConfig
from highway_env.planner.mode_definitions import MODE_NAMES


@dataclass(frozen=True)
class MappingStateSummary:
    checkpoint: str
    sample_index: int
    env_seed: int
    noise_seed: int
    vehicle_role: int
    selected_mode: int
    selected_mode_name: str

    expert_min_margin_m: float
    expert_offroad: int

    coarse_ade_m: float
    coarse_fde_m: float
    coarse_endpoint_dx_m: float
    coarse_endpoint_dy_m: float
    coarse_min_margin_m: float
    coarse_offroad: int
    coarse_first_offroad_time_s: float

    zero_diffusion_ade_m: float
    zero_diffusion_fde_m: float
    zero_diffusion_endpoint_dx_m: float
    zero_diffusion_endpoint_dy_m: float
    zero_diffusion_min_margin_m: float
    zero_diffusion_offroad: int
    zero_diffusion_first_offroad_time_s: float

    group0_ade_m: float
    group0_fde_m: float
    group0_endpoint_dx_m: float
    group0_endpoint_dy_m: float
    group0_min_margin_m: float
    group0_offroad: int
    group0_first_offroad_time_s: float

    margin_best_group_id: int
    margin_best_ade_m: float
    margin_best_fde_m: float
    margin_best_endpoint_dx_m: float
    margin_best_endpoint_dy_m: float
    margin_best_min_margin_m: float
    margin_best_offroad: int
    margin_best_first_offroad_time_s: float

    coarse_to_zero_ade_m: float
    coarse_to_zero_fde_m: float
    zero_to_group0_ade_m: float
    zero_to_group0_fde_m: float
    coarse_margin_to_zero_delta_m: float
    coarse_onroad_but_zero_offroad: int
    coarse_onroad_but_group0_offroad: int
    zero_worse_margin_than_coarse: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Curved Coarse->Diffusion Mapping Diagnostic. For a fixed production-selected mode, "
            "compare Expert, selected Coarse anchor, zero-noise diffusion output, Group0, and "
            "road-margin-best of G stochastic candidates. This separates anchor errors from "
            "diffusion mapping/refinement errors."
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
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/curved_coarse_diffusion_mapping_diagnostic"),
    )
    return p.parse_args()


def _trajectory_error(pred: np.ndarray, expert: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    delta = pred - expert
    dist = np.linalg.norm(delta, axis=-1)
    return {
        "ade": float(np.mean(dist)),
        "fde": float(dist[-1]),
        "dx": float(delta[-1, 0]),
        "dy": float(delta[-1, 1]),
    }


def _pair_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    dist = np.linalg.norm(delta, axis=-1)
    return float(np.mean(dist)), float(dist[-1])


def _min_margin(profile: np.ndarray) -> float:
    return float(np.min(np.asarray(profile, dtype=np.float64)))


def _nearest_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    line = np.asarray(polyline, dtype=np.float64)
    if line.ndim != 2 or line.shape[0] == 0 or line.shape[1] < 2:
        return np.full((p.shape[0],), np.nan, dtype=np.float64)
    d2 = np.sum((p[:, None, :2] - line[None, :, :2]) ** 2, axis=-1)
    return np.sqrt(np.min(d2, axis=1))


def _sparse_heading_curvature(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    extended = np.concatenate([np.zeros((1, 2), dtype=np.float64), pts[:, :2]], axis=0)
    seg = np.diff(extended, axis=0)
    heading = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0] + 1e-12))
    ds = np.linalg.norm(seg, axis=1)
    curvature = np.zeros_like(heading)
    if len(heading) > 1:
        dpsi = np.diff(heading)
        denom = np.maximum(0.5 * (ds[1:] + ds[:-1]), 1e-3)
        curvature[1:] = dpsi / denom
        curvature[0] = curvature[1]
    return heading, curvature


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(checkpoint: str, rows: list[MappingStateSummary]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for role in range(3):
        subset = [r for r in rows if r.vehicle_role == role]
        if not subset:
            continue
        def arr(name: str) -> np.ndarray:
            return np.asarray([getattr(r, name) for r in subset], dtype=np.float64)
        modes: dict[str, int] = {}
        for r in subset:
            modes[r.selected_mode_name] = modes.get(r.selected_mode_name, 0) + 1
        out.append({
            "checkpoint": checkpoint,
            "vehicle_role": role,
            "state_count": len(subset),
            "selected_mode_counts": json.dumps(modes, sort_keys=True),
            "expert_offroad_fraction": float(np.mean(arr("expert_offroad"))),
            "coarse_offroad_fraction": float(np.mean(arr("coarse_offroad"))),
            "zero_diffusion_offroad_fraction": float(np.mean(arr("zero_diffusion_offroad"))),
            "group0_offroad_fraction": float(np.mean(arr("group0_offroad"))),
            "margin_best_offroad_fraction": float(np.mean(arr("margin_best_offroad"))),
            "mean_expert_min_margin_m": float(np.mean(arr("expert_min_margin_m"))),
            "mean_coarse_min_margin_m": float(np.mean(arr("coarse_min_margin_m"))),
            "mean_zero_diffusion_min_margin_m": float(np.mean(arr("zero_diffusion_min_margin_m"))),
            "mean_group0_min_margin_m": float(np.mean(arr("group0_min_margin_m"))),
            "mean_margin_best_min_margin_m": float(np.mean(arr("margin_best_min_margin_m"))),
            "mean_coarse_ade_m": float(np.mean(arr("coarse_ade_m"))),
            "mean_zero_diffusion_ade_m": float(np.mean(arr("zero_diffusion_ade_m"))),
            "mean_group0_ade_m": float(np.mean(arr("group0_ade_m"))),
            "mean_margin_best_ade_m": float(np.mean(arr("margin_best_ade_m"))),
            "mean_coarse_fde_m": float(np.mean(arr("coarse_fde_m"))),
            "mean_zero_diffusion_fde_m": float(np.mean(arr("zero_diffusion_fde_m"))),
            "mean_group0_fde_m": float(np.mean(arr("group0_fde_m"))),
            "mean_margin_best_fde_m": float(np.mean(arr("margin_best_fde_m"))),
            "mean_coarse_endpoint_dy_m": float(np.mean(arr("coarse_endpoint_dy_m"))),
            "mean_zero_diffusion_endpoint_dy_m": float(np.mean(arr("zero_diffusion_endpoint_dy_m"))),
            "mean_group0_endpoint_dy_m": float(np.mean(arr("group0_endpoint_dy_m"))),
            "mean_margin_best_endpoint_dy_m": float(np.mean(arr("margin_best_endpoint_dy_m"))),
            "mean_coarse_to_zero_ade_m": float(np.mean(arr("coarse_to_zero_ade_m"))),
            "mean_coarse_to_zero_fde_m": float(np.mean(arr("coarse_to_zero_fde_m"))),
            "coarse_onroad_but_zero_diffusion_offroad_fraction": float(np.mean(arr("coarse_onroad_but_zero_offroad"))),
            "coarse_onroad_but_group0_offroad_fraction": float(np.mean(arr("coarse_onroad_but_group0_offroad"))),
            "zero_worse_margin_than_coarse_fraction": float(np.mean(arr("zero_worse_margin_than_coarse"))),
        })
    return out


def _plot_galleries(
    args: argparse.Namespace,
    checkpoint: CheckpointSpec,
    records: dict[int, list[dict[str, Any]]],
) -> None:
    out = args.output_dir / checkpoint.name / "plots"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.visualization_seed))

    for role in range(3):
        pool = records[role]
        if not pool:
            continue
        count = min(int(args.states_per_gallery), len(pool), 9)
        ids = sorted(rng.choice(len(pool), size=count, replace=False).tolist())
        chosen = [pool[i] for i in ids]

        # Geometry: Expert -> Coarse -> zero-noise diffusion -> Group0 -> MarginBest.
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
        for ax_idx, ax in enumerate(axes.ravel()):
            if ax_idx >= len(chosen):
                ax.axis("off")
                continue
            rec = chosen[ax_idx]
            for line in rec["lane_lines"]:
                ax.plot(line[:, 0], line[:, 1], linewidth=0.8, alpha=0.4)
            target = rec["target_lane"]
            ax.plot(target[:, 0], target[:, 1], linewidth=1.4, linestyle="--", label="Target lane")
            ax.plot(rec["expert"][:, 0], rec["expert"][:, 1], linewidth=2.2, label="Expert")
            ax.plot(rec["coarse"][:, 0], rec["coarse"][:, 1], linewidth=2.0, label="Coarse")
            ax.plot(rec["zero"][:, 0], rec["zero"][:, 1], linewidth=2.0, label="Zero-noise diff")
            ax.plot(rec["group0"][:, 0], rec["group0"][:, 1], linewidth=1.7, label="Group0")
            ax.plot(rec["margin_best"][:, 0], rec["margin_best"][:, 1], linewidth=2.0, label="MarginBest")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.18)
            ax.set_title(
                f"id={rec['sample_index']:02d} {rec['mode_name']}\n"
                f"margin E/C/Z/G/M={rec['expert_min']:+.2f}/{rec['coarse_min']:+.2f}/"
                f"{rec['zero_min']:+.2f}/{rec['group0_min']:+.2f}/{rec['margin_best_min']:+.2f}m",
                fontsize=8,
            )
            if ax_idx == 0:
                ax.legend(fontsize=7, loc="best")
        fig.suptitle(
            f"Curved Coarse→Diffusion Mapping | {checkpoint.name} | role{role}\n"
            "Same selected mode: Expert → Coarse anchor → Zero-noise diffusion → Group0 → MarginBest",
            fontsize=11,
        )
        fig.savefig(out / f"curved_mapping_geometry_role{role}_3x3_{checkpoint.name}.png", dpi=180)
        plt.close(fig)

        # Sparse lateral error profile vs expert.
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
        t = np.arange(1, 9, dtype=np.float64) * 0.5
        for ax_idx, ax in enumerate(axes.ravel()):
            if ax_idx >= len(chosen):
                ax.axis("off")
                continue
            rec = chosen[ax_idx]
            exp = rec["expert"]
            ax.axhline(0.0, linewidth=0.8, alpha=0.5)
            for key, label in (("coarse", "Coarse"), ("zero", "Zero-noise diff"), ("group0", "Group0"), ("margin_best", "MarginBest")):
                ax.plot(t, rec[key][:, 1] - exp[:, 1], marker="o", markersize=2.5, linewidth=1.4, label=label)
            ax.set_xlabel("time [s]", fontsize=7)
            ax.set_ylabel("Δy vs expert [m]", fontsize=7)
            ax.grid(True, alpha=0.2)
            ax.set_title(f"id={rec['sample_index']:02d} {rec['mode_name']}", fontsize=8)
            ax.tick_params(labelsize=7)
            if ax_idx == 0:
                ax.legend(fontsize=7)
        fig.suptitle(
            f"Curved lateral mapping error | {checkpoint.name} | role{role}\n"
            "If Coarse≈0 but diffusion moves away, the mapping/refinement stage is the failure source.",
            fontsize=11,
        )
        fig.savefig(out / f"curved_mapping_lateral_error_role{role}_3x3_{checkpoint.name}.png", dpi=180)
        plt.close(fig)

        # Exact production road margin profile.
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
        for ax_idx, ax in enumerate(axes.ravel()):
            if ax_idx >= len(chosen):
                ax.axis("off")
                continue
            rec = chosen[ax_idx]
            tt = rec["margin_times"]
            ax.axhline(0.0, linewidth=0.9, alpha=0.6)
            for key, label in (("expert_margin", "Expert"), ("coarse_margin", "Coarse"), ("zero_margin", "Zero-noise diff"), ("group0_margin", "Group0"), ("margin_best_margin", "MarginBest")):
                ax.plot(tt, rec[key], linewidth=1.4, label=label)
            ax.set_xlabel("time [s]", fontsize=7)
            ax.set_ylabel("signed road margin [m]", fontsize=7)
            ax.grid(True, alpha=0.2)
            ax.set_title(f"id={rec['sample_index']:02d} {rec['mode_name']}", fontsize=8)
            ax.tick_params(labelsize=7)
            if ax_idx == 0:
                ax.legend(fontsize=7)
        fig.suptitle(
            f"Curved road-margin mapping profile | {checkpoint.name} | role{role}",
            fontsize=11,
        )
        fig.savefig(out / f"curved_mapping_road_margin_role{role}_3x3_{checkpoint.name}.png", dpi=180)
        plt.close(fig)


def _run_checkpoint(args: argparse.Namespace, checkpoint: CheckpointSpec) -> None:
    device = torch.device(args.device)
    adapter, model = _build_model(checkpoint, device)
    sampler = _build_sampler(args, model)
    reward_config = TrajectoryModeRewardConfig()

    state_rows: list[MappingStateSummary] = []
    step_rows: list[dict[str, Any]] = []
    plot_records: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}

    collected = 0
    attempts = 0
    while collected < int(args.samples):
        if attempts >= int(args.max_seed_attempts):
            raise RuntimeError(f"could not collect {args.samples} valid curved states")
        env_seed = _state_seed(args.seed, 0, attempts)
        attempts += 1
        env = CurvedLaneChangeEnv(
            config={"show_trajectories": False, "show_future_trajectories": False},
            render_mode=None,
        )
        try:
            env.reset(seed=env_seed)
            _, _, terminated, truncated, _ = env.step(int(args.group_action))
            if bool(terminated) or bool(truncated):
                continue

            features = _scenario_features(env, adapter)
            context, selected_mode_idx = _frozen_reward_context(model, features, env)
            poses = _planning_poses(env)
            expert = _expert_sparse(env)
            frozen_argmax = _as_numpy(context.frozen_argmax_joint_trajectories).astype(np.float64)
            coarse_all = _as_numpy(features["coarse_trajectories"]).astype(np.float64)
            target_lanes = _as_numpy(features["target_lane_polyline"])[..., :2].astype(np.float64)

            # Zero stochastic input: same reverse-diffusion mapping, but base_noise=0.
            with torch.no_grad():
                zero_base = model._runtime_base_sample(
                    features,
                    base_noise=torch.zeros_like(features["coarse_trajectories"]),
                )
            zero_candidates = _as_numpy(zero_base["trajectory_candidates"]).astype(np.float64)

            noise_seed = _sample_noise_seed(args.noise_seed, 0, collected)
            generator = torch.Generator(device=device.type).manual_seed(noise_seed)
            with torch.no_grad():
                trace = sampler.sample(features, generator=generator)
            stochastic_candidates = _as_numpy(trace.candidates).astype(np.float64)

            for role in range(3):
                mode = int(selected_mode_idx[role].detach().cpu().item())
                mode_name = MODE_NAMES[mode] if 0 <= mode < len(MODE_NAMES) else str(mode)
                exp = expert[role]
                coarse = coarse_all[role, mode]
                zero = zero_candidates[role, mode]
                groups = stochastic_candidates[role, :, mode]

                # Exact production road-margin profiles for all stochastic candidates.
                _, group_margins, margin_times = _dense_profiles(
                    groups,
                    role=role,
                    frozen_argmax=frozen_argmax,
                    pose=poses[role],
                    road=env.road,
                    config=reward_config,
                )
                group_min = np.min(group_margins, axis=1)
                margin_best_gid = int(np.argmax(group_min))
                group0 = groups[0]
                margin_best = groups[margin_best_gid]

                _, exp_margin, exp_times = _expert_profile(
                    expert,
                    role=role,
                    pose=poses[role],
                    road=env.road,
                    config=reward_config,
                )
                if not np.allclose(margin_times, exp_times):
                    raise RuntimeError("expert/candidate road-margin grids differ")

                def one_profile(traj: np.ndarray) -> np.ndarray:
                    _, margins, tt = _dense_profiles(
                        np.asarray(traj, dtype=np.float64)[None],
                        role=role,
                        frozen_argmax=frozen_argmax,
                        pose=poses[role],
                        road=env.road,
                        config=reward_config,
                    )
                    if not np.allclose(tt, margin_times):
                        raise RuntimeError("mapping road-margin time grids differ")
                    return margins[0]

                coarse_margin = one_profile(coarse)
                zero_margin = one_profile(zero)
                group0_margin = group_margins[0]
                margin_best_margin = group_margins[margin_best_gid]

                ce = _trajectory_error(coarse, exp)
                ze = _trajectory_error(zero, exp)
                ge = _trajectory_error(group0, exp)
                me = _trajectory_error(margin_best, exp)
                c2z_ade, c2z_fde = _pair_error(zero, coarse)
                z2g_ade, z2g_fde = _pair_error(group0, zero)

                coarse_off = int(_offroad(coarse_margin))
                zero_off = int(_offroad(zero_margin))
                group0_off = int(_offroad(group0_margin))
                mb_off = int(_offroad(margin_best_margin))
                cmin = _min_margin(coarse_margin)
                zmin = _min_margin(zero_margin)

                summary = MappingStateSummary(
                    checkpoint=checkpoint.name,
                    sample_index=collected,
                    env_seed=env_seed,
                    noise_seed=noise_seed,
                    vehicle_role=role,
                    selected_mode=mode,
                    selected_mode_name=mode_name,
                    expert_min_margin_m=_min_margin(exp_margin),
                    expert_offroad=int(_offroad(exp_margin)),
                    coarse_ade_m=ce["ade"], coarse_fde_m=ce["fde"],
                    coarse_endpoint_dx_m=ce["dx"], coarse_endpoint_dy_m=ce["dy"],
                    coarse_min_margin_m=cmin, coarse_offroad=coarse_off,
                    coarse_first_offroad_time_s=_first_offroad_time(coarse_margin, margin_times),
                    zero_diffusion_ade_m=ze["ade"], zero_diffusion_fde_m=ze["fde"],
                    zero_diffusion_endpoint_dx_m=ze["dx"], zero_diffusion_endpoint_dy_m=ze["dy"],
                    zero_diffusion_min_margin_m=zmin, zero_diffusion_offroad=zero_off,
                    zero_diffusion_first_offroad_time_s=_first_offroad_time(zero_margin, margin_times),
                    group0_ade_m=ge["ade"], group0_fde_m=ge["fde"],
                    group0_endpoint_dx_m=ge["dx"], group0_endpoint_dy_m=ge["dy"],
                    group0_min_margin_m=_min_margin(group0_margin), group0_offroad=group0_off,
                    group0_first_offroad_time_s=_first_offroad_time(group0_margin, margin_times),
                    margin_best_group_id=margin_best_gid,
                    margin_best_ade_m=me["ade"], margin_best_fde_m=me["fde"],
                    margin_best_endpoint_dx_m=me["dx"], margin_best_endpoint_dy_m=me["dy"],
                    margin_best_min_margin_m=_min_margin(margin_best_margin), margin_best_offroad=mb_off,
                    margin_best_first_offroad_time_s=_first_offroad_time(margin_best_margin, margin_times),
                    coarse_to_zero_ade_m=c2z_ade, coarse_to_zero_fde_m=c2z_fde,
                    zero_to_group0_ade_m=z2g_ade, zero_to_group0_fde_m=z2g_fde,
                    coarse_margin_to_zero_delta_m=float(zmin - cmin),
                    coarse_onroad_but_zero_offroad=int((not coarse_off) and bool(zero_off)),
                    coarse_onroad_but_group0_offroad=int((not coarse_off) and bool(group0_off)),
                    zero_worse_margin_than_coarse=int(zmin < cmin - 1e-6),
                )
                state_rows.append(summary)

                trajectories = {
                    "expert": exp,
                    "coarse": coarse,
                    "zero_diffusion": zero,
                    "group0": group0,
                    "margin_best": margin_best,
                }
                kinematics = {name: _sparse_heading_curvature(traj) for name, traj in trajectories.items()}
                target_lane = target_lanes[role]
                target_distances = {
                    name: _nearest_polyline_distance(traj, target_lane)
                    for name, traj in trajectories.items()
                }
                for step in range(exp.shape[0]):
                    row: dict[str, Any] = {
                        "checkpoint": checkpoint.name,
                        "sample_index": collected,
                        "env_seed": env_seed,
                        "vehicle_role": role,
                        "selected_mode": mode,
                        "selected_mode_name": mode_name,
                        "step_index": step,
                        "time_s": float((step + 1) * 0.5),
                    }
                    for name, traj in trajectories.items():
                        heading, curvature = kinematics[name]
                        row[f"{name}_x_m"] = float(traj[step, 0])
                        row[f"{name}_y_m"] = float(traj[step, 1])
                        row[f"{name}_heading_rad"] = float(heading[step])
                        row[f"{name}_curvature_1pm"] = float(curvature[step])
                        row[f"{name}_target_lane_distance_m"] = float(target_distances[name][step])
                        if name != "expert":
                            row[f"{name}_dx_vs_expert_m"] = float(traj[step, 0] - exp[step, 0])
                            row[f"{name}_dy_vs_expert_m"] = float(traj[step, 1] - exp[step, 1])
                            row[f"{name}_heading_error_vs_expert_rad"] = float(
                                np.arctan2(
                                    np.sin(heading[step] - kinematics["expert"][0][step]),
                                    np.cos(heading[step] - kinematics["expert"][0][step]),
                                )
                            )
                            row[f"{name}_curvature_error_vs_expert_1pm"] = float(
                                curvature[step] - kinematics["expert"][1][step]
                            )
                    step_rows.append(row)

                plot_records[role].append({
                    "sample_index": collected,
                    "mode_name": mode_name,
                    "lane_lines": _lane_lines_local(env, poses[role]),
                    "target_lane": target_lane,
                    "expert": exp,
                    "coarse": coarse,
                    "zero": zero,
                    "group0": group0,
                    "margin_best": margin_best,
                    "margin_times": margin_times,
                    "expert_margin": exp_margin,
                    "coarse_margin": coarse_margin,
                    "zero_margin": zero_margin,
                    "group0_margin": group0_margin,
                    "margin_best_margin": margin_best_margin,
                    "expert_min": _min_margin(exp_margin),
                    "coarse_min": cmin,
                    "zero_min": zmin,
                    "group0_min": _min_margin(group0_margin),
                    "margin_best_min": _min_margin(margin_best_margin),
                })

            latest = state_rows[-3:]
            print(
                f"[mapping] ckpt={checkpoint.name} sample={collected + 1:03d}/{args.samples} "
                + " ".join(
                    f"r{r.vehicle_role}:{r.selected_mode_name} "
                    f"off C/Z/G/M={r.coarse_offroad}/{r.zero_diffusion_offroad}/{r.group0_offroad}/{r.margin_best_offroad} "
                    f"margin={r.coarse_min_margin_m:+.2f}/{r.zero_diffusion_min_margin_m:+.2f}/"
                    f"{r.group0_min_margin_m:+.2f}/{r.margin_best_min_margin_m:+.2f}"
                    for r in latest
                )
            )
            collected += 1
        finally:
            env.close()

    out = args.output_dir / checkpoint.name
    _write_rows(out / "curved_mapping_state_summary.csv", [asdict(r) for r in state_rows])
    _write_rows(out / "curved_mapping_step_profile.csv", step_rows)
    overall = _aggregate(checkpoint.name, state_rows)
    _write_rows(out / "curved_mapping_overall_summary.csv", overall)
    _plot_galleries(args, checkpoint, plot_records)

    with (out / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            vars(args) | {"checkpoint": checkpoint.name, "checkpoint_path": str(checkpoint.path)},
            fh,
            indent=2,
            default=str,
        )

    print("=== CURVED COARSE->DIFFUSION MAPPING SUMMARY ===")
    for row in overall:
        print(
            f"role={row['vehicle_role']} modes={row['selected_mode_counts']} "
            f"off E/C/Z/G/M={row['expert_offroad_fraction']:.3f}/"
            f"{row['coarse_offroad_fraction']:.3f}/{row['zero_diffusion_offroad_fraction']:.3f}/"
            f"{row['group0_offroad_fraction']:.3f}/{row['margin_best_offroad_fraction']:.3f} "
            f"margin E/C/Z/G/M={row['mean_expert_min_margin_m']:+.3f}/"
            f"{row['mean_coarse_min_margin_m']:+.3f}/{row['mean_zero_diffusion_min_margin_m']:+.3f}/"
            f"{row['mean_group0_min_margin_m']:+.3f}/{row['mean_margin_best_min_margin_m']:+.3f} "
            f"coarse->zero_ADE={row['mean_coarse_to_zero_ade_m']:.3f} "
            f"C_on->Z_off={row['coarse_onroad_but_zero_diffusion_offroad_fraction']:.3f}"
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
    print("[OK] Curved Coarse->Diffusion Mapping Diagnostic completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
