from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.grpo_noise_probe import (
    DEFAULT_IMPROVEMENT_EPS,
    ProbeGroupSummary,
    ProbeSampleSummary,
    ProbeVehicleModeSummary,
    aggregate_sample_summaries,
    analyze_group_rewards,
    maybe_plot_summary,
    write_dataclass_csv,
    write_dict_csv,
    write_summary_json,
)
from highway_env.envs.scenarios.curved_lane_change_env import CurvedLaneChangeEnv
from highway_env.envs.scenarios.merge_in_env import MergeInEnv
from highway_env.envs.scenarios.merge_out_env import MergeOutEnv
from highway_env.envs.scenarios.straight_lane_change_env import StraightLaneChangeEnv
from highway_env.planner.diffusion.config import StructuredDiffusionConfig
from highway_env.planner.diffusion.grpo.checkpoint import load_pretrained
from highway_env.planner.diffusion.grpo.reward_adapter import (
    CandidateRewardAdapter,
    GRPORewardContext,
    resolve_reward_evaluator,
    w4_to_grpo_rewards,
)
from highway_env.planner.diffusion.grpo.sampling import GroupDiffusionSampler
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from highway_env.planner.diffusion.tensor_adapter import BOOL_KEYS, FLOAT_KEYS, PlannerTensorAdapter


FEATURE_KEYS = (*FLOAT_KEYS, *BOOL_KEYS)
SCENARIOS = {
    "straight": StraightLaneChangeEnv,
    "curved": CurvedLaneChangeEnv,
    "merge_in": MergeInEnv,
    "merge_out": MergeOutEnv,
}


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: Path


def _parse_checkpoint(value: str) -> CheckpointSpec:
    name, sep, raw_path = value.partition("=")
    if not sep:
        path = Path(value)
        name = path.parent.parent.parent.name or path.stem
    else:
        name = name.strip()
        path = Path(raw_path.strip())
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    return CheckpointSpec(name=name, path=path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe current GRPO group sampling without training. For each fixed "
            "environment state, generate G trajectories from different diffusion "
            "noise, score them with the production W4 reward, and compare every "
            "group against group 0."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_parse_checkpoint,
        required=True,
        help="repeatable NAME=PATH; checkpoints are loaded one at a time",
    )
    parser.add_argument(
        "--scenario",
        choices=("all", *SCENARIOS.keys()),
        default="all",
    )
    parser.add_argument("--samples-per-scenario", type=int, default=25)
    parser.add_argument("--group-size", type=int, default=48)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--group-action", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7, help="base environment seed")
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=700000,
        help="base seed for diffusion group sampling; shared across checkpoints",
    )
    parser.add_argument(
        "--improvement-eps",
        type=float,
        default=DEFAULT_IMPROVEMENT_EPS,
    )
    parser.add_argument(
        "--max-seed-attempts",
        type=int,
        default=1000,
        help="maximum deterministic seed attempts per scenario to collect valid states",
    )
    parser.add_argument("--reward-fn", default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/grpo_noise_probe_100ep"),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip matplotlib summary plots",
    )
    return parser.parse_args()


def _scenario_features(env: object, adapter: PlannerTensorAdapter) -> Dict[str, torch.Tensor]:
    values = getattr(env, "latest_planner_features", None)
    if values is None:
        raise RuntimeError(
            "scenario did not expose latest_planner_features; call env.step(group_action) first"
        )
    missing = [key for key in FEATURE_KEYS if key not in values]
    if missing:
        raise RuntimeError(f"latest_planner_features missing keys: {missing}")
    return adapter.to_torch({key: values[key] for key in FEATURE_KEYS})


def _frozen_reward_context(
    model: StructuredDiffusionPlanner,
    features: Dict[str, torch.Tensor],
    env: object,
) -> GRPORewardContext:
    model.eval()
    with torch.no_grad():
        frozen = model.infer_multimodal(features)
    return GRPORewardContext(
        env=env,
        frozen_all_mode_trajectories=frozen["trajectory_candidates"],
        frozen_argmax_joint_trajectories=frozen["trajectory"],
        valid_mode_mask=features["mode_valid_mask"],
    )


def _scenario_order(args: argparse.Namespace) -> list[str]:
    if args.scenario == "all":
        return list(SCENARIOS.keys())
    return [str(args.scenario)]


def _state_seed(base_seed: int, scenario_index: int, attempt_index: int) -> int:
    return int(base_seed + scenario_index * 100000 + attempt_index)


def _sample_noise_seed(base_seed: int, scenario_index: int, sample_index: int) -> int:
    return int(base_seed + scenario_index * 100000 + sample_index)


def _build_model(checkpoint: CheckpointSpec, device: torch.device):
    config = StructuredDiffusionConfig()
    adapter = PlannerTensorAdapter(config, device)
    model = StructuredDiffusionPlanner(config, adapter).to(device)
    load_info = load_pretrained(model, checkpoint.path, strict=True)
    model.eval()
    model.requires_grad_(False)
    print(f"[checkpoint] name={checkpoint.name} path={checkpoint.path}")
    print(f"[checkpoint] strict load: {load_info}")
    return adapter, model


def _run_checkpoint(
    args: argparse.Namespace,
    checkpoint: CheckpointSpec,
    *,
    device: torch.device,
) -> tuple[list[ProbeSampleSummary], list[ProbeGroupSummary], list[ProbeVehicleModeSummary]]:
    adapter, model = _build_model(checkpoint, device)
    sampler = GroupDiffusionSampler(
        model,
        group_size=int(args.group_size),
        eta=float(args.eta),
    )
    reward_adapter = CandidateRewardAdapter(resolve_reward_evaluator(args.reward_fn))

    sample_rows: list[ProbeSampleSummary] = []
    group_rows: list[ProbeGroupSummary] = []
    vm_rows: list[ProbeVehicleModeSummary] = []

    scenarios = _scenario_order(args)
    for scenario_index, scenario_name in enumerate(scenarios):
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
            env = env_cls(
                config={
                    "show_trajectories": False,
                    "show_future_trajectories": False,
                },
                render_mode=None,
            )
            try:
                env.reset(seed=env_seed)
                _, _, terminated, truncated, _ = env.step(int(args.group_action))
                if bool(terminated) or bool(truncated):
                    print(
                        f"[skip] checkpoint={checkpoint.name} scenario={scenario_name} "
                        f"seed={env_seed} terminated={terminated} truncated={truncated}"
                    )
                    continue

                features = _scenario_features(env, adapter)
                context = _frozen_reward_context(model, features, env)
                noise_seed = _sample_noise_seed(
                    args.noise_seed,
                    scenario_index,
                    collected,
                )
                generator = torch.Generator(device=device.type).manual_seed(noise_seed)

                with torch.no_grad():
                    trace = sampler.sample(features, generator=generator)
                    reward_result = reward_adapter.evaluate_result(
                        trace.candidates,
                        features=features,
                        context=context,
                    )
                    rewards = w4_to_grpo_rewards(
                        reward_result.rewards,
                        device=trace.candidates.device,
                        dtype=trace.candidates.dtype,
                    )

                sample_summary, one_group_rows, one_vm_rows = analyze_group_rewards(
                    rewards,
                    features["mode_valid_mask"],
                    checkpoint=checkpoint.name,
                    scenario=scenario_name,
                    sample_index=collected,
                    env_seed=env_seed,
                    noise_seed=noise_seed,
                    improvement_eps=float(args.improvement_eps),
                )
                sample_rows.append(sample_summary)
                group_rows.extend(one_group_rows)
                vm_rows.extend(one_vm_rows)

                print(
                    f"[probe] ckpt={checkpoint.name} scenario={scenario_name} "
                    f"sample={collected + 1:03d}/{args.samples_per_scenario} "
                    f"R0={sample_summary.group0_reward:.5f} "
                    f"Rbest={sample_summary.best_group_reward:.5f} "
                    f"gain={sample_summary.best_group_gain:+.5f} "
                    f"better_groups={sample_summary.better_group_count}/{args.group_size - 1} "
                    f"({sample_summary.better_group_fraction:.3f}) "
                    f"pair_better={sample_summary.pair_better_fraction:.3f} "
                    f"rank0={sample_summary.group0_rank}/{args.group_size}"
                )
                collected += 1
            finally:
                env.close()

    del sampler
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sample_rows, group_rows, vm_rows


def _write_outputs(
    args: argparse.Namespace,
    sample_rows: list[ProbeSampleSummary],
    group_rows: list[ProbeGroupSummary],
    vm_rows: list[ProbeVehicleModeSummary],
) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_dataclass_csv(output_dir / "sample_summary.csv", sample_rows)
    write_dataclass_csv(output_dir / "group_summary.csv", group_rows)
    write_dataclass_csv(output_dir / "vehicle_mode_summary.csv", vm_rows)

    checkpoints = [spec.name for spec in args.checkpoint]
    summary_rows = []
    for checkpoint in checkpoints:
        summary_rows.append(
            aggregate_sample_summaries(sample_rows, checkpoint=checkpoint, scenario="all")
        )
        for scenario in _scenario_order(args):
            summary_rows.append(
                aggregate_sample_summaries(
                    sample_rows,
                    checkpoint=checkpoint,
                    scenario=scenario,
                )
            )

    write_dict_csv(output_dir / "overall_summary.csv", summary_rows)
    write_summary_json(output_dir / "overall_summary.json", summary_rows)
    if not args.no_plots:
        maybe_plot_summary(output_dir / "plots", sample_rows)

    print(f"[OK] sample CSV       : {output_dir / 'sample_summary.csv'}")
    print(f"[OK] group CSV        : {output_dir / 'group_summary.csv'}")
    print(f"[OK] vehicle-mode CSV : {output_dir / 'vehicle_mode_summary.csv'}")
    print(f"[OK] overall summary  : {output_dir / 'overall_summary.csv'}")

    print("\n=== OVERALL ===")
    for row in summary_rows:
        if row["scenario"] != "all":
            continue
        print(
            f"{row['checkpoint']}: "
            f"improvable_state={row['improvable_state_fraction']:.3f} | "
            f"better_group={row['mean_better_group_fraction']:.3f} | "
            f"pair_better={row['mean_pair_better_fraction']:.3f} | "
            f"best_gain={row['mean_best_group_gain']:+.5f} | "
            f"reward_range={row['mean_group_reward_range']:.5f} | "
            f"group0_rank={row['mean_group0_rank']:.2f}"
        )


def main() -> int:
    args = parse_args()
    if args.samples_per_scenario < 1:
        raise SystemExit("--samples-per-scenario must be >= 1")
    if args.group_size < 2:
        raise SystemExit("--group-size must be >= 2")
    if args.improvement_eps < 0:
        raise SystemExit("--improvement-eps must be >= 0")
    if args.max_seed_attempts < args.samples_per_scenario:
        raise SystemExit("--max-seed-attempts must be >= --samples-per-scenario")

    checkpoint_names = [spec.name for spec in args.checkpoint]
    if len(checkpoint_names) != len(set(checkpoint_names)):
        raise SystemExit("checkpoint names must be unique")
    for spec in args.checkpoint:
        if not spec.path.is_file():
            raise SystemExit(f"checkpoint not found: {spec.path}")

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device(args.device)
    print(
        f"[config] device={device} group_size={args.group_size} eta={args.eta} "
        f"scenario={args.scenario} samples_per_scenario={args.samples_per_scenario} "
        f"improvement_eps={args.improvement_eps:g}"
    )

    all_sample_rows: list[ProbeSampleSummary] = []
    all_group_rows: list[ProbeGroupSummary] = []
    all_vm_rows: list[ProbeVehicleModeSummary] = []

    for checkpoint in args.checkpoint:
        sample_rows, group_rows, vm_rows = _run_checkpoint(
            args,
            checkpoint,
            device=device,
        )
        all_sample_rows.extend(sample_rows)
        all_group_rows.extend(group_rows)
        all_vm_rows.extend(vm_rows)

    _write_outputs(args, all_sample_rows, all_group_rows, all_vm_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
