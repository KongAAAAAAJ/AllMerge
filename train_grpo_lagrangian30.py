from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

from highway_env.envs.scenarios.curved_lane_change_env import CurvedLaneChangeEnv
from highway_env.envs.scenarios.merge_in_env import MergeInEnv
from highway_env.envs.scenarios.merge_out_env import MergeOutEnv
from highway_env.envs.scenarios.straight_lane_change_env import StraightLaneChangeEnv
from highway_env.planner.diffusion.config import build_structured_diffusion_config
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from pretraining.checkpoint_io import load_checkpoint_file
from highway_env.planner.diffusion.tensor_adapter import BOOL_KEYS, FLOAT_KEYS, PlannerTensorAdapter
from highway_env.planner.diffusion.grpo import (
    CandidateRewardAdapter,
    GRPOConfig,
    GRPOTrainer,
    fake_progress_reward,
    resolve_reward_evaluator,
    save_grpo_checkpoint,
)
from highway_env.planner.diffusion.grpo.reward_adapter import GRPORewardContext

FEATURE_KEYS = (*FLOAT_KEYS, *BOOL_KEYS)
SCENARIOS = {
    "straight": StraightLaneChangeEnv,
    "curved": CurvedLaneChangeEnv,
    "merge_in": MergeInEnv,
    "merge_out": MergeOutEnv,
}


def _lookup_array(data: np.lib.npyio.NpzFile, key: str):
    for candidate in (key, f"feature_{key}", f"features_{key}", f"features/{key}"):
        if candidate in data:
            return data[candidate]
    return None


def load_feature_arrays(path: Path) -> Dict[str, np.ndarray]:
    """Load W1 feature arrays for fake-reward/offline GRPO smoke only."""
    with np.load(path, allow_pickle=True) as data:
        arrays = {key: _lookup_array(data, key) for key in FEATURE_KEYS}
        if all(value is not None for value in arrays.values()):
            return arrays
        if "features" in data:
            records = data["features"]
            if records.dtype == object and len(records):
                return {
                    key: np.stack([
                        record.item().get(key) if hasattr(record, "item") else record[key]
                        for record in records
                    ], axis=0)
                    for key in FEATURE_KEYS
                }
        missing = [key for key, value in arrays.items() if value is None]
        raise KeyError(f"Could not find planner feature arrays {missing} in {path}")


def iter_batches(
    arrays: Dict[str, np.ndarray],
    *,
    batch_size: int,
    adapter: PlannerTensorAdapter,
) -> Iterable[Dict[str, torch.Tensor]]:
    n = len(arrays["ego_state"])
    if n == 0:
        raise ValueError("empty dataset shard")
    start = 0
    while True:
        indices = np.arange(start, start + batch_size) % n
        yield adapter.to_torch({key: value[indices] for key, value in arrays.items()})
        start = (start + batch_size) % n


def _scenario_features(env: object, adapter: PlannerTensorAdapter) -> Dict[str, torch.Tensor]:
    values = getattr(env, "latest_planner_features", None)
    if values is None:
        raise RuntimeError(
            "scenario did not expose latest_planner_features; "
            "call env.step(group_action) before GRPO scoring"
        )
    missing = [key for key in FEATURE_KEYS if key not in values]
    if missing:
        raise RuntimeError(f"latest_planner_features missing keys: {missing}")
    return adapter.to_torch({key: values[key] for key in FEATURE_KEYS})


def _frozen_reward_context(
    trainer: GRPOTrainer,
    features: Dict[str, torch.Tensor],
    env: object,
) -> GRPORewardContext:
    """Build the exact W4 frozen Stage-1 context from the fixed reference model."""
    trainer.reference_model.eval()
    with torch.no_grad():
        frozen = trainer.reference_model.infer_multimodal(features)
    return GRPORewardContext(
        env=env,
        frozen_all_mode_trajectories=frozen["trajectory_candidates"],
        frozen_argmax_joint_trajectories=frozen["trajectory"],
        valid_mode_mask=features["mode_valid_mask"],
    )



# FIXED-STATE PAIRED VALIDATION V1

def _build_fixed_validation_states(
    args,
    adapter: PlannerTensorAdapter,
    trainer: GRPOTrainer,
):
    """Create deterministic validation env states once and keep them frozen."""
    count = int(args.fixed_validation_states)
    if count <= 0:
        return []

    env_cls = SCENARIOS[args.scenario]
    fixed_states = []
    print(
        f"[fixed-val] building {count} states | "
        f"scenario={args.scenario} "
        f"seed_offset={args.fixed_validation_seed_offset}"
    )

    try:
        for index in range(count):
            state_seed = (
                int(args.seed)
                + int(args.fixed_validation_seed_offset)
                + index
            )
            sample_seed = (
                int(args.seed)
                + 2 * int(args.fixed_validation_seed_offset)
                + index
            )

            env = env_cls(
                config={
                    "show_trajectories": False,
                    "show_future_trajectories": False,
                },
                render_mode=None,
            )
            env.reset(seed=state_seed)

            _, _, terminated, truncated, _ = env.step(int(args.group_action))
            if bool(terminated) or bool(truncated):
                env.close()
                raise RuntimeError(
                    "fixed validation state terminated on construction: "
                    f"index={index}, seed={state_seed}"
                )

            raw_features = _scenario_features(env, adapter)
            features = {
                key: value.detach().clone()
                for key, value in raw_features.items()
            }
            context = _frozen_reward_context(trainer, features, env)

            fixed_states.append(
                {
                    "index": index,
                    "state_seed": state_seed,
                    "sample_seed": sample_seed,
                    "env": env,
                    "features": features,
                    "context": context,
                }
            )
    except Exception:
        for item in fixed_states:
            try:
                item["env"].close()
            except Exception:
                pass
        raise

    print(f"[fixed-val] cached={len(fixed_states)}")
    return fixed_states


def _close_fixed_validation_states(fixed_states) -> None:
    for item in fixed_states:
        try:
            item["env"].close()
        except Exception:
            pass


def _fixed_state_paired_validation(
    trainer: GRPOTrainer,
    fixed_states,
    *,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate current/frozen policies on identical cached states and noise."""
    if not fixed_states:
        return {}

    trainer.model.eval()
    trainer.reference_model.eval()

    per_state = []
    with torch.no_grad():
        for item in fixed_states:
            current_generator = torch.Generator(
                device=device.type
            ).manual_seed(int(item["sample_seed"]))
            frozen_generator = torch.Generator(
                device=device.type
            ).manual_seed(int(item["sample_seed"]))

            features = item["features"]
            context = item["context"]

            current_trace = trainer.sampler.sample(
                features,
                generator=current_generator,
            )
            current_rewards = trainer.score_candidates(
                current_trace.candidates,
                features=features,
                context=context,
            )
            metrics = trainer.paired_validation(
                current_trace,
                current_rewards,
                features,
                context=context,
                generator=frozen_generator,
            )
            per_state.append(metrics)

    def array(key: str) -> np.ndarray:
        return np.asarray(
            [float(row[key]) for row in per_state],
            dtype=np.float64,
        )

    gains = array("validation/paired_group_reward_gain")
    current_rewards = array("validation/paired_group_current_reward_mean")
    frozen_rewards = array("validation/paired_group_frozen_reward_mean")
    candidate_delta = array("validation/paired_candidate_delta_m")
    selected_gain = array("validation/selected_vehicle_reward_gain")
    selected_current = array(
        "validation/selected_vehicle_reward_mean"
    )
    selected_frozen = array(
        "validation/selected_pretrain_vehicle_reward_mean"
    )
    group_size = int(trainer.config.group_size)
    result: Dict[str, float] = {
        "fixed_validation/state_count": float(len(fixed_states)),
        "fixed_validation/current_reward_mean": float(current_rewards.mean()),
        "fixed_validation/frozen_reward_mean": float(frozen_rewards.mean()),
        "fixed_validation/paired_reward_gain_mean": float(gains.mean()),
        "fixed_validation/paired_reward_gain_median": float(np.median(gains)),
        "fixed_validation/paired_reward_gain_p05": float(np.quantile(gains, 0.05)),
        "fixed_validation/paired_reward_gain_p95": float(np.quantile(gains, 0.95)),
        "fixed_validation/positive_fraction": float(np.mean(gains > 1e-6)),
        "fixed_validation/candidate_delta_m_mean": float(candidate_delta.mean()),
        f"fixed_validation/paired_n{group_size}_reward_gain_mean": float(gains.mean()),
        "fixed_validation/selected_vehicle_reward_mean": float(selected_current.mean()),
        "fixed_validation/selected_pretrain_vehicle_reward_mean": float(
            selected_frozen.mean()
        ),
        "fixed_validation/selected_vehicle_reward_gain_mean": float(
            selected_gain.mean()
        ),
        "fixed_validation/selected_vehicle_reward_gain_median": float(
            np.median(selected_gain)
        ),
        "fixed_validation/selected_positive_fraction": float(
            np.mean(selected_gain > 1e-6)
        ),
    }

    role_gain_means = []

    selected_role_gain_means = []

    constraint_keys = [
        "validation/current_constraint_feasible_fraction",
        "validation/frozen_constraint_feasible_fraction",
        "validation/constraint_feasible_fraction_gain",
        "validation/current_constraint_max_violation_mean",
        "validation/frozen_constraint_max_violation_mean",
        "validation/constraint_max_violation_change",
    ]
    for name in trainer.config.constraint_names:
        constraint_keys.extend(
            [
                f"validation/current_constraint_{name}_violation_mean",
                f"validation/frozen_constraint_{name}_violation_mean",
                f"validation/constraint_{name}_violation_change",
            ]
        )
    for key in constraint_keys:
        values = array(key)
        suffix = key.removeprefix("validation/")
        result[f"fixed_validation/{suffix}_mean"] = float(values.mean())


    for role in range(3):

        role_gains = array(f"validation/vehicle_{role}_reward_gain")

        role_mean = float(role_gains.mean())

        result[f"fixed_validation/vehicle_{role}_reward_gain_mean"] = role_mean

        role_gain_means.append(role_mean)

        selected_role_gains = array(

            f"validation/selected_vehicle_{role}_reward_gain"

        )

        selected_role_mean = float(selected_role_gains.mean())

        result[

            f"fixed_validation/selected_vehicle_{role}_reward_gain_mean"

        ] = selected_role_mean

        selected_role_gain_means.append(selected_role_mean)


    result["fixed_validation/worst_vehicle_reward_gain_mean"] = float(

        min(role_gain_means)

    )

    result["fixed_validation/best_vehicle_reward_gain_mean"] = float(

        max(role_gain_means)

    )

    result["fixed_validation/vehicle_reward_gain_spread"] = float(

        max(role_gain_means) - min(role_gain_means)

    )

    result["fixed_validation/worst_selected_vehicle_reward_gain_mean"] = float(

        min(selected_role_gain_means)

    )


    return result


def _fixed_validation_csv_path(args) -> Path:
    if args.fixed_validation_csv is not None:
        return Path(args.fixed_validation_csv)
    output = Path(args.output)
    return output.with_name(output.stem + ".fixed_validation.csv")


def _write_fixed_validation_csv(
    path: Path,
    *,
    step: int,
    metrics: Dict[str, float],
    initialize: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", *sorted(metrics.keys())]
    mode = "w" if initialize else "a"

    with path.open(mode, newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if initialize:
            writer.writeheader()
        row = {"step": int(step)}
        row.update({key: float(value) for key, value in metrics.items()})
        writer.writerow(row)


def _print_fixed_validation(step: int, metrics: Dict[str, float]) -> None:
    print(
        f"[fixed-val step {step:03d}] "
        f"states={int(metrics['fixed_validation/state_count'])} "
        f"gain_mean={metrics['fixed_validation/paired_reward_gain_mean']:.5f} "
        f"gain_median={metrics['fixed_validation/paired_reward_gain_median']:.5f} "
        f"positive_fraction={metrics['fixed_validation/positive_fraction']:.3f} "
        f"selected_gain={metrics['fixed_validation/selected_vehicle_reward_gain_mean']:.5f} "
        f"worst_role={metrics['fixed_validation/worst_vehicle_reward_gain_mean']:.5f} "
        f"candidate_delta_m={metrics['fixed_validation/candidate_delta_m_mean']:.5f}"
    )


def _validate_fixed_step_zero(metrics: Dict[str, float]) -> None:
    gain = abs(float(metrics["fixed_validation/paired_reward_gain_mean"]))
    delta = float(metrics["fixed_validation/candidate_delta_m_mean"])
    if gain >= 1e-3:
        raise RuntimeError(
            "fixed validation step-0 paired reward gain is not near zero: "
            f"{gain:.9g}"
        )
    if delta >= 1e-3:
        raise RuntimeError(
            "fixed validation step-0 paired candidate delta exceeds 1 mm: "
            f"{delta:.9g} m"
        )


    selected_gain = abs(
        float(
            metrics[
                "fixed_validation/selected_vehicle_reward_gain_mean"
            ]
        )
    )
    if selected_gain >= 1e-3:
        raise RuntimeError(
            "fixed validation step-0 selected reward gain "
            f"is not near zero: {selected_gain:.9g}"
        )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AllMerge migration-first GRPO trainer")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--dataset-shard",
        type=Path,
        default=None,
        help="W1 shard; required only with --fake-reward",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default="straight",
        help="online scenario used by production W4 reward",
    )
    parser.add_argument("--group-action", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3, choices=(3, 10, 30, 100))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument(
        "--task-reward-type",
        choices=("legacy_w4", "progress_comfort"),
        default="legacy_w4",
        help=(
            "task objective; progress_comfort removes collision/road/TTC/gap "
            "terms so safety is handled only by constraints"
        ),
    )
    parser.add_argument(
        "--constraint-strategy",
        choices=("none", "lagrangian", "hard_worst", "soft_active"),
        default="none",
        help=(
            "multi-constraint GRPO strategy; 'none' preserves the current "
            "reward-only behavior"
        ),
    )
    parser.add_argument(
        "--constraint-names",
        default="collision,road,ttc,background_gap,teammate_gap",
        help=(
            "comma-separated W4 constraints used by constrained GRPO; "
            "available: collision,road,ttc,background_gap,teammate_gap"
        ),
    )
    parser.add_argument(
        "--constraint-residual-cap", type=float, default=5.0,
        help="cap for normalized signed constraint residuals",
    )
    parser.add_argument(
        "--lagrangian-dual-lr", type=float, default=0.05,
        help="dual ascent learning rate for Lagrangian constrained GRPO",
    )
    parser.add_argument(
        "--lagrangian-lambda-init", type=float, default=0.0,
        help="initial multiplier shared by all enabled constraints",
    )
    parser.add_argument(
        "--lagrangian-lambda-max", type=float, default=20.0,
        help="upper projection bound for Lagrangian multipliers",
    )
    parser.add_argument(
        "--active-feasible-fraction", type=float, default=0.50,
        help="current feasible-candidate fraction required to switch restoration to reward mode",
    )
    parser.add_argument(
        "--active-cvar-alpha", type=float, default=0.25,
        help="upper-tail fraction used for per-constraint group severity",
    )
    parser.add_argument(
        "--active-temperature", type=float, default=0.20,
        help="soft_active temperature; lower values approach hard worst-constraint selection",
    )
    parser.add_argument(
        "--active-support-gate", action="store_true",
        help="skip restoration actor updates when no candidate improves on frozen reference violation",
    )
    parser.add_argument(
        "--active-support-margin", type=float, default=1e-3,
        help="minimum pretrained-relative violation improvement required by support gate",
    )
    parser.add_argument("--reward-fn", default="auto", help="module:function or auto")
    parser.add_argument("--fake-reward", action="store_true")
    parser.add_argument(
        "--fixed-validation-states",
        type=int,
        default=0,
        help=(
            "number of cached fixed online states; "
            "0 disables fixed-state validation"
        ),
    )
    parser.add_argument(
        "--fixed-validation-interval",
        type=int,
        default=10,
        help=(
            "optimizer-step interval for fixed-state validation; "
            "step 0 and final step are always evaluated"
        ),
    )
    parser.add_argument(
        "--fixed-validation-seed-offset",
        type=int,
        default=10000,
        help="deterministic seed offset for fixed validation states",
    )
    parser.add_argument(
        "--fixed-validation-csv",
        type=Path,
        default=None,
        help=(
            "optional CSV path; default is "
            "<output-stem>.fixed_validation.csv"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/grpo_smoke.pt"))
    return parser.parse_args()


def _build_trainer(args: argparse.Namespace):
    device = torch.device(args.device)
    state_dict, stored_model_config = load_checkpoint_file(
        args.checkpoint, map_location="cpu"
    )
    model_config = build_structured_diffusion_config(**stored_model_config)
    adapter = PlannerTensorAdapter(model_config, device)
    model = StructuredDiffusionPlanner(model_config, adapter).to(device)
    load_result = model.load_state_dict(state_dict, strict=True)
    load_info = {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "reg_head_type": model_config.reg_head_type,
    }
    print(f"[checkpoint] loaded={args.checkpoint} {load_info}")

    evaluator = fake_progress_reward if args.fake_reward else resolve_reward_evaluator(args.reward_fn)
    constraint_names = tuple(
        name.strip() for name in str(args.constraint_names).split(",") if name.strip()
    )
    trainer = GRPOTrainer(
        model,
        CandidateRewardAdapter(evaluator),
        config=GRPOConfig(
            group_size=args.group_size,
            learning_rate=args.lr,
            eta=args.eta,
            clip_eps=args.clip_eps,
            kl_coef=args.kl_coef,
            max_grad_norm=args.max_grad_norm,
            update_epochs=args.update_epochs,
            task_reward_type=args.task_reward_type,
            constraint_strategy=args.constraint_strategy,
            constraint_names=constraint_names,
            constraint_residual_cap=args.constraint_residual_cap,
            lagrangian_dual_lr=args.lagrangian_dual_lr,
            lagrangian_lambda_init=args.lagrangian_lambda_init,
            lagrangian_lambda_max=args.lagrangian_lambda_max,
            active_feasible_fraction=args.active_feasible_fraction,
            active_cvar_alpha=args.active_cvar_alpha,
            active_temperature=args.active_temperature,
            active_support_gate=args.active_support_gate,
            active_support_margin=args.active_support_margin,
        ),
    )
    print(f"[grpo] trainable_params={trainer.trainable_parameter_count:,}")
    print(f"[grpo] update_epochs={args.update_epochs} clip_eps={args.clip_eps}")
    print(f"[grpo] task_reward_type={args.task_reward_type}")
    print(
        f"[grpo] constraint_strategy={args.constraint_strategy} "
        f"constraints={constraint_names}"
    )
    if args.constraint_strategy == "lagrangian":
        print(
            "[grpo] lagrangian "
            f"dual_lr={args.lagrangian_dual_lr} "
            f"lambda_init={args.lagrangian_lambda_init} "
            f"lambda_max={args.lagrangian_lambda_max} "
            f"residual_cap={args.constraint_residual_cap}"
        )
    if args.constraint_strategy in {"hard_worst", "soft_active"}:
        print(
            "[grpo] active_constraint "
            f"feasible_fraction={args.active_feasible_fraction} "
            f"cvar_alpha={args.active_cvar_alpha} "
            f"temperature={args.active_temperature} "
            f"support_gate={args.active_support_gate} "
            f"support_margin={args.active_support_margin}"
        )
    return adapter, trainer, device


def _run_fake(args, adapter, trainer, device, generator) -> Dict[str, float]:
    if args.dataset_shard is None:
        raise SystemExit("--dataset-shard is required with --fake-reward")
    arrays = load_feature_arrays(args.dataset_shard)
    batches = iter_batches(arrays, batch_size=args.batch_size, adapter=adapter)
    metrics: Dict[str, float] = {}
    for step in range(1, args.steps + 1):
        metrics = trainer.train_step(next(batches), generator=generator)
        compact = " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
        print(f"[step {step:03d}/{args.steps}] {compact}")
    return metrics


def _run_w4(args, adapter, trainer, device, generator) -> Dict[str, float]:
    if args.batch_size != 1:
        print(
            "[INFO] --batch-size is ignored for online W4 reward; "
            "scenario batch is the 3-vehicle platoon"
        )

    env_cls = SCENARIOS[args.scenario]
    env = env_cls(
        config={
            "show_trajectories": False,
            "show_future_trajectories": False,
        },
        render_mode=None,
    )

    fixed_states = []
    metrics: Dict[str, float] = {}
    try:
        fixed_states = _build_fixed_validation_states(args, adapter, trainer)
        fixed_csv = _fixed_validation_csv_path(args) if fixed_states else None

        if fixed_states:
            fixed_metrics = _fixed_state_paired_validation(
                trainer,
                fixed_states,
                device=device,
            )
            _validate_fixed_step_zero(fixed_metrics)
            _print_fixed_validation(0, fixed_metrics)
            assert fixed_csv is not None
            _write_fixed_validation_csv(
                fixed_csv,
                step=0,
                metrics=fixed_metrics,
                initialize=True,
            )
            print(f"[fixed-val] csv={fixed_csv}")

        env.reset(seed=int(args.seed))

        for step in range(1, args.steps + 1):
            _, _, terminated, truncated, _ = env.step(int(args.group_action))
            features = _scenario_features(env, adapter)
            context = _frozen_reward_context(trainer, features, env)


            metrics = trainer.train_step(
                features,
                context=context,
                generator=generator,
                paired_validation=False,
            )

            fixed_interval = int(args.fixed_validation_interval)
            fixed_now = bool(fixed_states) and (
                step == args.steps
                or step % fixed_interval == 0
            )
            if fixed_now:
                fixed_metrics = _fixed_state_paired_validation(
                    trainer,
                    fixed_states,
                    device=device,
                )
                metrics.update(fixed_metrics)
                _print_fixed_validation(step, fixed_metrics)

                assert fixed_csv is not None
                _write_fixed_validation_csv(
                    fixed_csv,
                    step=step,
                    metrics=fixed_metrics,
                    initialize=False,
                )

            compact = " ".join(
                f"{key}={value:.5f}" for key, value in metrics.items()
            )
            print(
                f"[step {step:03d}/{args.steps}] "
                f"scenario={args.scenario} {compact}"
            )

            if bool(terminated) or bool(truncated):
                env.reset(seed=int(args.seed) + step)
    finally:
        env.close()
        _close_fixed_validation_states(fixed_states)

    return metrics


def main() -> int:
    args = parse_args()
    if args.fixed_validation_states < 0:
        raise SystemExit("--fixed-validation-states must be >= 0")
    if args.fixed_validation_interval < 1:
        raise SystemExit("--fixed-validation-interval must be >= 1")
    if args.fake_reward and args.constraint_strategy != "none":
        raise SystemExit(
            "constrained GRPO requires the production W4 reward evaluator; "
            "--fake-reward only supports --constraint-strategy none"
        )
    if args.fake_reward and args.task_reward_type != "legacy_w4":
        raise SystemExit(
            "--task-reward-type progress_comfort requires the production W4 evaluator"
        )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    adapter, trainer, device = _build_trainer(args)
    generator = torch.Generator(device=device.type).manual_seed(args.seed)

    if args.fake_reward:
        metrics = _run_fake(args, adapter, trainer, device, generator)
    else:
        metrics = _run_w4(args, adapter, trainer, device, generator)

    save_grpo_checkpoint(
        args.output,
        model=trainer.model,
        optimizer=trainer.optimizer,
        step=args.steps,
        config=trainer.config,
        metrics=metrics,
    )
    print(f"[OK] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# GRPO_FORMAL_METRICS_V3

# LAGRANGIAN_CONSTRAINED_GRPO_BASELINE_V2_SEMANTIC

# ACTIVE_CONSTRAINT_GRPO_V1_SEMANTIC

# DECOUPLED_TASK_CONSTRAINT_V2_SEMANTIC
