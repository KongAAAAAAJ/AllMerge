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
from highway_env.planner.diffusion.config import StructuredDiffusionConfig
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from highway_env.planner.diffusion.tensor_adapter import BOOL_KEYS, FLOAT_KEYS, PlannerTensorAdapter
from highway_env.planner.diffusion.grpo import (
    CandidateRewardAdapter,
    GRPOConfig,
    GRPOTrainer,
    fake_progress_reward,
    load_pretrained,
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
            current_rewards = trainer.reward_adapter(
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
    }

    for role in range(3):
        role_gains = array(f"validation/vehicle_{role}_reward_gain")
        result[f"fixed_validation/vehicle_{role}_reward_gain_mean"] = float(
            role_gains.mean()
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



# GRPO DIAGNOSTICS BASE V1

def _diagnostics_csv_path(args) -> Path:
    if args.diagnostics_csv is not None:
        return Path(args.diagnostics_csv)
    output = Path(args.output)
    return output.with_name(output.stem + ".diagnostics.csv")


def _write_diagnostics_csv(
    path: Path,
    *,
    step: int,
    metrics: Dict[str, float],
    initialize: bool,
) -> None:
    diagnostics = {
        key: float(value)
        for key, value in metrics.items()
        if key.startswith("diagnostics/")
    }
    if not diagnostics:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if initialize:
        fieldnames = ["step", *sorted(diagnostics.keys())]
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            row = {"step": int(step), **diagnostics}
            writer.writerow(row)
        return

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        fieldnames = next(reader)

    row = {name: "" for name in fieldnames}
    row["step"] = int(step)
    for key, value in diagnostics.items():
        if key not in row:
            raise RuntimeError(
                "diagnostics CSV schema changed after initialization: " + key
            )
        row[key] = value

    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(row)


def _print_diagnostics(step: int, metrics: Dict[str, float]) -> None:
    gain = metrics.get("diagnostics/online_paired_reward_gain", float("nan"))
    positive = metrics.get(
        "diagnostics/online_paired_positive_fraction", float("nan")
    )
    reward_std = metrics.get(
        "diagnostics/within_group_reward_std", float("nan")
    )
    pair_distance = metrics.get(
        "diagnostics/candidate_pairwise_distance_m", float("nan")
    )
    message = (
        f"[diag step {step:03d}] "
        f"online_gain={gain:.5f} "
        f"positive_fraction={positive:.3f} "
        f"group_reward_std={reward_std:.5f} "
        f"pairwise_distance_m={pair_distance:.5f}"
    )
    if "diagnostics/policy_grad_norm" in metrics:
        message += (
            f" policy_g={metrics['diagnostics/policy_grad_norm']:.3f}"
            f" weighted_kl_g={metrics['diagnostics/reference_kl_weighted_grad_norm']:.3f}"
            f" policy_kl_cos={metrics['diagnostics/policy_kl_grad_cosine']:.3f}"
        )
        for key in (
            "diagnostics/vehicle_01_grad_cosine",
            "diagnostics/vehicle_02_grad_cosine",
            "diagnostics/vehicle_12_grad_cosine",
        ):
            if key in metrics:
                message += f" {key.rsplit('/', 1)[-1]}={metrics[key]:.3f}"
    print(message)

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
    parser.add_argument("--steps", type=int, default=3, choices=(3, 10, 100))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--update-epochs", type=int, default=2)
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
    parser.add_argument(
        "--diagnostics-interval",
        type=int,
        default=0,
        help=(
            "online same-state/same-noise reward + group diagnostics interval; "
            "0 disables. Step 1 and final step are always included when enabled"
        ),
    )
    parser.add_argument(
        "--gradient-diagnostics-interval",
        type=int,
        default=0,
        help=(
            "policy/KL/vehicle gradient diagnostics interval; 0 disables. "
            "Step 1 and final step are always included when enabled"
        ),
    )
    parser.add_argument(
        "--diagnostics-csv",
        type=Path,
        default=None,
        help="optional diagnostics CSV path; default is <output-stem>.diagnostics.csv",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/grpo_smoke.pt"))
    return parser.parse_args()


def _build_trainer(args: argparse.Namespace):
    device = torch.device(args.device)
    model_config = StructuredDiffusionConfig()
    adapter = PlannerTensorAdapter(model_config, device)
    model = StructuredDiffusionPlanner(model_config, adapter).to(device)
    load_info = load_pretrained(model, args.checkpoint, strict=True)
    print(f"[checkpoint] loaded={args.checkpoint} {load_info}")

    evaluator = fake_progress_reward if args.fake_reward else resolve_reward_evaluator(args.reward_fn)
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
        ),
    )
    print(f"[grpo] trainable_params={trainer.trainable_parameter_count:,}")
    print(f"[grpo] update_epochs={args.update_epochs} clip_eps={args.clip_eps}")
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

        diagnostics_enabled = (
            int(args.diagnostics_interval) > 0
            or int(args.gradient_diagnostics_interval) > 0
        )
        diagnostics_csv = (
            _diagnostics_csv_path(args)
            if diagnostics_enabled
            else None
        )
        diagnostics_csv_initialized = False

        for step in range(1, args.steps + 1):
            _, _, terminated, truncated, _ = env.step(int(args.group_action))
            features = _scenario_features(env, adapter)
            context = _frozen_reward_context(trainer, features, env)


            diagnostic_now = (
                int(args.diagnostics_interval) > 0
                and (
                    step == 1
                    or step == args.steps
                    or step % int(args.diagnostics_interval) == 0
                )
            )
            gradient_diagnostic_now = (
                int(args.gradient_diagnostics_interval) > 0
                and (
                    step == 1
                    or step == args.steps
                    or step % int(args.gradient_diagnostics_interval) == 0
                )
            )
            diagnostic_now = bool(
                diagnostic_now or gradient_diagnostic_now
            )

            metrics = trainer.train_step(
                features,
                context=context,
                generator=generator,
                paired_validation=False,
                diagnostics=diagnostic_now,
                gradient_diagnostics=gradient_diagnostic_now,
            )

            if diagnostic_now:
                _print_diagnostics(step, metrics)
                assert diagnostics_csv is not None
                _write_diagnostics_csv(
                    diagnostics_csv,
                    step=step,
                    metrics=metrics,
                    initialize=not diagnostics_csv_initialized,
                )
                diagnostics_csv_initialized = True

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
    if args.diagnostics_interval < 0:
        raise SystemExit("--diagnostics-interval must be >= 0")
    if args.gradient_diagnostics_interval < 0:
        raise SystemExit("--gradient-diagnostics-interval must be >= 0")
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
