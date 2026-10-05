from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from train_grpo import (
    SCENARIOS,
    _build_trainer,
    _frozen_reward_context,
    _fixed_state_paired_validation,
    _scenario_features,
    _validate_fixed_step_zero,
)
from highway_env.planner.diffusion.grpo import save_grpo_checkpoint

# PPO_VALUE_BASELINE_V1_1_20261005
from highway_env.planner.diffusion.ppo import PPOConfig, PPOTrainer

# Route helper functions defined in train_grpo.py to PPO too.
try:
    import train_grpo as _ppo_train_grpo
    _ppo_train_grpo.GRPOConfig = PPOConfig
    _ppo_train_grpo.GRPOTrainer = PPOTrainer
except Exception as _ppo_patch_exc:
    _ppo_train_grpo = None

GRPOConfig = PPOConfig
GRPOTrainer = PPOTrainer

# Fail early if a future refactor accidentally routes back to GRPO.
assert GRPOConfig is PPOConfig
assert GRPOTrainer is PPOTrainer

MARKER = "VANILLA_GRPO_FIXED_STATE_LEARNING_PROBE_V1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fixed-state Vanilla GRPO learnability probe. The environment state is frozen "
            "during training; evaluation reuses a fixed bank of noise seeds so reward trends "
            "reflect policy learning rather than state/noise drift."
        )
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scenario", choices=tuple(SCENARIOS), default="curved")
    p.add_argument("--group-action", type=int, default=3)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--group-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--update-epochs", type=int, default=2)
    p.add_argument(
        "--task-reward-type",
        choices=("legacy_w4", "progress_comfort"),
        default="progress_comfort",
    )
    p.add_argument("--reward-fn", default="auto")
    p.add_argument("--state-seed", type=int, default=7)
    p.add_argument("--train-noise-seed", type=int, default=70007)
    p.add_argument("--eval-noise-seed-base", type=int, default=170007)
    p.add_argument("--eval-noise-seeds", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--trend-window", type=int, default=10)
    p.add_argument("--ema-alpha", type=float, default=0.25)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/grpo_vanilla_learning_probe/current_cfg"),
    )
    return p.parse_args()


def _trainer_args(args: argparse.Namespace) -> SimpleNamespace:
    # _build_trainer is intentionally reused so this probe exercises the exact
    # production Vanilla GRPO model/reward/optimizer construction path.
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        device=args.device,
        fake_reward=False,
        reward_fn=args.reward_fn,
        group_size=args.group_size,
        lr=args.lr,
        eta=args.eta,
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        task_reward_type=args.task_reward_type,
        constraint_strategy="none",
        constraint_names="collision,road,ttc,background_gap,teammate_gap",
        constraint_residual_cap=5.0,
        lagrangian_dual_lr=0.05,
        lagrangian_lambda_init=0.0,
        lagrangian_lambda_max=20.0,
        active_feasible_fraction=0.50,
        active_cvar_alpha=0.25,
        active_temperature=0.20,
        active_support_gate=False,
        active_support_margin=1e-3,
    )


def _slope(values: List[float], window: int) -> float:
    if len(values) < 2:
        return 0.0
    y = np.asarray(values[-max(2, int(window)):], dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denom)


def _write_rows(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = ["step"]
    all_keys = sorted({key for row in rows for key in row if key != "step"})
    fields.extend(all_keys)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _eval_bank(trainer, env, features, context, args, device):
    states = []
    for i in range(int(args.eval_noise_seeds)):
        states.append(
            {
                "index": i,
                "state_seed": int(args.state_seed),
                "sample_seed": int(args.eval_noise_seed_base) + i,
                "env": env,
                "features": features,
                "context": context,
            }
        )
    return _fixed_state_paired_validation(trainer, states, device=device)


def _compact(step: int, row: Dict[str, float]) -> None:
    keys = (
        "probe/current_reward_mean",
        "probe/frozen_reward_mean",
        "probe/reward_gain",
        "probe/reward_gain_ema",
        "probe/reward_slope",
        "probe/positive_step_fraction",
        "probe/selected_reward_gain",
        "probe/candidate_delta_m",
        "train/reward_mean",
        "train/approx_kl",
        "train/reference_kl",
        "train/clip_fraction",
        "train/parameter_update_norm",
    )
    text = " ".join(f"{k.split('/')[-1]}={row[k]:.6f}" for k in keys if k in row)
    print(f"[learn-probe step {step:03d}] {text}")


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.group_size < 2:
        raise SystemExit("--group-size must be >= 2")
    if args.eval_noise_seeds < 1:
        raise SystemExit("--eval-noise-seeds must be >= 1")
    if args.eval_every < 1:
        raise SystemExit("--eval-every must be >= 1")
    if not (0.0 < args.ema_alpha <= 1.0):
        raise SystemExit("--ema-alpha must be in (0,1]")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    adapter, trainer, device = _build_trainer(_trainer_args(args))
    if trainer.config.constraint_strategy != "none":
        raise RuntimeError("learning probe must run Vanilla GRPO with constraint_strategy=none")

    env_cls = SCENARIOS[args.scenario]
    env = env_cls(
        config={"show_trajectories": False, "show_future_trajectories": False},
        render_mode=None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "learning_probe.csv"
    ckpt_path = args.output_dir / "vanilla_learning_probe.pt"

    rows: List[Dict[str, float]] = []
    reward_curve: List[float] = []
    gain_curve: List[float] = []
    ema = None
    positive_steps = 0
    previous_reward = None

    try:
        env.reset(seed=int(args.state_seed))
        _, _, terminated, truncated, _ = env.step(int(args.group_action))
        if bool(terminated) or bool(truncated):
            raise RuntimeError("fixed training state terminated during construction")
        features = {
            k: v.detach().clone()
            for k, v in _scenario_features(env, adapter).items()
        }
        context = _frozen_reward_context(trainer, features, env)

        # Evaluation noise is reset on every evaluation. Training noise is a
        # separate advancing stream so optimization still sees fresh G samples.
        train_generator = torch.Generator(device=device.type).manual_seed(
            int(args.train_noise_seed)
        )

        initial = _eval_bank(trainer, env, features, context, args, device)
        _validate_fixed_step_zero(initial)
        initial_reward = float(initial["fixed_validation/current_reward_mean"])
        frozen_reward = float(initial["fixed_validation/frozen_reward_mean"])
        previous_reward = initial_reward
        ema = 0.0
        reward_curve.append(initial_reward)
        gain_curve.append(0.0)
        row0 = {
            "step": 0,
            "probe/current_reward_mean": initial_reward,
            "probe/frozen_reward_mean": frozen_reward,
            "probe/reward_gain": 0.0,
            "probe/reward_gain_ema": 0.0,
            "probe/reward_slope": 0.0,
            "probe/positive_step_fraction": 0.0,
            "probe/selected_reward_gain": float(
                initial["fixed_validation/selected_vehicle_reward_gain_mean"]
            ),
            "probe/candidate_delta_m": float(
                initial["fixed_validation/candidate_delta_m_mean"]
            ),
        }
        rows.append(row0)
        _compact(0, row0)

        last_train_metrics: Dict[str, float] = {}
        for step in range(1, int(args.steps) + 1):
            train_metrics = trainer.train_step(
                features,
                context=context,
                generator=train_generator,
                paired_validation=False,
            )
            last_train_metrics = train_metrics

            if step % int(args.eval_every) != 0 and step != int(args.steps):
                continue

            ev = _eval_bank(trainer, env, features, context, args, device)
            current = float(ev["fixed_validation/current_reward_mean"])
            frozen = float(ev["fixed_validation/frozen_reward_mean"])
            gain = float(ev["fixed_validation/paired_reward_gain_mean"])
            reward_curve.append(current)
            gain_curve.append(gain)

            if previous_reward is not None and current > previous_reward + 1e-9:
                positive_steps += 1
            previous_reward = current
            transitions = max(1, len(reward_curve) - 1)
            positive_fraction = positive_steps / transitions
            ema = gain if ema is None else (
                float(args.ema_alpha) * gain + (1.0 - float(args.ema_alpha)) * ema
            )

            row: Dict[str, float] = {
                "step": step,
                "probe/current_reward_mean": current,
                "probe/frozen_reward_mean": frozen,
                "probe/reward_gain": gain,
                "probe/reward_gain_ema": float(ema),
                "probe/reward_slope": _slope(reward_curve, args.trend_window),
                "probe/gain_slope": _slope(gain_curve, args.trend_window),
                "probe/positive_step_fraction": float(positive_fraction),
                "probe/selected_reward_gain": float(
                    ev["fixed_validation/selected_vehicle_reward_gain_mean"]
                ),
                "probe/positive_state_fraction": float(
                    ev["fixed_validation/positive_fraction"]
                ),
                "probe/candidate_delta_m": float(
                    ev["fixed_validation/candidate_delta_m_mean"]
                ),
            }
            for key in (
                "reward_mean",
                "reward_std",
                "loss",
                "policy_loss",
                "reference_kl",
                "approx_kl",
                "clip_fraction",
                "ratio_mean",
                "ratio_std",
                "grad_norm",
                "parameter_update_norm",
            ):
                if key in train_metrics:
                    row[f"train/{key}"] = float(train_metrics[key])
            rows.append(row)
            _write_rows(csv_path, rows)
            _compact(step, row)

        _write_rows(csv_path, rows)
        final = rows[-1]
        final_gain = float(final["probe/reward_gain"])
        final_slope = float(final["probe/reward_slope"])
        monotonic = float(final["probe/positive_step_fraction"])
        selected_gain = float(final["probe/selected_reward_gain"])

        if final_gain > 0.0 and final_slope > 0.0 and monotonic >= 0.60:
            verdict = "CLEAR_LEARNING_SIGNAL"
        elif final_gain > 0.0 and final_slope > 0.0:
            verdict = "WEAK_POSITIVE_LEARNING_SIGNAL"
        else:
            verdict = "NO_STABLE_LEARNING_SIGNAL"

        print("\n===== Vanilla GRPO fixed-state learnability summary =====")
        print(f"verdict={verdict}")
        print(f"initial_reward={initial_reward:.8f}")
        print(f"final_reward={final['probe/current_reward_mean']:.8f}")
        print(f"final_gain={final_gain:.8f}")
        print(f"final_reward_slope={final_slope:.8f} reward/step")
        print(f"positive_step_fraction={monotonic:.3f}")
        print(f"selected_reward_gain={selected_gain:.8f}")
        print(f"csv={csv_path}")

        save_grpo_checkpoint(
            ckpt_path,
            model=trainer.model,
            optimizer=trainer.optimizer,
            step=int(args.steps),
            config=trainer.config,
            metrics={
                **last_train_metrics,
                "probe/final_reward_gain": final_gain,
                "probe/final_reward_slope": final_slope,
                "probe/positive_step_fraction": monotonic,
            },
        )
        print(f"checkpoint={ckpt_path}")
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# VANILLA_GRPO_FIXED_STATE_LEARNING_PROBE_V1
