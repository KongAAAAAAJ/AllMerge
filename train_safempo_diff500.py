# GRPO_5000STEP_SUPPORT_20261008
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from highway_env.planner.diffusion.config import build_structured_diffusion_config
from highway_env.planner.diffusion.grpo import (
    CandidateRewardAdapter,
    resolve_reward_evaluator,
    save_grpo_checkpoint,
)
from highway_env.planner.diffusion.safempo_diff import (
    SafeMPODiffConfig,
    SafeMPODiffTrainer,
)
from highway_env.planner.diffusion.structured_model import StructuredDiffusionPlanner
from highway_env.planner.diffusion.tensor_adapter import PlannerTensorAdapter
from pretraining.checkpoint_io import load_checkpoint_file

# Reuse the exact frozen-state construction/evaluation/run loop used by the
# 500-step Lagrangian baseline so only the policy-update mechanism changes.
from train_grpo_lagrangian500 import SCENARIOS, _run_w4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SafeMPO-Diff V1: G48 multi-constraint q* + KL distillation"
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--scenario", choices=tuple(SCENARIOS), default="curved")
    p.add_argument("--group-action", type=int, default=3)
    p.add_argument("--steps", type=int, default=3, choices=(3, 10, 30, 100, 500, 5000))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--group-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument(
        "--clip-eps",
        type=float,
        default=0.10,
        help="compatibility/config field only; SafeMPO-Diff V1 does not use PPO clipping",
    )
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--update-epochs", type=int, default=2)
    p.add_argument(
        "--task-reward-type",
        choices=("progress_comfort",),
        default="progress_comfort",
    )
    p.add_argument(
        "--constraint-names",
        default="collision,road,ttc,background_gap,teammate_gap",
    )
    p.add_argument("--constraint-residual-cap", type=float, default=5.0)

    # Paper-faithful SafeMPO defaults unless explicitly overridden.
    p.add_argument("--safempo-kl-epsilon", type=float, default=0.10)
    p.add_argument("--safempo-kappa", type=float, default=10.0)
    p.add_argument("--safempo-constraint-beta", type=float, default=1.0)
    p.add_argument("--safempo-lambda-init", type=float, default=1.0)
    p.add_argument("--safempo-lambda-min", type=float, default=1e-6)
    p.add_argument("--safempo-lambda-max", type=float, default=10000.0)
    p.add_argument("--safempo-nu-init", type=float, default=1.0)
    p.add_argument("--safempo-nu-min", type=float, default=1e-5)
    p.add_argument("--safempo-nu-max", type=float, default=1000.0)
    p.add_argument("--safempo-active-range-eps", type=float, default=1e-6)
    p.add_argument("--safempo-dual-maxiter", type=int, default=128)
    p.add_argument("--safempo-dual-ftol", type=float, default=1e-9)

    # SAFEMPO_DIFF_V11_TRUST_ANCHOR_20261007: independent M-step switches.
    p.add_argument("--safempo-local-trust-enabled", action="store_true")
    p.add_argument("--safempo-local-kl-target", type=float, default=0.01)
    p.add_argument("--safempo-local-dual-init", type=float, default=0.0)
    p.add_argument("--safempo-local-dual-lr", type=float, default=25.0)
    p.add_argument("--safempo-local-dual-min", type=float, default=0.0)
    p.add_argument("--safempo-local-dual-max", type=float, default=10.0)
    p.add_argument("--safempo-global-anchor-enabled", action="store_true")
    p.add_argument("--safempo-global-kl-target", type=float, default=0.15)
    p.add_argument("--safempo-global-dual-init", type=float, default=0.05)
    p.add_argument("--safempo-global-dual-lr", type=float, default=2.0)
    p.add_argument("--safempo-global-dual-min", type=float, default=0.05)
    p.add_argument("--safempo-global-dual-max", type=float, default=5.0)

    p.add_argument("--fixed-validation-states", type=int, default=0)
    p.add_argument("--fixed-validation-interval", type=int, default=10)
    p.add_argument("--fixed-validation-seed-offset", type=int, default=10000)
    p.add_argument("--fixed-validation-csv", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/safempo_diff_v1/safempo_diff.pt"),
    )
    return p.parse_args()


def _build_trainer(args: argparse.Namespace):
    device = torch.device(args.device)
    state_dict, stored_model_config = load_checkpoint_file(
        args.checkpoint, map_location="cpu"
    )
    model_config = build_structured_diffusion_config(**stored_model_config)
    adapter = PlannerTensorAdapter(model_config, device)
    model = StructuredDiffusionPlanner(model_config, adapter).to(device)
    load_result = model.load_state_dict(state_dict, strict=True)
    print(
        f"[checkpoint] loaded={args.checkpoint} "
        f"missing={list(load_result.missing_keys)} "
        f"unexpected={list(load_result.unexpected_keys)} "
        f"reg_head_type={model_config.reg_head_type}"
    )

    constraint_names = tuple(
        name.strip() for name in str(args.constraint_names).split(",") if name.strip()
    )
    trainer = SafeMPODiffTrainer(
        model,
        CandidateRewardAdapter(resolve_reward_evaluator("auto")),
        config=SafeMPODiffConfig(
            group_size=args.group_size,
            learning_rate=args.lr,
            eta=args.eta,
            clip_eps=args.clip_eps,
            kl_coef=args.kl_coef,
            max_grad_norm=args.max_grad_norm,
            update_epochs=args.update_epochs,
            task_reward_type=args.task_reward_type,
            constraint_strategy="none",
            constraint_names=constraint_names,
            constraint_residual_cap=args.constraint_residual_cap,
            safempo_kl_epsilon=args.safempo_kl_epsilon,
            safempo_kappa=args.safempo_kappa,
            safempo_constraint_beta=args.safempo_constraint_beta,
            safempo_lambda_init=args.safempo_lambda_init,
            safempo_lambda_min=args.safempo_lambda_min,
            safempo_lambda_max=args.safempo_lambda_max,
            safempo_nu_init=args.safempo_nu_init,
            safempo_nu_min=args.safempo_nu_min,
            safempo_nu_max=args.safempo_nu_max,
            safempo_active_range_eps=args.safempo_active_range_eps,
            safempo_dual_maxiter=args.safempo_dual_maxiter,
            safempo_dual_ftol=args.safempo_dual_ftol,
            # SAFEMPO_DIFF_V11_TRUST_ANCHOR_20261007
            safempo_local_trust_enabled=args.safempo_local_trust_enabled,
            safempo_local_kl_target=args.safempo_local_kl_target,
            safempo_local_dual_init=args.safempo_local_dual_init,
            safempo_local_dual_lr=args.safempo_local_dual_lr,
            safempo_local_dual_min=args.safempo_local_dual_min,
            safempo_local_dual_max=args.safempo_local_dual_max,
            safempo_global_anchor_enabled=args.safempo_global_anchor_enabled,
            safempo_global_kl_target=args.safempo_global_kl_target,
            safempo_global_dual_init=args.safempo_global_dual_init,
            safempo_global_dual_lr=args.safempo_global_dual_lr,
            safempo_global_dual_min=args.safempo_global_dual_min,
            safempo_global_dual_max=args.safempo_global_dual_max,
        ),
    )
    print(f"[safempo-diff] trainable_params={trainer.trainable_parameter_count:,}")
    print(
        "[safempo-diff] V1 "
        f"G={args.group_size} lr={args.lr:g} eta={args.eta:g} "
        f"update_epochs={args.update_epochs} ref_kl_coef={args.kl_coef:g}"
    )
    print(
        "[safempo-diff] E-step "
        f"epsilon={args.safempo_kl_epsilon:g} "
        f"kappa={args.safempo_kappa:g} beta={args.safempo_constraint_beta:g} "
        f"dual_maxiter={args.safempo_dual_maxiter}"
    )
    print(
        "[safempo-diff] constraints="
        f"{constraint_names} residual_cap={args.constraint_residual_cap:g}"
    )
    print(
        "[safempo-diff] M-step=KL(q*||p_theta,G) with "
        "p_theta=softmax(sum_reverse_steps(logp_new-logp_old)); "
        "PPO clip is not used"
    )
    # SAFEMPO_DIFF_V11_TRUST_ANCHOR_20261007
    print(
        "[safempo-diff-v1.1] local_trust="
        f"{int(args.safempo_local_trust_enabled)} target={args.safempo_local_kl_target:g} "
        f"dual_init={args.safempo_local_dual_init:g} dual_lr={args.safempo_local_dual_lr:g}; "
        "adaptive_global_anchor="
        f"{int(args.safempo_global_anchor_enabled)} target={args.safempo_global_kl_target:g} "
        f"dual_init={args.safempo_global_dual_init:g} dual_lr={args.safempo_global_dual_lr:g}"
    )
    return adapter, trainer, device


def main() -> int:
    args = parse_args()
    if args.fixed_validation_states < 0:
        raise SystemExit("--fixed-validation-states must be >= 0")
    if args.fixed_validation_interval < 1:
        raise SystemExit("--fixed-validation-interval must be >= 1")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    adapter, trainer, device = _build_trainer(args)
    generator = torch.Generator(device=device.type).manual_seed(args.seed)
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
