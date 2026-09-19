from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

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

FEATURE_KEYS = (*FLOAT_KEYS, *BOOL_KEYS)


def _lookup_array(data: np.lib.npyio.NpzFile, key: str):
    for candidate in (key, f"feature_{key}", f"features_{key}", f"features/{key}"):
        if candidate in data:
            return data[candidate]
    return None


def load_feature_arrays(path: Path) -> Dict[str, np.ndarray]:
    """Load W1-style feature arrays without imposing a second dataset format."""
    with np.load(path, allow_pickle=True) as data:
        arrays = {key: _lookup_array(data, key) for key in FEATURE_KEYS}
        if all(value is not None for value in arrays.values()):
            return arrays
        if "features" in data:
            records = data["features"]
            if records.dtype == object and len(records):
                return {
                    key: np.stack([record.item().get(key) if hasattr(record, "item") else record[key]
                                   for record in records], axis=0)
                    for key in FEATURE_KEYS
                }
        missing = [key for key, value in arrays.items() if value is None]
        raise KeyError(
            f"Could not find planner feature arrays {missing} in {path}. "
            "Reuse the W2 dataset loader instead of converting the dataset; "
            "or expose the nine planner feature keys directly in the shard."
        )


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
        numpy_batch = {key: value[indices] for key, value in arrays.items()}
        yield adapter.to_torch(numpy_batch)
        start = (start + batch_size) % n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AllMerge migration-first GRPO smoke trainer")
    parser.add_argument("--checkpoint", required=True, type=Path, help="W2 pretrained checkpoint")
    parser.add_argument("--dataset-shard", required=True, type=Path, help="one W1 .npz shard")
    parser.add_argument("--steps", type=int, default=3, choices=(3, 10, 100))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--reward-fn", default="auto", help="module:function or auto")
    parser.add_argument("--fake-reward", action="store_true", help="smoke only; bypass W4 reward")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/grpo_smoke.pt"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
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
        ),
    )
    print(f"[grpo] trainable_params={trainer.trainable_parameter_count:,}")

    arrays = load_feature_arrays(args.dataset_shard)
    batches = iter_batches(arrays, batch_size=args.batch_size, adapter=adapter)
    generator = torch.Generator(device=device.type).manual_seed(args.seed)
    metrics = {}
    for step in range(1, args.steps + 1):
        metrics = trainer.train_step(next(batches), generator=generator)
        compact = " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
        print(f"[step {step:03d}/{args.steps}] {compact}")

    save_grpo_checkpoint(
        args.output,
        model=model,
        optimizer=trainer.optimizer,
        step=args.steps,
        config=trainer.config,
        metrics=metrics,
    )
    print(f"[OK] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
