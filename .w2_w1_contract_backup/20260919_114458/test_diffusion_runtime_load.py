"""Load an exported pretraining checkpoint through the frozen DiffusionPlannerRuntime."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data_pipeline.expert_dataset import make_dataloader
from highway_env.planner.diffusion.runtime import DiffusionPlannerRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="Runtime checkpoint load/inference test.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="data/expert/allmerge_planner_v1")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    model_config = dict(payload.get("model_config") or {}) if isinstance(payload, dict) else {}
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    runtime = DiffusionPlannerRuntime({
        "device": device,
        "checkpoint": str(checkpoint),
        "strict_checkpoint": True,
        "allow_random_weights": False,
        "model": model_config,
        "deterministic_seed": 0,
    })

    loader = make_dataloader(
        args.dataset_root,
        split=args.split,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        validate_samples=True,
    )
    batch = next(iter(loader))
    numpy_features = {
        key: value.detach().cpu().numpy()
        for key, value in batch["features"].items()
    }
    output = runtime.infer(numpy_features)
    assert tuple(output["trajectory"].shape) == (1, runtime.config.horizon_steps, 2)
    assert tuple(output["trajectory_candidates"].shape) == (
        1,
        runtime.config.num_modes,
        runtime.config.horizon_steps,
        2,
    )
    print("PASS: DiffusionPlannerRuntime loaded exported W2 checkpoint")
    print(f"trajectory={output['trajectory'].shape}")
    print(f"candidates={output['trajectory_candidates'].shape}")
    print(f"latency_ms={float(output['latency_ms']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
