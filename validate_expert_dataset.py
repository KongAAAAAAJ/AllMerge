from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from torch.utils.data import DataLoader

from expert_dataset import (
    BOOL_FEATURE_KEYS,
    FEATURE_KEYS,
    FLOAT_FEATURE_KEYS,
    build_dataset,
)


NUM_MODES = 10
HORIZON_STEPS = 8
TRAJECTORY_DIMS = 2
NUM_SEMANTICS = 4

REQUIRED_KEYS = set(FEATURE_KEYS) | {
    "expert_trajectory_xy",
    "target_mode",
    "target_semantic",
    "trajectory_time_s",
    "frame_index",
    "episode_index",
    "episode_step",
    "seed",
    "ego_index",
    "sample_index",
    "contract_ok",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate AllMerge expert shards and smoke-test Dataset/DataLoader."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=("all", "train", "val", "test"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-loader-samples", type=int, default=64)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def _shape_without_batch(array: np.ndarray) -> Tuple[int, ...]:
    return tuple(array.shape[1:])


def _validate_shard(
    shard_path: Path,
    reference_schema: Dict | None,
) -> tuple[Dict, Dict]:
    with np.load(shard_path, allow_pickle=False) as shard:
        keys = set(shard.files)
        missing = sorted(REQUIRED_KEYS - keys)
        if missing:
            raise KeyError(
                f"{shard_path} missing keys: {missing}"
            )

        sample_count = int(
            shard["expert_trajectory_xy"].shape[0]
        )

        for key in REQUIRED_KEYS:
            if shard[key].shape[0] != sample_count:
                raise ValueError(
                    f"{shard_path}: key={key} first dimension "
                    f"{shard[key].shape[0]} != {sample_count}"
                )

        if _shape_without_batch(
            shard["expert_trajectory_xy"]
        ) != (HORIZON_STEPS, TRAJECTORY_DIMS):
            raise ValueError(
                f"{shard_path}: expert_trajectory_xy shape "
                f"{shard['expert_trajectory_xy'].shape}"
            )

        if tuple(
            shard["coarse_trajectories"].shape[-3:]
        ) != (NUM_MODES, HORIZON_STEPS, TRAJECTORY_DIMS):
            raise ValueError(
                f"{shard_path}: coarse_trajectories shape "
                f"{shard['coarse_trajectories'].shape}"
            )

        if shard["mode_valid_mask"].shape[-1] != NUM_MODES:
            raise ValueError(
                f"{shard_path}: mode_valid_mask shape "
                f"{shard['mode_valid_mask'].shape}"
            )

        target_mode = np.asarray(
            shard["target_mode"],
            dtype=np.int64,
        ).reshape(-1)
        if not np.all(
            (target_mode >= 0)
            & (target_mode < NUM_MODES)
        ):
            raise ValueError(
                f"{shard_path}: target_mode out of range"
            )

        target_semantic = np.asarray(
            shard["target_semantic"],
            dtype=np.int64,
        ).reshape(-1)
        if not np.all(
            (target_semantic >= 0)
            & (target_semantic < NUM_SEMANTICS)
        ):
            raise ValueError(
                f"{shard_path}: target_semantic out of range"
            )

        for key in FLOAT_FEATURE_KEYS | {"expert_trajectory_xy"}:
            values = np.asarray(shard[key])
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{shard_path}: non-finite values in {key}"
                )

        for key in BOOL_FEATURE_KEYS | {"contract_ok"}:
            if shard[key].dtype != np.bool_:
                raise TypeError(
                    f"{shard_path}: {key} dtype={shard[key].dtype}, "
                    "expected bool"
                )

        contract = np.asarray(
            shard["contract_ok"],
            dtype=bool,
        ).reshape(-1)
        if not bool(contract.all()):
            raise ValueError(
                f"{shard_path}: contract_ok contains false"
            )

        schema = {
            key: {
                "sample_shape": list(shard[key].shape[1:]),
                "dtype": str(shard[key].dtype),
            }
            for key in shard.files
        }

        if (
            reference_schema is not None
            and schema != reference_schema
        ):
            raise ValueError(
                f"{shard_path}: schema differs from first shard"
            )

        summary = {
            "samples": sample_count,
            "target_mode_min": (
                int(target_mode.min())
                if sample_count
                else None
            ),
            "target_mode_max": (
                int(target_mode.max())
                if sample_count
                else None
            ),
            "contract_ok_ratio": (
                float(contract.mean())
                if sample_count
                else 1.0
            ),
        }
        return schema, summary


def main() -> None:
    args = parse_args()

    shard_dir = args.dataset_root / "shards"
    shard_paths = sorted(
        shard_dir.glob("shard_*.npz")
    )
    if not shard_paths:
        raise FileNotFoundError(
            f"No shards found under {shard_dir}"
        )

    reference_schema = None
    total_samples = 0
    shard_summaries = []

    for shard_path in shard_paths:
        schema, summary = _validate_shard(
            shard_path,
            reference_schema,
        )
        if reference_schema is None:
            reference_schema = schema
        total_samples += int(summary["samples"])
        shard_summaries.append(
            {"name": shard_path.name, **summary}
        )

    dataset = build_dataset(
        dataset_root=args.dataset_root,
        split=args.split,
        include_metadata=True,
        max_samples=args.max_loader_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    features, targets, _metadata = next(iter(loader))

    if tuple(targets["trajectory"].shape[1:]) != (
        HORIZON_STEPS,
        TRAJECTORY_DIMS,
    ):
        raise ValueError(
            "DataLoader trajectory batch has wrong shape: "
            f"{tuple(targets['trajectory'].shape)}"
        )

    if tuple(
        features["coarse_trajectories"].shape[-3:]
    ) != (
        NUM_MODES,
        HORIZON_STEPS,
        TRAJECTORY_DIMS,
    ):
        raise ValueError(
            "DataLoader coarse_trajectories has wrong shape: "
            f"{tuple(features['coarse_trajectories'].shape)}"
        )

    report = {
        "dataset_root": str(args.dataset_root),
        "shards": len(shard_paths),
        "samples": total_samples,
        "loader_split": args.split,
        "loader_dataset_samples_checked": len(dataset),
        "loader_first_batch_size": int(
            targets["trajectory"].shape[0]
        ),
        "first_batch_feature_shapes": {
            key: list(tensor.shape)
            for key, tensor in features.items()
        },
        "first_batch_target_shapes": {
            key: list(tensor.shape)
            for key, tensor in targets.items()
            if key != "diagnostics"
        },
        "shard_summaries": shard_summaries,
        "status": "PASS",
    }

    report_path = (
        args.report_path
        if args.report_path
        else (
            args.dataset_root
            / "reports"
            / "validation_report.json"
        )
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("VALIDATION PASS")
    print(
        f"shards={len(shard_paths)} samples={total_samples}"
    )
    print("first batch:")
    for key, tensor in features.items():
        print(
            f"  feature {key:<24s} "
            f"{tuple(tensor.shape)} {tensor.dtype}"
        )
    for key, tensor in targets.items():
        if key == "diagnostics":
            continue
        print(
            f"  target  {key:<24s} "
            f"{tuple(tensor.shape)} {tensor.dtype}"
        )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
