from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from highway_env.planner.a1_anchor_manager import A1AnchorManager
from highway_env.planner.anchor_scoring.manager import AdaptiveAnchorManager


A1_RAW_KEYS = (
    "road_lane_geometry",
    "road_lane_point_mask",
    "road_lane_mask",
    "road_current_lane_slot",
    "road_target_lane_slot",
    "road_lane_connections",
    "road_lane_meta",
)
A2_FEATURE_KEYS = (
    "ego_state",
    "agent_states",
    "agent_valid_mask",
    "map_polylines",
    "map_valid_mask",
    "target_point",
    "target_lane_polyline",
)


def _scalar_text(x):
    x = np.asarray(x).reshape(-1)[0]
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def _read_test_shards(dataset_root: Path):
    split_file = dataset_root / "splits" / "test.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing held-out split file: {split_file}")
    names = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not names:
        raise RuntimeError(f"Test split is empty: {split_file}")
    paths = [dataset_root / "shards" / name for name in names]
    missing = [str(x) for x in paths if not x.is_file()]
    if missing:
        raise FileNotFoundError("Test split references missing shards:\n" + "\n".join(missing[:20]))
    return paths


def _collect_episode_samples(paths, max_episodes: int, seed: int):
    rng = np.random.default_rng(seed)
    if max_episodes > 0 and len(paths) > max_episodes:
        # Keep the split's episode order deterministic before seeded sampling.
        indices = np.sort(rng.choice(len(paths), size=max_episodes, replace=False))
        paths = [paths[int(i)] for i in indices]

    rows = []
    scenario_counts = {}
    payloads, mus = set(), set()
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            n = int(z["ego_state"].shape[0])
            if n <= 0:
                raise RuntimeError(f"Empty episode shard: {path}")
            idx = int(rng.integers(0, n))
            required = set(A1_RAW_KEYS) | set(A2_FEATURE_KEYS) | {
                "a1_coarse_controls", "a1_coarse_trajectories",
                "a1_dense_trajectories", "a1_valid_mask",
                "coarse_controls", "coarse_trajectories", "mode_valid_mask",
                "scenario", "koopman_payload_kg", "koopman_road_mu",
            }
            missing = sorted(required - set(z.files))
            if missing:
                raise KeyError(f"{path.name}: missing fields required by real-data test: {missing}")
            row = {key: np.array(z[key][idx], copy=True)
                   for key in (set(A1_RAW_KEYS) | set(A2_FEATURE_KEYS))}
            row.update({
                "a1_coarse_controls": np.array(z["a1_coarse_controls"][idx], copy=True),
                "a1_coarse_trajectories": np.array(z["a1_coarse_trajectories"][idx], copy=True),
                "a1_dense_trajectories": np.array(z["a1_dense_trajectories"][idx], copy=True),
                "a1_valid_mask": np.array(z["a1_valid_mask"][idx], copy=True),
                "coarse_controls": np.array(z["coarse_controls"][idx], copy=True),
                "coarse_trajectories": np.array(z["coarse_trajectories"][idx], copy=True),
                "mode_valid_mask": np.array(z["mode_valid_mask"][idx], copy=True),
                "scenario": _scalar_text(z["scenario"][idx]),
                "payload_kg": float(z["koopman_payload_kg"][idx]),
                "road_mu": float(z["koopman_road_mu"][idx]),
                "episode_index": int(z["episode_index"][idx]),
                "frame_index": int(z["frame_index"][idx]),
            })
            rows.append(row)
            scenario_counts[row["scenario"]] = scenario_counts.get(row["scenario"], 0) + 1
            payloads.add(row["payload_kg"])
            mus.add(row["road_mu"])
    if not rows:
        raise RuntimeError("No test samples loaded")
    return rows, scenario_counts, sorted(payloads), sorted(mus), len(paths)


def _stack(rows, keys):
    return {key: np.stack([row[key] for row in rows], axis=0) for key in keys}


def _batches(rows, batch_size):
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def _a1_test(rows, dataset_root, config_path, batch_size, device):
    manager = A1AnchorManager(
        {"a1_config": str(config_path)},
        control_steps=40, dt=0.1, wheelbase_m=5.5,
        max_accel_mps2=2.5, max_steer_rad=0.08,
    )
    manifest = manager.validate_dataset_contract(dataset_root)

    totals = {
        "samples": 0,
        "control_max_abs_error": 0.0,
        "trajectory_max_abs_error": 0.0,
        "dense_max_abs_error": 0.0,
        "valid_mask_mismatches": 0,
        "control_samples_within_tolerance": 0,
        "trajectory_samples_within_tolerance": 0,
        "dense_samples_within_tolerance": 0,
    }
    atol = 1e-6
    for batch_rows in _batches(rows, batch_size):
        features = _stack(batch_rows, ("ego_state",) + A1_RAW_KEYS)
        features["ego_state"] = torch.as_tensor(features["ego_state"], dtype=torch.float32, device=device)
        features.update({key: value for key, value in _stack(batch_rows, A1_RAW_KEYS).items()})
        features["_a1_online_runtime"] = True

        started = time.perf_counter()
        online = manager.select(features)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        pred_u = online["coarse_controls"].detach().cpu().numpy()
        pred_xy = online["coarse_trajectories"].detach().cpu().numpy()
        pred_dense = online["a1_dense_trajectories"].detach().cpu().numpy()
        pred_mask = online["mode_valid_mask"].detach().cpu().numpy().astype(bool)

        ref_u = np.stack([r["a1_coarse_controls"] for r in batch_rows])
        ref_xy = np.stack([r["a1_coarse_trajectories"] for r in batch_rows])
        ref_dense = np.stack([r["a1_dense_trajectories"] for r in batch_rows])
        ref_mask = np.stack([r["a1_valid_mask"] for r in batch_rows]).astype(bool)

        err_u = np.max(np.abs(pred_u - ref_u), axis=(1, 2, 3))
        err_xy = np.max(np.abs(pred_xy - ref_xy), axis=(1, 2, 3))
        err_dense = np.max(np.abs(pred_dense - ref_dense), axis=(1, 2, 3))
        totals["samples"] += len(batch_rows)
        totals["control_max_abs_error"] = max(totals["control_max_abs_error"], float(err_u.max(initial=0)))
        totals["trajectory_max_abs_error"] = max(totals["trajectory_max_abs_error"], float(err_xy.max(initial=0)))
        totals["dense_max_abs_error"] = max(totals["dense_max_abs_error"], float(err_dense.max(initial=0)))
        totals["valid_mask_mismatches"] += int(np.count_nonzero(pred_mask != ref_mask))
        totals["control_samples_within_tolerance"] += int(np.count_nonzero(err_u <= atol))
        totals["trajectory_samples_within_tolerance"] += int(np.count_nonzero(err_xy <= atol))
        totals["dense_samples_within_tolerance"] += int(np.count_nonzero(err_dense <= atol))

    n = totals["samples"]
    passed = (
        totals["valid_mask_mismatches"] == 0
        and totals["control_samples_within_tolerance"] == n
        and totals["trajectory_samples_within_tolerance"] == n
        and totals["dense_samples_within_tolerance"] == n
    )
    return {
        "passed": passed,
        "dataset_schema": manifest.get("schema_version"),
        "a1_config": str(config_path),
        "tolerance_atol": atol,
        **totals,
    }


def _a2_test(rows, checkpoint, batch_size, device, top_r, num_selected):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("architecture") != "TRACE_Adaptive_Anchor_Safety_Scorer_V1":
        raise ValueError(f"Unsupported A2 checkpoint architecture: {payload.get('architecture')}")
    teacher = payload.get("teacher_config") or {}
    teacher_dimensions = {
        "ego_length_m": teacher.get("ego_length_m"),
        "ego_width_m": teacher.get("ego_width_m"),
    }
    truck_dimensions_match = (
        teacher_dimensions["ego_length_m"] is not None
        and teacher_dimensions["ego_width_m"] is not None
        and abs(float(teacher_dimensions["ego_length_m"]) - 8.0) < 1e-6
        and abs(float(teacher_dimensions["ego_width_m"]) - 2.6) < 1e-6
    )

    manager = AdaptiveAnchorManager({
        "scorer_checkpoint": str(checkpoint),
        "candidate_generation": payload.get("candidate_generation") or {},
        "num_selected": num_selected,
        "top_r": top_r,
    }, device=str(device))

    total = 0
    valid_counts = []
    latencies = []
    score_min, score_max = float("inf"), float("-inf")
    unique_selected_min = num_selected
    for batch_rows in _batches(rows, batch_size):
        arrays = _stack(batch_rows, A2_FEATURE_KEYS)
        features = {}
        for key, value in arrays.items():
            dtype = torch.bool if key in ("agent_valid_mask", "map_valid_mask") else torch.float32
            features[key] = torch.as_tensor(value, dtype=dtype, device=device)

        selected = manager.select(features)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        scores = manager.last_scores.detach()
        indices = manager.last_selected_indices.detach()
        vc = manager.last_valid_counts
        if scores.shape != (len(batch_rows), manager.builder.num_candidates):
            raise RuntimeError(f"A2 scores shape mismatch: {tuple(scores.shape)}")
        if indices.shape != (len(batch_rows), num_selected):
            raise RuntimeError(f"A2 selected index shape mismatch: {tuple(indices.shape)}")
        if not torch.isfinite(scores).all():
            raise RuntimeError("A2 produced non-finite candidate scores")
        if selected["coarse_controls"].shape != (len(batch_rows), num_selected, 40, 2):
            raise RuntimeError("A2 selected control tensor has an invalid shape")
        if selected["coarse_trajectories"].shape != (len(batch_rows), num_selected, 8, 2):
            raise RuntimeError("A2 selected trajectory tensor has an invalid shape")
        if selected["mode_valid_mask"].shape != (len(batch_rows), num_selected):
            raise RuntimeError("A2 selected validity mask has an invalid shape")
        if not torch.isfinite(selected["coarse_controls"]).all() or not torch.isfinite(selected["coarse_trajectories"]).all():
            raise RuntimeError("A2 selected non-finite anchors")
        if not selected["mode_valid_mask"].bool().all():
            raise RuntimeError("A2 selected an invalid candidate")
        unique_per_sample = [int(torch.unique(row).numel()) for row in indices]
        unique_selected_min = min(unique_selected_min, min(unique_per_sample))
        if min(vc) < num_selected:
            raise RuntimeError(f"A2 candidate pool has fewer than {num_selected} valid candidates: {vc}")
        valid_counts.extend(int(x) for x in vc)
        score_min = min(score_min, float(scores.min().cpu()))
        score_max = max(score_max, float(scores.max().cpu()))
        latencies.append(float(manager.last_latency_ms))
        total += len(batch_rows)

    if unique_selected_min != num_selected:
        raise RuntimeError(f"A2 selected duplicate indices; minimum unique count={unique_selected_min}")
    return {
        "passed": bool(truck_dimensions_match),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "candidate_count": int(manager.builder.num_candidates),
        "top_r": int(top_r),
        "num_selected": int(num_selected),
        "teacher_vehicle_dimensions_m": teacher_dimensions,
        "teacher_dimensions_match_truck_8x2_6": truck_dimensions_match,
        "samples": total,
        "valid_candidates_min": min(valid_counts),
        "valid_candidates_mean": float(np.mean(valid_counts)),
        "score_min": score_min,
        "score_max": score_max,
        "selected_unique_min": unique_selected_min,
        "batch_latency_ms_mean": float(np.mean(latencies)),
        "batch_latency_ms_p95": float(np.percentile(latencies, 95)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Real-data A1 offline/online equivalence and A2 online inference validation."
    )
    parser.add_argument("--dataset-root", required=True,
                        help="Completed A1/K2 dataset containing shards/, splits/test.txt and reports/manifest.json")
    parser.add_argument("--scorer-checkpoint", required=True,
                        help="Trained A2 scorer checkpoint, e.g. outputs/anchor_ablation/ckpt/best.pt")
    parser.add_argument("--a1-config", default="configs/anchors/a1.yaml")
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="0 tests every held-out test episode; otherwise deterministic episode cap.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-r", type=int, default=20)
    parser.add_argument("--num-selected", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="outputs/anchor_ablation/real_data_runtime_validation.json")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    checkpoint = Path(args.scorer_checkpoint).resolve()
    a1_config = Path(args.a1_config).resolve()
    for path in (root, checkpoint, a1_config):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.batch_size <= 0 or args.max_episodes < 0:
        raise ValueError("batch-size must be positive and max-episodes nonnegative")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    paths = _read_test_shards(root)
    rows, scenario_counts, payloads, mus, episode_count = _collect_episode_samples(
        paths, args.max_episodes, args.seed
    )
    print(f"[REAL DATA] dataset={root}", flush=True)
    print(f"[REAL DATA] heldout_episodes={episode_count} sampled={len(rows)} "
          f"scenarios={scenario_counts} payload_levels={len(payloads)} mu_levels={len(mus)}", flush=True)

    a1 = _a1_test(rows, root, a1_config, args.batch_size, device)
    print("[A1 REAL-DATA]", json.dumps(a1, ensure_ascii=False), flush=True)

    a2 = _a2_test(rows, checkpoint, args.batch_size, device, args.top_r, args.num_selected)
    print("[A2 ONLINE]", json.dumps(a2, ensure_ascii=False), flush=True)

    result = {
        "dataset_root": str(root),
        "test_split_episodes": len(paths),
        "sampled_episodes": episode_count,
        "sampled_scenarios": scenario_counts,
        "sampled_payload_levels_kg": payloads,
        "sampled_mu_levels": mus,
        "device": str(device),
        "a1_offline_online_equivalence": a1,
        "a2_online_inference": a2,
        "overall_passed": bool(a1["passed"] and a2["passed"]),
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[REAL-DATA VALIDATION {'PASS' if result['overall_passed'] else 'FAIL'}] {output}", flush=True)
    if not result["overall_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
