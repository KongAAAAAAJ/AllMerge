"""Configuration contract for the single TRACE Polynomial/TruckSim expert pipeline."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/data/expert_collection.yaml"
SCENARIOS = ("straight", "curved", "merge_in", "merge_out")


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _positive_number(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def _positive_int(value, name, *, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be an integer >= {0 if allow_zero else 1}")
    return value


def load_config(filename=DEFAULT_CONFIG):
    path = resolve_path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Expert collection config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Expert collection YAML must be a mapping")
    cfg = deepcopy(raw)

    # This project has one supported expert data-generation process. Kinematic
    # legacy collection and the old mode/anchor-labelled schema are rejected.
    if cfg.get("vehicle_model") != "trucksim":
        raise ValueError("The unified TRACE collector requires vehicle_model: trucksim")
    scenarios = cfg.get("scenarios")
    if (not isinstance(scenarios, list) or not scenarios or
            any(x not in SCENARIOS for x in scenarios) or len(set(scenarios)) != len(scenarios)):
        raise ValueError(f"scenarios must be unique members of {SCENARIOS}")
    cfg["scenarios"] = list(scenarios)

    koopman = cfg.get("koopman_collection_config", "configs/koopman/collect.yaml")
    cfg["koopman_collection_config"] = str(resolve_path(koopman))
    conditions = cfg.setdefault("conditions", {})
    if not isinstance(conditions, dict):
        raise ValueError("conditions must be a mapping")
    conditions.setdefault("selection", "round_robin_payload_friction")
    if conditions["selection"] != "round_robin_payload_friction":
        raise ValueError("Only round_robin_payload_friction condition selection is supported")
    conditions["copy_instances_beside_source_dll"] = bool(
        conditions.get("copy_instances_beside_source_dll", True))

    collection = cfg.setdefault("collection", {})
    if not isinstance(collection, dict):
        raise ValueError("collection must be a mapping")
    defaults = {
        "dataset_name": "expert_trucksim_k2_raw_100k",
        "output_root": "outputs/expert_dataset",
        "target_samples": 100000,
        "min_episodes": 0,
        "episode_duration_s": 12.0,
        "min_contiguous_control_steps": 40,
        "max_episodes": 10000,
        "timeout_s": 300.0,
        "worker_retries": 2,
        "resume": True,
        "start_seed": 2026,
        "start_episode_index": 0,
        "group_action": 3,
        "max_consecutive_empty_episodes": 20,
        "train_split_ratio": 0.8,
        "val_split_ratio": 0.1,
        "test_split_ratio": 0.1,
        "split_seed": 0,
    }
    for key, value in defaults.items():
        collection.setdefault(key, value)
    for key in ("target_samples", "max_episodes", "worker_retries", "max_consecutive_empty_episodes"):
        collection[key] = _positive_int(collection[key], f"collection.{key}")
    for key in ("start_seed", "start_episode_index", "group_action", "split_seed", "min_episodes"):
        collection[key] = _positive_int(collection[key], f"collection.{key}", allow_zero=True)
    collection['min_contiguous_control_steps'] = _positive_int(
        collection['min_contiguous_control_steps'], 'collection.min_contiguous_control_steps', allow_zero=True)
    if collection["group_action"] not in (0, 1, 2, 3):
        raise ValueError("collection.group_action must be in [0,3]")
    for key in ("episode_duration_s", "timeout_s"):
        collection[key] = _positive_number(collection[key], f"collection.{key}")
    for key in ("train_split_ratio", "val_split_ratio", "test_split_ratio"):
        collection[key] = _positive_number(collection[key], f"collection.{key}", allow_zero=True)
    if not math.isclose(sum(collection[k] for k in ("train_split_ratio", "val_split_ratio", "test_split_ratio")), 1.0, abs_tol=1e-8):
        raise ValueError("train/val/test split ratios must sum to 1")
    name = collection["dataset_name"]
    if not isinstance(name, str) or not name.strip() or any(x in name for x in ("/", "\\", ":")) or name in (".", ".."):
        raise ValueError("collection.dataset_name must be a plain directory name")
    collection["output_root"] = str(resolve_path(collection["output_root"]))
    if not isinstance(collection['resume'], bool):
        raise ValueError('collection.resume must be a YAML boolean')

    dims = cfg.setdefault("vehicle_dimensions", {"length_m": 8.0, "width_m": 2.6})
    road = cfg.setdefault("road_geometry", {})
    dims.setdefault("length_m", 8.0); dims.setdefault("width_m", 2.6)
    road_defaults = {"forward_min_m": 120.0, "forward_max_m": 300.0,
                     "point_spacing_m": 1.5, "max_lanes": 16, "max_points_per_lane": 256}
    for key, value in road_defaults.items(): road.setdefault(key, value)
    for key in ("length_m", "width_m"):
        dims[key] = _positive_number(dims[key], f"vehicle_dimensions.{key}")
    for key in ("forward_min_m", "forward_max_m", "point_spacing_m"):
        road[key] = _positive_number(road[key], f"road_geometry.{key}")
    if road["forward_max_m"] < road["forward_min_m"]:
        raise ValueError("road_geometry.forward_max_m must be >= forward_min_m")
    for key in ("max_lanes", "max_points_per_lane"):
        road[key] = _positive_int(road[key], f"road_geometry.{key}")

    # Remove stale configuration branches that belonged to the retired
    # standard collector or the former second raw-collector entry point.
    for key in ("raw_expert_collection", "samples_per_shard", "vehicle_model_legacy"):
        cfg.pop(key, None)
    return cfg


def collection_args(cfg, *, config_path=DEFAULT_CONFIG, check_only=False, overrides=None):
    c = dict(cfg["collection"])
    c.update(overrides or {})
    return argparse.Namespace(
        config_path=resolve_path(config_path),
        check_only=bool(check_only),
        vehicle_model="trucksim",
        scenarios=list(cfg["scenarios"]),
        koopman_config=Path(cfg["koopman_collection_config"]),
        copy_trucksim_dll_instances=bool(cfg["conditions"]["copy_instances_beside_source_dll"]),
        ego_length_m=float(cfg["vehicle_dimensions"]["length_m"]),
        ego_width_m=float(cfg["vehicle_dimensions"]["width_m"]),
        road_geometry=dict(cfg["road_geometry"]),
        dataset_name=c["dataset_name"],
        output_root=Path(c["output_root"]),
        target_samples=int(c["target_samples"]),
        min_episodes=int(c["min_episodes"]),
        trucksim_model_id=c.get("trucksim_model_id"),
        duration=float(c["episode_duration_s"]),
        min_contiguous_control_steps=int(c['min_contiguous_control_steps']),
        max_episodes=int(c["max_episodes"]),
        timeout_s=float(c["timeout_s"]),
        retries=int(c["worker_retries"]),
        resume=bool(c["resume"]),
        start_seed=int(c["start_seed"]),
        start_episode_index=int(c["start_episode_index"]),
        group_action=int(c["group_action"]),
        max_consecutive_empty_episodes=int(c["max_consecutive_empty_episodes"]),
        train_split_ratio=float(c["train_split_ratio"]),
        val_split_ratio=float(c["val_split_ratio"]),
        test_split_ratio=float(c["test_split_ratio"]),
        split_seed=int(c["split_seed"]),
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Unified TRACE Polynomial/TruckSim expert data collection")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--target-samples", type=int)
    p.add_argument("--min-episodes", type=int)
    p.add_argument("--trucksim-model-id", default=None)
    p.add_argument("--dataset-name")
    p.add_argument("--output-root")
    p.add_argument("--duration", type=float)
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--timeout", type=float)
    p.add_argument("--retries", type=int)
    p.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    p.add_argument("--resume", dest="resume", action="store_true", default=None)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    overrides = {"target_samples": a.target_samples, "dataset_name": a.dataset_name,
                 "output_root": resolve_path(a.output_root) if a.output_root else None,
                 "episode_duration_s": a.duration, "max_episodes": a.max_episodes,
                 "min_episodes": a.min_episodes, "trucksim_model_id": a.trucksim_model_id,
                 "timeout_s": a.timeout, "worker_retries": a.retries, "resume": a.resume}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    for key in ("target_samples", "max_episodes", "worker_retries"):
        if key in overrides and int(overrides[key]) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if a.min_episodes is not None and a.min_episodes < 0:
        p.error("--min-episodes must be nonnegative")
    for key in ("episode_duration_s", "timeout_s"):
        if key in overrides and float(overrides[key]) <= 0:
            p.error(f"invalid positive value for {key}")
    if "dataset_name" in overrides:
        name = str(overrides["dataset_name"])
        if not name.strip() or any(x in name for x in ("/", "\\", ":")) or name in (".", ".."):
            p.error("--dataset-name must be a plain directory name")
    # Scenario override is applied as a one-item batch; all remains YAML-driven.
    if a.scenario != "all": cfg["scenarios"] = [a.scenario]
    return collection_args(cfg, config_path=a.config, check_only=a.check_only, overrides=overrides)
