"""Online A1 anchor generation using the same builder as offline dataset enrichment.

Offline training consumes the paired A1 anchors already stored in each shard.
Online inference rebuilds those anchors from the current ego state and the
full-resolution road geometry exported by the live environment.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import numpy as np
import torch
import yaml

from highway_env.planner.a1_anchor_builder import A1AnchorBuilder


A1_ROAD_GEOMETRY_KEYS = (
    "road_lane_geometry",
    "road_lane_point_mask",
    "road_lane_mask",
    "road_current_lane_slot",
    "road_target_lane_slot",
    "road_lane_connections",
    "road_lane_meta",
)


class A1AnchorManager:
    """Shared offline/online A1 adapter with a single ``select`` contract."""

    runtime_kind = "a1"
    NUM_ANCHORS = 10
    CONTROL_STEPS = 40
    SPARSE_STEPS = 8

    def __init__(
        self,
        config=None,
        *,
        control_steps: int = 40,
        dt: float = 0.1,
        wheelbase_m: float = 5.5,
        max_accel_mps2: float = 2.5,
        max_steer_rad: float = 0.08,
    ):
        self.config = dict(config or {})
        if self.NUM_ANCHORS <= 0:
            raise RuntimeError("Invalid A1 anchor count")
        if int(control_steps) != self.CONTROL_STEPS:
            raise ValueError(f"A1 requires {self.CONTROL_STEPS} control steps")
        self.control_steps = int(control_steps)
        self.dt = float(dt)

        cfg_value = self.config.get(
            "a1_config", self.config.get("a1_config_path", "configs/anchors/a1.yaml")
        )
        from highway_env.planner.diffusion.config.experiment import resolve_repo_file
        cfg_path = resolve_repo_file(cfg_value)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"A1 configuration not found: {cfg_path}")
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        a1_cfg = raw.get("a1")
        if not isinstance(a1_cfg, dict):
            raise ValueError("A1 config must contain an 'a1' mapping")
        self.builder_config = dict(a1_cfg)
        if int(self.builder_config.get("control_steps", -1)) != self.CONTROL_STEPS:
            raise ValueError("A1 control_steps must equal the control-diffusion horizon (40)")
        if not np.isclose(float(self.builder_config.get("dt", -1)), self.dt, atol=1e-8):
            raise ValueError(
                f"A1 dt={self.builder_config.get('dt')} does not match diffusion dt={self.dt}"
            )
        if not np.isclose(float(self.builder_config.get("wheelbase_m", -1)),
                          float(wheelbase_m), atol=1e-6):
            raise ValueError("A1 wheelbase_m differs from the control-diffusion model")
        if not np.isclose(float(self.builder_config.get("max_accel_mps2", -1)),
                          float(max_accel_mps2), atol=1e-6):
            raise ValueError("A1 max_accel_mps2 differs from the control-diffusion model")
        if not np.isclose(float(self.builder_config.get("max_steer_rad", -1)),
                          float(max_steer_rad), atol=1e-6):
            raise ValueError("A1 max_steer_rad differs from the control-diffusion model")
        self.road_geometry_config = dict(raw.get("road_geometry") or {})
        if not self.road_geometry_config:
            raise ValueError(
                "A1 config must define road_geometry so online and offline geometry sampling match"
            )
        self.config_path = str(cfg_path)
        self.builder = A1AnchorBuilder(self.builder_config)
        self.last_latency_ms = 0.0
        self.last_valid_counts = None
        self.last_invalid_reasons = None

    @staticmethod
    def _numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def validate_dataset_contract(self, dataset_root):
        """Reject offline A1 training data built with different A1/map settings."""
        dataset_root = Path(dataset_root)
        manifest_path = dataset_root / "reports" / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"A1 training requires the final dataset manifest: {manifest_path}"
            )
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != "trace_expert_road_geometry_a1_k2_v2"
                or not manifest.get("complete")):
            raise ValueError("Selected dataset is not a completed TRACE A1/K2 dataset")
        recorded_a1 = manifest.get("a1_anchor_parameters") or {}
        for key, expected in self.builder_config.items():
            if key not in recorded_a1:
                raise ValueError(f"A1 dataset manifest is missing parameter {key!r}")
            actual = recorded_a1[key]
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                if not np.isclose(float(actual), float(expected), atol=1e-8, rtol=0.0):
                    raise ValueError(
                        f"A1 dataset/config mismatch for {key}: dataset={actual}, "
                        f"runtime={expected}"
                    )
            elif actual != expected:
                raise ValueError(
                    f"A1 dataset/config mismatch for {key}: dataset={actual!r}, "
                    f"runtime={expected!r}"
                )
        recorded_road = manifest.get("road_geometry") or {}
        for key, expected in self.road_geometry_config.items():
            if key not in recorded_road:
                raise ValueError(f"A1 dataset manifest is missing road_geometry.{key}")
            if not np.isclose(float(recorded_road[key]), float(expected), atol=1e-8, rtol=0.0):
                raise ValueError(
                    f"A1 road geometry mismatch for {key}: dataset={recorded_road[key]}, "
                    f"runtime={expected}"
                )
        return manifest

    def _validate_offline(self, features):
        required = ("coarse_controls", "coarse_trajectories", "mode_valid_mask")
        missing = [key for key in required if key not in features]
        if missing:
            raise RuntimeError(
                "A1 offline training requires prebuilt paired anchors in the dataset; "
                f"missing {missing}. Build the A1 dataset before training."
            )
        controls = features["coarse_controls"]
        trajectories = features["coarse_trajectories"]
        valid = features["mode_valid_mask"]
        if tuple(controls.shape[1:]) != (10, 40, 2):
            raise ValueError(f"A1 offline controls must be [B,10,40,2], got {tuple(controls.shape)}")
        if tuple(trajectories.shape[1:]) != (10, 8, 2):
            raise ValueError(
                f"A1 offline trajectories must be [B,10,8,2], got {tuple(trajectories.shape)}"
            )
        if tuple(valid.shape[1:]) != (10,):
            raise ValueError(f"A1 offline validity mask must be [B,10], got {tuple(valid.shape)}")
        if not torch.as_tensor(valid).bool().any(dim=1).all():
            raise ValueError("A1 offline dataset contains a sample with no valid anchor")
        return features

    @torch.no_grad()
    def select(self, features):
        """Use stored anchors offline, or rebuild them from live road geometry.

        The runtime explicitly sets ``_a1_online_runtime``. This prevents a
        stale legacy coarse-anchor field from being mistaken for the live A1
        result when the environment has not supplied current road geometry.
        """
        online = bool(features.get("_a1_online_runtime", False))
        if not online:
            return self._validate_offline(features)

        missing = [key for key in A1_ROAD_GEOMETRY_KEYS if key not in features]
        if missing:
            raise KeyError(f"Online A1 generation is missing live road geometry: {missing}")
        if "ego_state" not in features:
            raise KeyError("Online A1 generation requires ego_state")

        started = time.perf_counter()
        arrays = {key: self._numpy(features[key]) for key in A1_ROAD_GEOMETRY_KEYS}
        ego = self._numpy(features["ego_state"])
        if ego.ndim != 2 or ego.shape[1] < 1:
            raise ValueError(f"Online A1 ego_state must be [B,D], got {ego.shape}")
        batch_size = int(ego.shape[0])
        for key, value in arrays.items():
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"Online A1 road field {key} batch={value.shape[0]} != ego batch={batch_size}"
                )

        built = []
        for batch_index in range(batch_size):
            sample = {"ego_state": ego[batch_index]}
            sample.update({key: value[batch_index] for key, value in arrays.items()})
            built.append(self.builder.build(sample))

        device = features["ego_state"].device
        dtype = features["ego_state"].dtype
        controls = torch.as_tensor(
            np.stack([row["a1_coarse_controls"] for row in built]),
            device=device, dtype=dtype,
        )
        trajectories = torch.as_tensor(
            np.stack([row["a1_coarse_trajectories"] for row in built]),
            device=device, dtype=dtype,
        )
        valid = torch.as_tensor(
            np.stack([row["a1_valid_mask"] for row in built]),
            device=device, dtype=torch.bool,
        )
        valid_counts = valid.sum(dim=1)
        if (valid_counts <= 0).any():
            bad = torch.nonzero(valid_counts <= 0).flatten().cpu().tolist()
            raise RuntimeError(f"Online A1 produced no valid anchors for batch items {bad}")

        selected = {
            key: value for key, value in features.items()
            if key not in A1_ROAD_GEOMETRY_KEYS and key != "_a1_online_runtime"
        }
        selected["coarse_controls"] = controls
        selected["coarse_trajectories"] = trajectories
        selected["mode_valid_mask"] = valid
        selected["a1_dense_trajectories"] = torch.as_tensor(
            np.stack([row["a1_dense_trajectories"] for row in built]),
            device=device, dtype=dtype,
        )
        self.last_valid_counts = valid_counts.detach().cpu().tolist()
        self.last_invalid_reasons = [
            row["a1_invalid_reasons"].tolist() for row in built
        ]
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return selected
