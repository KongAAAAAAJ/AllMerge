"""Generate A1 anchors from current observed scene and raw road geometry."""
from __future__ import annotations

import time

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
    """Online A1 adapter with a single ``select`` contract."""

    runtime_kind = "a1"
    CONTROL_STEPS = 40
    SPARSE_STEPS = 8

    def __init__(
        self,
        config=None,
        *,
        num_selected: int | None = None,
        control_steps: int = 40,
        dt: float = 0.1,
        wheelbase_m: float = 5.5,
        max_accel_mps2: float = 2.5,
        max_steer_rad: float = 0.08,
    ):
        self.config = dict(config or {})
        self.num_selected = int(num_selected if num_selected is not None else self.config.get("num_selected", 10))
        if self.num_selected <= 0:
            raise ValueError("A1 num_selected must be positive")
        self.config["num_selected"] = self.num_selected
        if int(control_steps) != self.CONTROL_STEPS:
            raise ValueError(f"A1 requires {self.CONTROL_STEPS} control steps")
        self.control_steps = int(control_steps)
        self.dt = float(dt)

        cfg_value = self.config.get(
            "a1_config", self.config.get("a1_config_path", "configs/anchors/a1.yaml")
        )
        from highway_env.planner.diffusion.config.experiment import resolve_repo_file
        cfg_path = resolve_repo_file(cfg_value)
        raw = self.config.get("resolved_a1")
        if raw is None:
            if not cfg_path.is_file():
                raise FileNotFoundError(f"A1 configuration not found: {cfg_path}")
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        a1_cfg = raw.get("a1")
        if not isinstance(a1_cfg, dict):
            raise ValueError("A1 config must contain an 'a1' mapping")
        self.builder_config = dict(a1_cfg)
        self.builder_config["num_selected"] = self.num_selected
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
                "A1 config must define road_geometry so training and live geometry sampling match"
            )
        self.config_path = str(cfg_path)
        self.builder = A1AnchorBuilder(self.builder_config)
        self.builder_config['candidate_generation'] = self.builder.candidate_builder.resolved_config
        self.config['resolved_a1'] = {
            'a1': self.builder_config, 'road_geometry': self.road_geometry_config,
        }
        self.last_latency_ms = 0.0
        self.last_valid_counts = None
        self.last_invalid_reasons = None

    @staticmethod
    def _numpy(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @torch.no_grad()
    def select(self, features):
        """Build K anchors online; stored anchor tensors never replace geometry."""
        missing = [key for key in A1_ROAD_GEOMETRY_KEYS if key not in features]
        if missing:
            raise KeyError(f"Online A1 generation is missing live road geometry: {missing}")
        if "ego_state" not in features:
            raise KeyError("Online A1 generation requires ego_state")

        started = time.perf_counter()
        ego = self._numpy(features["ego_state"])
        batch_size = int(ego.shape[0])
        built = []
        if "a2_candidate_controls" in features:
            from highway_env.planner.candidate_cache import CANDIDATE_KEYS
            pools = {key: self._numpy(features[value]) for key, value in CANDIDATE_KEYS.items()}
            for batch_index in range(batch_size):
                pool = {key: value[batch_index] for key, value in pools.items()}
                built.append(self.builder.select_pool(pool, ego[batch_index]))
        else:
            arrays = {key: self._numpy(features[key]) for key in A1_ROAD_GEOMETRY_KEYS}
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
        if not torch.isfinite(controls).all() or not torch.isfinite(trajectories).all():
            raise ValueError('Online A1 generated nonfinite controls or trajectories')
        if (valid_counts != self.num_selected).any():
            bad = torch.nonzero(valid_counts != self.num_selected).flatten().cpu().tolist()
            raise RuntimeError(f"Online A1 did not produce K valid anchors for batch items {bad}")

        selected = {
            key: value for key, value in features.items()
            if key not in A1_ROAD_GEOMETRY_KEYS and not key.startswith("a2_candidate_")
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
