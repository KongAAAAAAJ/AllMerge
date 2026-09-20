from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .config import (
    StructuredDiffusionConfig,
    build_structured_diffusion_config,
)
from .structured_model import (
    StructuredDiffusionPlanner,
)
from .tensor_adapter import PlannerTensorAdapter
from .trajectory_spline import ClampedCubicTrajectorySpline  # STAGE_A_SPLINE_RUNTIME_V1


class DiffusionPlannerRuntime:
    """
    Persistent inference runtime.

    Instantiate once per environment, never once per planner step.
    """

    def __init__(
        self,
        runtime_config: Optional[dict] = None,
    ) -> None:
        runtime_config = dict(
            runtime_config or {}
        )

        device_name = runtime_config.get(
            "device",
            "cuda"
            if torch.cuda.is_available()
            else "cpu",
        )

        self.device = torch.device(
            device_name
        )

        model_overrides = dict(
            runtime_config.get(
                "model",
                {},
            )
        )

        self.config = (
            build_structured_diffusion_config(
                **model_overrides
            )
        )

        self.adapter = PlannerTensorAdapter(
            self.config,
            self.device,
        )

        self.model = (
            StructuredDiffusionPlanner(
                self.config,
                self.adapter,
            )
            .to(self.device)
        )

        self.checkpoint = (
            runtime_config.get(
                "checkpoint"
            )
        )

        self.strict_checkpoint = bool(
            runtime_config.get(
                "strict_checkpoint",
                False,
            )
        )

        self.allow_random_weights = bool(
            runtime_config.get(
                "allow_random_weights",
                True,
            )
        )

        self.deterministic_seed = (
            runtime_config.get(
                "deterministic_seed",
                0,
            )
        )

        # STAGE_A_SPLINE_RUNTIME_V1
        self.diffusion_spline_enabled = bool(
            runtime_config.get("diffusion_spline_enabled", False)
        )
        self.diffusion_execution_enabled = bool(
            runtime_config.get("diffusion_execution_enabled", False)
        )
        self.spline_dense_dt = float(
            runtime_config.get("spline_dense_dt", 0.1)
        )
        if self.diffusion_execution_enabled and not self.diffusion_spline_enabled:
            raise ValueError(
                "diffusion_execution_enabled=True requires "
                "diffusion_spline_enabled=True"
            )

        self.trajectory_spline = None
        if self.diffusion_spline_enabled:
            self.trajectory_spline = ClampedCubicTrajectorySpline(
                horizon_s=(
                    float(self.config.horizon_steps)
                    * float(self.config.trajectory_dt)
                ),
                sparse_dt=float(self.config.trajectory_dt),
                dense_dt=self.spline_dense_dt,
            ).to(self.device)

        self._load_checkpoint_if_needed()

        self.model.eval()

        self.last_latency_ms = None

    def _load_checkpoint_if_needed(
        self,
    ) -> None:
        if not self.checkpoint:
            if not self.allow_random_weights:
                raise RuntimeError(
                    "No Diffusion checkpoint configured and "
                    "allow_random_weights=False"
                )
            return

        path = Path(
            self.checkpoint
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Diffusion checkpoint not found: {path}"
            )

        payload = torch.load(
            path,
            map_location="cpu",
        )

        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        elif isinstance(payload, dict) and "planner_state_dict" in payload:
            # EVAL_VIZ_V1: native W2 pretraining checkpoint.
            state_dict = payload["planner_state_dict"]
        else:
            state_dict = payload

        missing, unexpected = (
            self.model.load_state_dict(
                state_dict,
                strict=self.strict_checkpoint,
            )
        )

        if (
            not self.strict_checkpoint
            and (missing or unexpected)
        ):
            print(
                "[DiffusionPlannerRuntime] "
                f"checkpoint loaded non-strictly; "
                f"missing={len(missing)}, "
                f"unexpected={len(unexpected)}"
            )

    @torch.no_grad()
    def infer(
        self,
        numpy_features: Dict[
            str,
            np.ndarray,
        ],
    ) -> Dict[str, np.ndarray]:
        torch_features = (
            self.adapter.to_torch(
                numpy_features
            )
        )

        generator = None

        if self.deterministic_seed is not None:
            generator = torch.Generator(
                device=self.device
            )
            generator.manual_seed(
                int(
                    self.deterministic_seed
                )
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        start = time.perf_counter()

        output = self.model.infer_multimodal(
            torch_features,
            generator=generator,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        self.last_latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        # STAGE_A_SPLINE_RUNTIME_V1
        # Spline is a selected-mode post-process only. It never changes the
        # sparse [B,8,2] Diffusion output or candidate mode selection.
        if self.diffusion_spline_enabled:
            if self.trajectory_spline is None:
                raise RuntimeError("Spline runtime was enabled without a decoder")

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            spline_start = time.perf_counter()

            selected_sparse = output["trajectory"]
            start_xy = torch.zeros_like(selected_sparse[..., 0, :])
            start_velocity_xy = torch_features["ego_state"][..., 0:2]

            dense_position, dense_velocity, dense_acceleration = (
                self.trajectory_spline.evaluate(
                    selected_sparse,
                    start_xy=start_xy,
                    start_velocity_xy=start_velocity_xy,
                )
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            spline_decode_ms = (time.perf_counter() - spline_start) * 1000.0

            dense_at_sparse = dense_position.index_select(
                -2,
                self.trajectory_spline.sparse_dense_indices,
            )
            waypoint_error = (dense_at_sparse - selected_sparse).abs().amax()

            vx = dense_velocity[..., 0]
            vy = dense_velocity[..., 1]
            ax = dense_acceleration[..., 0]
            ay = dense_acceleration[..., 1]
            speed_sq = vx.square() + vy.square()
            cross = vx * ay - vy * ax
            curvature = torch.where(
                speed_sq > 1e-4,
                cross / (speed_sq.sqrt() * speed_sq + 1e-6),
                torch.zeros_like(cross),
            )
            delta_curvature = curvature[..., 1:] - curvature[..., :-1]

            output["trajectory_dense"] = dense_position
            output["trajectory_dense_velocity"] = dense_velocity
            output["trajectory_dense_acceleration"] = dense_acceleration
            output["trajectory_dense_curvature"] = curvature
            output["trajectory_dense_time_s"] = (
                self.trajectory_spline.dense_time_s.to(
                    device=dense_position.device,
                    dtype=dense_position.dtype,
                )
            )
            output["spline_decode_ms"] = torch.as_tensor(
                spline_decode_ms,
                dtype=dense_position.dtype,
                device=dense_position.device,
            )
            output["spline_sparse_point_count"] = torch.as_tensor(
                self.config.horizon_steps,
                dtype=torch.int32,
                device=dense_position.device,
            )
            output["spline_dense_point_count"] = torch.as_tensor(
                dense_position.shape[-2],
                dtype=torch.int32,
                device=dense_position.device,
            )
            output["spline_dense_dt"] = torch.as_tensor(
                self.spline_dense_dt,
                dtype=dense_position.dtype,
                device=dense_position.device,
            )
            output["spline_max_abs_waypoint_interpolation_error"] = waypoint_error
            output["spline_max_curvature"] = curvature.abs().amax()
            output["spline_mean_abs_curvature"] = curvature.abs().mean()
            output["spline_max_abs_delta_curvature"] = (
                delta_curvature.abs().amax()
                if delta_curvature.numel()
                else torch.zeros((), device=dense_position.device)
            )

        numpy_output = {
            key: value.detach().cpu().numpy()
            for key, value
            in output.items()
        }

        numpy_output[
            "latency_ms"
        ] = np.asarray(
            self.last_latency_ms,
            dtype=np.float32,
        )

        numpy_output[
            "trajectory_time_s"
        ] = (
            np.arange(
                1,
                self.config.horizon_steps + 1,
                dtype=np.float32,
            )
            * float(self.config.trajectory_dt)
        )

        return numpy_output
