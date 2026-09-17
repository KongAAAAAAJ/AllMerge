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

        if isinstance(
            payload,
            dict,
        ) and "state_dict" in payload:
            state_dict = payload[
                "state_dict"
            ]
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

        return numpy_output
