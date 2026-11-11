"""Standalone control-diffusion runtime; old DiffusionPlannerRuntime stays unchanged."""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import torch

from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner
from highway_env.planner.diffusion.common.tensor_adapter import PlannerTensorAdapter


class ControlDiffusionRuntime:
    def __init__(self, checkpoint, device='auto', deterministic_seed=0, inference_steps=None):
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available()
                                   else 'cpu' if device == 'auto' else device)
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        self.model = ControlDiffusionPlanner(payload['model_config'], payload['control_config'])
        self.model.load_state_dict(payload['model_state'])
        self.model.to(self.device).eval()
        self.adapter = PlannerTensorAdapter(self.model.scene_config, self.device)
        self.deterministic_seed = deterministic_seed
        self.inference_steps = inference_steps
        self.last_latency_ms = None

    @torch.no_grad()
    def infer(self, numpy_features):
        features = self.adapter.to_torch(numpy_features)
        generator = None
        if self.deterministic_seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(self.deterministic_seed))
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        prediction = self.model.infer_multimodal(features, generator=generator,
                                                 num_steps=self.inference_steps)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.last_latency_ms = (time.perf_counter() - start) * 1000
        result = {k: v.detach().cpu().numpy() for k, v in prediction.items()}
        result['latency_ms'] = np.asarray(self.last_latency_ms, dtype=np.float32)
        result['trajectory_time_s'] = np.arange(1, 9, dtype=np.float32) * .5
        result['trajectory_dense_time_s'] = np.arange(1, 41, dtype=np.float32) * .1
        return result
