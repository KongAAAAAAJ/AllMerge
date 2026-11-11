"""Standalone control-diffusion runtime; old DiffusionPlannerRuntime stays unchanged."""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import torch

from scripts.training.train_diffusion_control import load_control_checkpoint
from highway_env.planner.diffusion.common.execution import K2_CONTEXT
from highway_env.planner.diffusion.common.tensor_adapter import PlannerTensorAdapter


class ControlDiffusionRuntime:
    def __init__(self, checkpoint, device='auto', deterministic_seed=0, inference_steps=None):
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available()
                                   else 'cpu' if device == 'auto' else device)
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        self.model, payload = load_control_checkpoint(path, device='cpu')
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
        context = {}
        if self.model.vehicle_model.kind == 'koopman_k2':
            absent = [k for k in K2_CONTEXT if k not in numpy_features]
            if absent:
                raise KeyError(f'K2 runtime requires synchronized observed context: {absent}')
            context = {k: torch.as_tensor(numpy_features[k], dtype=torch.float32,
                                           device=self.device) for k in K2_CONTEXT}
        prediction = self.model.infer_multimodal(features, generator=generator,
                                                 num_steps=self.inference_steps,
                                                 context=context)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.last_latency_ms = (time.perf_counter() - start) * 1000
        result = {k: v.detach().cpu().numpy() for k, v in prediction.items()}
        result['latency_ms'] = np.asarray(self.last_latency_ms, dtype=np.float32)
        result['trajectory_time_s'] = np.arange(1, 9, dtype=np.float32) * .5
        result['trajectory_dense_time_s'] = np.arange(1, 41, dtype=np.float32) * .1
        return result
