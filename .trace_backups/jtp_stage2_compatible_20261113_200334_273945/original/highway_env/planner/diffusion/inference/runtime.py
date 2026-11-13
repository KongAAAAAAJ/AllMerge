"""Stable planner runtime entry point with trajectory/control mode selection."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
from .trajectory import TrajectoryDiffusionRuntime


class DiffusionPlannerRuntime(TrajectoryDiffusionRuntime):
    """Create once per environment; select the configured inference backend."""

    def __init__(self, runtime_config: Optional[dict] = None) -> None:
        # TRACE_D1_RUNTIME_MODE_SWITCH_V3: read YAML without changing legacy kwargs.
        if isinstance(runtime_config, (str, Path)):
            from highway_env.planner.diffusion.config.experiment import load_mode_yaml, runtime_settings
            runtime_config = runtime_settings(load_mode_yaml(runtime_config))
        else:
            runtime_config = dict(runtime_config or {})
            config_yaml = runtime_config.pop('config_yaml', None)
            if config_yaml is not None:
                from highway_env.planner.diffusion.config.experiment import load_mode_yaml, runtime_settings
                merged = runtime_settings(load_mode_yaml(config_yaml))
                merged.update(runtime_config)
                runtime_config = merged
        if runtime_config.get('device') == 'auto':
            runtime_config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        # TRACE_D1_RUNTIME_CONTROL_V3: opt-in control-space runtime.
        if runtime_config.get('output_space', 'trajectory') == 'control':
            from highway_env.planner.diffusion.inference.control import ControlDiffusionRuntime
            if (runtime_config.get('diffusion_spline_enabled') or
                    runtime_config.get('diffusion_execution_enabled')):
                raise ValueError('Control-space runtime uses its own dense rollout; disable spline/execution')
            checkpoint = runtime_config.get('checkpoint')
            if not checkpoint:
                raise ValueError('Control-space runtime requires a trained checkpoint')
            self._control_delegate = ControlDiffusionRuntime(
                checkpoint,
                device=runtime_config.get('device', 'auto'),
                deterministic_seed=runtime_config.get('deterministic_seed', 0),
                inference_steps=runtime_config.get('control_inference_steps'),
            )
            self.device = self._control_delegate.device
            self.model = self._control_delegate.model
            self.adapter = self._control_delegate.adapter
            self.config = self.model.scene_config
            self.diffusion_spline_enabled = False
            self.diffusion_execution_enabled = False
            self.last_latency_ms = None
            return

        super().__init__(runtime_config)

    def infer(self, numpy_features):
        # TRACE_D1_RUNTIME_INFER_V3: forward control-space inference.
        if hasattr(self, '_control_delegate'):
            result = self._control_delegate.infer(numpy_features)
            self.last_latency_ms = self._control_delegate.last_latency_ms
            return result
        return super().infer(numpy_features)
