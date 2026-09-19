"""Checkpoint compatibility between W2 pretraining and DiffusionPlannerRuntime."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch


def _strip_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def load_planner_state_dict(checkpoint_obj) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Return planner-only state_dict plus stored model config when available.

    Supported inputs:
      * W2 training checkpoint: ``planner_state_dict`` is preferred.
      * legacy W2 ``state_dict`` with ``planner.`` prefix.
      * runtime checkpoint containing a direct planner ``state_dict``.
      * direct state_dict mapping.
    """
    model_config = {}
    if isinstance(checkpoint_obj, dict):
        model_config = dict(checkpoint_obj.get("allmerge_model_config") or {})
        if "planner_state_dict" in checkpoint_obj:
            return dict(checkpoint_obj["planner_state_dict"]), model_config
        state_dict = checkpoint_obj.get("state_dict", checkpoint_obj)
    else:
        state_dict = checkpoint_obj

    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a state_dict mapping")

    state_dict = dict(state_dict)
    prefixes = (
        "planner.",
        "module.planner.",
        "model.planner.",
    )
    for prefix in prefixes:
        if any(key.startswith(prefix) for key in state_dict):
            stripped = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if stripped:
                return stripped, model_config

    # Direct runtime/planner state dict.
    return state_dict, model_config


def load_checkpoint_file(path: str | Path, *, map_location="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location=map_location)
    return load_planner_state_dict(obj)


def export_runtime_checkpoint(
    source_checkpoint: str | Path,
    output_path: str | Path,
) -> Path:
    source_checkpoint = Path(source_checkpoint)
    output_path = Path(output_path)
    checkpoint_obj = torch.load(source_checkpoint, map_location="cpu")
    state_dict, model_config = load_planner_state_dict(checkpoint_obj)

    payload = {
        "state_dict": state_dict,
        "model_config": model_config,
        "source_checkpoint": str(source_checkpoint),
        "format": "allmerge_diffusion_runtime_v1",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path
