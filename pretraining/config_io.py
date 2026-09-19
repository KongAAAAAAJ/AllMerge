"""Dependency-light config loading for diffusion pretraining."""
from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Load JSON config; YAML remains optional when PyYAML is available."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is YAML but PyYAML is not installed. "
                "Use configs/diffusion_pretrain.json instead."
            ) from exc
        with path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    return data


def resolve_pin_memory(value) -> bool:
    import torch

    if value == "auto":
        return torch.cuda.is_available()
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
