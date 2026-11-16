"""Dependency-light config loading for diffusion pretraining."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


def load_yaml_config(path: str | Path, *, _parents: tuple[Path, ...] = ()) -> dict:
    """Load YAML, recursively merging an optional parent relative to this file.

    Mappings merge; child scalars, lists and null values replace parent values.
    Only extends paths are file-relative; dataset/checkpoint paths are unchanged.
    """
    import yaml

    filename = Path(path).expanduser().resolve()
    if filename in _parents:
        chain = ' -> '.join(str(p) for p in (*_parents, filename))
        raise ValueError(f'Circular YAML extends: {chain}')
    data = yaml.safe_load(filename.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'YAML configuration must be a mapping: {filename}')
    parent = data.pop('extends', None)
    if parent is None:
        return data
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError(f'extends must be a nonempty file path: {filename}')
    base = load_yaml_config(filename.parent / parent, _parents=(*_parents, filename))

    def merge(base, override):
        result = deepcopy(base)
        for key, value in override.items():
            result[key] = (merge(result[key], value)
                           if isinstance(result.get(key), dict) and isinstance(value, dict)
                           else deepcopy(value))
        return result

    return merge(base, data)


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
                "Install PyYAML to read this configuration."
            ) from exc
        data = load_yaml_config(path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    if 'output_space' in data and 'trajectory' in data:
        # This loader serves trajectory pretraining/evaluation, irrespective of
        # the mode selected for the unified training/runtime entry point.
        data = data['trajectory']
        if not isinstance(data, dict) or not all(isinstance(data.get(k), dict) for k in ('model', 'training')):
            raise ValueError('trajectory must contain model and training mappings')
    return data


def resolve_pin_memory(value) -> bool:
    import torch

    if value == "auto":
        return torch.cuda.is_available()
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
