"""Unified YAML output-space selection for trajectory and control diffusion."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_repo_file(value: str | Path, *, root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    base = root or project_root()
    if (base / path).exists():
        return (base / path).resolve()
    if path.exists():
        return path.resolve()
    return (base / path).resolve()


def load_mode_yaml(path: str | Path) -> dict[str, Any]:
    filename = resolve_repo_file(path)
    if not filename.is_file():
        raise FileNotFoundError(filename)
    data = yaml.safe_load(filename.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Diffusion mode YAML must be a mapping')
    if data.get('output_space') not in ('trajectory', 'control'):
        raise ValueError("output_space must be 'trajectory' or 'control'")
    selected = data[data['output_space']] if data['output_space'] in data else None
    if isinstance(selected, dict) and 'training' in selected:
        # Adapt grouped YAML to the existing trainer/runtime configuration API.
        data = dict(data)
        data['model'] = selected.get('model', {})
        data['training'] = selected['training']
        if data['output_space'] == 'control':
            data['control'] = selected['dynamics']
            data['execution'] = selected.get('execution', {'model': 'kinematic'})
            data['losses'] = selected.get('losses', {})
    return data


def runtime_settings(cfg: dict) -> dict:
    """Convert a mode YAML to the existing DiffusionPlannerRuntime kwargs."""
    selected = cfg['output_space']
    values = dict(cfg.get('runtime') or {})
    checkpoints = values.pop('checkpoint', None)
    if isinstance(checkpoints, dict):
        selected_checkpoint = checkpoints.get(selected)
    else:
        selected_checkpoint = checkpoints
    settings = {'output_space': selected, **values}
    if selected_checkpoint:
        settings['checkpoint'] = str(resolve_repo_file(selected_checkpoint))
    if selected == 'control':
        if values.get('diffusion_spline_enabled') or values.get('diffusion_execution_enabled'):
            raise ValueError('Control mode already provides dense rollout; disable spline/execution')
        settings['control_inference_steps'] = int(cfg['control']['inference_steps'])
    return settings
