"""Load independent evaluation defaults without modifying training YAML."""
from __future__ import annotations
from pathlib import Path


def defaults(path, section):
    import yaml
    filename=Path(path)
    if not filename.is_file():
        # Project-root relative even when Python launched from another folder.
        filename=Path(__file__).resolve().parents[2]/filename
    if not filename.is_file():
        raise FileNotFoundError(f'TRACE evaluation config not found: {filename}')
    data=yaml.safe_load(filename.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or not isinstance(data.get(section),dict):
        raise ValueError(f'{filename} must define {section} mapping')
    return dict(data[section])
