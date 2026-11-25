"""Regenerate the JTP paper figure from a saved, pickle-free figure-data NPZ."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ablation.jtp.visualization import draw_projection_figure


def _resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _unpack_ragged(data, key):
    if key not in data.files:
        return []
    packed = data[key]
    lengths_key = f'{key}_lengths'
    if lengths_key not in data.files:
        raise ValueError(f'{key} exists but {lengths_key} is missing; regenerate figure data with the updated evaluator')
    lengths = data[lengths_key]
    return [packed[i, :int(length)].copy() for i, length in enumerate(lengths)]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Replot TRACE JTP ablation figure from saved NPZ')
    parser.add_argument('--figure-data', required=True, help='projection_ablation_figure_data.npz')
    parser.add_argument('--jtp-config', default='configs/projection/jtp.yaml')
    parser.add_argument('--output', default='outputs/jtp_ablation/evaluation/projection_ablation.pdf')
    args = parser.parse_args(argv)

    data_path = _resolve_path(args.figure_data)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    config_path = _resolve_path(args.jtp_config)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    with np.load(data_path, allow_pickle=False) as data:
        figure = {key: data[key].copy() for key in data.files
                  if not key.endswith('_lengths') and key not in ('road_edge_world', 'road_safety_edge_world')}
        figure['road_edge_world'] = _unpack_ragged(data, 'road_edge_world')
        figure['road_safety_edge_world'] = _unpack_ragged(data, 'road_safety_edge_world')
        figure['external_actors'] = [row.copy() for row in data['external_actors']]
    output = _resolve_path(args.output)
    draw_projection_figure(
        output, figure,
        dt=float(cfg['execution']['dt_s']),
        kappa_lim=float(cfg['geometry']['curvature_limit_per_m']),
    )
    print(f'[COMPLETE] Wrote {output} and {output.with_suffix(".png")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
