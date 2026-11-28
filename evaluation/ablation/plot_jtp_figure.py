"""Regenerate the JTP paper figure from a saved, pickle-free figure-data NPZ."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ablation.jtp.visualization import draw_projection_figures


def _resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _unpack_ragged(data, key):
    packed = data[key]
    lengths_key = f'{key}_lengths'
    lengths = data[lengths_key]
    return [packed[i, :int(length)].copy() for i, length in enumerate(lengths)]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Replot TRACE JTP ablation figure from saved NPZ')
    parser.add_argument('--figure-data', nargs='+', required=True, help='Three projection_ablation_geometry_<1..3>_data.npz files, or stability/control data files')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)

    cases = {}
    for value in args.figure_data:
        with np.load(_resolve_path(value), allow_pickle=False) as data:
            figure = {key: data[key].copy() for key in data.files
                      if not key.endswith('_lengths') and key not in ('road_edge_world', 'road_safety_edge_world')}
            figure['road_edge_world'] = _unpack_ragged(data, 'road_edge_world')
            figure['road_safety_edge_world'] = _unpack_ragged(data, 'road_safety_edge_world')
            kind = str(data['case_kind'])
            if kind == 'geometry':
                cases.setdefault(kind, []).append(figure)
            else:
                cases[kind] = figure
    output = _resolve_path(args.output_dir)
    draw_projection_figures(output, cases)
    print(f'[COMPLETE] Wrote {", ".join(cases)} PDFs to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
