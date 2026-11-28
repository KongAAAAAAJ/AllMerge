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


def main(argv=None):
    parser = argparse.ArgumentParser(description='Replot TRACE JTP ablation figure from saved NPZ')
    parser.add_argument('--figure-data', nargs='+', required=True, help='projection_ablation_stability_data.npz and/or projection_ablation_control_data.npz')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)

    cases = {}
    for value in args.figure_data:
        with np.load(_resolve_path(value), allow_pickle=False) as data:
            figure = {key: data[key].copy() for key in data.files}
            cases[str(data['case_kind'])] = figure
    output = _resolve_path(args.output_dir)
    draw_projection_figures(output, cases)
    print(f'[COMPLETE] Wrote {", ".join(cases)} PDFs to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
