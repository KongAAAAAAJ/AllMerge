"""Canonical TRACE Control Diffusion open-loop CLI.

Updated in 2026-11: the former W2 StructuredDiffusionPlanner evaluator has
been replaced. No legacy model/checkpoint loading is performed by this entry.
"""
from __future__ import annotations
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from evaluation.trace_planning.open_loop import main

if __name__ == '__main__':
    raise SystemExit(main())
