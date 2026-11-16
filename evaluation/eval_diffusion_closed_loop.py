"""Canonical manuscript-aware native TruckSim closed-loop evaluation entry.

Aggregates the two-panel table from independently completed open-loop runs.
No missing baseline is synthesized. See evaluation.trace_planning README.
"""
from __future__ import annotations
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.trace_planning.run_paper_evaluation import main
if __name__ == '__main__':
    raise SystemExit(main())
