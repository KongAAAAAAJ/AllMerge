"""Project-native CLI for TRACE fuel-aware open-loop ablation."""
from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from highway_env.planner.diffusion.evaluation.fuel_ablation import main

if __name__ == '__main__':
    raise SystemExit(main())
