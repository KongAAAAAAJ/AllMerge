"""Project-native CLI for the final TRACE Fuel-Aware open-loop evaluation."""
from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from highway_env.planner.diffusion.evaluation.fuel_aware.runner import main

if __name__ == '__main__':
    raise SystemExit(main())
