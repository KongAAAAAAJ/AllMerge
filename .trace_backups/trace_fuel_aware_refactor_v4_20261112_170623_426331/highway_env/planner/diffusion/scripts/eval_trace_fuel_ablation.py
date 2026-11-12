"""TRACE Fuel-Aware V3 evaluator entrypoint (project-native location)."""
from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from highway_env.planner.diffusion.evaluation.fuel_ablation_v3 import main

if __name__ == '__main__':
    raise SystemExit(main())
