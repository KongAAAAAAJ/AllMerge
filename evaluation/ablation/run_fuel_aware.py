"""Evaluate the paper fuel-aware training ablation."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.ablation.fuel_aware.runner import main

if __name__ == '__main__':
    raise SystemExit(main())
