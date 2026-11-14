"""Evaluate the paper A0/A1/A2 adaptive-anchor ablation."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.ablation.anchor_ablation.runner import run

if __name__ == '__main__':
    run()
