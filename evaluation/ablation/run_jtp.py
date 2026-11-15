"""Entry point for the third TRACE paper ablation: Joint Trajectory Projection."""
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from evaluation.ablation.jtp.runner import main
if __name__=='__main__':
    raise SystemExit(main())
