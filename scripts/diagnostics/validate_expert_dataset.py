"""Convenience CLI for validating the final unified TRACE expert dataset."""
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.expert_dataset_validation import main

if __name__ == "__main__":
    raise SystemExit(main())
