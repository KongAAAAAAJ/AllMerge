"""Compatibility forwarding only; the canonical evaluator is execution_dense."""
from __future__ import annotations
from evaluate_rl_training_execution_dense import main

if __name__ == "__main__":
    raise SystemExit(main())
