from __future__ import annotations

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse

from learning.checkpoint import export_runtime_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Export W2 training checkpoint for DiffusionPlannerRuntime.")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    path = export_runtime_checkpoint(args.checkpoint, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
