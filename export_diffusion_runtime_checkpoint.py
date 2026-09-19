from __future__ import annotations

import argparse

from pretraining.checkpoint_io import export_runtime_checkpoint


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
