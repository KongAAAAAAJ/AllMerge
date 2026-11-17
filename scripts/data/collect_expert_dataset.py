"""The single public entry point for TRACE expert data collection."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker":
        from data_pipeline.expert_collection.worker import main as worker_main
        return worker_main(argv[1:])
    from data_pipeline.expert_collection.config import parse_args
    from data_pipeline.expert_collection.runner import main as batch_main
    return batch_main(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
