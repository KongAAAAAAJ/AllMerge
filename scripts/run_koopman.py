"""TRACE Stage 1 CLI: dry-run, collect, worker, audit.

Run from AllMerge parent folder:
    python all_merge/scripts/run_koopman.py collect --check-only
    python all_merge/scripts/run_koopman.py collect
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description="TRACE Koopman TruckSim experiment collection")
    parser.add_argument("command", choices=["collect", "worker", "audit", "smoke"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "koopman.yaml"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--model", default=None, help="Collect/audit a single enabled trucksim.models id")
    parser.add_argument("--speed-tier", default=None, help="Run only this SV_VXS initial speed tier (reuses the physical model), e.g. V60")
    parser.add_argument("--job", default=None, help="JSON job spec (worker mode only)")
    parser.add_argument("--output", default=None, help="Worker output directory")
    args = parser.parse_args(argv)

    from highway_env.vehicle.koopman.collect import read_yaml, validate, collect_batch, run_worker
    from highway_env.vehicle.koopman.data import audit_npz
    if args.command == "smoke":
        # The uploaded project archive has no tests/test_koopman.py. Run the
        # self-contained physical-model / SV_VXS scheduling regression suite.
        from tests.test_koopman_speed_grid import run_tests
        run_tests()
        return 0
    if args.command == "audit":
        from pathlib import Path
        cfg = read_yaml(args.config)
        d = Path(cfg["batch"]["output_dir"])
        if not d.is_absolute():
            d = ROOT/d
        if args.model and args.speed_tier:
            glob_pattern = f"{args.model}_{args.speed_tier}__*.npz"
        elif args.model:
            glob_pattern = f"{args.model}_*__*.npz"
        elif args.speed_tier:
            glob_pattern = f"*_{args.speed_tier}__*.npz"
        else:
            glob_pattern = "*.npz"
        paths = sorted(d.glob(glob_pattern))
        if not paths:
            raise FileNotFoundError(f"No .npz files in {d}")
        errors = []
        for file in paths:
            result = audit_npz(file)
            print("[AUDIT]", result)
            if not result["valid"]:
                errors.append(result)
        if errors:
            raise RuntimeError(f"{len(errors)} data files failed the audit")
        return 0
    cfg = read_yaml(args.config)
    if args.command == "collect":
        collect_batch(cfg, args.config, ROOT, dry_run=args.check_only, model_id=args.model, speed_tier=args.speed_tier)
        return 0
    if args.command == "worker":
        if not args.job or not args.output:
            parser.error("worker requires --job and --output")
        with open(args.job, encoding="utf-8") as f:
            job = json.load(f)
        run_worker(cfg, job, args.output)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
