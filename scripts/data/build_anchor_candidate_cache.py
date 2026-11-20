"""Build shared A1/A2 candidate pools without a scorer checkpoint."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from highway_env.planner.candidate_cache import build_cache
from highway_env.planner.diffusion.config.experiment import resolve_repo_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--candidate-config", default="configs/anchors/control_candidate_pool.yaml")
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args(argv)
    splits = args.splits.split(",")
    if not splits or any(split not in ("train", "val", "test") for split in splits):
        parser.error("--splits must be comma-separated train,val,test names")
    build_cache(resolve_repo_file(args.dataset_root), resolve_repo_file(args.cache_dir),
                {"config_path": str(resolve_repo_file(args.candidate_config))}, splits)


if __name__ == "__main__":
    main()
