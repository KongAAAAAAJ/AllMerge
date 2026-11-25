"""Replot full JTP distributions without model inference."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.ablation.jtp.distributions import draw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else ROOT / path
    draw(resolve(args.data), resolve(args.output_dir))


if __name__ == '__main__':
    main()
