"""Expert collection entry point; the collection method is selected by YAML."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Private route used only by the isolated TruckSim batch controller.
    if argv and argv[0] == '--worker':
        from data_pipeline.expert_collection.collector import main as collect, parse_args
        collect(parse_args(argv[1:]))
        return 0

    from data_pipeline.expert_collection.config import parse_args
    options = parse_args(argv)
    if options.vehicle_model == 'trucksim':
        from data_pipeline.expert_collection.runner import main as run
        return run(options)
    if options.check_only:
        print(f'[PLAN] kinematic; scenarios={options.scenarios}; '
              f'output={options.output_root / options.dataset_name}')
        return 0
    from data_pipeline.expert_collection.collector import main as collect
    if not options.resume and options.start_episode_index:
        options.episode_index = options.start_episode_index
    collect(options)
    return 0



if __name__ == '__main__':
    raise SystemExit(main())
