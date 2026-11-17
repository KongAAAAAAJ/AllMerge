# AllMerge / TRACE

## Unified expert data pipeline

The project has exactly one expert-data collection workflow. The Polynomial
expert and four scenario definitions are retained, while the controlled ego
vehicles use the configured TruckSim payload/friction plants. Each episode is
run in an isolated OS process with independent TruckSim DLL instances.

The public entry point is:

```bash
python scripts/data/collect_expert_dataset.py --check-only
python scripts/data/collect_expert_dataset.py
```

Configuration is centralized in `configs/data/expert_collection.yaml`.
The collector performs the following stages as one managed workflow:

1. Batch collection of anchor-independent Polynomial expert samples, full live
   road geometry, vehicle dimensions, and synchronized Koopman K2 context.
2. Offline A1 lattice anchor generation from the captured road geometry, using `configs/anchors/a1.yaml`.
3. Atomic episode-shard packaging and deterministic train/validation/test split.
4. Manifest generation, shard validation, native-worker logs, and resumable
   progress tracking.

Raw episode files are an internal staging format only. The only supported
training dataset is the final `shards/shard_*.npz` dataset containing A1 paired
control/trajectory anchors and K2 context. Legacy standard expert datasets and
separate raw-collection commands are unsupported.

## Diffusion training and evaluation

Both trajectory-space and control-space Diffusion configurations use the same
unified dataset root configured in `configs/train/diffusion.yaml`:

```bash
python scripts/diagnostics/validate_expert_dataset.py --help
python scripts/training/run_diffusion.py --help
python -m highway_env.planner.diffusion.training.control --help
python evaluation/trace_planning/run_paper_evaluation.py --help
python evaluation/ablation/run_fuel_aware.py --help
python -m pytest
```

Run commands from the project root. Dataset paths in YAML are resolved relative
to this repository. TruckSim model assets and the engine-map calibration must be
available at the paths specified by `configs/koopman/collect.yaml`.
