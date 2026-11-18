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
The collection command writes only persistent anchor-independent raw episodes to
`outputs/expert_dataset/expert_trucksim_k2_raw_100k/episodes/`, together with
resume state and reports. Build final training shards separately:

```bash
python scripts/data/build_a1_anchors.py --check-only
python scripts/data/build_a1_anchors.py
python scripts/diagnostics/validate_expert_dataset.py
```

The A1 builder uses `configs/anchors/a1.yaml`, reads the completed raw dataset
without modifying it, and creates `expert_trucksim_k2_anchor_pair_100k` with
paired A1 control/trajectory anchors, K2 context and deterministic episode
splits. Use a new `--output-dir` for each A1 parameter variant. The final
training format is unchanged. Do not run collection and A1 build concurrently.
Legacy standard datasets and separate raw-collection commands are unsupported.

## Diffusion training and evaluation

A0 fixed anchors can be built directly from the completed raw dataset, without
building A1 shards. The builder recovers contiguous 40-step executed expert
commands from recorded histories, using only the episode-level training split:

```bash
python scripts/data/build_a0_fixed_anchors.py --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k --raw-out-of-bounds drop --output outputs/anchor_ablation/ckpt/A0_anchors/a0_fixed_control_anchors.npz
```

`drop` excludes whole windows outside the configured control bounds; it never
clips control values. Without this explicit option, out-of-bounds windows cause
an error. Short or discontinuous histories are not padded or interpolated.

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
