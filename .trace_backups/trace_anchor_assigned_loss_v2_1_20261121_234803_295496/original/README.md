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
resume state and reports. These raw episodes are used directly by training
and evaluation. A1 anchors are generated online from observed ego state and road
geometry using `configs/anchors/a1.yaml`; K is `control.model.num_modes` in
`configs/train/diffusion.yaml`. No paired A1 dataset or offline build is needed.

```bash
python scripts/diagnostics/validate_expert_dataset.py --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k
```

## Diffusion training and evaluation

A0 fixed anchors can be built directly from the completed raw dataset, without
building A1 shards. The builder recovers contiguous 40-step executed expert
commands from recorded histories, using only the episode-level training split:

```bash
python scripts/data/build_a0_fixed_anchors.py --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k --output outputs/anchor_ablation/ckpt/A0_anchors/a0_fixed_control_anchors.npz
```

Before clustering, only complete control windows within `--max-accel-mps2`
and `--max-steer-rad` are retained; values are never clipped. Per-channel
exclusion counts are reported (they may overlap). The resulting centroids are
also checked. Short or discontinuous histories are not padded or interpolated.

Control-space Diffusion uses the raw dataset root configured in `configs/train/diffusion.yaml`:

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
