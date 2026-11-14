# Project entry points

Run commands from `all_merge/`. Script directories contain user entry points;
implementations live in their corresponding business packages.

| Entry | Purpose |
| --- | --- |
| `data/collect_expert_dataset.py` | YAML-selected expert collection |
| `data/split_expert_dataset.py` | Dataset split generation |
| `training/run_diffusion.py` | Trajectory/control diffusion training |
| `training/train_grpo.py` | GRPO fine-tuning |
| `diagnostics/analyze_expert_dataset.py` | Expert dataset statistics |
| `diagnostics/overfit_diagnostics.py` | Model/data fitting diagnostics |
| `simulation/run_test.py` | Scenario execution |

```powershell
python scripts/data/collect_expert_dataset.py --config configs/data/expert_collection.yaml
python scripts/data/collect_expert_dataset.py --config configs/data/expert_collection.yaml --check-only
python scripts/training/run_diffusion.py --config configs/train/diffusion.yaml
```

Expert collection selects `vehicle_model: kinematic` or `trucksim` in YAML.
TruckSim batches launch isolated workers through a private `--worker` route;
users do not need a separate TruckSim entry script.

Implementation locations:

- `evaluation/`: unified offline evaluation entry points and shared evaluation modules; see [evaluation/README.md](../evaluation/README.md).
- `data_pipeline/expert_collection/collector.py`: scenario collection and shards.
- `data_pipeline/expert_collection/runner.py`: TruckSim episode scheduling and recovery.
- `highway_env/planner/diffusion/training/trajectory.py` and `control.py`: trainers.
- `learning/export_checkpoint.py`: runtime checkpoint export (`python -m learning.export_checkpoint`).
- `evaluation/expert_dataset_validation.py`: dataset validation (`python -m evaluation.expert_dataset_validation`).
- `evaluation/legacy_open_loop_cli.py`: retained legacy evaluator (`python -m evaluation.legacy_open_loop_cli`).
- `tests/verify_trucksim_modes.py`: no-DLL compatibility validation.

The old per-backend scripts have been removed; use the unified entry points.
