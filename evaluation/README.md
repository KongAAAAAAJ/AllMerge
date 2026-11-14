# Offline evaluation

Run these commands from the project root (`all_merge/`). Evaluation entry
points and shared evaluation modules live together in this package.

| File | Purpose | Command |
| --- | --- | --- |
| `eval_diffusion_open_loop.py` | Diffusion checkpoint ADE/FDE and mode metrics | `python -m evaluation.eval_diffusion_open_loop --help` |
| `ablation/run_anchor.py` | Paper adaptive-anchor ablation | `python evaluation/ablation/run_anchor.py --help` |
| `ablation/run_fuel_aware.py` | Paper fuel-aware training ablation | `python evaluation/ablation/run_fuel_aware.py --help` |
| `validate_jtp_single_vehicle.py` | Offline single-vehicle control diffusion + frozen K2 + JTP validation | `python -m evaluation.validate_jtp_single_vehicle --help` |
| `expert_dataset_validation.py` | Expert dataset validation | `python -m evaluation.expert_dataset_validation --help` |
| `legacy_open_loop_cli.py` | Legacy open-loop evaluation CLI | `python -m evaluation.legacy_open_loop_cli --help` |
| `open_loop_eval.py` | Shared open-loop evaluation implementation | Imported by evaluation entry points |
| `open_loop_metrics.py` | Shared evaluation metrics | Imported by evaluation modules |
| `open_loop_visualization.py` | Evaluation plots and visualizations | Imported by evaluation modules |

The evaluation entry points also support direct execution, for example:

```powershell
python evaluation/eval_diffusion_open_loop.py --help
python evaluation/ablation/run_anchor.py --help
python evaluation/ablation/run_fuel_aware.py --help
python evaluation/validate_jtp_single_vehicle.py --help
```

Arguments and output locations are unchanged by this directory merge.
The two main paper ablations are centralized under `evaluation/ablation/`;
see [ablation/README.md](ablation/README.md) for commands and module ownership.
Koopman evaluation remains under `highway_env/vehicle/koopman/evaluation/`.
