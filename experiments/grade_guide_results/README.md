# Reward-Gradient Mean Guidance | in-project V5 module

## Scope

This standalone experiment lives in `all_merge-guideDistill/experiments/grade_duide_results/` (folder spelling retained exactly as requested). No files in `highway_env/`, `train_grpo.py`, or pretrained weights are changed. Sampling/evaluation semantics are carried over from V5, including the narrow numerical parity fallback from the October 17 fix.

- Same frozen checkpoint for all seeds; independent state/noise/random perturbation seeds.
- `8 -> 0` stochastic DDIM mean perturbations, baseline replay, and 41-point 10 Hz Clamped Cubic Spline production Reward.
- Original V5 measurements, no changes to step sizes, reward or random-control definitions.
- `samples.csv` (55 original fields), `trajectories.csv`, and per-seed CSVs are the source of truth; `data_plot.py` is model-independent.
- `random_sign` preserves absolute per-coordinate perturbation values and Gaussian KL; `random_gaussian` matches approximate axis RMS budgets.
- Curvature parity: strict check first, only accept bounded numerical discrepancies; otherwise fail.

## Commands (Windows, from AllMerge root)

```bat
cd /d D:\KONG_Files\AllMerge
run_grade_duide_results_3seed.bat --resume
plot_grade_duide_results.bat
```

Or from the target worktree:

```bat
cd /d D:\KONG_Files\AllMerge\all_merge-guideDistill
python -m experiments.grade_duide_results.multi_seed --root . --checkpoint ..\all_merge-gru\outputs\gru_coarse_goal_ablation_100ep_3seed\seed_0\mlp_dense\run_1\checkpoints\best.pt --output-dir outputs\reward_guided_mean_v5 --seeds 0,1,2 --resume
python experiments\grade_duide_results\data_plot.py --input-dir outputs\reward_guided_mean_v5 --gallery-amplitude-m 0.6 --gallery-count 8
```

Programmatic plotting interface (exact required entry point):

```python
from experiments.grade_duide_results.data_plot import plot_results
result = plot_results('outputs/reward_guided_mean_v5', gallery_amplitude_m=0.6, gallery_count=8)
# also available: from experiments.grade_duide_results.data_plot import plot
```

For direct programmatic collection:

```python
from experiments.grade_duide_results.collector import run, cli
args = cli().parse_args(['--root', '.', '--checkpoint', r'..\all_merge-gru\outputs\...\best.pt'])
run(args)
```

## Files and outputs

- `collector.py`: single-seed collection; writes `samples.csv`, `trajectories.csv`, `summary.json`, failures.
- `multi_seed.py`: deterministic 3-seed orchestration, resume and CSV merge. Main output `outputs/reward_guided_mean_v5/`.
- `data_plot.py`: offline CSV statistics, multi-seed mean ± sample SD curves, reward table PNG, positive-reward fraction PNG, selected state galleries with zoomed local windows.
- `core.py`, `bridge.py`, `math_utils.py`, `random_controls.py`, `stats.py`, `parity.py`: experiment-only helpers.

The gallery annotation *never* equates local centerline-distance improvement to local total Reward improvement; a zoom fallback shows geometry differences only.

**Reproducibility:** Distinct seed values change the sampled states and noise, but all runs use a common checkpoint. These are inference-seed replications, not three separately trained models. For a true across-model robustness study, run independently pretrained checkpoints as a different experiment design.
