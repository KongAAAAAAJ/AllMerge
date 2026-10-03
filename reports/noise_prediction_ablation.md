# Noise prediction ablation

This branch adds an explicit diffusion prediction parameterization switch:

- `prediction_type="sample"`: historical AllMerge behavior. The denoiser predicts clean sparse trajectory `x0`; the selected mode is optimized by physical-space SmoothL1 to the expert trajectory.
- `prediction_type="epsilon"`: comparison method. The selected expert trajectory is noised with `q(x_t | x0_expert)` and the denoiser predicts the sampled Gaussian epsilon with MSE. At runtime epsilon is converted back to `x0` before the deterministic DDIM step.

The scene encoder, semantic mode assignment, classifier, MLP denoiser capacity, GRU refinement option, optimizer, seed, and open-loop evaluation remain shared so the experiment isolates the diffusion target parameterization as much as possible.

## Why epsilon training must noise the expert target

The historical AllMerge path adds noise around a coarse dynamic anchor but directly regresses the expert `x0`. If epsilon MSE were naively applied to that same sampled noise, the mathematically implied clean sample would be the coarse anchor itself, so the network would not learn the expert correction. Therefore the epsilon branch replaces only the selected mode's clean sample with the normalized expert trajectory before calling `add_noise`; non-selected modes retain their anchors and are excluded from epsilon MSE.

## 100-step matched A/B

Run from the `Noise` worktree/repository root:

```bat
python scripts/run_noise_ablation_100step.py ^
  --dataset-root outputs/expert_dataset/allmerge_expert_50k_dense10hz_v2 ^
  --max-steps 100 ^
  --limit-val-samples 1000 ^
  --limit-eval-samples 100 ^
  --seed 0
```

The script trains x0/sample first, reuses its exact `checkpoints/initial.pt` as the epsilon run initialization, evaluates both with the same inference seed, and writes:

- `outputs/noise_ablation_100step/x0/run_*/...`
- `outputs/noise_ablation_100step/epsilon/run_*/...`
- `outputs/noise_ablation_100step/evaluation/x0_open_loop.csv`
- `outputs/noise_ablation_100step/evaluation/epsilon_open_loop.csv`
- `outputs/noise_ablation_100step/evaluation/prediction_type_comparison.json`

Do not compare raw `val_prediction_loss` numerically across parameterizations because SmoothL1 in meters and epsilon-MSE are in different units. Use `val_x0_reconstruction_loss` and the open-loop ADE/FDE/miss/off-road metrics for a fair result comparison.
