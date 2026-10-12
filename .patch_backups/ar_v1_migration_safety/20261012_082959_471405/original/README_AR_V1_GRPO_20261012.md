# AR V1 — Denoiser-Reused Gaussian Autoregressive Planner

**Source snapshot:** `AllMerge-all_merge-grpo(20261011-234634).zip` (**all_merge-grpo**). This is **not** the GuideDistill V5.2 snapshot. V5.2 Unified Guidance/MATP improvements are not introduced into this code.

## Scope and guarantees

- `StructuredTrajectoryDenoiser`, `StructuredSceneEncoder`, `GRU Refinement`, `DenseSparseResidualHead`, interpolation and W1 feature schema have **zero weight/shape changes**. `StructuredAutoregressivePlanner` subclasses the original `StructuredDiffusionPlanner` and only changes the sampling/training methods.
- `train_diffusion_pretrain.py` uses Diffusion by default, unchanged. Explicit `--generator-type ar_v1` switches to AR with the same model config, expert W1 loader, AdamW groups, scheduler, optimization steps, dense loss, expert mode assignment, safety evaluation, and runtime.
- AR parameters: fixed diagonal Gaussian in physical metres, `sigma_x=2.0 m`, `sigma_y=0.5 m`, configurable. No trainable log variance.
- 8 sequential calls to **the same Denoiser**, using `t=1..8` as a generation-step index. Previously generated or teacher-forced points are placed in prefix slots; unknown future slots retain the original coarse mode anchors. This is *not* diffusion noise or an imitation of the DDIM schedule.
- A step-1 forward on the pure anchors supplies mode logits, so classification cannot see expert prefix coordinates. Training NLL uses the exact semantic-constrained W1 target mode; stepwise GT prefix is used *only* during Teacher Forcing. The target mode's log likelihood is conditional on mode, **without** the discrete mode-selection probability.
- Original GRPO/Guidance reward or training code is **not** modified. `sample_group(...,group_size=48)` supplies compatible candidate shapes, and `joint_trajectory_log_prob(...)` exactly replays the **joint all-mode** Gaussian policy likelihood for *future* AR-GRPO adaptation. The existing denoiser has mode self-attention, so individual mode likelihood cannot be replayed accurately without the other modes' sampled histories. AR-GRPO itself is NOT implemented in V1.
- Dense Residual Head remains trained with exactly the old runtime-consistent spline-based auxiliary objective and isolated gradients. It uses AR rollout instead of DDIM rollout.
- Runtime identifies AR via checkpoint `generator_type=ar_v1` and auto-selects the new sample path. `trajectory_noisy_initial` is retained as a legacy key, but it now means the *unchanged coarse anchors*, not noisy diffusion data. AR figure labels are adjusted accordingly.

## Modified existing files

1. `pretraining/trainer.py` — optional AR planner factory, AR sigma, and training/resume checkpoint metadata. Diffusion remains default.
2. `train_diffusion_pretrain.py` — optional `--generator-type`, `--ar-sigma-x-m`, `--ar-sigma-y-m`, forward to same trainer.
3. `pretraining/checkpoint_io.py` — preserve AR type and Gaussian scales when exporting runtime checkpoints.
4. `highway_env/planner/diffusion/runtime.py` — choose AR vs Diffusion from saved metadata before loading identical network weights.
5. `evaluation/open_loop_visualization.py` — show AR initial coarse anchors correctly rather than claiming it is diffusion noise.

## New files

6. `highway_env/planner/diffusion/autoregressive.py` — AR forward rollout, teacher-forcing NLL, policy log probability, G-group sampler, inherited Dense Residual auxiliary.
7. `train_ar_pretrain.py` — convenience wrapper around the original training entrypoint.
8. `scripts/run_ar_open_loop_eval.py` — wrapper around the original Open-Loop evaluator; metadata drives model selection.
9. `scripts/compare_ar_diffusion_open_loop.py` — join on the actual **state/ego** record, produce state-paired CSV differences; no G48 pseudoreplication.
10. `test_ar_v1.py` — CPU smoke of shape/weights, NLL gradients, Dense gradients, seeded G sampling, and runtime checkpoint loading.

**Unmodified:** `structured_model.py`, `config.py`, `encoders.py`, `tensor_adapter.py`, `diffusion_schedule.py`, `mode_assignment.py`, `dense_supervision.py`, `trajectory_spline.py`, GRPO code, Reward code, environment scenarios.

## Verify

From `D:\KONG_Files\AllMerge`:

```bat
python install_ar_v1_from_grpo_20261012.py --root all_merge-grpo --check-only
python install_ar_v1_from_grpo_20261012.py --root all_merge-grpo
cd /d D:\KONG_Files\AllMerge\all_merge-grpo
python test_ar_v1.py
```

The installer checks source file SHA256 from the uploaded snapshot, makes backups, does not overwrite arbitrary local revisions, and can roll back its own modifications.

## Supervised 3-step smoke

```bat
cd /d D:\KONG_Files\AllMerge\all_merge-grpo
python train_ar_pretrain.py ^
  --dataset-root outputs\expert_dataset\allmerge_expert_50k_dense10hz_v2 ^
  --output-dir outputs\ar_v1_pretrain\smoke3 ^
  --max-steps 3 ^
  --max-epochs 1 ^
  --batch-size 4 ^
  --num-workers 0 ^
  --limit-train-samples 16 ^
  --limit-val-samples 16 ^
  --ar-sigma-x-m 2.0 ^
  --ar-sigma-y-m 0.5
```

The smoke presumes the local expert dataset exists at that location. No pretraining was executed in the provided environment (the dataset and normal Gymnasium dependencies are absent). For regular training, use original pretrain flags including `--max-epochs 100`, LR, seed/config, and the same dense supervision options as the Diffusion run. `train_ar_pretrain.py` chooses `outputs/ar_v1_pretrain` by default if `--output-dir` is omitted.

## Open-loop paired comparison

Run both original Diffusion and AR checkpoints over the same held-out states, fixed `--inference-seed`, `--limit-samples`, split, and `--batch-size`:

```bat
python scripts\run_open_loop_eval.py ^
  --checkpoint outputs\diffusion_pretrain\run_1\checkpoints\best.pt ^
  --dataset-root outputs\expert_dataset\allmerge_expert_50k_dense10hz_v2 ^
  --split test --limit-samples 100 --inference-seed 20261108 ^
  --output-csv outputs\evaluation\ar_v1_pair\diffusion.csv --no-visualization

python scripts\run_ar_open_loop_eval.py ^
  --checkpoint outputs\ar_v1_pretrain\run_1\checkpoints\best.pt ^
  --dataset-root outputs\expert_dataset\allmerge_expert_50k_dense10hz_v2 ^
  --split test --limit-samples 100 --inference-seed 20261108 ^
  --output-csv outputs\evaluation\ar_v1_pair\ar.csv --no-visualization

python scripts\compare_ar_diffusion_open_loop.py ^
  --diffusion-csv outputs\evaluation\ar_v1_pair\diffusion.csv ^
  --ar-csv outputs\evaluation\ar_v1_pair\ar.csv ^
  --output-csv outputs\evaluation\ar_v1_pair\comparison.csv
```

**Important:** paired samples are paired by identical *state records*, not by same Gaussian innovations across dissimilar sampling algorithms. Both runs must use independently pretrained checkpoints and matched budgets; do not compare a fresh AR checkpoint to a 100-epoch Diffusion checkpoint and claim a fair result. The sample paths shown are examples and must match your actual local checkpoint directories.

## Limitations

- AR NLL has physical-unit normalization while Diffusion has its own regression loss: compare validation ADE/FDE and safety, not cross-family raw training loss magnitude.
- G48 grouped sampler is provided, but the installed `GRPOTrainer`/`GroupDiffusionSampler` use diffusion-transition likelihoods and remain unmodified; they **cannot** train AR without a separate AR-GRPO implementation.
- With fixed Gaussian std, AR may under/over-explore particular scenarios. A learned std head would violate the strict no-architecture-change V1 constraint and belongs in V2.
- The existing 2-step inference DDIM and 8-step AR may have materially different latency; measure instead of assuming AR is faster.
