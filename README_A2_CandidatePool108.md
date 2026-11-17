# TRACE A2 Candidate Pool 108 / Shared A1-A2 Generation

## Candidate-count contract

The candidate pool size and the final diffusion anchor count K are separate:

| Parameter | Value | Meaning |
|---|---:|---|
| `candidate_pool_size` | 108 | Shared raw control/trajectory candidates |
| `top_r` | 60 | A2 safety-scored prescreening |
| `num_modes` | 10 by default | Final anchors passed to diffusion; configurable in `control.model.num_modes` |

The shared definition is `configs/anchors/control_candidate_pool.yaml`. Both the A1 dynamic manager and A2 adaptive manager load this same file.

The 108 slots are:
- Current lane: 15 speed samples x 3 steering residuals = 45
- Left lane: 9 speed samples x 3 steering residuals = 27
- Right lane: 9 speed samples x 3 steering residuals = 27
- Braking: 8 deceleration levels + emergency stop = 9

Unavailable adjacent lanes retain masked slots. Therefore the tensor shape remains 108, while the number of valid candidates depends on road topology. A current-lane-only scene has 54 valid candidates.

## A1 and A2 selection

- A1 uses the shared 108-candidate pool and deterministic joint control/trajectory diversity selection over all valid candidates. It does not use a safety scorer.
- A2 uses the same pool, applies the trained safety scorer, keeps Top-R, and then performs the same joint diversity selection to obtain K anchors.

The diffusion model defaults to `num_modes=10`, configurable under `control.model.num_modes` in `configs/train/diffusion.yaml`. A1/A2 managers derive their selected-anchor count from this value. Increasing K changes the planner input mode axis but does not add per-mode learned parameters because the denoiser weights are shared across modes.

## A2 scorer training implementation

The uploaded 2026-11-17 project uses **Cached V2** training:
- `training_data.py` precomputes candidate controls, trajectories and teacher safety labels once.
- Arrays are stored as memory-mapped NumPy files.
- Training epochs load cached supervision rather than regenerating candidates and safety labels.
- The trainer supports cache-only preparation, cache rebuild, evaluation, warm start and optimizer-state resume.

This patch upgrades the cache schema to V3 and makes all candidate array dimensions dynamic. The cache fingerprint includes the complete resolved candidate configuration, teacher configuration, source shard metadata and temperature. A 36-candidate cache cannot be reused for the 108-candidate pool.

## Paths

- Scorer checkpoint directory: `outputs/anchor_ablation/ckpt`
- Best checkpoint: `outputs/anchor_ablation/ckpt/best.pt`
- Last checkpoint: `outputs/anchor_ablation/ckpt/last.pt`
- Candidate cache: `outputs/anchor_ablation/cache_v3`

The diffusion config points to `outputs/anchor_ablation/ckpt/best.pt`.

## Training

Run from `D:\KONG_Files\AllMerge\all_merge`:

```bat
python -m highway_env.planner.anchor_scoring.train --config configs/train/adaptive_anchors.yaml --max-epochs 1 --max-batches 5
```

After the smoke test succeeds, prepare the full train/validation caches:

```bat
python -m highway_env.planner.anchor_scoring.train --config configs/train/adaptive_anchors.yaml --prepare-cache-only
```

Then run formal training:

```bat
python -m highway_env.planner.anchor_scoring.train --config configs/train/adaptive_anchors.yaml --device auto
```

For a changed candidate-pool config, the cache fingerprint changes automatically. Use `--rebuild-cache` only when intentionally rebuilding the same fingerprint.

## Storage estimate

At 108 candidates, the cached candidate/control/label arrays require approximately 43.5 KB per sample, or about 4.4 GB for 100,000 samples, excluding filesystem overhead. Ensure several additional GB are free before cache preparation.
