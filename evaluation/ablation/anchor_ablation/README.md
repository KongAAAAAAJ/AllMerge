# TRACE Anchor Ablation (eight metrics, raw K2 safety evaluation)

This evaluator compares A0 Fixed Anchors, A1 Dynamic Anchors and A2 Ours.
It uses the current anchor-independent raw expert dataset and the online
A1AnchorManager/A1AnchorBuilder interface.

## Paper metrics

### Anchor quality (4 metrics)

1. **Anchor minADE (m) ↓**: minimum ADE between the K online anchors and the
   8-point expert trajectory.
2. **Safe Ratio (%) ↑**: hard-safe anchors divided by K, using the common A2
   safety teacher's constant-velocity external-actor collision and
   drivable-road corridor checks.
3. **Trajectory Diversity (m) ↑**: mean pairwise pointwise XY distance among
   the K anchor trajectories.
4. **Control Diversity (-) ↑**: mean pairwise pointwise control distance after
   channel normalization by `[2.5 m/s^2, 0.08 rad]`.

### Diffusion planning performance (4 metrics)

5. **Planning minADE (m) ↓**: minimum ADE among valid diffusion-generated
   8-point K2 execution trajectories.
6. **CR (%) ↓**: percentage of selected planning trajectories with at least
   one collision during the prediction horizon. Uncontrolled background
   actors are extrapolated with constant velocity; synchronized controlled
   trucks are checked against their own raw K2 predictions.
7. **DAC (%) ↑**: mean percentage of future time steps for which the complete
   ego vehicle footprint remains inside the union of recognized drivable-lane
   corridors. The footprint uses the configured multi-circle vehicle envelope.
8. **TTC (%) ↑**: percentage of selected planning trajectories whose minimum
   predicted TTC to uncontrolled actors or synchronized controlled trucks
   is greater than or equal to the configured threshold. The default threshold
   is 1.0 s and can be changed in `configs/evaluation/anchor_ablation.yaml`.

**Planning safety metrics are evaluated on the deployment-selected raw K2
trajectory before JTP.** The raw K2 poses are in the ego-at-planning-time local
frame and are transformed to the world frame only for synchronized
inter-truck and road-corridor checks. JTP is not instantiated or invoked:
there is no JTP constraint evaluation, iterative projection, projection
correction or projected-trajectory fallback.

The existing `--jtp-config` argument is retained for command compatibility.
Its YAML is used only to read the vehicle footprint and execution time step
for collision, TTC and drivable-area calculations. It does not invoke JTP.

## Staged entry points

- `--stage anchors` computes the four anchor-quality metrics using independent
  raw controlled-ego samples. It does not group the full dataset into
  synchronized frames. A1 is regenerated online from `ego_state` and raw
  road geometry. It generates `anchor_ablation.pdf`.
- `--stage feasibility` is retained as the legacy stage name. It computes the
  four planning metrics using complete synchronized frames, because
  inter-truck collision and TTC require synchronized controlled-vehicle
  predictions. It generates `anchor_planning_ablation.pdf`.
- `--stage all` computes all eight metrics and both PDFs using complete
  synchronized frames.

## Smoke commands

Anchor-only A0/A1 smoke (A2 scorer is not required):

```bat
python evaluation/ablation/run_anchor.py --stage anchors --methods A0 A1 --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k --a0-bank outputs/anchor_ablation/ckpt/a0_fixed_control_anchors.npz --num-samples 500 --batch-size 128 --num-workers 0 --device auto
```

After A2 scorer training:

```bat
python evaluation/ablation/run_anchor.py --stage anchors --methods A0 A1 A2 --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k --a0-bank outputs/anchor_ablation/ckpt/a0_fixed_control_anchors.npz --a2-scorer outputs/anchor_ablation/ckpt/A2_anchors/best.pt --num-samples 5000 --batch-size 128 --num-workers 0 --device auto
```

Planning-only evaluation (legacy stage name `feasibility`):

```bat
python evaluation/ablation/run_anchor.py --stage feasibility --methods A0 A1 A2 --checkpoint-a0 outputs/anchor_ablation/diffusion/A0/best.pt --checkpoint-a1 outputs/anchor_ablation/diffusion/A1/best.pt --checkpoint-a2 outputs/anchor_ablation/diffusion/A2/best.pt --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k --num-frames 200 --batch-size 16 --num-workers 0 --seeds 0,1,2 --device auto
```

Complete eight-metric run (existing command arguments are unchanged):

```bat
python evaluation/ablation/run_anchor.py --stage all --methods A0 A1 A2 --checkpoint-a0 outputs/anchor_ablation/diffusion/A0/best.pt --checkpoint-a1 outputs/anchor_ablation/diffusion/A1/best.pt --checkpoint-a2 outputs/anchor_ablation/diffusion/A2/best.pt --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k --num-frames 200 --batch-size 16 --num-workers 0 --seeds 0,1,2 --device auto
```

`--num-frames` controls synchronized frames in `feasibility/all`.
`--num-samples` controls independent raw ego samples in `anchors`. For
backwards convenience, `--num-frames` is accepted as an alias for
`--num-samples` only in `anchors`.

## Outputs

- `anchor_quality_per_sample.csv`
- `planning_per_sample.csv` (`feasibility/all`), including per-sample CR,
  DAC, external/inter-truck TTC diagnostics and TTC compliance
- `anchor_ablation_summary.csv` with the eight paper metrics plus sample-count
  and TTC diagnostics
- `anchor_ablation_table.tex` with four Anchor Quality and four Planning
  Performance columns
- `anchor_ablation.pdf` (`anchors/all`)
- `anchor_planning_ablation.pdf` (`feasibility/all`)
- `evaluation_manifest.json`, `evaluation_summary.json`

Figures retain the existing PDF layout, Times New Roman typography and vehicle
rendering. The planning figure displays the selected unprojected raw K2 path.

A1 is generated online from raw scenes; no precomputed A1 dataset is required.
The default `num_workers: 0` avoids extra PyTorch/CUDA processes on Windows.

## Anchor figure selection

When both A1 and A2 are evaluated, `figure_candidate_samples: 100` randomly
samples up to 100 observed ego scenes from the evaluated subset, using
`figure_seed`. It reuses the existing anchor-quality pass; it does not run an
additional model pass. `figure_scenes: 3` controls the final number of rows.
Increase the candidate count with `--figure-candidate-samples 200`; the pool
cannot exceed the number of evaluated samples.

Scenes are ranked by the reduction in unsafe anchors from A1 to A2 (collision
or road violation, counted once per anchor). Road and collision reductions
break ties. Positive improvements are preferred, with distinct episodes and
unique frames where possible. If too few positive examples exist, the best
available scenes are shown and the evaluator prints a notice.

These are **selected A2 advantage examples**, not a random illustration of
overall performance. Quantitative averages still use every evaluation sample.
`anchor_figure_selection.json` and `evaluation_manifest.json` record the
selection rule, all candidate safety counts, ranks and selected scene IDs.
Runs without both A1 and A2 retain random scene selection.

In `anchor_ablation.pdf`, unsafe A0/A1 anchors (collision or road violation)
use translucent red dashed lines from the same per-anchor safety labels used
for Safe Ratio. Safe anchors and A2 curves retain their normal colors. The
legend identifies the A0/A1 highlighting explicitly. Each subplot shows its
method name in a rectangular label with a translucent method-specific fill.
Style constants at the top of `visualization.py` control the unsafe-line color
and opacity, label position in axes fractions, and label fill colors/opacity.
