# TRACE Anchor Ablation Online V1 (seven-metric staged evaluator)

This version is rebuilt against the 2026-11-18 AllMerge project snapshot and
uses the current anchor-independent raw expert dataset plus the current online
A1AnchorManager/A1AnchorBuilder interface. It does not support legacy datasets.

## Seven paper metrics

### Anchor quality stage (4 metrics)
1. Anchor minADE (m): minimum ADE between K online anchors and the 8-point expert trajectory.
2. Safe Ratio (%): hard-safe selected anchors divided by K, using the A2 safety teacher's constant-velocity external-actor collision and drivable-road boundary checks.
3. Trajectory Diversity (m): mean pairwise pointwise XY distance among K anchors.
4. Control Diversity (normalized): mean pairwise pointwise control distance normalized by [2.5 m/s^2, 0.08 rad].

### Diffusion planning stage (3 metrics)
5. Planning minADE (m): minimum ADE among valid diffusion-generated 8-point execution trajectories.
6. Pre-projection Feasibility (%): JTP nominal constraints plus independent constant-velocity external-actor and synchronized inter-truck collision checks, before projection.
7. Projection Correction (m): mean pointwise XY displacement between raw and successfully projected execution trajectories. Projection success/failure counts and rate are reported separately.

## Staged entry points

`--stage anchors` computes the four anchor metrics using independent raw controlled-ego samples. It does not scan/group the entire dataset into synchronized frames. A1 is regenerated online from `ego_state` and raw road geometry. It generates `anchor_ablation.pdf` using the existing publication figure layout and style.

`--stage feasibility` computes the three diffusion/JTP metrics using complete synchronized frames, because the inter-truck collision component requires synchronized ego groups. It generates `anchor_planning_ablation.pdf`.

`--stage all` computes all seven metrics and both PDFs. It uses complete synchronized frames for a consistent planning/anchor joint comparison.

## Smoke commands

Anchor-only A0/A1 smoke (A2 scorer not required):

```bat
python evaluation/ablation/run_anchor.py --stage anchors --methods A0 A1 --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k --a1-dataset-root outputs/anchor_ablation/ckpt/A1_anchors --a0-bank outputs/anchor_ablation/ckpt/a0_fixed_control_anchors.npz --num-samples 500 --batch-size 128 --num-workers 4 --device auto
```

After A2 scorer training:

```bat
python evaluation/ablation/run_anchor.py --stage anchors --methods A0 A1 A2 --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k --a1-dataset-root outputs/anchor_ablation/ckpt/A1_anchors --a0-bank outputs/anchor_ablation/ckpt/a0_fixed_control_anchors.npz --a2-scorer outputs/anchor_ablation/ckpt/A2_anchors/best.pt --num-samples 5000 --batch-size 128 --num-workers 4 --device auto
```

## Planning-stage entry points

```bat
python evaluation/ablation/run_anchor.py --stage feasibility --methods A0 A1 A2 --checkpoint-a0 outputs/anchor_ablation/diffusion/A0/best.pt --checkpoint-a1 outputs/anchor_ablation/diffusion/A1/best.pt --checkpoint-a2 outputs/anchor_ablation/diffusion/A2/best.pt --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k --num-frames 200 --batch-size 16 --num-workers 4 --seeds 0,1,2 --device auto
```

Complete seven-metric run:

```bat
python evaluation/ablation/run_anchor.py --stage all --methods A0 A1 A2 --checkpoint-a0 outputs/anchor_ablation/diffusion/A0/best.pt --checkpoint-a1 outputs/anchor_ablation/diffusion/A1/best.pt --checkpoint-a2 outputs/anchor_ablation/diffusion/A2/best.pt --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k --num-frames 200 --batch-size 16 --num-workers 4 --seeds 0,1,2 --device auto
```

`--num-frames` controls synchronized frames in `feasibility/all`. `--num-samples`
controls independent raw ego samples in `anchors`. For backwards convenience,
`--num-frames` is accepted as an alias for `--num-samples` only in `anchors`.

## Outputs

- `anchor_quality_per_sample.csv`
- `planning_per_sample.csv` (feasibility/all)
- `anchor_ablation_summary.csv` with all seven columns (stage-inapplicable columns blank)
- `anchor_ablation_table.tex`
- `anchor_ablation.pdf` (anchors/all)
- `anchor_planning_ablation.pdf` (feasibility/all)
- `evaluation_manifest.json`, `evaluation_summary.json`

The existing `visualization.py` is retained unchanged, preserving the PDF format,
Times New Roman typography, panel arrangement, line styles, legend placement,
vehicle rendering and trajectory display semantics.
