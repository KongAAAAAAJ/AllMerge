# TRACE Adaptive Anchor Ablation

The evaluator preserves three stages:

- `anchors`: Anchor Quality only; uses the fixed A0 bank, paired A1 anchors in expert shards, and the trained A2 safety scorer. No diffusion checkpoints are required.
- `feasibility`: Anchor-conditioned diffusion planning; requires the selected diffusion checkpoints and evaluates raw plans before JTP projection, then measures JTP correction.
- `all`: Runs both parts with one common test subset and representative scene. Use this stage for the complete seven-metric paper result.

## Seven paper metrics

1. Anchor minADE (m)
2. Safe Ratio (%)
3. Trajectory Diversity (m)
4. Control Diversity (normalized)
5. Planning minADE (m)
6. Pre-projection Feasibility (%)
7. Projection Correction (m)

A1 `mode_valid_mask` is ignored for geometric validity because it mixes traffic and geometry availability. Instead, the evaluator keeps only complete synchronized frames where every A1 ego sample has ten finite, geometrically instantiated trajectories. Zero-filled unavailable slots are never counted as anchors.

Pre-projection feasibility calls the same `JointTrajectoryProjector.evaluate_preprojection()` constraint evaluator used by JTP, without applying projection. It includes JTP internal constraint families plus separate constant-velocity external-actor and synchronized inter-truck collision checks. `JointTrajectoryProjector.project()` is then applied to measure the correction. Projection success rate and failure counts are reported as diagnostics.

## Commands

```bat
python evaluation/ablation/run_anchor.py --stage anchors --methods A0 A1 A2

python evaluation/ablation/run_anchor.py --stage feasibility --methods A0 ^
  --checkpoint-a0 outputs/anchor_ablation/A0_fixed/best.pt

python evaluation/ablation/run_anchor.py --stage all ^
  --checkpoint-a0 outputs/anchor_ablation/A0_fixed/best.pt ^
  --checkpoint-a1 outputs/anchor_ablation/A1_dynamic/best.pt ^
  --checkpoint-a2 outputs/anchor_ablation/A2_adaptive/best.pt
```

Optional arguments include `--a0-bank`, `--a2-scorer`, `--scorer-config`, `--jtp-config`, `--dataset-root`, `--split`, `--num-frames`, `--batch-size`, `--num-workers`, `--seeds`, `--sample-seed`, `--device`, and `--output-dir`.

Outputs are isolated under `outputs/anchor_ablation/evaluation/<stage>/<methods>/` unless `--output-dir` is explicitly supplied. The complete `all` stage creates both 1×3 paper figures, the seven-metric CSV/LaTeX summary, and an evaluation manifest.

The evaluator does not use MATP. It uses the existing JTP implementation for joint constraint evaluation and projection.
