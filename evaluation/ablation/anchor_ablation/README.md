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
   raw road-union center-point checks.
3. **Trajectory Diversity (m) ↑**: mean pairwise pointwise XY distance among
   the K anchor trajectories.
4. **Control Diversity (-) ↑**: mean pairwise pointwise control distance after
   channel normalization by `[2.5 m/s^2, 0.08 rad]`.

### Diffusion planning performance (4 metrics)

5. **Planning minADE (m) ↓**: minimum ADE among valid diffusion-generated
   8-point K2 execution trajectories.
6. **CR (%) ↓**: percentage of selected planning trajectories with at least
   one body-rectangle overlap (zero safety margin; boundary contact excluded).
   Controlled-truck dimensions come from the raw manifest (currently 8 x 2.6 m);
   background vehicles use their observed dimensions. Uncontrolled background
   actors are extrapolated with constant velocity; synchronized controlled
   trucks are checked against their own raw K2 predictions.
   All methods and Expert use the same per-frame prefix before the earliest
   detected expert lane-end clipping. Earlier collisions are retained; samples
   without valid future nodes are excluded from the CR denominator. CSV/JSON
   record the valid future steps, cutoff, clipping reason and CR sample counts.
   This clipping rule changes only CR, not minADE, DAC, TTC or Safe Ratio.
7. **DAC (%) ↑**: mean per-trajectory percentage of valid future steps without confirmed lateral
   road-boundary violations, using the configured multi-circle vehicle envelope.
   End-cap unknown steps are excluded unless a local side violation is confirmed.
   Trajectories with no valid steps have no DAC value and are omitted from the mean.
8. **TTC (%) ↑**: percentage of selected planning trajectories whose minimum
   predicted TTC to uncontrolled actors or synchronized controlled trucks
   is greater than or equal to the configured threshold. The default threshold
   is 1.0 s and can be changed in `configs/evaluation/anchor_ablation.yaml`.

**Planning safety metrics are evaluated on the deployment-selected raw K2
trajectory before JTP.** The raw K2 poses are in the ego-at-planning-time local
frame and are transformed to the world frame only for synchronized
inter-truck and road-union checks. JTP is not instantiated or invoked:
there is no JTP constraint evaluation, iterative projection, projection
correction or projected-trajectory fallback.

The existing `--jtp-config` argument is retained for command compatibility.
Its YAML supplies the execution time step and the multi-circle footprint for
TTC and drivable-area calculations. CR uses body dimensions from the raw
manifest instead of the multi-circle footprint. It does not invoke JTP.

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

`feasibility/all` also adds an **Expert trajectory** reference row to the table
and CSV/JSON summaries. Its four anchor-quality entries are `-` in LaTeX. The
planning minADE is zero by definition (expert reference against itself); CR,
DAC and TTC use the YAML-selected `trajectory_source` directly, without
diffusion or K2 prediction. `polynomial` reads `polynomial_future_xy`, with
headings estimated from path tangents. `trucksim_closed_loop` reads
`trucksim_future_poses`, including actual body yaw. The observed initial
local pose is zero.

Expert checks use exactly the same evaluated synchronized scenes, vehicle
footprint, road geometry, CV background prediction and safety thresholds as
the model rows. Controlled trucks use their synchronized expert paths in this
row. Each expert scene is evaluated once rather than repeated for inference
seeds. `planning_per_sample.csv` uses method `Expert` and an empty inference
seed; the manifest documents the reference protocol. `anchors` alone omits
this row because it does not evaluate synchronized planning safety.

Controlled-truck collision flags are per vehicle: a collision between two
trucks marks those two trucks, not every truck in the synchronized frame.
TTC is also evaluated per truck. The existing 8 x 2.6 m collection/anchor
settings and the existing JTP footprint are retained.

New Polynomial expert predictions follow connected route lanes. Beyond a
terminal lane, positions continue along its endpoint tangent instead of all
being clamped to one endpoint. This prediction extension does not extend the
simulator's drivable road. Previously stored clipped expert labels are not
rewritten; existing-data retests still include those legacy label artifacts.

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

The figure omits Target markers and uses thinner expert lines. Axis names are
shared across the figure; each row keeps its own tick values in the first
column because scene limits can differ. Row and column gaps are set in inches
inside `_finish_anchor_grid`.

A1/A2 panels include zoom insets with selection rectangles and connecting
lines. Both columns in the same row always use identical zoom limits:

```python
ANCHOR_INSET_DEFAULTS = {
    'enabled': True,
    'bounds': (0.055, 0.15, 0.48, 0.48),  # left, bottom, width, height; axes fractions
}
ANCHOR_INSET_ROW_LIMITS = {
    0: {'xlim': None, 'ylim': None},  # first row; None = automatic
    1: {'xlim': None, 'ylim': None},
    2: {'xlim': None, 'ylim': None},
}
```

For example, set row `0` to `{'xlim': (40, 85), 'ylim': (-3, 12)}` to apply that
range in metres to both A1/A2 panels. Change individual inset positions/sizes
with `ANCHOR_INSET_OVERRIDES = {(0, 'A1'): {'bounds': (0.03, 0.08, 0.50, 0.50)}}`.
Set `enabled` to `False` globally or in an override to hide insets. Insets
reuse the plotted trajectories and safety labels without additional evaluation.

DAC uses a separate evaluation-only copy of the raw lane-edge polygon union.
Closed holes whose 1-mm inward buffer is empty are filled as numerical seams
(a fixed sub-2-mm-width assumption); the exterior is never buffered and larger
holes and disconnected areas remain. Exposed boundary segments are matched to
raw lane end cross-sections with 0.1-mm tolerance. Both actual lane ends and
preview truncations are unknown beyond their caps; side segments use local
projection and signed inward distance, without extending endpoint tangents.
Confirmed side violations take precedence over endpoint exclusion. Circle
footprints, contact tolerance and the future time grid are unchanged.

Per-sample CSV/JSON records `dac_valid_steps`, `dac_endpoint_excluded_steps`
and `dac_side_failure_steps`; summaries include their totals. DAC remains a
mean of per-trajectory percentages rather than a pooled time-step rate. A zero
valid denominator is empty (`--` in the table), not 100%. Different methods may
have different effective denominators; compare the reported counts as well.
Old evaluation outputs require new inference to obtain revised DAC values.

Shared road geometry, anchor teacher labels, training losses, CR and TTC are
unchanged. The anchor teacher still checks centres against the original raw
road union, including holes, with its existing hard threshold `1e-4 m`.
Shapely 2.x is required.

## 双轨迹数据与评估来源

仅支持新格式 `trace_expert_raw_k2_v3`，不兼容旧数据。评估 YAML 顶层设置：

```yaml
trajectory_source: polynomial  # 或 trucksim_closed_loop
```

该标签统一选择 minADE 参考、专家基准和图中的专家轨迹。所加载 diffusion
checkpoint 的训练来源必须一致；缺失或不一致直接报错。输出 manifest 记录来源。
TruckSim 来源表示持续重规划下的实际未来 4 秒，不是固定当前规划的跟踪轨迹；
不使用 Polynomial 的冻结尾部裁剪规则。两种来源始终使用同一组完整未来样本。
A0 控制库和 A2 scorer 不增加来源开关；共享候选池可供新数据的两种来源共用。
