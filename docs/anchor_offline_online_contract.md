# TRACE A0/A1/A2 anchor offline-online runtime contract

The three anchor strategies share the same high-level lifecycle but have
different online behavior. Offline artifacts are training inputs or learned
parameters; they are not replayed as scene-specific future trajectories.

## A0: fixed control anchors

- Offline: `scripts/data/build_a0_fixed_anchors.py` clusters the training split
  control sequences in normalized 40-step control space and writes K-means
  centroids to the fixed-anchor bank.
- Online: `FixedAnchorManager` loads that bank once, broadcasts the fixed
  controls, and rolls them out from the current observed ego speed.
- Config: `control.anchors.strategy: a0` (or legacy alias `fixed`) and
  `bank_path`.

A0 reads the learned/fitted fixed control bank online. The associated XY
trajectories are regenerated for the current initial speed.

## A1: road-conditioned lattice anchors

- Offline: `scripts/data/build_a1_anchors.py` runs `A1AnchorBuilder` over the
  immutable Raw Expert episodes and writes paired A1 controls, trajectories,
  validity masks, and episode-level train/val/test shards.
- Online: `A1AnchorManager` invokes the same `A1AnchorBuilder` using the live
  ego state and full-resolution current road geometry. It never reads the
  offline sample's stored trajectories during online inference.
- Config: `control.anchors.strategy: a1` and
  `a1_config: configs/anchors/a1.yaml`.

Offline training with strategy `a1` consumes the already-built
`coarse_controls`, `coarse_trajectories`, and `mode_valid_mask` from the
selected A1 dataset. Runtime inference sets an explicit online flag and
requires live `road_lane_geometry`, masks, lane slots, connections, and lane
metadata. Missing live road geometry is a hard error; it does not fall back to
the legacy feature anchors.

## A2: learned safety-ranked adaptive anchors

- Offline: `AdaptiveCandidateBuilder` creates the 36-candidate training pool;
  the safety teacher creates deterministic collision/road-boundary labels;
  `anchor_scoring.train` caches these targets and trains
  `AnchorSafetyScorer`.
- Online: `AdaptiveAnchorManager` creates the same 36-candidate pool from
  current scene features, runs the frozen scorer, ranks candidates, and applies
  Top-R plus trajectory/control diversity selection to return `num_selected`
  anchors.
- Config: `control.anchors.strategy: a2` (or legacy alias `adaptive`),
  `scorer_checkpoint`, and the exact `candidate_generation` configuration
  used during training.

A2 is a distinct 36-candidate generation/scoring pipeline. It does not score
the ten fixed semantic A1 slots unless its candidate generator is explicitly
changed and retrained. The checkpoint's candidate-generation contract must
match the online configuration.

## Runtime feature contract for A1

`AllMergeFeatureBuilder` continues to create the standard model features.
When the loaded control-diffusion checkpoint uses strategy `a1`, the action
pipeline additionally calls `build_road_geometry_batch()` at every online
planning cycle and passes that geometry to `ControlDiffusionRuntime`.
`A1AnchorManager` converts the live road arrays into ten controls
`[B,10,40,2]`, sparse trajectories `[B,10,8,2]`, and validity masks
`[B,10]`. These replace any legacy/stale coarse-anchor fields for that cycle.

A1's builder parameters must match the control-diffusion horizon, sample time,
wheelbase, acceleration limit, and steering limit. Current supported contract:
40 control steps at 0.1 s and ten semantic slots.

## Configuration examples

A0:
```yaml
anchors:
  strategy: a0
  num_selected: 10
  bank_path: outputs/anchor_ablation/a0_fixed_control_anchors.npz
```

A1:
```yaml
anchors:
  strategy: a1
  num_selected: 10
  a1_config: configs/anchors/a1.yaml
```

A2:
```yaml
anchors:
  strategy: a2
  num_selected: 10
  top_r: 20
  scorer_checkpoint: outputs/trace_adaptive_anchors_formal/best.pt
  candidate_generation: {}
  trajectory_weight: 1.0
  control_weight: 1.0
```

## Validation before closed-loop evaluation

1. Verify each offline artifact: A0 bank hash, A1 dataset manifest and A2
   scorer architecture/candidate-generation metadata.
2. Run the corresponding runtime manager on a held-out live scene.
3. Check output shapes, finite controls/trajectories, and valid candidate
   counts.
4. For A1, compare online generation with offline `A1AnchorBuilder` on the
   same captured state and road geometry; outputs must agree within numerical
   tolerance.
5. For A2, record candidate generation and scoring latency and confirm the
   online generator configuration exactly matches the scorer checkpoint.
6. Only then enable the selected strategy in the control-diffusion checkpoint
   and proceed to full closed-loop testing.
