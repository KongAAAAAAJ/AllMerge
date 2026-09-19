# HANDOFF_EXPERT

## Window 01 result

This package productionizes the existing AllMerge expert pilot without creating
a second collector, expert, feature schema, or label implementation.

Files:

```text
EXPERT_PORTING_MAP.md
collect_expert_dataset.py
expert_dataset.py
split_expert_dataset.py
validate_expert_dataset.py
HANDOFF_EXPERT.md
```

The authoritative data path remains:

```text
AllMerge scenario
    ↓
Polynomial expert
    ↓
latest_planner_features
latest_expert_alignment
    ↓
collect_expert_pilot.build_frame_record()
    ↓
per-ego flatten only
    ↓
NPZ shards + splits + manifest
```

## Stored contract

One stored sample is one controlled ego at one valid 10-Hz planning instant.

Features:

```text
ego_state
agent_states
agent_valid_mask
map_polylines
map_valid_mask
target_point
target_lane_polyline
coarse_trajectories
mode_valid_mask
```

Training targets:

```text
expert_trajectory_xy   [8, 2]
target_mode             scalar, 0..9
target_semantic         scalar
```

Provenance/diagnostics are stored in the same shard, including episode, frame,
ego and global sample indices.

## 1. 100-sample overfit set

Run from the outer AllMerge project root (where `all_merge/` lives):

```bash
python all_merge/collect_expert_dataset.py \
  --target-samples 100 \
  --samples-per-shard 100 \
  --output-root outputs/expert_dataset \
  --dataset-name allmerge_overfit100 \
  --start-seed 1
```

Validate:

```bash
python all_merge/validate_expert_dataset.py \
  --dataset-root all_merge/outputs/expert_dataset/allmerge_overfit100
```

Acceptance:

```text
VALIDATION PASS
samples=100
contract_ok = 100%
trajectory batch shape = [B, 8, 2]
coarse_trajectories batch tail = [10, 8, 2]
```

## 2. 1k smoke set

```bash
python all_merge/collect_expert_dataset.py \
  --target-samples 1000 \
  --samples-per-shard 256 \
  --output-root outputs/expert_dataset \
  --dataset-name allmerge_smoke1k \
  --start-seed 1
```

```bash
python all_merge/validate_expert_dataset.py \
  --dataset-root all_merge/outputs/expert_dataset/allmerge_smoke1k
```

## 3. Formal bulk collection

Example:

```bash
python collect_expert_dataset.py \
  --target-samples 50000 \
  --samples-per-shard 2048 \
  --output-root /path/to/AllMerge_Data \
  --dataset-name allmerge_expert_50k \
  --start-seed 1
```

Interrupted run:

```bash
python collect_expert_dataset.py \
  --target-samples 50000 \
  --samples-per-shard 2048 \
  --output-root /path/to/AllMerge_Data \
  --dataset-name allmerge_expert_50k \
  --start-seed 1 \
  --resume
```

Resume state is reconstructed from existing shard contents. The manifest is a
reporting artifact, not the only recovery source.

## 4. Split regeneration

The collector already creates train/val/test shard lists. To regenerate:

```bash
python split_expert_dataset.py \
  --dataset-root /path/to/AllMerge_Data/allmerge_expert_50k \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 0
```

## 5. Pretraining-side consumption

```python
from torch.utils.data import DataLoader

from expert_dataset import build_dataset

train_set = build_dataset(
    "outputs/expert_dataset/allmerge_smoke1k",
    split="train",
)

loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

features, targets = next(iter(loader))

output = model.forward_train(
    features,
    targets["trajectory"],
    targets["target_semantic"],
)

loss = output["loss"]
```

`PlannerTensorAdapter` remains the only place for physical-unit scaling. Do not
create a second normalized feature cache unless a later profiling result
justifies it.

## 6. Acceptance gates before Window 02 pretraining

```text
[ ] 100-sample dataset validates
[ ] 100-sample overfit training drives loss down strongly
[ ] 1k dataset validates
[ ] DataLoader returns the exact 9-key batched contract
[ ] target_mode is always in [0, 9]
[ ] contract_ok is 100%
[ ] no NaN / Inf in floating features or expert trajectory
[ ] resume adds samples without overwriting existing shards
[ ] manifest records the actual AllMerge git commit
```

## Design decisions

### No mandatory visual preprocessing stage

Diffusion-metadrive's `run_diffusion_preprocess.sh` is tied to the previous
visual-feature pipeline and is optional even there. AllMerge now stores the
structured numeric planner input directly, while `PlannerTensorAdapter`
performs deterministic scaling at runtime. A second cached representation would
duplicate the contract.

### No offline abstract-anchor migration

Diffusion-metadrive's abstract-anchor flow clusters fixed trajectory anchors.
AllMerge now supplies dynamic `coarse_trajectories` with 10 modes for every
sample. These anchors must remain aligned with `mode_assignment.py`.

### Why flatten by ego

The pilot keeps multiple controlled vehicles together for diagnostics. The
planner API uses the leading dimension as batch. Flattening at the storage
boundary lets `DataLoader` produce exactly `[B,...]`, with no extra
`[batch, num_ego, ...]` dimension and no training-side workaround.
