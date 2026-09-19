# EXPERT_PORTING_MAP

## Baseline and migration rule

This window follows the migration-first rule: reuse the existing AllMerge pilot
contract and port only the production data-engineering pieces that are missing.

The migration deliberately does **not** redesign:

- the 9-key planner feature contract;
- the Polynomial expert;
- F.1/F.2 mode assignment;
- the 8 × 0.5 s trajectory horizon;
- the 10 dynamic modes;
- the already migrated diffusion planner trunk.

The actual AllMerge repository commit is recorded at collection time with
`git rev-parse HEAD`. No new hard-coded main-branch commit is introduced.

## Porting table

| Diffusion-metadrive source | Decision | AllMerge target | Notes |
| --- | --- | --- | --- |
| `metadrive/exp_dataset/collect_expert.py::ShardWriter` | ADAPT | `collect_expert_dataset.py::ShardWriter` | Preserve compressed NPZ sharding. Add schema-drift checks and atomic writes. |
| `collect_expert.py::detect_existing_state` | ADAPT | `collect_expert_dataset.py::detect_existing_state` | Resume state is reconstructed from shard contents, not only `manifest.json`. |
| `collect_expert.py::write_manifest` | ADAPT | `collect_expert_dataset.py::write_manifest` | Preserve manifest/resume history; replace MetaDrive visual/IDM fields with AllMerge feature/label metadata. |
| `collect_expert.py` MetaDrive frame extraction | DROP | existing `collect_expert_pilot.py` | AllMerge already exposes `latest_planner_features` and `latest_expert_alignment`. |
| `collect_expert.py` IDM/PPO expert logic | DROP | existing Polynomial expert | Do not create a second expert policy. |
| `collect_expert.py` camera/LiDAR/BEV storage | DROP | 9-key structured features | AllMerge structured planner does not consume the old visual observation contract. |
| `collect_expert.py` trajectory correction/filter | DROP | existing Polynomial trajectory + label contract | Dataset production must not alter expert geometry. |
| `metadrive/exp_dataset/metadrive_dataset.py::MetaDriveShardDataset` | ADAPT | `expert_dataset.py::AllMergeExpertShardDataset` | Preserve lazy shard indexing/cache; replace keys with AllMerge structured features and targets. |
| `metadrive_dataset.py::split_shards` | COPY/ADAPT | `expert_dataset.py::split_shards` | Preserve deterministic shard-level train/val/test splitting. |
| `scripts/run_dataset_collect.sh` | ADAPT | command recipes in `HANDOFF_EXPERT.md` | Same output-root / dataset-name / target-samples workflow. |
| `scripts/run_diffusion_preprocess.sh` | DROP as mandatory stage | validation + existing `PlannerTensorAdapter` | Structured numeric features are already training-ready; physical scaling remains online in `PlannerTensorAdapter`. |
| `scripts/run_abstract_anchors.sh` + `abstract_anchors.py` | DROP | existing dynamic `coarse_trajectories` | AllMerge already supplies 10 per-sample dynamic anchors. Offline clustered anchors would conflict with the current planner. |

## AllMerge-specific glue added

### 1. Frame-to-ego flattening

`collect_expert_pilot.py` is a diagnostic collector. One record contains all
controlled vehicles at one planning frame.

The production dataset stores **one controlled ego as one sample**. Therefore
PyTorch `DataLoader` produces the model batch dimension directly:

```text
ego_state              [B, D_ego]
agent_states           [B, N_agent, D_agent]
agent_valid_mask       [B, N_agent]
map_polylines          [B, N_map, P, D_map]
map_valid_mask         [B, N_map]
target_point           [B, 2]
target_lane_polyline   [B, P, D_map]
coarse_trajectories    [B, 10, 8, 2]
mode_valid_mask        [B, 10]
```

No model-side extra vehicle dimension is introduced.

### 2. Single source of truth for labels

The production collector imports and calls the existing:

```python
collect_expert_pilot.build_frame_record(...)
```

It does **not** reimplement `assign_expert_mode()` or recompute
`target_mode/target_semantic`.

### 3. Sharding and crash-safe resume

Production output:

```text
<dataset_root>/
├── shards/
│   ├── shard_000000.npz
│   ├── shard_000001.npz
│   └── ...
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── reports/
    ├── manifest.json
    └── validation_report.json
```

Resume scans the shard contents and recovers:

- sample count;
- next shard index;
- next episode index;
- next frame index;
- next global sample index.

This remains recoverable even if a previous process died before rewriting the
manifest.

### 4. Dataset/DataLoader contract

`expert_dataset.py` returns:

```python
features = {
    # unchanged 9 keys
}

targets = {
    "trajectory": expert_trajectory_xy,  # [8, 2]
    "target_mode": target_mode,
    "target_semantic": target_semantic,
}
```

Optional metadata and diagnostics can also be requested.

### 5. Validation

`validate_expert_dataset.py` verifies:

- required keys;
- cross-shard shape/dtype consistency;
- finite floating-point values;
- trajectory `[8,2]`;
- dynamic anchors `[10,8,2]`;
- `target_mode ∈ [0,9]`;
- `contract_ok == True`;
- actual DataLoader batch shapes.

## Frozen files

This migration does not modify:

```text
highway_env/planner/diffusion/config.py
highway_env/planner/diffusion/tensor_adapter.py
highway_env/planner/diffusion/encoders.py
highway_env/planner/diffusion/diffusion_schedule.py
highway_env/planner/diffusion/mode_assignment.py
highway_env/planner/diffusion/structured_model.py
highway_env/planner/diffusion/runtime.py
```

It also does not modify the Polynomial expert or scenario feature schema.
