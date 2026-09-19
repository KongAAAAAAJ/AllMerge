# W2 Diffusion Pretraining Migration — Static Verification

Date: 2026-09-19
Baseline contract: AllMerge `d946cef`.

## Passed in this build environment

1. `python -m py_compile` on all W2 Python files: **PASS**.
2. W1 synthetic shard -> real `ExpertDataset` -> `collate_expert_samples` -> W2 `validate_training_batch`: **PASS**.
3. `pretraining.checkpoint_io` planner-prefix stripping + runtime checkpoint export: **PASS**.
4. Migrated `WarmupCosLR` state/step smoke: **PASS**.
5. Lightning wrapper API/optimizer smoke using a stub with the exact frozen `forward_train(features, target_trajectory, target_semantic)` interface: **PASS**.
6. `apply_diffusion_pretraining_migration.py` dry-run and install into a fake compatible checkout: **PASS**.
7. Frozen planner ownership check: hash of sentinel `structured_model.py` unchanged before/after installer: **PASS**.
8. After adding the real W1 `data_pipeline` package to that installed checkout, `python -m pretraining.smoke_test_contract`: **PASS**.

## Source-level interface verification

The frozen AllMerge `StructuredDiffusionPlanner.forward_train()` owns:

- semantic-constrained expert mode assignment;
- random diffusion timestep / noise construction;
- denoising prediction;
- selected-mode Smooth-L1 trajectory regression;
- raw-logit mode classification;
- combined training loss and assignment diagnostics.

Therefore W2 intentionally does not reproduce those calculations.

## Not claimed here

These require the actual AllMerge checkout plus real W1 shards and should be run after W1/W2 integration:

- real Planner `forward_train()` numerical step;
- 100-sample overfit;
- 1k-sample smoke training;
- strict `DiffusionPlannerRuntime` load with a newly trained real checkpoint;
- formal pretraining.

The corresponding scripts and report templates are included in this migration bundle.
