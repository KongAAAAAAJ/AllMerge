# W2 Diffusion Pretraining — PRETRAIN_PORTING_MAP

Baseline: AllMerge `d946cef` + W1 `data_pipeline` handoff.
Rule: migrate first, adapt second, add only AllMerge-specific glue/tests.

| Diffusion-metadrive source | Decision | AllMerge target | Reason / adaptation |
|---|---|---|---|
| `scripts/run_diffusion_train.sh` | ADAPT | `scripts/run_diffusion_pretrain.sh` | Preserve one-command training entrypoint; replace MetaDrive dataset/anchor arguments with W1 dataset root + W2 config. |
| `metadrive.policy.diffusion_policy.train_transfuser` | ADAPT | `train_diffusion_pretrain.py` | Preserve Lightning `Trainer`, run directories, TensorBoard, ModelCheckpoint, precision and validation cadence. Replace dataset and model APIs only. |
| `metadrive.policy.diffusion_policy.transfuser_agent.TransfuserAgent` | ADAPT | `pretraining/lightning_module.py` | Keep LightningModule / optimizer / scheduler structure. Replace `V2TransfuserModel` with existing `StructuredDiffusionPlanner`; call `forward_train()` directly. |
| `modules/scheduler.WarmupCosLR` | COPY | `pretraining/warmup_cos_lr.py` | Framework-independent; copied with import-path cleanup only. |
| `transfuser_config.py` training fields | ADAPT | `configs/diffusion_pretrain.yaml` | Keep AdamW, weight decay, warmup/cosine, epochs, precision, dataloader workers. Camera/LiDAR/BEV fields are dropped. |
| `transfuser_features.MetaDriveTransfuserDataset` | DROP | W1 `data_pipeline.expert_dataset.ExpertDataset` | Replaced by frozen `allmerge_planner_v1` shards and collate contract. |
| `V2TransfuserModel.forward/loss` | DROP | existing `StructuredDiffusionPlanner.forward_train()` | All target-mode assignment, diffusion-noise construction, trajectory regression and classification loss already live in AllMerge model. |
| `transfuser_loss.py` | DROP | none | Recomputing loss in Trainer is explicitly forbidden. |
| `transfuser_callback.py` camera/LiDAR visualizer | DROP | none | MetaDrive perception-specific. W2 validation is metric/log based. |
| `eval_transfuser_open_loop.py` | ADAPT | `eval_diffusion_open_loop.py` | Reuse ADE/FDE concept; add minADE/minFDE and raw/masked mode metrics for 10-mode structured planner; drop camera/LiDAR rendering. |
| `scripts/run_diffusion_open_loop_eval.sh` | ADAPT | `scripts/run_diffusion_open_loop_eval.sh` | New dataset/model arguments. |
| `scripts/run_diffusion_test.sh` closed-loop MetaDrive test | DROP | `test_diffusion_runtime_load.py` | W2 scope is pretraining + checkpoint/runtime compatibility. Scenario closed-loop evaluation belongs integration/evaluation windows. |
| checkpoint/resume | ADAPT | Lightning `.ckpt` + `pretraining/checkpoint_io.py` | Lightning checkpoint remains resumeable; planner-only state is additionally embedded and exported to Runtime-compatible `.pt`. |
| MetaDrive/NavSim camera/LiDAR/NavSim dependencies | DROP | none | AllMerge structured features are authoritative. |
| W1/W2 schema guard | NEW | `pretraining/contract.py` | Minimal glue to prevent silent cross-window interface drift. |
| W2 runtime load test | NEW | `test_diffusion_runtime_load.py` | Required because frozen runtime expects a planner-only `state_dict`, while Lightning resume checkpoints contain wrapper prefixes. |

## Frozen ownership boundary

W2 does **not** modify:

```text
highway_env/planner/diffusion/config.py
highway_env/planner/diffusion/tensor_adapter.py
highway_env/planner/diffusion/encoders.py
highway_env/planner/diffusion/diffusion_schedule.py
highway_env/planner/diffusion/mode_assignment.py
highway_env/planner/diffusion/structured_model.py
highway_env/planner/diffusion/runtime.py
```

W2 consumes W1 directly:

```text
batch["features"]
batch["expert_trajectory"]  -> StructuredDiffusionPlanner.forward_train(..., target_trajectory=...)
batch["expert_semantic"]    -> StructuredDiffusionPlanner.forward_train(..., target_semantic=...)
batch["expert_mode"]        -> diagnostic only
```

No target mode, diffusion target, regression loss, or classification loss is recomputed in W2.
