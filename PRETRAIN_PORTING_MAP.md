# PRETRAIN_PORTING_MAP

## Baseline

AllMerge main after W1 expert-dataset acceptance. The frozen planner trunk remains unchanged:
`config.py`, `tensor_adapter.py`, `encoders.py`, `diffusion_schedule.py`,
`mode_assignment.py`, `structured_model.py`, `runtime.py`.

## Dependency rule

W2 must run in the existing AllMerge environment. `requirements.txt` provides
PyTorch and TensorBoard but does not require PyTorch Lightning or PyYAML.
Therefore the migrated orchestration uses pure PyTorch and a JSON config.
No new training-framework dependency is introduced.

## Porting table

| Diffusion-metadrive source | Decision | AllMerge target | Adaptation |
| --- | --- | --- | --- |
| diffusion train entry / train-val loop | ADAPT | `train_diffusion_pretrain.py`, `pretraining/trainer.py` | Preserve train/val/optimizer/checkpoint flow; replace Lightning wrapper with dependency-free PyTorch orchestration. |
| optimizer | COPY/ADAPT | `pretraining/trainer.py` | AdamW retained. |
| warmup + cosine scheduler | COPY | `pretraining/warmup_cos_lr.py` | Mathematical schedule retained. |
| checkpoint / resume | ADAPT | `pretraining/trainer.py`, `checkpoint_io.py` | Save planner, optimizer, scheduler, scaler, epoch and global step; runtime export stays planner-only. |
| TensorBoard logging | COPY/ADAPT | `pretraining/trainer.py` | Uses `torch.utils.tensorboard`; JSONL fallback if TensorBoard is unavailable. |
| old dataset | DROP | W1 `expert_dataset.py` | Consume W1 `build_dataset()` directly. |
| V2Transfuser model/loss | DROP | `StructuredDiffusionPlanner.forward_train()` | Planner owns target assignment and supervised losses. |
| camera/LiDAR/NavSim preprocessing | DROP | structured W1 features | Not part of current AllMerge contract. |
| open-loop evaluation | ADAPT | `eval_diffusion_open_loop.py` | ADE/FDE, minADE/minFDE and mode metrics retained for current output format. |

## Hard boundary

The trainer must not recompute `target_mode`, semantic assignment, diffusion
noise targets, trajectory regression loss, or classification loss. Those remain
inside `StructuredDiffusionPlanner.forward_train()`.
