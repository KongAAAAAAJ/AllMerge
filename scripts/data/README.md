采集专家数据命令：
python scripts\data\collect_expert_dataset.py ^
  --config configs\data\expert_collection.yaml ^
  --dataset-name smoke_actuator_constraints_v2 ^
  --trucksim-model-id L0_MU08 ^
  --target-samples 3000 ^
  --min-episodes 4 ^
  --duration 12 ^
  --max-episodes 300 ^
  --no-resume

构建 A0 anchor 命令：
python scripts/data/build_a0_fixed_anchors.py --max-accel-mps2 2.5 --max-steer-rad 0.08 ^
--dataset-root outputs/expert_dataset/smoke_actuator_constraints_v2 ^
--output outputs/anchor_ablation/ckpt/A0_anchors/A0_control_anchors_smoke2.npz

A1 在 diffusion 训练、开环/闭环推理及消融评估时根据 raw 场景在线生成，无需单独构建或保存锚点库。数量读取 diffusion.yaml 的 control.model.num_modes，生成参数位于 configs/anchors/a1.yaml。

训练 A2 命令：
python -m highway_env.planner.anchor_scoring.train ^
  --config configs/train/adaptive_anchors.yaml ^
  --dataset-root outputs/expert_dataset/smoke_actuator_constraints_v2 ^
  --max-epochs 2 --max-batches 2 --max-samples 64 ^
  --output-dir outputs/anchor_ablation/ckpt/A2_smoke2 ^
  --cache-dir outputs/anchor_ablation/cache_A2_smoke2

评估A0 A1 A2命令：
python evaluation/ablation/run_anchor.py --stage anchors --methods A0 A1 A2