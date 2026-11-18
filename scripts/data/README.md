构建 A0 anchor 命令：
python scripts/data/build_a0_fixed_anchors.py --max-accel-mps2 2.5 --max-steer-rad 0.08 ^
--dataset-root outputs/expert_dataset/smoke_actuator_constraints_v2 ^
--output outputs/anchor_ablation/ckpt/A0_anchors/A0_control_anchors_smoke2.npz

构建 A1 anchor 命令：
python scripts/data/build_a1_anchors.py ^
  --input-dir outputs/expert_dataset/smoke_actuator_constraints_v2 ^
  --output-dir outputs/anchor_ablation/ckpt/A1_anchors_smoke2 ^
  --config configs/anchors/a1.yaml ^
  --diffusion-config configs/train/diffusion.yaml

评估A0 A1 A2命令：
python evaluation/ablation/run_anchor.py --stage anchors --methods A0 A1 A2

训练 A2 命令：
python -m highway_env.planner.anchor_scoring.train --max-epochs 2 --max-batches 2 --max-samples 64 --output-dir outputs/anchor_ablation/ckpt/A2_smoke2 --cache-dir outputs/anchor_ablation/cache_A2_smoke2

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