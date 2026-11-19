采集专家数据命令：
python scripts\data\collect_expert_dataset.py ^
  --config configs\data\expert_collection.yaml ^
  --dataset-name expert_trucksim_50k ^
  --target-samples 50000 ^
  --min-episodes 252 ^
  --duration 12 ^
  --max-episodes 30000 ^
  --no-resume

构建 A0 anchor 命令：
python scripts/data/build_a0_fixed_anchors.py --max-accel-mps2 2.5 --max-steer-rad 0.08 ^
--dataset-root outputs/expert_dataset/expert_trucksim_50k ^
--output outputs/anchor_ablation/ckpt/A0_anchors/A0_control_anchors.npz

构建 A1 anchoor：
A1 在 diffusion 训练、开环/闭环推理及消融评估时根据 raw 场景在线生成，无需单独构建或保存锚点库。数量读取 diffusion.yaml 的 control.model.num_modes，生成参数位于 configs/anchors/a1.yaml。

训练 A2 命令：
python scripts/data/train_a2_anchors.py ^
  --config configs/train/a2_scorer.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_50k ^
  --max-epochs 30 ^
  --output-dir outputs/anchor_ablation/ckpt/A2_anchors ^
  --cache-dir outputs/anchor_ablation/cache_A2

Anchor Ablation，评估A0 A1 A2命令：
开环指标：
python evaluation/ablation/run_anchor.py --stage anchors --methods A0 A1 A2
开环 + 闭环指标：
python evaluation/ablation/run_anchor.py ^
  --stage all ^
  --methods A0 A1 A2 ^
  --split test ^
  --checkpoint-a0 outputs/anchor_ablation/diffusion_ckpt/A0/best.pt ^
  --checkpoint-a1 outputs/anchor_ablation/diffusion_ckpt/A1/best.pt ^
  --checkpoint-a2 outputs/anchor_ablation/diffusion_ckpt/A2/best.pt ^
  --jtp-config configs/projection/jtp.yaml ^
  --num-workers 0

  --stage anchors 前四项指标      --stage feasibility Planning minADE 和 Pre-proj. Feas.      --stage all 六项指标；不执行迭代投影，Proj. Corr. 为 --
