Diffusion 训练命令 A0：
python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_a0_fixed.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_50k ^
  --max-epochs 2 ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/A0_smoke

Diffusion 训练命令 A1：
python scripts/training/run_diffusion.py ^
  --config configs/train/diffusion.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_50k ^
  --max-epochs 2 ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/A1_smoke

Diffusion 训练命令 A2：
python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_a2_adaptive.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_50k ^
  --max-epochs 60 ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/A2

从已保存的 A0 checkpoint 续训：
python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_a0_fixed.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_50k ^
  --resume outputs/anchor_ablation/diffusion_ckpt/A0/last.pt ^
  --max-epochs 60 ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/A0

--resume 恢复模型、优化器、轮次、历史最佳验证损失和早停计数。
--max-epochs 是总轮次上限；第 8 轮 checkpoint 会从第 9 轮继续。
