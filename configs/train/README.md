Diffusion 训练命令 A0：
python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_a0_fixed.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_50k ^
  --max-epochs 60 ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/A0

Diffusion 训练命令 A1：
python scripts/training/run_diffusion.py ^
  --config configs/train/diffusion.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_50k ^
  --max-epochs 60 ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/A1

Diffusion 训练命令 A2：
python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_a2_adaptive.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_50k ^
  --max-epochs 60 ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/A2
