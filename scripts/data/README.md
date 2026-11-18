构建 A0 anchor 命令：
python scripts/data/build_a0_fixed_anchors.py --dataset-root outputs/expert_dataset/expert_trucksim_k2_raw_100k --diffusion-config configs/train/diffusion.yaml --max-accel-mps2 4.0 --max-steer-rad 0.5 --output outputs/anchor_ablation/ckpt/A0_anchors/A0_control_anchors.npz

评估A0 A1命令：
cd /d D:\KONG_Files\AllMerge\all_merge

python evaluation/ablation/run_anchor.py ^
  --stage anchors ^
  --methods A0 A1