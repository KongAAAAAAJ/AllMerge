Fuel Ablation评估命令：
python evaluation/ablation/run_fuel_aware.py ^
  --config configs/train/diffusion.yaml ^
  --checkpoint F0=outputs\fuel_ablation\F0\best.pt ^
  --checkpoint F3=outputs\fuel_ablation\F3\best.pt ^
  --split test


JTP Ablation评估命令：
python evaluation/ablation/run_jtp.py ^
  --checkpoint outputs\fuel_ablation\F0\best.pt ^
  --split test ^
  --num-frames 100 ^
  --seeds 0,1,2,3,4 ^
  --output-dir outputs\jtp_ablation\formal_100x5_v2