构建 A0 anchor 命令：
python scripts/data/build_a0_fixed_anchors.py --max-accel-mps2 4.0 --max-steer-rad 0.5

评估A0 A1 A2命令：
cd /d D:\KONG_Files\AllMerge\all_merge
python evaluation/ablation/run_anchor.py --stage anchors --methods A0 A1 A2

训练 A2 命令：
python -m highway_env.planner.anchor_scoring.train --config configs/train/adaptive_anchors.yaml --output-dir outputs/anchor_ablation/ckpt/A2_anchors --cache-dir outputs/anchor_ablation/cache_A2