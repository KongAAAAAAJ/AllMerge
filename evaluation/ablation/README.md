## Anchor Ablation，评估A0 A1 A2命令：
开环指标：
python evaluation/ablation/run_anchor.py --stage anchors --methods A0 A1 A2
开环 + 闭环指标：
python evaluation/ablation/run_anchor.py ^
  --stage all ^
  --methods A0 ^
  --split test ^
  --checkpoint-a0 outputs/anchor_ablation/diffusion_ckpt/dual/A0/best_through_epoch_0049.pt ^
  --jtp-config configs/projection/jtp.yaml ^
  --num-workers 0

  --stage anchors 前四项指标      --stage feasibility Planning minADE 和 Pre-proj. Feas.      --stage all 六项指标；不执行迭代投影，Proj. Corr. 为 --


## JTP Ablation，评估投影前后模型命令：
python evaluation/ablation/run_jtp.py ^
  --config configs/evaluation/jtp_ablation.yaml ^
  --diffusion-config configs/train/ablation/diffusion_a0_fixed.yaml ^
  --jtp-config configs/projection/jtp.yaml ^
  --checkpoint outputs/anchor_ablation/diffusion_ckpt/dual/A0/best.pt ^
  --split test ^
  --num-frames 100 ^
  --seeds "0" ^
  --output-dir outputs/jtp_ablation/A0


已成功安装。
当前step 1: 新增 Candidate Scoring，替换 Target Point Selector，统一训练、推理、评估的最终选择逻辑。是否已经完成？
接下来实现step 2: 新增 Priority-weighted Loss Aggregation，让高优先级候选获得更充分的优化。
等step 2完成之后，统一再进行A0和A0_assignment评估