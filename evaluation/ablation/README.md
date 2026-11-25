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
  --num-frames 500 ^
  --seeds "0" ^
  --output-dir outputs/jtp_ablation/A0 ^
  --constraint-distributions --output-dir outputs/jtp_ablation/A0


已成功安装。
当前step 1: 新增 Candidate Scoring，替换 Target Point Selector，统一训练、推理、评估的最终选择逻辑。是否已经完成？
接下来实现step 2: 新增 Priority-weighted Loss Aggregation，让高优先级候选获得更充分的优化。
等step 2完成之后，统一再进行A0和A0_assignment评估
## JTP 十二项约束分布

在原有 `run_jtp.py` 命令后添加 `--constraint-distributions`，并使用新的输出目录，例如 `--output-dir outputs/jtp_ablation/A0_distributions`。默认不开启，不改变已有评估指标。

开启后保存 `jtp_constraint_distributions.npz` 及同名 JSON 元数据，并生成两张 3×2 组图（PDF、300 dpi PNG）：`jtp_distributions_geometry_stability` 和 `jtp_distributions_control_safety`。

在 `all_merge` 目录中独立重绘，无需加载模型：

```powershell
python evaluation/ablation/plot_jtp_distributions.py --data outputs/jtp_ablation/A0_distributions/jtp_constraint_distributions.npz --output-dir outputs/jtp_ablation/A0_distributions/replot
```

分布覆盖所有轨迹的未来时间步；曲率使用未来节点间 H−1 个区间，控制变化率包含历史最后控制到首步的连接。道路显示每车每步最小安全余量（阈值0），碰撞显示最小圆包络净间距（阈值为配置安全距离），不存在碰撞对象的样本不计入分布。车队内部间距按受控车辆统计，因此同一车对可贡献双方样本。

横摆角速度、侧偏角、侧倾角保留原始物理量，不显示动态阈值线；其余阈值线显示运行配置的名义阈值。每个子图同时标注有效时间步样本数、按评估容差计算的时间步违规率（step）与轨迹违规率（traj）。动态指标的违规率仍使用每个样本实际工况阈值。道路和碰撞的 step 以每车每步最不利余量判定。

旧 A0/A0_smoke 输出没有保存完整分布，必须重新运行评估后绘制，不能从汇总 CSV 还原。建议新建输出目录以保留历史结果。
