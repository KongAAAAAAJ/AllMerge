在 `all_merge/` 目录下运行。

## TRACE Adaptive Anchor Ablation

评估入口保留三个阶段：`anchors`、`feasibility` 和 `all`。分阶段运行用于独立诊断，论文最终七指标结果建议使用 `--stage all` 一次完成，确保 Anchor Quality 与 Anchor-conditioned Diffusion Planning 使用同一测试子集、相同代表性场景和一致的模型配置。

### 1. Anchor Quality

不需要扩散模型 checkpoint。A0 从固定 anchor bank 读取，A1 从专家数据中的 paired dynamic anchors 读取，A2 使用训练好的 Safety Scoring Network。

```bash
# Evaluate A0/A1/A2
python evaluation/ablation/run_anchor.py --stage anchors --methods A0 A1 A2 --figure-scenes 3 --figure-seed 2027

# Evaluate one method, e.g. A2
python evaluation/ablation/run_anchor.py --stage anchors --methods A2 ^
  --a2-scorer outputs/trace_adaptive_anchors_v2/best.pt
```

输出四项 Anchor Quality 指标：

- Anchor minADE (m)
- Safe Ratio (%)
- Trajectory Diversity (m)
- Control Diversity (normalized)

A1 的 `mode_valid_mask` 不作为 Anchor 几何有效性判据。评估器根据实际生成轨迹的几何有效性筛选完整同步帧，确保每个纳入样本均具有十条有效 Anchor；零填充占位轨迹不会被强制视为有效。

### 2. Anchor-conditioned Diffusion Planning

需要相应方法训练完成的 diffusion checkpoint。可单独评价一个或多个方法。

```bash
# Evaluate A0 only
python evaluation/ablation/run_anchor.py --stage feasibility --methods A0 ^
  --checkpoint-a0 outputs/anchor_ablation/A0_fixed/best.pt

# Evaluate all three methods
python evaluation/ablation/run_anchor.py --stage feasibility --methods A0 A1 A2 ^
  --checkpoint-a0 outputs/anchor_ablation/A0_fixed/best.pt ^
  --checkpoint-a1 outputs/anchor_ablation/A1_dynamic/best.pt ^
  --checkpoint-a2 outputs/anchor_ablation/A2_adaptive/best.pt
```

输出三项规划指标：

- Planning minADE (m)
- Pre-projection Feasibility (%)
- Projection Correction (m)

Pre-projection Feasibility 先调用 JTP 的 `evaluate_preprojection()`，使用与 JTP 投影相同的 K2 rollout、稳定性、道路、曲率和控制约束，且不执行投影。外部交通参与者碰撞和编队车辆间碰撞由独立同步检查补充。随后调用同一个 JTP 的 `project()`，统计投影前后执行轨迹的平均点位移。投影成功率及失败数量作为诊断量单独记录。

JTP 配置默认使用 `configs/projection/jtp.yaml`，也可通过 `--jtp-config` 指定。

### 3. Complete Evaluation

一次完成七项论文指标，并输出两张同一代表性场景的 1×3 对比图。

```bash
python evaluation/ablation/run_anchor.py --stage all ^
  --checkpoint-a0 outputs/anchor_ablation/A0_fixed/best.pt ^
  --checkpoint-a1 outputs/anchor_ablation/A1_dynamic/best.pt ^
  --checkpoint-a2 outputs/anchor_ablation/A2_adaptive/best.pt
```

### 4. Outputs

默认输出目录按阶段和方法隔离，避免分阶段运行相互覆盖：

```text
outputs/anchor_ablation/evaluation/
  anchors/A0_A1_A2/
  feasibility/A0_A1_A2/
  all/A0_A1_A2/
```

完整 `all` 阶段输出：

```text
anchor_quality_per_sample.csv
planning_per_sample.csv
anchor_ablation_summary.csv
anchor_ablation_table.tex
anchor_ablation.pdf
anchor_planning_ablation.pdf
evaluation_manifest.json
evaluation_summary.json
```

`anchor_ablation.pdf` 展示 Anchor 分布；`anchor_planning_ablation.pdf` 展示 Anchor、Diffusion 生成轨迹、Expert Trajectory，以及 JTP 修正前后的选中轨迹。两张图使用同一个代表性场景。

分阶段结果各自保存，缺失阶段的指标以 `--` 表示，不会伪造或自动拼接来自不同数据子集的结果。需要完整论文表格时请运行 `--stage all`。

## Fuel-aware Ablation

```bash
python evaluation/ablation/run_fuel_aware.py --config configs/train/diffusion.yaml --checkpoint F0=outputs\fuel_ablation\F0\best.pt --checkpoint F3=outputs\fuel_ablation\F3\best.pt --split test
```
