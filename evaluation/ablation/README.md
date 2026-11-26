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
  --num-frames 10 ^
  --seeds "0" ^
  --workers 1 ^
  --worker-threads 1 ^
  --output-dir outputs/jtp_ablation/A0 ^
  --constraint-distributions


已成功安装。
当前step 1: 新增 Candidate Scoring，替换 Target Point Selector，统一训练、推理、评估的最终选择逻辑。是否已经完成？
接下来实现step 2: 新增 Priority-weighted Loss Aggregation，让高优先级候选获得更充分的优化。
等step 2完成之后，统一再进行A0和A0_assignment评估
## JTP trajectory / step 统计

### 并行评估

默认 `--workers 1` 保持串行评估。在原命令后添加 `--workers 2 --worker-threads 1`，可用两个独立进程并行评估不同场景帧（Windows使用spawn）；每个进程只在启动时加载一次模型。同一场景内所有受控车辆仍由JTP联合求解。命令行覆盖评估YAML中的 `workers`、`worker_threads`。

单张GPU建议先比较1与2个worker的实际吞吐量。各进程共享所选GPU，但分别占用模型和求解显存；worker更多不保证更快。CPU模式可通过 `--device cpu --workers 4 --worker-threads 1` 使用多核。随机种子仍由场景和推理种子决定，结果按原输入顺序合并，统计、图表及写文件均由主进程统一完成。

本机RTX 4080、当前A0配置的小规模验证：16帧串行13.84秒，双进程13.82秒（评估阶段），未观察到明显提速；4帧则双进程更慢。两种模式的逐步记录完全相同，逐轨迹记录仅延迟字段不同。此测试不代表500帧性能，默认仍保留串行。

`jtp_manifest.json` 的 `execution` 记录进程数、设备、评估墙钟时间及场景/种子对吞吐量（含worker启动、帧准备及主进程统计采集，不含最终bootstrap和绘图）。并行时单帧projection latency包含资源争用，不能作为独占设备算法延迟与串行值直接比较。模型初始化有固定开销，少量帧可能比串行慢。内存中最多预提交2×workers个任务；任一任务失败即中止，不跳帧或回退。

`run_jtp.py` 新增 `--statistics-unit both|trajectory|step`，默认 `both`。在原评估命令后添加 `--statistics-unit step` 可只输出step报告。建议采用新的 `--output-dir outputs/jtp_ablation/A0_units`，避免混用历史文件。

每种口径在对应的 `trajectory/` 或 `step/` 子目录下生成 `jtp_per_vehicle_<unit>.csv`、`jtp_per_seed_<unit>.csv`、`jtp_summary_<unit>.csv`、`jtp_paired_statistics_<unit>.json` 和 `projection_ablation_table_<unit>.tex`。不再生成旧的根目录无单位统计文件，也不自动删除已有历史文件。原projection_ablation场景示例图、其数据、manifest、求解器诊断及可选分布图仍保留在根目录。

两张表均报告Joint Feas.（整队可行率）、Feas.（单车可行率）、Geometric Viol.、Stability Viol.、Deviation。trajectory的Joint要求同一场景所有车辆整段轨迹可行，Feas.按单车轨迹统计；step的Joint要求同一场景同一步所有车辆可行，Feas.按单车时间步统计。几何、稳定性等合计使用各子项违规标志的逻辑并集。

trajectory保持未来1…H步评估范围（控制变化率仅2…H），偏移平均1…H步。step将全部12项约束和偏移统一到2…H步，曲率区间归属结束节点。1500条轨迹、H=40时step有58500个单车样本，500个场景对应19500个整队步骤（1个推理种子）。step不依赖`--constraint-distributions`。H=1时step计数为0、比例和偏移为null，表中为N/A。两种口径的置信区间均按物理episode/frame做簇bootstrap，保留簇内所有种子、车辆及时间步。

## JTP 十二项约束分布

在原有 `run_jtp.py` 命令后添加 `--constraint-distributions`，并使用新的输出目录，例如 `--output-dir outputs/jtp_ablation/A0_distributions`。默认不开启，不改变已有评估指标。

开启后保存 `jtp_constraint_distributions.npz` 及同名 JSON 元数据，并生成两张 3×2 组图（PDF、300 dpi PNG）：`jtp_distributions_geometry_stability` 和 `jtp_distributions_control_safety`。

在 `all_merge` 目录中独立重绘，无需加载模型：

```powershell
python evaluation/ablation/plot_jtp_distributions.py --data outputs/jtp_ablation/A0_distributions/jtp_constraint_distributions.npz --output-dir outputs/jtp_ablation/A0_distributions/replot
```

`--num-frames` 指同步场景帧数，不是单车轨迹数。500帧×3辆车×1个种子=每种方法1500条轨迹；H=40时多数子图包含60000个时间步数值，曲率为58500个。蓝、橙两组使用同一分箱，以并排不透明柱状图显示时间步数值的概率密度。

分布覆盖所有轨迹的未来时间步；曲率使用未来节点间 H−1 个区间。独立评估中的加速度变化率与转向角变化率仅计算预测内部相邻控制的差分，排除历史最后控制到预测首步的连接；对应分布保存真实步编号2…H。H=40、1500条轨迹时，两项变化率各有58500个数值。道路显示每车每步最小安全余量（阈值0），碰撞显示最小圆包络净间距（阈值为配置安全距离），不存在碰撞对象的样本不计入分布。车队内部间距按受控车辆统计，因此同一车对可贡献双方样本。

首步变化率排除默认作用于全部独立评估结果：逐车、逐种子、summary、联合可行率、配对统计与论文表格，包括违规标志及超限程度；其他约束仍覆盖原有步骤。JTP投影器和其内部诊断仍约束历史到首步的变化率，因此内部可行标志与独立评估可能不同。H=1时变化率无可评估区间，违规为false、超限程度为0，分布为空。

新分布元数据版本为2，记录首步排除范围；旧版本NPZ重绘仍保留原有数据及口径，不自动排除首步。重新评估请使用新的输出目录，勿混用新旧summary与分布。

图中trajectory违规率按每条轨迹任意有效一步超限统计，对应trajectory报告中的分项。分布图step比例使用该指标全部有效步骤；step报告统一为2…H，因此道路、稳定性、幅值与碰撞等分布图比例不一定与step表完全一致。汇总几何率取道路与曲率违规标志的并集，稳定性率取4项稳定性违规标志的并集，控制变化率与碰撞率同理，不能相加各子图比例。没有碰撞对象的轨迹不进入分布子图分母，汇总则视其该项无违规。

横摆角速度、侧偏角、侧倾角保留原始物理量，不显示动态阈值线；其余阈值线显示运行配置的名义阈值。每个子图同时标注有效时间步样本数、按评估容差计算的时间步违规率（step）与轨迹违规率（traj）。动态指标的违规率仍使用每个样本实际工况阈值。道路和碰撞的 step 以每车每步最不利余量判定。

旧 A0/A0_smoke 输出没有保存完整分布，必须重新运行评估后绘制，不能从汇总 CSV 还原。建议新建输出目录以保留历史结果。
