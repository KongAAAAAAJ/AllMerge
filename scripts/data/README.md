采集专家数据命令（新）：
python scripts\data\collect_expert_dataset.py ^
  --config configs\data\expert_collection.yaml ^
  --dataset-name expert_trucksim_dual_100k ^
  --scenario all ^
  --target-samples 100000 ^
  --duration 12 ^
  --max-episodes 10000 ^
  --resume

构建 A0 anchor 命令：
python scripts/data/build_a0_fixed_anchors.py --max-accel-mps2 2.5 --max-steer-rad 0.08 ^
--dataset-root outputs/expert_dataset/expert_trucksim_dual_100k ^
--output outputs/anchor_ablation/ckpt/A0_anchors/A0_control_anchors.npz

构建 A1 anchoor：
A1 在 diffusion 训练、开环/闭环推理及消融评估时根据 raw 场景在线生成，无需单独构建或保存锚点库。数量读取 diffusion.yaml 的 control.model.num_modes，生成参数位于 configs/anchors/a1.yaml。

训练 A2 命令：
python scripts/data/train_a2_anchors.py ^
  --config configs/train/a2_scorer.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k ^
  --max-epochs 30 ^
  --output-dir outputs/anchor_ablation/ckpt/A2_anchors_straight ^
  --cache-dir outputs/anchor_ablation/cache_A2_straight

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


## A1/A2 共享候选池缓存

在 `all_merge` 目录执行。缓存只保存筛选前的候选池，不需要 A2 scorer：

```powershell
python scripts/data/build_anchor_candidate_cache.py --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k --cache-dir outputs/anchor_ablation/cache_candidates
```

默认生成 train、val；可用 `--splits train,val,test` 和 `--candidate-config` 指定划分与候选配置。目标 split 已存在时直接报错，不自动覆盖。缓存绑定原始数据、样本顺序和候选生成配置。

```powershell
# A1
python scripts/training/run_diffusion.py --anchor-cache-dir outputs/anchor_ablation/cache_candidates
# A2
python scripts/training/run_diffusion.py --config configs/train/ablation/diffusion_a2_adaptive.yaml --anchor-cache-dir outputs/anchor_ablation/cache_candidates
```

不传 `--anchor-cache-dir` 即在线生成。缓存缺失或不匹配会直接报错；A1 仍执行多样性筛选，A2 仍执行 scorer 评分和 Top-R 筛选，因此缓存不会消除 A2 的评分开销。数据目录应与训练配置或 `--dataset-root` 一致。缓存不会改变模型或 checkpoint 格式。




## 双轨迹格式 v3

当前采集仍为独立 episode 进程、10 Hz 逐帧记录、episode 结束后原子写入压缩 `.npz`。
每个有效样本同时包含 `polynomial_future_xy [40,2]`（当前规划的参考轨迹）和
`trucksim_future_poses [40,3]`（随后闭环实际执行的位置、车身航向），两者都在
规划时刻 `koopman_pose0_si` 的局部坐标系中。后者包含后续重规划，不能视作固定
当前规划的单次跟踪结果。稀疏 8 点由第 5、10、…、40 步派生，不重复存储。

仅支持 `trace_expert_raw_k2_v3`。旧数据不转换、不回退；需使用新数据目录重新采集。
默认新数据集名为 `expert_trucksim_dual_raw_100k_v3`。episode 尾部不足完整 4 秒的
样本按三车同步帧舍弃，不外推、不补点、不延长 episode；有效样本数会相应减少。

专家控制器的车辆几何参数在 `configs/data/expert_collection.yaml` 中统一配置：
`vehicle_dimensions.length_m` 和 `width_m` 用于车身包络，`wheelbase_m` 用于
Pure Pursuit 与路径曲率反算。`wheelbase_m` 不得用车身总长代替，默认 5.5 m 与
TRACE 锚点/动力学配置一致；正式采集前必须核对所选 TruckSim `.sim` 模型的真实轴距。
TruckSim 导出的车身纵、横向速度分别转换为 m/s，并投影到目标车道 Frenet 坐标系，
作为 Polynomial 参考轨迹的初始速度边界条件。修改该控制器几何或初始速度定义后，
采集器会将其写入 collection contract，避免在同一数据集目录中混合不同配置生成的样本。

训练在 `configs/train/diffusion.yaml` 的 `control.training` 下设置：

```yaml
trajectory_source: polynomial  # 或 trucksim_closed_loop
```

该选项统一切换监督目标和其他受控车的参考轨迹。评估时在
`configs/evaluation/anchor_ablation.yaml` 顶层设置同名标签，与 checkpoint 来源保持一致。
checkpoint 和 resolved config 记录来源；切换来源应使用新的训练输出目录启动新训练，
不可用另一来源的 `--resume` 延续历史。已有旧 checkpoint 缺少来源时不能续训。
同一份 v3 数据生成的共享候选池可供两种来源共用；旧数据的缓存不迁移。
