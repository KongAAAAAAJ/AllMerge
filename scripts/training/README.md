## A1/A2 共享候选池缓存
python scripts/data/build_anchor_candidate_cache.py ^
  --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k ^
  --cache-dir outputs/anchor_ablation/cache_candidates_parallel ^
  --workers 4

## 修改yaml配置：
旧Loss: training_scheme: goal_endpoint
or
新的Anchor Loss: training_scheme: anchor_consistent
## 改yaml后记得改ckpt保存路径

## Diffusion 训练命令 A0：
python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_a0_fixed.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k ^
  --max-epochs 60 ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/dual/A0

## Diffusion 训练命令 A1：
python scripts/training/run_diffusion.py ^
  --config configs/train/diffusion.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k ^
  --anchor-cache-dir outputs/anchor_ablation/cache_candidates_parallel ^
  --max-epochs 60 ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/dual/A1

## Diffusion 训练命令 A2：
python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_a2_adaptive.yaml ^
  --anchor-cache-dir outputs/anchor_ablation/cache_candidates ^
  --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k ^
  --max-epochs 60 ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/dual/A2

## 从已保存的 A0 checkpoint 续训：
python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_a0_fixed.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k ^
  --resume outputs/anchor_ablation/diffusion_ckpt/A0/last.pt ^
  --max-epochs 60 ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/A0

--resume 恢复模型、优化器、轮次、历史最佳验证损失和早停计数。
--max-epochs 是总轮次上限；第 8 轮 checkpoint 会从第 9 轮继续。

## 双轨迹格式 v3

当前采集仍为独立 episode 进程、10 Hz 逐帧记录、episode 结束后原子写入压缩 `.npz`。
每个有效样本同时包含 `polynomial_future_xy [40,2]`（当前规划的参考轨迹）和
`trucksim_future_poses [40,3]`（随后闭环实际执行的位置、车身航向），两者都在
规划时刻 `koopman_pose0_si` 的局部坐标系中。后者包含后续重规划，不能视作固定
当前规划的单次跟踪结果。稀疏 8 点由第 5、10、…、40 步派生，不重复存储。

仅支持 `trace_expert_raw_k2_v3`。旧数据不转换、不回退；需使用新数据目录重新采集。
默认新数据集名为 `expert_trucksim_dual_raw_100k_v3`。episode 尾部不足完整 4 秒的
样本按三车同步帧舍弃，不外推、不补点、不延长 episode；有效样本数会相应减少。

训练在 `configs/train/diffusion.yaml` 的 `control.training` 下设置：

```yaml
trajectory_source: polynomial  # 或 trucksim_closed_loop
```

该选项统一切换监督目标和其他受控车的参考轨迹。评估时在
`configs/evaluation/anchor_ablation.yaml` 顶层设置同名标签，与 checkpoint 来源保持一致。
checkpoint 和 resolved config 记录来源；切换来源应使用新的训练输出目录启动新训练，
不可用另一来源的 `--resume` 延续历史。已有旧 checkpoint 缺少来源时不能续训。
同一份 v3 数据生成的共享候选池可供两种来源共用；旧数据的缓存不迁移。


### 候选池缓存的 CPU 并行与进度

缓存生成默认使用 4 个 CPU worker，按 episode 并行计算，由主进程按原始样本索引写入缓存。
可用 `--workers` 调整进程数；例如：

```cmd
python scripts/data/build_anchor_candidate_cache.py --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k --cache-dir outputs/anchor_ablation/cache_candidates_parallel --workers 4
```

开始时输出索引与进程池状态，每完成一个 episode 输出样本完成数、百分比、episode 完成数、
samples/s、已耗时和预计剩余秒数。进度按完成的 episode 更新，worker 启动或首个 episode
计算期间可能暂时没有新输出。增加 worker 会增加内存占用和磁盘读取压力，可从 4 开始调整。
候选算法、样本顺序和缓存格式不变。

已运行的旧进程不会自动切换到并行版本。若另行启动新版本，请使用独立缓存目录，
避免与正在生成或已完成的缓存共用目标目录；本入口不提供缓存断点续建。
