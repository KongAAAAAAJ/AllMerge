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
  --auto-resume ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/dual/A0_anchor_assignment_new

## Diffusion 训练命令 A1：
python scripts/training/run_diffusion.py ^
  --config configs/train/diffusion.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k ^
  --anchor-cache-dir outputs/anchor_ablation/cache_candidates_parallel ^
  --max-epochs 60 ^
  --auto-resume ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/dual/A1_anchor_assignment_new

## Diffusion 训练命令 A2：
python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_a2_adaptive.yaml ^
  --anchor-cache-dir outputs/anchor_ablation/cache_candidates_parallel ^
  --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k ^
  --max-epochs 60 ^
  --auto-resume ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/dual/A2_new

## 从已保存的 A0 checkpoint 续训：
python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_a0_fixed.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k ^
  --resume outputs/anchor_ablation/diffusion_ckpt/dual/A0_anchor_assignment/last.pt ^
  --auto-resume ^
  --max-epochs 60 ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/dual/A0_anchor_assignment

## 从已保存的 A1 checkpoint 续训：
python scripts/training/run_diffusion.py ^
  --config configs/train/diffusion.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k ^
  --anchor-cache-dir outputs/anchor_ablation/cache_candidates_parallel ^
  --resume outputs/anchor_ablation/diffusion_ckpt/dual/A1_anchor_assignment_new/last.pt ^
  --max-epochs 60 ^
  --auto-resume ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/dual/A1_anchor_assignment_new

## 从已保存的 A2 checkpoint 续训：
python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_a2_adaptive.yaml ^
  --anchor-cache-dir outputs/anchor_ablation/cache_candidates_parallel ^
  --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k ^
  --resume outputs/anchor_ablation/diffusion_ckpt/dual/A2_new/last.pt ^
  --max-epochs 60 ^
  --auto-resume ^
  --output-dir outputs/anchor_ablation/diffusion_ckpt/dual/A2_new

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


## 固定首次参数与异常自动续训

新训练首次启动时，无论是否启用自动模式，都会在输出目录保存 `training_run.json`，
记录展开后的 YAML、有效训练配置和命令行参数（数据/输出/缓存目录、设备、总轮次、
每轮 batch 限制和样本限制）。原始 YAML 与有效配置分别记录，以区分 YAML 修改和首次 CLI 覆盖。

后续 `--resume` 和 `--auto-resume --resume` 都读取 checkpoint 同目录的 `training_run.json`。
当前 YAML 若与首次 YAML 不一致，会发出 `RuntimeWarning`，并继续使用原快照；
续训命令中的其他训练参数不覆盖快照，包含 `--max-epochs`、数据目录和输出目录。
因此修改当前 YAML 的 loss 不会改变已有任务的训练方案。切换 loss 请使用新输出目录从头训练。
原配置引用的外部模型、anchor 配置和数据仍需保持不变，已有恢复一致性检查继续生效。

从头训练并自动重试，在原训练命令追加：

```cmd
--auto-resume
```

给已有的新训练任务启用自动续训：

```cmd
python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_a0_fixed.yaml ^
  --resume outputs/anchor_ablation/diffusion_ckpt/dual/A0/last.pt ^
  --auto-resume
```

有 `training_run.json` 时按原快照恢复完整参数。旧任务缺少快照时发出 warning，
采用本次 YAML 和命令行覆盖参数续训，并在实际输出目录保存新的 `training_run.json`。
这代表此次采用的参数，并非还原旧任务首次启动的所有参数。无快照续训时，未指定
`--output-dir` 则输出到 checkpoint 所在目录；显式指定新目录则只向新目录写快照。
后续从新目录生成的 checkpoint 续训即可读取该快照。

旧任务仍须显式给出 `--resume`；仅传 `--auto-resume` 不会自动采用已有 `last.pt`。
loss、轨迹来源、anchor、K2 等一致性校验不变，不兼容时仍报错。快照在首次续训尝试前
写入，因此不能据其存在判断 checkpoint 已校验通过；失败后不会自动覆盖快照。
若需要修正采用的配置，先明确移除该次生成的 `training_run.json` 后再启动。
新输出目录若已有快照也不会覆盖。

例如旧任务首次兼容续训并启用自动模式：

```cmd
python scripts/training/run_diffusion.py ^
  --config configs/train/ablation/diffusion_a0_fixed.yaml ^
  --dataset-root outputs/expert_dataset/expert_trucksim_dual_100k ^
  --resume outputs/anchor_ablation/diffusion_ckpt/dual/A0/last.pt ^
  --max-epochs 60 ^
  --auto-resume
```
自动模式会在输出目录另存本次监督使用的 `auto_resume_run.json`；每次重试的子进程直接读取它，
不再读取原实验 YAML。已运行的旧训练进程不会自动获得此功能。

每次训练尝试使用独立 Python 子进程，日志直接显示在终端。异常退出后等待 10 秒，
从本次最新 `last.pt` 恢复；若尚未保存新 checkpoint，则保留首次 resume 文件，或从头重新尝试。
有新 epoch 保存就清零连续无进展计数；连续 3 次异常均没有新 epoch 时停止并返回失败。
达到首次目标轮次，或因原有 `patience` 早停正常退出，都结束监督，不重新启动。
按 Ctrl+C 停止当前子进程及其子进程并退出，不自动重启。

自动模式从头训练要求输出目录没有现存 `.pt`，不会擅自采用旧 checkpoint。
`last.pt` 和 `best.pt` 先写同目录临时文件，再原子替换，写入失败时保留原完整文件。
中断 epoch 中尚未保存的 batch 会重新执行，本功能不提供 batch 级无损恢复。
