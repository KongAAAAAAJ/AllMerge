# AllMerge

Current package ownership and migration details: [docs/architecture.md](docs/architecture.md).

| Directory | Responsibility |
| --- | --- |
| `configs/` | Experiment settings |
| `scripts/` | Command entry points |
| `highway_env/` | Scenarios, planners and vehicle dynamics |
| `data_pipeline/` | Expert collection and datasets |
| `learning/` | Shared training infrastructure |
| `simulation/` | Scenario orchestration, decisions and recording |
| `evaluation/` | Offline evaluation |
| `tests/` | Regression tests |
| `docs/` | Architecture and usage |
| `outputs/` | Datasets, checkpoints, logs, plots and videos |

## 运行方式

在项目根目录执行，参数沿用原脚本：

```bash
python -m data_pipeline.expert_collection.collector --help
python -m evaluation.expert_dataset_validation --help
python -m highway_env.planner.diffusion.training.trajectory --help
python -m scripts.training.train_grpo --help
python -m scripts.evaluation.eval_diffusion_open_loop --help
python -m scripts.diagnostics.overfit_diagnostics --help
python -m scripts.simulation.run_test
python -m pytest
```

也支持直接运行入口文件，例如 `python highway_env/planner/diffusion/training/trajectory.py --help`。相对参数路径仍以当前工作目录为基准；推荐在项目根目录运行。`scripts/run_diffusion_*.sh` 已同步使用新的模块入口。

共享模块导入路径改为 `data_pipeline.expert_dataset`、`data_pipeline.expert_collection.core`、`simulation.data_save` 和 `simulation.env_reset`。数据采集核心原有的工作目录切换行为保留，默认数据、回合记录和视频输出位置保持一致。

原根目录的 7 个 `test_*.py` 已迁入 `tests/`，其中 `test_diffusion_runtime_load.py` 是需要实际检查点和数据集的命令行验证工具：

```bash
python tests/test_diffusion_runtime_load.py --checkpoint PATH --dataset-root PATH
```

运行相应入口需要原项目依赖，包括 PyTorch、NumPy、Gymnasium、stable-baselines3 等；自动化测试还需要 pytest。

`MIGRATION_MANIFEST.json` 记录的是此前预训练迁移时的路径与哈希，作为历史记录保留；其中根目录脚本的当前路径以本目录结构为准。
