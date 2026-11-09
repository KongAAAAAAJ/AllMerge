# AllMerge

根目录 Python 文件已按用途分类；模型、配置、检查点和输出目录保持原有布局。

| 目录 | 用途 |
| --- | --- |
| `data_pipeline/` | 专家数据集读取、划分及采集核心逻辑 |
| `simulation/` | 仿真环境初始化、回合数据保存 |
| `scripts/data/` | 专家数据采集、分析、划分、校验入口 |
| `scripts/training/` | Diffusion 预训练、GRPO 训练、运行时检查点导出 |
| `scripts/evaluation/` | Diffusion 开环评估入口 |
| `scripts/diagnostics/` | 过拟合诊断入口 |
| `scripts/simulation/` | 全链路闭环仿真入口 `run_test.py` |
| `tests/` | 自动化测试及已有本地调试工具 |
| `pretraining/`、`evaluation/` | 训练与评估的共享实现 |
| `highway_env/`、`decide/`、`mode_selector/` | 环境、规划及决策实现 |

## 运行方式

在项目根目录执行，参数沿用原脚本：

```bash
python -m scripts.data.collect_expert_dataset --help
python -m scripts.data.validate_expert_dataset --help
python -m scripts.training.train_diffusion_pretrain --help
python -m scripts.training.train_grpo --help
python -m scripts.evaluation.eval_diffusion_open_loop --help
python -m scripts.diagnostics.overfit_diagnostics --help
python -m scripts.simulation.run_test
python -m pytest
```

也支持直接运行入口文件，例如 `python scripts/training/train_diffusion_pretrain.py --help`。相对参数路径仍以当前工作目录为基准；推荐在项目根目录运行。`scripts/run_diffusion_*.sh` 已同步使用新的模块入口。

共享模块导入路径改为 `data_pipeline.expert_dataset`、`data_pipeline.expert_collection_core`、`simulation.data_save` 和 `simulation.env_reset`。数据采集核心原有的工作目录切换行为保留，默认数据、回合记录和视频输出位置保持一致。

原根目录的 7 个 `test_*.py` 已迁入 `tests/`，其中 `test_diffusion_runtime_load.py` 是需要实际检查点和数据集的命令行验证工具：

```bash
python tests/test_diffusion_runtime_load.py --checkpoint PATH --dataset-root PATH
```

运行相应入口需要原项目依赖，包括 PyTorch、NumPy、Gymnasium、stable-baselines3 等；自动化测试还需要 pytest。

`MIGRATION_MANIFEST.json` 记录的是此前预训练迁移时的路径与哈希，作为历史记录保留；其中根目录脚本的当前路径以本目录结构为准。
