AllMerge-main/
│
├── highway_env/
│   ├── vehicle/
│   │   ├── trucksim_dynamics.py              # 已有
│   │   ├── dynamics_EightDoF_Model.py         # 已有
│   │   ├── trucksim_simulation/               # 已有
│   │   │
│   │   └── koopman/                           # 新增
│   │       ├── __init__.py
│   │       ├── model.py                       # K0/K1/K2统一模型
│   │       ├── edmd.py                        # EDMDc基线
│   │       ├── data.py                        # 数据集和预处理
│   │       ├── collect.py                     # TruckSim采集接口
│   │       ├── train.py                       # 统一训练入口
│   │       ├── evaluate.py                    # 统一预测评估
│   │       └── metrics.py                     # 评价指标与绘图
│   │
│   └── planner/
│       └── diffusion/                         # 现有，暂不修改
│
├── configs/
│   └── koopman.yaml                           # 新增统一配置
│
├── scripts/
│   └── run_koopman.py                         # 新增统一CLI
│
├── outputs/
│   └── koopman/                               # 运行时生成
│       ├── data/
│       ├── checkpoints/
│       └── evaluation/
│
└── tests/
    └── test_koopman.py                        # 新增