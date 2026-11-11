# 数据采集
python all_merge\highway_env\vehicle\koopman\scripts\run_dataset.py collect

# 数据划分（只划分一次，除非数据集修改）
python all_merge\highway_env\vehicle\koopman\scripts\run_training.py split

# 训练
python highway_env\vehicle\koopman\scripts\run_training.py train --model all --epochs 2

# 评测
python highway_env\vehicle\koopman\scripts\run_evaluation.py compare --split val