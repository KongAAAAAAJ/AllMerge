# 数据采集
python all_merge\highway_env\vehicle\koopman\scripts\run_dataset.py collect

# 数据划分（只划分一次，除非数据集修改）
python all_merge\highway_env\vehicle\koopman\scripts\run_training.py split

# 训练
python highway_env\vehicle\koopman\scripts\run_training.py train --model all --epochs 2

# 评测
python highway_env\vehicle\koopman\scripts\run_evaluation.py compare --split val

Compare five models, excluding Parametric EDMDc (omit `--models` for all six):

```bat
python highway_env\vehicle\koopman\scripts\run_evaluation.py compare --split test --models EDMDc K0 K1-P K1-F K2
```

`--models` controls evaluation, summary CSVs, and both comparison PDFs. The
requested order is used in tables and plots. Names containing spaces must be
quoted, for example `--models EDMDc "Parametric EDMDc" K2`.
Output filenames remain unchanged; rerunning for the same split replaces the
existing comparison CSVs and PDFs.

`koopman_chassis_responses.pdf` uses independent one-step predictions with
recorded true history refreshed at every input time. With `horizon_steps: 40`,
input origins 0 through 40 predict targets 1 through 41 (0.1 through 4.1 s
relative to the first input when `dt_s: 0.1`). Each origin retains the configured
`history_steps`; predicted states are never fed into the next origin's history.
The metric tables and `koopman_prediction_errors.pdf` still use multistep rollouts.
