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

`--models` controls evaluation, summary CSVs, and comparison PDFs. The
requested order is used in tables and plots. Names containing spaces must be
quoted, for example `--models EDMDc "Parametric EDMDc" K2`.
Output filenames remain unchanged; rerunning for the same split replaces the
existing comparison CSVs and PDFs.

The final paper chassis figure defaults to a 3-step lead:
`koopman_chassis_responses_3step.pdf` (example metadata:
`koopman_chassis_examples_3step.csv`). Every input origin resets to recorded
true history, then predicts three transitions. With `horizon_steps: 40`, input
origins 0 through 40 predict targets 3 through 43. TruckSim is gray and each of
the nine panels has an inset. The existing ten-step sampling boundary is kept
to preserve previously sampled examples for the same seed/data.
The metric tables and `koopman_prediction_errors.pdf` still use multistep rollouts.

Generate only the paper chassis figure, without full metric evaluation:

```bat
python highway_env\vehicle\koopman\scripts\run_evaluation.py chassis --split test --models EDMDc K0 K1-P K1-F K2
```

`--chassis-steps` defaults to `3` for both `compare` and `chassis`; other leads
remain available explicitly, e.g. `--chassis-steps 1 3 5 8 10`.
The `chassis` command loads the chosen checkpoints and performs short
predictions. It accepts `--output` as an optional output directory.

In `evaluation/evaluate.py`, edit `CHASSIS_INSET_DEFAULTS` for all panels or
`CHASSIS_INSET_OVERRIDES[(row, col)]` for one panel (zero-based indices):

```python
(0, 0): {
    'bounds': (0.54, 0.12, 0.43, 0.40),
    'xlim': (1.2, 2.0),
    'ylim': (-0.02, 0.06),
    'connector_corners': (2, 4),
},
```

`bounds` means (left margin, bottom margin, width, height), all as fractions of
that panel's axes. Right margin = 1 - left - width; top margin = 1 - bottom -
height. `xlim` uses seconds from the first input; `ylim` uses rad/s, degrees,
or LTR according to the column. `None` selects ranges automatically.
`window_s` sets the automatic window duration; `ylim_pad` sets vertical padding.
Connector corners: 1 upper right, 2 upper left, 3 lower left, 4 lower right.

Automatic selection maximizes mean baseline absolute error minus K2 absolute
error over a time window, averaging the selected K0/K1-P/K1-F baselines.
Insets copy all selected model curves; automatic vertical limits focus on
truth and neural models. No advantage is asserted when the gain is nonpositive.
The example CSV records the displayed limits and automatic gain for inspection.
