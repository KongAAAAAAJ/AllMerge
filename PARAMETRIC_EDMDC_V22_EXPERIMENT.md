# Parametric EDMDc V2.2: validation-selected shift-preserving stabilization

## Scope

V2.2 changes Parametric EDMDc only. Original EDMDc retains its legacy fitting
and stabilization path. K0, K1-P, K1-F and K2 are untouched. The 49-dimensional
dictionary, condition-affine parameterization, Ridge identification and frozen
train/validation/test split are retained.

## V2.2 changes

1. Preserve exact deterministic state-history and control-history shifts.
2. Search a set of learned-row scaling factors that satisfy the existing
   spectral-radius cap over the observed train + validation condition grid.
3. Evaluate candidates on the frozen validation episodes using the same
   horizon, warm-up and stride as the paper evaluation.
4. Select the factor with the lowest normalized multi-metric validation score:
   ADE 0.30, FDE 0.25, episode P95 FDE 0.15, worst condition mean FDE 0.10,
   LTR 0.10, yaw RMSE 0.05, roll RMSE 0.05. Each metric is normalized by the
   best value among feasible candidates. Ties prefer the larger factor.
5. Fit a second LTR decoder on model-generated TRAIN rollouts. The frozen
   validation split selects between the original teacher-forced Ridge decoder
   and rollout-aligned Ridge decoder using LTR RMSE only. No validation labels
   enter either regression fit.

## Training and evaluation

Run from the AllMerge repository root:

```bash
python highway_env/vehicle/koopman/scripts/run_training.py train --model "Parametric EDMDc"
python highway_env/vehicle/koopman/scripts/run_evaluation.py evaluate --model "Parametric EDMDc" --split val
python highway_env/vehicle/koopman/scripts/run_evaluation.py compare --split test
```

Only Parametric EDMDc is retrained. Do not regenerate the frozen split or tune
on the test split. The checkpoint stats include candidate factors, validation
metrics/objectives, selected factor, spectral radii, and LTR decoder choice.

## Runtime

Validation search is performed during fitting. The number of candidate
rollouts is intentionally bounded (coarse scan plus local refinement). This
adds training-time CPU work but does not add inference-time cost.
