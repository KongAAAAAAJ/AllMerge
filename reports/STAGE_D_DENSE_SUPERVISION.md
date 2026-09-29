# Stage D — Dense Spline Supervision for Diffusion Pretraining

Marker: `STAGED_DENSE_SUPERVISION_V1`

## Important adaptation to the real current AllMerge objective

The current `StructuredDiffusionPlanner` uses `prediction_type="sample"` semantics:
its denoiser predicts `x0` directly (`predicted_x0_norm = noisy_norm + residual`).
The current pretraining objective is therefore:

```text
L_base = L_sparse_regression + L_mode_classification
```

It is **not** an epsilon-prediction `L_noise` implementation. Stage D does not invent
`recover_x0()` or rewrite the scheduler parameterization.

With dense supervision enabled:

```text
L_total = L_sparse_regression
        + L_mode_classification
        + lambda_p * terminal_weight * L_dense
```

`L_dense` is SmoothL1 by default in physical ego-local metres.

## Terminal auxiliary pass

The main random-t pass remains unchanged. When `dense_loss_enabled=true` and
`dense_loss_lambda_p>0`, the same batch receives one additional denoiser pass at
scheduler timestep 0 (the closest-to-clean timestep for this scheduler):

```text
same dynamic anchors
  -> add_noise(..., t=0)
  -> denoiser predicts clean x0 candidates
  -> gather the already-assigned expert mode
  -> denormalize to metres
  -> ClampedCubicTrajectorySpline [8,2] -> [41,2]
  -> drop t=0
  -> compare [40,2] with real native 10 Hz expert target
```

No full reverse chain is unrolled.

## Baseline compatibility

- default `dense_loss_enabled=false`
- `lambda_p=0` skips the auxiliary pass entirely
- old datasets remain usable when dense loss is disabled
- dense loss enabled with `lambda_p>0` requires real Stage-3 dense targets
- sparse output shape/action definition remains `[8,2]`
- no GRPO changes
