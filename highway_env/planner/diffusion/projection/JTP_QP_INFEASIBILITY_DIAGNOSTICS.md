# JTP G2 Primal-Infeasibility Diagnostics

The optional diagnostic captures the exact OSQP problem whenever the
`road_curvature` group returns `primal infeasible` or
`primal infeasible inaccurate`.

## Enable

In `configs/projection/jtp.yaml`:

```yaml
projection:
  qp:
    diagnostics:
      save_primal_infeasible: true
      output_dir: outputs/jtp_qp_diagnostics
```

The same settings are inherited by copied JTP YAML files when omitted. The
diagnostic is disabled by default and does not change the QP formulation.

Run the usual `evaluation/ablation/run_jtp.py` command. The evaluation runner
passes seed, episode, and frame identifiers into the projector. One artifact
pair is written for each failed road/curvature group QP.

## Files

Each case produces:

- `<case>.npz`: exact sparse OSQP matrices `P` and `A` in CSC components,
  objective vector `q`, lower/upper row bounds, primal/dual infeasibility
  certificates, solver vectors, state/protection Jacobians and residuals,
  slack bounds, absolute control limits, and group-entry control correction
  bounds.
- `<case>.json`: scene and vehicle identifiers, QP group order and index,
  solver settings/status, variable and constraint block ranges, physical
  row labels, and certificate contribution analysis.

The `A` matrix row order is:

1. `control_box`
2. `state_inequality`
3. `state_slack_nonnegative`
4. `protection_inequality`
5. `protection_slack_nonnegative`

`row_labels` and `row_families` in the NPZ map each row to its constraint
family and time/physical quantity. `variable_labels` maps decision variables
to control corrections and slack variables.

## Certificate interpretation

For OSQP's primal infeasibility certificate `y`, the diagnostic computes:

- `||A.T @ y||_inf`, which should be close to zero;
- `sum(u_i * max(y_i, 0) + l_i * min(y_i, 0))`, omitting infinite-bound terms;
- the same bound contribution aggregated by QP row block and constraint family.

A negative separation value is the certificate's infeasibility witness.
Negative contributions identify the row blocks/families that support the
contradiction. Inspect `certificate_supported_rows` to locate individual
rows. Interpret the certificate jointly with the saved `A`, `l`, and `u`;
a family with a large row count is not, by itself, evidence that it caused
infeasibility.

## Reconstruct sparse matrices

```python
import numpy as np
import scipy.sparse as sp

data = np.load("case.npz")
A = sp.csc_matrix(
    (data["A_data"], data["A_indices"], data["A_indptr"]),
    shape=tuple(data["A_shape"]),
)
P = sp.csc_matrix(
    (data["P_data"], data["P_indices"], data["P_indptr"]),
    shape=tuple(data["P_shape"]),
)
```

The diagnostic only records failed road/curvature QPs. It does not alter solver
status handling, tolerances, constraints, priority order, line search, or
nonlinear acceptance.
