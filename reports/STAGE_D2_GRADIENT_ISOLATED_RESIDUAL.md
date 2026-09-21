# Stage D2 — Gradient-isolated sparse residual head

- Same batch / same iteration / same optimizer step.
- `L_sparse + L_cls` update the original Diffusion planner.
- `L_dense` updates only `dense_residual_head`.
- Diffusion-derived mode features and sparse points are detached before the residual MLP.
- The auxiliary terminal forward runs under `torch.no_grad()` and `torch.random.fork_rng()` so it neither creates a gradient path nor consumes the base planner RNG stream.
- Gradient clipping is performed independently for base-planner parameters and residual-head parameters.
- Raw `output["trajectory"]` remains unchanged.
- `output["trajectory_execution_sparse"] = raw + residual` is added for spline execution.
- Runtime spline decodes `trajectory_execution_sparse` when available.
- Residual head is a small MLP and its final layer is zero-initialized, making initial/disabled behavior exactly identity.
