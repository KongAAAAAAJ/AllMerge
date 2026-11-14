# Diffusion planner

The package separates model computation, inference, rewards, and diagnostics.
Paths supplied in `configs/train/diffusion.yaml` remain relative to `all_merge/`.

| Package | Responsibility |
| --- | --- |
| `config/` | Model configuration and experiment YAML loading |
| `common/` | Scene tensors, encoders, mode assignment, splines, bicycle rollout, dense supervision, fuel and LTR objectives |
| `models/` | Trajectory/control planners, shared neural blocks, DDIM schedule |
| `inference/` | Public runtime facade and trajectory/control implementations |
| `rewards/trajectory_mode/` | Production candidate scoring, geometry, and risk |
| `grpo/` | Group sampling, reward adaptation, objectives, and fine-tuning |
| `diagnostics/` | Background prediction, interaction analysis, validation reports |

Public imports remain available from the package root:

```python
from highway_env.planner.diffusion import (
    DiffusionPlannerRuntime,
    StructuredDiffusionPlanner,
    build_structured_diffusion_config,
)
```

Root exports are loaded on demand. Internal modules use explicit package paths.
Models share `models/blocks.py`; the control planner does not import the
trajectory planner to obtain attention blocks or time embeddings.

`inference/runtime.py` selects the backend. Create the runtime once per
environment and reuse it across planning steps. Moving modules does not change
the saved model parameter names or tensor contracts.

Run validation from `all_merge/`:

```powershell
python -m pytest tests/test_diffusion_control_d1.py tests/test_diffusion_spline_runtime.py tests/grpo -q
python -m tests.test_structured_diffusion_smoke
python -m highway_env.planner.diffusion.diagnostics.validation.smoke_test
```

Training entry points remain in `scripts/training/`; the unified configuration
is `configs/train/diffusion.yaml`.

## TRACE control-space inference with Joint Trajectory Projection

JTP is an optional inference-only post-process for the control-space planner.
It does not modify the denoiser architecture, diffusion checkpoint, or training
losses. Enable it in the environment's `Planner.Diffusion` configuration:

```yaml
output_space: control
jtp_enabled: true
jtp_config: configs/projection/jtp.yaml
jtp_runtime_config: configs/projection/jtp_runtime.yaml
jtp_control_execution_enabled: false
# A control checkpoint trained with execution.model=koopman_k2 is required.
checkpoint: outputs/your_control_checkpoint/best.pt
diffusion_spline_enabled: false
diffusion_execution_enabled: false
```

The online environment collects five consecutive 10-Hz TruckSim states and
four previously applied reference-control pairs. The first four planner calls
are warm-up only; K2 inference and JTP begin once the synchronized history is
available. A timing gap resets the history window instead of fabricating or
misaligning samples.

Set `runtime.context.payload_kg` and `runtime.context.road_mu` in
`configs/projection/jtp_runtime.yaml`, or expose these physical values on each
controlled vehicle. They are explicit operating conditions and are never
inferred from the scene. The numerical constraint limits and QP/SCP parameters
remain in `configs/projection/jtp.yaml`. The road corridor is built from the current and target
road-network lane groups. Background traffic is predicted with the same
constant-velocity assumption used by Adaptive Anchor evaluation.

If `jtp_control_execution_enabled` is true, the controller applies the first
projected `[acceleration, front-wheel steering]` pair only when JTP returns a
nonlinear-constraint-verified feasible result. During warm-up or after a JTP
failure, the existing Polynomial controller remains active. With this option
false, JTP runs in shadow mode and publishes its projected trajectories and
controls for inspection without changing vehicle actuation.

Runtime diagnostics include `jtp_success`, `jtp_status`, `jtp_max_violation`,
`jtp_iterations`, `jtp_latency_ms`, and the nominal/projected control sequences.
