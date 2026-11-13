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
