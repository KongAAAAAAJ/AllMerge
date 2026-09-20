# Stage A — Spline Runtime code-path record

Marker: `STAGE_A_SPLINE_RUNTIME_V1`

## Diffusion inference

```text
highway_env/envs/common/action.py
  MultiAgentAction._build_planner_features()
  -> MultiAgentAction._run_diffusion_shadow()

highway_env/planner/diffusion/runtime.py
  DiffusionPlannerRuntime.infer()

highway_env/planner/diffusion/structured_model.py
  StructuredDiffusionPlanner.infer_multimodal()
  -> trajectory_mode_logits_masked
  -> argmax mode selector
  -> selected trajectory [B,8,2]

highway_env/planner/diffusion/trajectory_spline.py
  selected [B,8,2]
  + current ego-local [0,0]
  + ego body velocity [vx,vy]
  -> clamped cubic spline [B,41,2] @ 10 Hz

highway_env/envs/common/action.py
  -> local-to-world publication per controlled vehicle

highway_env/vehicle/controller.py
  -> existing FOLLOWVehicle.trajectory_steering_control()
     only when Diffusion.diffusion_execution_enabled=true
```

## Pretraining path observed at Stage A baseline

```text
pretraining/trainer.py
  -> StructuredDiffusionPlanner.forward_train()
  -> output["loss"]

structured_model.py currently computes:
  trajectory_regression_loss = SmoothL1(selected predicted x0 trajectory, expert)
  classification_loss = cross_entropy(raw mode logits, target mode)
  total = trajectory_regression_loss + classification_loss
```

Important: Stage A does not modify this training objective.

## Stage A switches

```text
Planner.Diffusion.diffusion_spline_enabled: true/false
Planner.Diffusion.diffusion_execution_enabled: true/false
Planner.Diffusion.spline_dense_dt: 0.1
Planner.Diffusion.spline_tracking_index: 10
```

Execution defaults OFF. Therefore Stage A adds the runtime decoder without silently
changing the existing Polynomial controller behavior.
