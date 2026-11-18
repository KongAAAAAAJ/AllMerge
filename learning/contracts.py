"""Shared model dimension checks for runtime tools."""
def validate_model_schema(model_config) -> None:
    if int(model_config.horizon_steps) != 8:
        raise RuntimeError(f"Expected 8-step horizon, got {model_config.horizon_steps}")
    if int(model_config.num_modes) <= 0:
        raise RuntimeError(f"num_modes must be positive, got {model_config.num_modes}")
    if abs(float(model_config.trajectory_dt) - 0.5) > 1e-9:
        raise RuntimeError(f"Expected trajectory_dt=0.5, got {model_config.trajectory_dt}")
