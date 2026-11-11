"""Diffusion planners and rewards, exposed through lazy public imports."""
from importlib import import_module

_EXPORTS = {'StructuredDiffusionConfig': ('highway_env.planner.diffusion.config.model', 'StructuredDiffusionConfig'), 'build_structured_diffusion_config': ('highway_env.planner.diffusion.config.model', 'build_structured_diffusion_config'), 'DiffusionPlannerRuntime': ('highway_env.planner.diffusion.inference.runtime', 'DiffusionPlannerRuntime'), 'StructuredDiffusionPlanner': ('highway_env.planner.diffusion.models.trajectory', 'StructuredDiffusionPlanner'), 'TrajectoryModeCounterfactualReward': ('highway_env.planner.diffusion.rewards.trajectory_mode', 'TrajectoryModeCounterfactualReward'), 'TrajectoryModePretrainRewardResult': ('highway_env.planner.diffusion.rewards.trajectory_mode', 'TrajectoryModePretrainRewardResult'), 'TrajectoryModeRewardConfig': ('highway_env.planner.diffusion.rewards.trajectory_mode', 'TrajectoryModeRewardConfig'), 'TrajectoryModeRewardError': ('highway_env.planner.diffusion.rewards.trajectory_mode', 'TrajectoryModeRewardError'), 'TrajectoryModeRewardResult': ('highway_env.planner.diffusion.rewards.trajectory_mode', 'TrajectoryModeRewardResult'), 'evaluate_candidates': ('highway_env.planner.diffusion.rewards.trajectory_mode', 'evaluate_candidates')}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value
