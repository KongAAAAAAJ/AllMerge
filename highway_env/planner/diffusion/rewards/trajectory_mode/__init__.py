"""AllMerge trajectory-mode reward package.

The unit of comparison is one trajectory mode of one target controlled
vehicle. Teammate trajectories remain fixed to the Stage-1 argmax context.
"""

from highway_env.planner.diffusion.rewards.trajectory_mode.config import (
    TrajectoryModeRewardConfig,
    TrajectoryModeRewardError,
    trajectory_mode_reward_config_sha256,
)
from highway_env.planner.diffusion.rewards.trajectory_mode.contracts import (
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    TRAJECTORY_MODE_REWARD_CONTRACT,
    TRAJECTORY_MODE_REWARD_CONTRACT_SHA256,
)
from highway_env.planner.diffusion.rewards.trajectory_mode.counterfactual import (
    TrajectoryModeCounterfactualReward,
)
from highway_env.planner.diffusion.rewards.trajectory_mode.evaluator import evaluate_candidates
from highway_env.planner.diffusion.rewards.trajectory_mode.results import (
    RewardGeometryContext,
    TrajectoryModePretrainRewardResult,
    TrajectoryModeRewardResult,
)

__all__ = [
    "TrajectoryModeRewardConfig",
    "TrajectoryModeRewardError",
    "trajectory_mode_reward_config_sha256",
    "TRAJECTORY_MODE_REWARD_CONTRACT",
    "TRAJECTORY_MODE_REWARD_CONTRACT_SHA256",
    "GRPO_OPEN_REWARD_APPLICATION_CONTRACT",
    "RewardGeometryContext",
    "TrajectoryModePretrainRewardResult",
    "TrajectoryModeRewardResult",
    "TrajectoryModeCounterfactualReward",
    "evaluate_candidates",
]
