"""AllMerge trajectory-mode reward package.

The unit of comparison is one trajectory mode of one target controlled
vehicle. Teammate trajectories remain fixed to the Stage-1 argmax context.
"""

from .config import (
    TrajectoryModeRewardConfig,
    TrajectoryModeRewardError,
    trajectory_mode_reward_config_sha256,
)
from .contracts import (
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    TRAJECTORY_MODE_REWARD_CONTRACT,
    TRAJECTORY_MODE_REWARD_CONTRACT_SHA256,
)
from .counterfactual import (
    TrajectoryModeCounterfactualReward,
)
from .evaluator import evaluate_candidates
from .results import (
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
