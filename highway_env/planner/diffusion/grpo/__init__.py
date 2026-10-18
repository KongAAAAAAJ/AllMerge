"""Migration-first GRPO utilities for AllMerge StructuredDiffusionPlanner."""

from .checkpoint import load_pretrained, save_grpo_checkpoint
from .constraints import ConstraintBatch, SUPPORTED_CONSTRAINTS, evaluate_w4_constraints
from .constraint_strategy import (
    ActiveConstraintConfig, ActiveConstraintStrategy,
    LagrangianConstraintConfig, LagrangianConstraintStrategy,
)
from .objective import GRPOObjectiveResult, group_relative_advantage, grpo_clipped_objective
from .reward_adapter import CandidateRewardAdapter, fake_progress_reward, resolve_reward_evaluator
from .rollout_adapter import AllMergeRolloutAdapter
from .sampling import DiffusionTrace, GroupDiffusionSampler
from .scheduler import StochasticDDIMTransition, TransitionResult
from .trainer import GRPOConfig, GRPOTrainer
from .task_reward import SUPPORTED_TASK_REWARDS, task_reward_from_result

__all__ = [
    "AllMergeRolloutAdapter",
    "ActiveConstraintConfig",
    "ActiveConstraintStrategy",
    "CandidateRewardAdapter",
    "ConstraintBatch",
    "DiffusionTrace",
    "GRPOConfig",
    "GRPOObjectiveResult",
    "GRPOTrainer",
    "LagrangianConstraintConfig",
    "LagrangianConstraintStrategy",
    "GroupDiffusionSampler",
    "SUPPORTED_CONSTRAINTS",
    "SUPPORTED_TASK_REWARDS",
    "StochasticDDIMTransition",
    "TransitionResult",
    "evaluate_w4_constraints",
    "fake_progress_reward",
    "group_relative_advantage",
    "grpo_clipped_objective",
    "load_pretrained",
    "resolve_reward_evaluator",
    "save_grpo_checkpoint",
    "task_reward_from_result",
]

# LAGRANGIAN_CONSTRAINED_GRPO_BASELINE_V2_SEMANTIC

# ACTIVE_CONSTRAINT_GRPO_V1_SEMANTIC

# DECOUPLED_TASK_CONSTRAINT_V2_SEMANTIC
