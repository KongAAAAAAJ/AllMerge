"""Migration-first GRPO utilities for AllMerge StructuredDiffusionPlanner."""

from highway_env.planner.diffusion.grpo.checkpoint import load_pretrained, save_grpo_checkpoint
from highway_env.planner.diffusion.grpo.objective import GRPOObjectiveResult, group_relative_advantage, grpo_clipped_objective
from highway_env.planner.diffusion.grpo.reward_adapter import CandidateRewardAdapter, fake_progress_reward, resolve_reward_evaluator
from highway_env.planner.diffusion.grpo.rollout_adapter import AllMergeRolloutAdapter
from highway_env.planner.diffusion.grpo.sampling import DiffusionTrace, GroupDiffusionSampler
from highway_env.planner.diffusion.grpo.scheduler import StochasticDDIMTransition, TransitionResult
from highway_env.planner.diffusion.grpo.trainer import GRPOConfig, GRPOTrainer

__all__ = [
    "AllMergeRolloutAdapter",
    "CandidateRewardAdapter",
    "DiffusionTrace",
    "GRPOConfig",
    "GRPOObjectiveResult",
    "GRPOTrainer",
    "GroupDiffusionSampler",
    "StochasticDDIMTransition",
    "TransitionResult",
    "fake_progress_reward",
    "group_relative_advantage",
    "grpo_clipped_objective",
    "load_pretrained",
    "resolve_reward_evaluator",
    "save_grpo_checkpoint",
]
