from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple


@dataclass
class FeatureScaleConfig:
    """
    Fixed physical-unit scaling for the first AllMerge-native model.

    This is deliberately simple and deterministic. Later expert-data
    pretraining can replace it with dataset mean/std normalization without
    changing the environment feature contract.
    """

    ego: Tuple[float, ...] = (
        40.0,   # vx_body
        15.0,   # vy_body
        10.0,   # ax_body
        10.0,   # ay_body
        1.0,    # yaw_rate
        0.6,    # steering_command
        0.6,    # steering_feedback
        5.0,    # lane_lateral_offset
        1.0,    # sin_heading_error
        1.0,    # cos_heading_error
        0.5,    # roll
        1.5,    # roll_rate
        0.5,    # pitch
        1.5,    # pitch_rate
    )

    agent: Tuple[float, ...] = (
        120.0,  # x_rel
        24.0,   # y_rel
        40.0,   # vx_rel
        20.0,   # vy_rel
        12.0,   # ax_rel
        12.0,   # ay_rel
        1.0,    # sin_heading_rel
        1.0,    # cos_heading_rel
        20.0,   # length
        4.0,    # width
        1.0,    # is_controlled
    )

    map_point: Tuple[float, ...] = (
        120.0,  # x_rel
        24.0,   # y_rel
        1.0,    # sin_heading_rel
        1.0,    # cos_heading_rel
        6.0,    # lane_width
        40.0,   # speed_limit
        1.0,    # is_current_lane
        1.0,    # is_left_lane
        1.0,    # is_right_lane
        1.0,    # is_target_lane
    )

    target_point: Tuple[float, float] = (
        120.0,
        24.0,
    )

    trajectory_xy: Tuple[float, float] = (
        120.0,
        24.0,
    )


@dataclass
class StructuredDiffusionConfig:
    # Existing AllMerge feature contract.
    ego_dim: int = 14
    agent_dim: int = 11
    map_dim: int = 10
    max_agents: int = 16
    max_map_polylines: int = 8
    map_points: int = 32

    num_modes: int = 10
    horizon_steps: int = 8
    trajectory_dt: float = 0.5

    # Keep aligned with Diffusion-metadrive BEV "small" preset.
    d_model: int = 128
    d_ffn: int = 512
    num_heads: int = 4
    num_scene_layers: int = 1
    num_denoiser_layers: int = 2
    dropout: float = 0.1

    # Diffusion settings.
    num_train_timesteps: int = 1000
    train_timestep_max: int = 50

    # Truncated test-time diffusion:
    # x_t starts close to the dynamic anchor rather than pure Gaussian noise.
    inference_start_timestep: int = 8
    inference_timesteps: Tuple[int, ...] = (8, 0)
    inference_noise_scale: float = 1.0

    # Residual clean-sample prediction around the noisy trajectory.
    #
    # Bounds are specified in physical ego-centric coordinates instead of
    # one shared normalized scalar. Internally they are divided by
    # feature_scales.trajectory_xy before being applied.
    #
    # Current defaults:
    #   longitudinal x residual: +/-20 m
    #   lateral      y residual: +/-4 m
    max_residual_x_m: float = 20.0
    max_residual_y_m: float = 4.0

    # DENSE_RESIDUAL_HEAD_V2
    # A small execution-correction head is trained only by L_dense.
    # Its inputs are stop-gradient diffusion mode features plus the selected
    # sparse trajectory. The physical residual is bounded so this module
    # remains a local execution correction rather than a second planner.
    dense_residual_hidden_dim: int = 128
    dense_residual_max_x_m: float = 2.0
    dense_residual_max_y_m: float = 0.75

    feature_scales: FeatureScaleConfig = field(
        default_factory=FeatureScaleConfig
    )

    def validate(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive")
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if self.inference_start_timestep < 0:
            raise ValueError(
                "inference_start_timestep must be >= 0"
            )

        if self.max_residual_x_m <= 0.0:
            raise ValueError(
                "max_residual_x_m must be positive"
            )

        if self.max_residual_y_m <= 0.0:
            raise ValueError(
                "max_residual_y_m must be positive"
            )

        # DENSE_RESIDUAL_HEAD_V2
        if self.dense_residual_hidden_dim <= 0:
            raise ValueError("dense_residual_hidden_dim must be positive")
        if self.dense_residual_max_x_m <= 0.0:
            raise ValueError("dense_residual_max_x_m must be positive")
        if self.dense_residual_max_y_m <= 0.0:
            raise ValueError("dense_residual_max_y_m must be positive")

        if len(self.feature_scales.trajectory_xy) != 2:
            raise ValueError(
                "trajectory_xy scale must contain exactly x/y"
            )

        if any(
            scale <= 0.0
            for scale in self.feature_scales.trajectory_xy
        ):
            raise ValueError(
                "trajectory_xy scales must be positive"
            )

        if len(self.feature_scales.ego) != self.ego_dim:
            raise ValueError("ego scale dimension mismatch")
        if len(self.feature_scales.agent) != self.agent_dim:
            raise ValueError("agent scale dimension mismatch")
        if len(self.feature_scales.map_point) != self.map_dim:
            raise ValueError("map scale dimension mismatch")


def build_structured_diffusion_config(
    **overrides,
) -> StructuredDiffusionConfig:
    cfg = StructuredDiffusionConfig()

    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(
                f"Unknown StructuredDiffusionConfig field: {key}"
            )
        setattr(cfg, key, value)

    cfg.validate()
    return cfg
