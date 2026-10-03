from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .config import StructuredDiffusionConfig
from .tensor_adapter import PlannerTensorAdapter


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers = [
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
    ]
    if dropout > 0.0:
        layers.append(
            nn.Dropout(dropout)
        )

    layers.extend(
        [
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        ]
    )

    return nn.Sequential(*layers)


class EgoEncoder(nn.Module):
    def __init__(
        self,
        config: StructuredDiffusionConfig,
    ) -> None:
        super().__init__()
        self.net = make_mlp(
            config.ego_dim,
            config.d_model,
            config.d_model,
            config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # [B,D] -> [B,1,C]
        return self.net(x).unsqueeze(1)


class AgentEncoder(nn.Module):
    def __init__(
        self,
        config: StructuredDiffusionConfig,
    ) -> None:
        super().__init__()

        self.net = make_mlp(
            config.agent_dim,
            config.d_model,
            config.d_model,
            config.dropout,
        )

        # Optional lightweight context mixing among valid agents.
        if config.num_scene_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_ffn,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.context_encoder = nn.TransformerEncoder(
                layer,
                num_layers=config.num_scene_layers,
            )
        else:
            self.context_encoder = None

    def forward(
        self,
        states: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.net(states)

        if self.context_encoder is None:
            return tokens

        # Transformer all-masked rows can yield NaN. Ensure one temporary
        # unmasked zero-token, then zero the row again afterwards.
        padding_mask = ~valid_mask
        all_invalid = ~valid_mask.any(
            dim=1
        )

        safe_padding_mask = padding_mask.clone()

        if all_invalid.any():
            safe_padding_mask[
                all_invalid,
                0,
            ] = False
            tokens = tokens.clone()
            tokens[
                all_invalid,
                0,
            ] = 0.0

        tokens = self.context_encoder(
            tokens,
            src_key_padding_mask=safe_padding_mask,
        )

        tokens = tokens.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        return tokens


class PolylineEncoder(nn.Module):
    """
    Vectorized map encoder.

    Point MLP -> mean/max pooling -> polyline token.
    """

    def __init__(
        self,
        config: StructuredDiffusionConfig,
    ) -> None:
        super().__init__()

        self.point_encoder = make_mlp(
            config.map_dim,
            config.d_model,
            config.d_model,
            config.dropout,
        )

        self.polyline_fuser = nn.Sequential(
            nn.Linear(
                2 * config.d_model,
                config.d_model,
            ),
            nn.LayerNorm(
                config.d_model
            ),
            nn.GELU(),
        )

    def forward(
        self,
        polylines: torch.Tensor,
    ) -> torch.Tensor:
        # [B,N,P,D] or [B,P,D]
        squeeze_lane_dim = False

        if polylines.ndim == 3:
            polylines = polylines.unsqueeze(
                1
            )
            squeeze_lane_dim = True

        point_tokens = self.point_encoder(
            polylines
        )

        mean_pool = point_tokens.mean(
            dim=-2
        )
        max_pool = point_tokens.amax(
            dim=-2
        )

        tokens = self.polyline_fuser(
            torch.cat(
                [mean_pool, max_pool],
                dim=-1,
            )
        )

        if squeeze_lane_dim:
            tokens = tokens.squeeze(1)

        return tokens


class TargetEncoder(nn.Module):
    def __init__(
        self,
        config: StructuredDiffusionConfig,
        map_encoder: PolylineEncoder,
    ) -> None:
        super().__init__()

        self.map_encoder = map_encoder

        self.point_encoder = make_mlp(
            2,
            config.d_model,
            config.d_model,
            config.dropout,
        )

        self.fuser = nn.Sequential(
            nn.Linear(
                2 * config.d_model,
                config.d_model,
            ),
            nn.LayerNorm(
                config.d_model
            ),
            nn.GELU(),
        )

    def forward(
        self,
        target_point: torch.Tensor,
        target_lane: torch.Tensor,
    ) -> torch.Tensor:
        lane_token = self.map_encoder(
            target_lane
        )
        point_token = self.point_encoder(
            target_point
        )

        fused = self.fuser(
            torch.cat(
                [lane_token, point_token],
                dim=-1,
            )
        )

        return fused.unsqueeze(1)


@dataclass
class SceneEncoding:
    ego_token: torch.Tensor
    agent_tokens: torch.Tensor
    agent_padding_mask: torch.Tensor

    map_tokens: torch.Tensor
    map_padding_mask: torch.Tensor

    target_token: torch.Tensor


class StructuredSceneEncoder(nn.Module):
    """
    AllMerge-native replacement for camera/LiDAR + TransFuser backbone.
    """

    def __init__(
        self,
        config: StructuredDiffusionConfig,
        adapter: PlannerTensorAdapter,
    ) -> None:
        super().__init__()

        self.config = config
        self.adapter = adapter

        self.ego_encoder = EgoEncoder(
            config
        )
        self.agent_encoder = AgentEncoder(
            config
        )
        self.map_encoder = PolylineEncoder(
            config
        )
        self.target_encoder = TargetEncoder(
            config,
            self.map_encoder,
        )

    def forward(
        self,
        features,
    ) -> SceneEncoding:
        ego = self.adapter.normalize_ego(
            features["ego_state"]
        )
        agents = (
            self.adapter.normalize_agents(
                features["agent_states"]
            )
        )
        maps = self.adapter.normalize_map(
            features["map_polylines"]
        )
        target_point = (
            self.adapter.normalize_target_point(
                features["target_point"]
            )
        )
        target_lane = (
            self.adapter.normalize_map(
                features[
                    "target_lane_polyline"
                ]
            )
        )

        ego_token = self.ego_encoder(
            ego
        )

        agent_tokens = self.agent_encoder(
            agents,
            features[
                "agent_valid_mask"
            ],
        )

        map_tokens = self.map_encoder(
            maps
        )
        map_tokens = map_tokens.masked_fill(
            (
                ~features[
                    "map_valid_mask"
                ]
            ).unsqueeze(-1),
            0.0,
        )

        target_token = self.target_encoder(
            target_point,
            target_lane,
        )

        return SceneEncoding(
            ego_token=ego_token,
            agent_tokens=agent_tokens,
            agent_padding_mask=(
                ~features[
                    "agent_valid_mask"
                ]
            ),
            map_tokens=map_tokens,
            map_padding_mask=(
                ~features[
                    "map_valid_mask"
                ]
            ),
            target_token=target_token,
        )
