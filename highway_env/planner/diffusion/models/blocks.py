"""Shared attention blocks and diffusion time embeddings."""
from __future__ import annotations
import math
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from torch import nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
    ) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        half = self.dim // 2

        scale = math.log(
            10000.0
        ) / max(
            half - 1,
            1,
        )

        frequencies = torch.exp(
            torch.arange(
                half,
                device=timesteps.device,
                dtype=torch.float32,
            )
            * -scale
        )

        values = (
            timesteps.float()[:, None]
            * frequencies[None, :]
        )

        embedding = torch.cat(
            [
                values.sin(),
                values.cos(),
            ],
            dim=-1,
        )

        if embedding.shape[-1] < self.dim:
            embedding = F.pad(
                embedding,
                (0, 1),
            )

        return embedding


class CrossAttentionBlock(nn.Module):
    """
    Structured replacement for GridSampleCrossBEVAttention.

    Mode/trajectory tokens cross-attend:
        map tokens
        agent tokens
        ego + target tokens
    """

    def __init__(
        self,
        config: StructuredDiffusionConfig,
    ) -> None:
        super().__init__()

        d = config.d_model
        h = config.num_heads

        self.mode_self_attn = (
            nn.MultiheadAttention(
                d,
                h,
                dropout=config.dropout,
                batch_first=True,
            )
        )

        self.map_attn = nn.MultiheadAttention(
            d,
            h,
            dropout=config.dropout,
            batch_first=True,
        )

        self.agent_attn = (
            nn.MultiheadAttention(
                d,
                h,
                dropout=config.dropout,
                batch_first=True,
            )
        )

        self.global_attn = (
            nn.MultiheadAttention(
                d,
                h,
                dropout=config.dropout,
                batch_first=True,
            )
        )

        self.time_film = nn.Sequential(
            nn.Linear(
                d,
                2 * d,
            ),
            nn.SiLU(),
            nn.Linear(
                2 * d,
                2 * d,
            ),
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                d,
                config.d_ffn,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.d_ffn,
                d,
            ),
        )

        self.norm_self = nn.LayerNorm(d)
        self.norm_map = nn.LayerNorm(d)
        self.norm_agent = nn.LayerNorm(d)
        self.norm_global = nn.LayerNorm(d)
        self.norm_ffn = nn.LayerNorm(d)

    def forward(
        self,
        mode_tokens: torch.Tensor,
        scene: SceneEncoding,
        time_embed: torch.Tensor,
    ) -> torch.Tensor:
        x = mode_tokens

        residual = x
        q = self.norm_self(x)
        x_sa, _ = self.mode_self_attn(
            q,
            q,
            q,
            need_weights=False,
        )
        x = residual + x_sa

        # Map cross-attention.
        if (
            ~scene.map_padding_mask
        ).any():
            residual = x
            q = self.norm_map(x)

            map_out, _ = self.map_attn(
                q,
                scene.map_tokens,
                scene.map_tokens,
                key_padding_mask=(
                    scene.map_padding_mask
                ),
                need_weights=False,
            )
            x = residual + map_out

        # Agent cross-attention.
        if (
            ~scene.agent_padding_mask
        ).any():
            residual = x
            q = self.norm_agent(x)

            # Safe all-invalid handling row-by-row.
            padding = (
                scene.agent_padding_mask
            ).clone()
            tokens = (
                scene.agent_tokens
            )

            all_invalid = padding.all(
                dim=1
            )

            if all_invalid.any():
                padding[
                    all_invalid,
                    0,
                ] = False
                tokens = tokens.clone()
                tokens[
                    all_invalid,
                    0,
                ] = 0.0

            agent_out, _ = (
                self.agent_attn(
                    q,
                    tokens,
                    tokens,
                    key_padding_mask=padding,
                    need_weights=False,
                )
            )
            x = residual + agent_out

        # Ego + route/target conditioning.
        global_tokens = torch.cat(
            [
                scene.ego_token,
                scene.target_token,
            ],
            dim=1,
        )

        residual = x
        q = self.norm_global(x)

        global_out, _ = self.global_attn(
            q,
            global_tokens,
            global_tokens,
            need_weights=False,
        )
        x = residual + global_out

        # Diffusion-timestep FiLM modulation.
        scale_shift = self.time_film(
            time_embed
        )
        scale, shift = scale_shift.chunk(
            2,
            dim=-1,
        )
        x = (
            x
            * (
                1.0
                + scale[:, None, :]
            )
            + shift[:, None, :]
        )

        x = x + self.ffn(
            self.norm_ffn(x)
        )

        return x


