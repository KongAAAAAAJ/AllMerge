from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import StructuredDiffusionConfig
from .diffusion_schedule import TruncatedDDIMSchedule
from .encoders import (
    SceneEncoding,
    StructuredSceneEncoder,
)
from .tensor_adapter import PlannerTensorAdapter


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


class StructuredTrajectoryDenoiser(nn.Module):
    def __init__(
        self,
        config: StructuredDiffusionConfig,
        adapter: PlannerTensorAdapter,
    ) -> None:
        super().__init__()

        self.config = config
        self.adapter = adapter

        # Convert physical residual limits [m] into normalized
        # trajectory coordinates once at construction time.
        trajectory_scale_x = float(
            config.feature_scales.trajectory_xy[0]
        )
        trajectory_scale_y = float(
            config.feature_scales.trajectory_xy[1]
        )

        residual_limit_norm = torch.tensor(
            [
                float(config.max_residual_x_m)
                / trajectory_scale_x,
                float(config.max_residual_y_m)
                / trajectory_scale_y,
            ],
            dtype=torch.float32,
        )

        self.register_buffer(
            "residual_limit_norm",
            residual_limit_norm,
            persistent=False,
        )

        trajectory_dim = (
            config.horizon_steps
            * 2
        )

        self.anchor_encoder = nn.Sequential(
            nn.Linear(
                trajectory_dim,
                config.d_model,
            ),
            nn.LayerNorm(
                config.d_model
            ),
            nn.GELU(),
            nn.Linear(
                config.d_model,
                config.d_model,
            ),
        )

        self.mode_embedding = nn.Embedding(
            config.num_modes,
            config.d_model,
        )

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(
                config.d_model
            ),
            nn.Linear(
                config.d_model,
                4 * config.d_model,
            ),
            nn.SiLU(),
            nn.Linear(
                4 * config.d_model,
                config.d_model,
            ),
        )

        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    config
                )
                for _ in range(
                    config.num_denoiser_layers
                )
            ]
        )

        self.reg_head = nn.Sequential(
            nn.LayerNorm(
                config.d_model
            ),
            nn.Linear(
                config.d_model,
                config.d_ffn,
            ),
            nn.GELU(),
            nn.Linear(
                config.d_ffn,
                trajectory_dim,
            ),
        )

        self.cls_head = nn.Sequential(
            nn.LayerNorm(
                config.d_model
            ),
            nn.Linear(
                config.d_model,
                config.d_model,
            ),
            nn.GELU(),
            nn.Linear(
                config.d_model,
                1,
            ),
        )

    def forward(
        self,
        noisy_norm: torch.Tensor,
        timestep: torch.Tensor,
        scene: SceneEncoding,
    ):
        batch, modes, steps, dims = (
            noisy_norm.shape
        )

        flat = noisy_norm.reshape(
            batch,
            modes,
            steps * dims,
        )

        mode_tokens = self.anchor_encoder(
            flat
        )

        mode_ids = torch.arange(
            modes,
            device=noisy_norm.device,
        )
        mode_tokens = (
            mode_tokens
            + self.mode_embedding(
                mode_ids
            )[None, :, :]
        )

        time_embed = self.time_embedding(
            timestep
        )

        for layer in self.layers:
            mode_tokens = layer(
                mode_tokens,
                scene,
                time_embed,
            )

        residual = self.reg_head(
            mode_tokens
        ).reshape(
            batch,
            modes,
            steps,
            dims,
        )

        # Residual semantics remain unchanged:
        #
        #     x0_hat = x_t + delta_theta
        #
        # The x/y limits are specified in metres in the config and
        # converted into normalized trajectory coordinates in __init__.
        residual = torch.tanh(
            residual
        )

        residual = (
            residual
            * self.residual_limit_norm.to(
                dtype=residual.dtype,
            )
        )

        predicted_x0_norm = (
            noisy_norm + residual
        )

        logits = self.cls_head(
            mode_tokens
        ).squeeze(-1)

        return (
            predicted_x0_norm,
            logits,
        )


class StructuredDiffusionPlanner(nn.Module):
    """
    AllMerge-native diffusion + transformer planner.

    Replaces:
        camera/LiDAR -> TransFuser -> BEV cross-attention

    with:
        ego/agent/vector-map/target structured encoders
        -> map/agent/global cross attention
        -> diffusion trajectory denoising.

    Output trajectory is ego-centric [x, y].
    """

    def __init__(
        self,
        config: StructuredDiffusionConfig,
        adapter: PlannerTensorAdapter,
    ) -> None:
        super().__init__()

        config.validate()

        self.config = config
        self.adapter = adapter

        self.scene_encoder = (
            StructuredSceneEncoder(
                config,
                adapter,
            )
        )

        self.denoiser = (
            StructuredTrajectoryDenoiser(
                config,
                adapter,
            )
        )

        self.schedule = (
            TruncatedDDIMSchedule(
                num_train_timesteps=(
                    config.num_train_timesteps
                )
            )
        )

    @staticmethod
    def _masked_logits(
        logits: torch.Tensor,
        mode_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not mode_valid_mask.any(
            dim=1
        ).all():
            # AnchorBuilder should already guarantee a fallback mode.
            raise RuntimeError(
                "At least one batch item has no valid mode"
            )

        return logits.masked_fill(
            ~mode_valid_mask,
            torch.finfo(
                logits.dtype
            ).min,
        )

    @torch.no_grad()
    def infer_multimodal(
        self,
        features: Dict[str, torch.Tensor],
        *,
        generator: Optional[
            torch.Generator
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        self.eval()

        scene = self.scene_encoder(
            features
        )

        clean_anchor_norm = (
            self.adapter.normalize_trajectory(
                features[
                    "coarse_trajectories"
                ]
            )
        )

        batch = clean_anchor_norm.shape[0]

        start_t = int(
            self.config.inference_start_timestep
        )

        start_timesteps = torch.full(
            (batch,),
            start_t,
            dtype=torch.long,
            device=clean_anchor_norm.device,
        )

        noise = torch.randn(
            clean_anchor_norm.shape,
            dtype=clean_anchor_norm.dtype,
            device=clean_anchor_norm.device,
            generator=generator,
        ) * float(
            self.config.inference_noise_scale
        )

        sample = self.schedule.add_noise(
            clean_anchor_norm,
            noise,
            start_timesteps,
        )
        # EVAL_VIZ_V1: retain the physical noisy trajectory for
        # start-vs-end denoising visualization.
        initial_noisy = sample.clone()

        final_logits = None
        final_x0 = None

        timesteps = list(
            self.config.inference_timesteps
        )

        if not timesteps:
            timesteps = [start_t, 0]

        # Ensure the sampling chain starts where the sample was noised.
        if timesteps[0] != start_t:
            timesteps = [
                start_t,
                *timesteps,
            ]

        for index, timestep in enumerate(
            timesteps
        ):
            t_batch = torch.full(
                (batch,),
                int(timestep),
                dtype=torch.long,
                device=sample.device,
            )

            predicted_x0, logits = (
                self.denoiser(
                    sample,
                    t_batch,
                    scene,
                )
            )

            final_x0 = predicted_x0
            final_logits = logits

            prev_timestep = (
                timesteps[index + 1]
                if index + 1 < len(
                    timesteps
                )
                else None
            )

            sample = (
                self.schedule.step_predict_x0(
                    sample,
                    predicted_x0,
                    int(timestep),
                    (
                        int(prev_timestep)
                        if prev_timestep
                        is not None
                        else None
                    ),
                )
            )

        candidates = (
            self.adapter.denormalize_trajectory(
                final_x0
            )
        )

        masked_logits = (
            self._masked_logits(
                final_logits,
                features[
                    "mode_valid_mask"
                ],
            )
        )

        mode_index = masked_logits.argmax(
            dim=-1
        )

        gather_index = (
            mode_index[
                :,
                None,
                None,
                None,
            ]
            .expand(
                -1,
                1,
                self.config.horizon_steps,
                2,
            )
        )

        best = torch.gather(
            candidates,
            dim=1,
            index=gather_index,
        ).squeeze(1)

        return {
            "trajectory": best,
            "trajectory_candidates": candidates,
            "trajectory_mode_logits": final_logits,
            "trajectory_mode_logits_masked": masked_logits,
            "trajectory_mode_idx": mode_index,
            # EVAL_VIZ_V1
            "trajectory_noisy_initial": (
                self.adapter.denormalize_trajectory(initial_noisy)
            ),
        }

    def forward_train(
        self,
        features: Dict[str, torch.Tensor],
        target_trajectory: torch.Tensor,
        target_semantic: Optional[
            torch.Tensor
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Supervised multimodal pretraining with semantic-constrained
        expert-to-anchor assignment.

        The network structure remains unchanged:
            dynamic anchors
            -> diffusion refinement per mode
            -> mode logits
            -> selected-mode regression + mode classification

        As in DiffusionDrive, one expert trajectory is matched to a plan
        anchor and that same mode supervises both:
            1. trajectory regression;
            2. mode classification.

        F.2 changes ONLY how the target mode is constructed:

            old:
                global nearest among traffic-valid modes

            new:
                expected semantic
                AND geometrically existing modes
                -> nearest anchor by mean pointwise Euclidean distance

        Traffic `mode_valid_mask` is intentionally NOT used for training
        label assignment or classification masking. It remains active in
        infer_multimodal() for online safety/selectability filtering.

        `target_semantic` is optional:
            None:
                derive KEEP / LEFT_LC / RIGHT_LC from existing
                current/left/right/target map flags.

            explicit tensor [B]:
                supports future explicit STOP supervision and unusual
                topology without changing the model architecture.
        """
        from .mode_assignment import (
            assign_expert_mode,
        )

        scene = self.scene_encoder(
            features
        )

        anchors = features[
            "coarse_trajectories"
        ]

        target = target_trajectory

        assignment = (
            assign_expert_mode(
                features=features,
                target_trajectory=target,
                target_semantic=target_semantic,
            )
        )

        target_mode = (
            assignment.target_mode
        )

        clean_anchor_norm = (
            self.adapter.normalize_trajectory(
                anchors
            )
        )

        batch = anchors.shape[0]

        timesteps = torch.randint(
            low=0,
            high=self.config.train_timestep_max,
            size=(batch,),
            device=anchors.device,
        )

        noise = torch.randn_like(
            clean_anchor_norm
        )

        noisy = self.schedule.add_noise(
            clean_anchor_norm,
            noise,
            timesteps,
        )

        predicted_x0_norm, logits = (
            self.denoiser(
                noisy,
                timesteps,
                scene,
            )
        )

        candidates = (
            self.adapter.denormalize_trajectory(
                predicted_x0_norm
            )
        )

        gather_index = (
            target_mode[
                :,
                None,
                None,
                None,
            ]
            .expand(
                -1,
                1,
                self.config.horizon_steps,
                2,
            )
        )

        selected = torch.gather(
            candidates,
            dim=1,
            index=gather_index,
        ).squeeze(1)

        # Preserve the existing AllMerge regression loss so this patch
        # isolates label correctness rather than changing two variables
        # simultaneously.
        regression_loss = F.smooth_l1_loss(
            selected,
            target,
        )

        # IMPORTANT:
        # Training uses RAW logits.
        #
        # A correct expert target is allowed to be traffic-invalid according
        # to the online safety heuristic (F.1 Ego 1 / Ego 2 demonstrated
        # exactly this case). Masking that logit here would make the correct
        # class impossible to learn.
        classification_loss = (
            F.cross_entropy(
                logits,
                target_mode,
            )
        )

        loss = (
            regression_loss
            + classification_loss
        )

        mode_valid_mask = features[
            "mode_valid_mask"
        ].bool()

        target_mode_traffic_valid = (
            torch.gather(
                mode_valid_mask,
                dim=1,
                index=target_mode[
                    :,
                    None,
                ],
            )
            .squeeze(1)
        )

        target_assignment_distance = (
            torch.gather(
                assignment.anchor_distance,
                dim=1,
                index=target_mode[
                    :,
                    None,
                ],
            )
            .squeeze(1)
        )

        return {
            "loss":
                loss,

            "trajectory_regression_loss":
                regression_loss,

            "trajectory_classification_loss":
                classification_loss,

            "target_mode":
                target_mode,

            "target_semantic":
                assignment.target_semantic,

            "target_mode_assignment_distance":
                target_assignment_distance,

            "target_mode_geometry_valid":
                torch.gather(
                    assignment.geometry_mask,
                    dim=1,
                    index=target_mode[
                        :,
                        None,
                    ],
                ).squeeze(1),

            "target_mode_traffic_valid":
                target_mode_traffic_valid,

            "mode_geometry_mask_train":
                assignment.geometry_mask,

            "mode_semantic_mask_train":
                assignment.semantic_mask,

            "mode_assignment_candidate_mask_train":
                assignment.candidate_mask,

            "trajectory_candidates_train":
                candidates,

            "trajectory_mode_logits_train":
                logits,
        }
