from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import StructuredDiffusionConfig
from .diffusion_schedule import TruncatedDDIMSchedule
from .dense_supervision import (
    DenseSparseResidualHead,
    dense_position_loss,
    dense_timestep_weight,
)
from .trajectory_spline import ClampedCubicTrajectorySpline
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
        *,
        return_mode_tokens: bool = False,
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

        if return_mode_tokens:
            return (
                predicted_x0_norm,
                logits,
                mode_tokens,
            )

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

        # STAGED_DENSE_SUPERVISION_V1
        # Parameter-free differentiable decoder reused by the Stage-D objective.
        self.dense_trajectory_spline = ClampedCubicTrajectorySpline(
            horizon_s=float(config.horizon_steps) * float(config.trajectory_dt),
            sparse_dt=float(config.trajectory_dt),
            dense_dt=0.1,
        )

        # DENSE_RESIDUAL_HEAD_V2
        # L_dense trains only this small execution-correction MLP.
        self.dense_residual_head = DenseSparseResidualHead(
            feature_dim=int(config.d_model),
            horizon_steps=int(config.horizon_steps),
            hidden_dim=int(config.dense_residual_hidden_dim),
            max_residual_x_m=float(config.dense_residual_max_x_m),
            max_residual_y_m=float(config.dense_residual_max_y_m),
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
        final_mode_tokens = None  # DENSE_RESIDUAL_HEAD_V2

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

            predicted_x0, logits, mode_tokens = (
                self.denoiser(
                    sample,
                    t_batch,
                    scene,
                    return_mode_tokens=True,
                )
            )

            final_x0 = predicted_x0
            final_logits = logits
            final_mode_tokens = mode_tokens

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

        # DENSE_RESIDUAL_HEAD_V2
        # Keep raw Diffusion output unchanged for planner/open-loop metrics,
        # while exposing an execution-corrected sparse trajectory for spline.
        selected_x0_norm = torch.gather(
            final_x0,
            dim=1,
            index=gather_index,
        ).squeeze(1)
        feature_index = mode_index[:, None, None].expand(
            -1, 1, int(self.config.d_model)
        )
        selected_mode_feature = torch.gather(
            final_mode_tokens,
            dim=1,
            index=feature_index,
        ).squeeze(1)
        execution_residual_m = self.dense_residual_head(
            selected_mode_feature,
            selected_x0_norm,
        )
        execution_sparse = best + execution_residual_m

        return {
            "trajectory": best,
            "trajectory_execution_sparse": execution_sparse,
            "trajectory_execution_residual": execution_residual_m,
            "trajectory_candidates": candidates,
            "trajectory_mode_logits": final_logits,
            "trajectory_mode_logits_masked": masked_logits,
            "trajectory_mode_idx": mode_index,
            # EVAL_VIZ_V1
            "trajectory_noisy_initial": (
                self.adapter.denormalize_trajectory(initial_noisy)
            ),
        }

    # STAGED_DENSE_SUPERVISION_V1
    # DENSE_RESIDUAL_HEAD_V2
    def _dense_terminal_auxiliary(
        self,
        *,
        scene: SceneEncoding,
        clean_anchor_norm: torch.Tensor,
        noise: torch.Tensor,
        features: Dict[str, torch.Tensor],
        target_mode: torch.Tensor,
        target_trajectory_dense: torch.Tensor,
        dense_loss_type: str,
        dense_loss_lambda_p: float,
        dense_loss_terminal_timestep: int,
        dense_loss_weight_mode: str,
        dense_loss_terminal_weight: float,
    ) -> Dict[str, torch.Tensor]:
        """Dense execution supervision with a hard gradient boundary.

        The terminal Diffusion prediction and its mode feature are treated as
        fixed inputs. L_dense can only update ``dense_residual_head``.
        """
        batch = int(clean_anchor_norm.shape[0])
        terminal_t = int(dense_loss_terminal_timestep)
        if terminal_t < 0 or terminal_t >= int(self.config.num_train_timesteps):
            raise ValueError(
                "dense_loss_terminal_timestep must lie in scheduler range "
                f"[0,{int(self.config.num_train_timesteps) - 1}], got {terminal_t}"
            )
        if terminal_t >= int(self.config.train_timestep_max):
            raise ValueError(
                "dense_loss_terminal_timestep should be inside the training "
                f"timestep range [0,{int(self.config.train_timestep_max) - 1}]"
            )

        expected_dense_steps = int(round(
            float(self.config.horizon_steps)
            * float(self.config.trajectory_dt)
            / 0.1
        ))
        expected_shape = (batch, expected_dense_steps, 2)
        if tuple(target_trajectory_dense.shape) != expected_shape:
            raise ValueError(
                "real 10 Hz dense target has wrong shape: "
                f"got {tuple(target_trajectory_dense.shape)}, expected {expected_shape}"
            )
        if not torch.isfinite(target_trajectory_dense).all():
            raise ValueError("real 10 Hz dense target contains NaN/Inf")

        terminal_timesteps = torch.full(
            (batch,),
            terminal_t,
            dtype=torch.long,
            device=clean_anchor_norm.device,
        )
        terminal_noisy = self.schedule.add_noise(
            clean_anchor_norm,
            noise,
            terminal_timesteps,
        )

        # The auxiliary forward must not alter the Diffusion planner's RNG
        # stream (dropout etc.), otherwise the base planner could diverge even
        # with zero dense gradient. fork_rng restores CPU/CUDA RNG afterwards.
        cuda_devices = []
        if terminal_noisy.is_cuda:
            cuda_devices = [terminal_noisy.device.index or 0]
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            with torch.no_grad():
                terminal_x0_norm, _, terminal_mode_tokens = self.denoiser(
                    terminal_noisy,
                    terminal_timesteps,
                    scene,
                    return_mode_tokens=True,
                )
                terminal_candidates_m = self.adapter.denormalize_trajectory(
                    terminal_x0_norm
                )

        gather_index = (
            target_mode[:, None, None, None]
            .expand(-1, 1, self.config.horizon_steps, 2)
        )
        selected_terminal_m = torch.gather(
            terminal_candidates_m,
            dim=1,
            index=gather_index,
        ).squeeze(1).detach()
        selected_terminal_norm = torch.gather(
            terminal_x0_norm,
            dim=1,
            index=gather_index,
        ).squeeze(1).detach()
        feature_index = target_mode[:, None, None].expand(
            -1, 1, int(self.config.d_model)
        )
        selected_mode_feature = torch.gather(
            terminal_mode_tokens,
            dim=1,
            index=feature_index,
        ).squeeze(1).detach()

        execution_residual_m = self.dense_residual_head(
            selected_mode_feature,
            selected_terminal_norm,
        )
        execution_sparse_m = (
            selected_terminal_m.float() + execution_residual_m.float()
        )

        # Keep the spline solve in fp32 even under AMP.
        start_xy = torch.zeros_like(execution_sparse_m[:, 0, :])
        start_velocity_xy = features["ego_state"][:, 0:2].float().detach()
        pred_dense_all_m = self.dense_trajectory_spline(
            execution_sparse_m,
            start_xy=start_xy,
            start_velocity_xy=start_velocity_xy,
        )
        pred_dense_future_m = pred_dense_all_m[:, 1:, :]
        expert_dense_m = target_trajectory_dense.to(
            device=pred_dense_future_m.device,
            dtype=torch.float32,
        )

        raw = dense_position_loss(
            pred_dense_future_m,
            expert_dense_m,
            loss_type=dense_loss_type,
        )
        weight_per_sample = dense_timestep_weight(
            terminal_timesteps,
            mode=dense_loss_weight_mode,
            terminal_weight=dense_loss_terminal_weight,
        )
        weight = weight_per_sample.mean()
        weighted = raw * float(dense_loss_lambda_p) * weight
        ade_m = torch.linalg.vector_norm(
            pred_dense_future_m - expert_dense_m,
            dim=-1,
        ).mean()

        # Diagnostic baseline: same terminal sparse trajectory without residual.
        with torch.no_grad():
            base_dense_all_m = self.dense_trajectory_spline(
                selected_terminal_m.float(),
                start_xy=start_xy,
                start_velocity_xy=start_velocity_xy,
            )
            base_dense_future_m = base_dense_all_m[:, 1:, :]
            base_ade_m = torch.linalg.vector_norm(
                base_dense_future_m - expert_dense_m,
                dim=-1,
            ).mean()

        residual_abs = execution_residual_m.abs()
        residual_l2 = torch.linalg.vector_norm(execution_residual_m, dim=-1)
        residual_abs_x = residual_abs[..., 0]
        residual_abs_y = residual_abs[..., 1]
        x_bound = float(self.config.dense_residual_max_x_m)
        y_bound = float(self.config.dense_residual_max_y_m)
        x_saturation_ratio = (
            residual_abs_x >= (0.95 * x_bound)
        ).float().mean()
        y_saturation_ratio = (
            residual_abs_y >= (0.95 * y_bound)
        ).float().mean()

        return {
            "dense_loss_raw": raw,
            "dense_loss_weighted": weighted,
            "dense_ade_m": ade_m,
            "dense_base_ade_m": base_ade_m,
            "dense_ade_gain_m": base_ade_m - ade_m,
            "dense_residual_mean_abs_m": residual_abs.mean(),
            "dense_residual_max_abs_m": residual_abs.max(),
            "dense_residual_mean_l2_m": residual_l2.mean(),
            # RESIDUAL_XY_DIAGNOSTICS_V1
            "dense_residual_mean_abs_x_m": residual_abs_x.mean(),
            "dense_residual_mean_abs_y_m": residual_abs_y.mean(),
            "dense_residual_max_abs_x_m": residual_abs_x.max(),
            "dense_residual_max_abs_y_m": residual_abs_y.max(),
            "dense_residual_x_saturation_ratio": x_saturation_ratio,
            "dense_residual_y_saturation_ratio": y_saturation_ratio,
            "dense_weight_t": weight,
            "dense_terminal_fraction": raw.new_ones(()),
        }

    def forward_train(
        self,
        features: Dict[str, torch.Tensor],
        target_trajectory: torch.Tensor,
        target_semantic: Optional[
            torch.Tensor
        ] = None,
        *,
        target_trajectory_dense: Optional[torch.Tensor] = None,
        dense_loss_enabled: bool = False,
        dense_loss_lambda_p: float = 0.0,
        dense_loss_type: str = "smooth_l1",
        dense_loss_terminal_timestep: int = 0,
        dense_loss_weight_mode: str = "terminal_constant",
        dense_loss_terminal_weight: float = 1.0,
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

        # STAGED_DENSE_SUPERVISION_V1
        dense_loss_raw = regression_loss.new_zeros(())
        dense_loss_weighted = regression_loss.new_zeros(())
        dense_ade_m = regression_loss.new_zeros(())
        dense_base_ade_m = regression_loss.new_zeros(())
        dense_ade_gain_m = regression_loss.new_zeros(())
        dense_residual_mean_abs_m = regression_loss.new_zeros(())
        dense_residual_max_abs_m = regression_loss.new_zeros(())
        dense_residual_mean_l2_m = regression_loss.new_zeros(())
        # RESIDUAL_XY_DIAGNOSTICS_V1
        dense_residual_mean_abs_x_m = regression_loss.new_zeros(())
        dense_residual_mean_abs_y_m = regression_loss.new_zeros(())
        dense_residual_max_abs_x_m = regression_loss.new_zeros(())
        dense_residual_max_abs_y_m = regression_loss.new_zeros(())
        dense_residual_x_saturation_ratio = regression_loss.new_zeros(())
        dense_residual_y_saturation_ratio = regression_loss.new_zeros(())
        dense_weight_t = regression_loss.new_zeros(())
        dense_terminal_fraction = regression_loss.new_zeros(())

        dense_active = bool(dense_loss_enabled) and float(dense_loss_lambda_p) > 0.0
        if dense_active:
            if target_trajectory_dense is None:
                raise ValueError(
                    "dense supervision requires the real 10 Hz dense target from "
                    "Stage 3; no sparse-to-dense fallback is allowed"
                )
            dense_aux = self._dense_terminal_auxiliary(
                scene=scene,
                clean_anchor_norm=clean_anchor_norm,
                noise=noise,
                features=features,
                target_mode=target_mode,
                target_trajectory_dense=target_trajectory_dense,
                dense_loss_type=dense_loss_type,
                dense_loss_lambda_p=float(dense_loss_lambda_p),
                dense_loss_terminal_timestep=int(dense_loss_terminal_timestep),
                dense_loss_weight_mode=dense_loss_weight_mode,
                dense_loss_terminal_weight=float(dense_loss_terminal_weight),
            )
            dense_loss_raw = dense_aux["dense_loss_raw"]
            dense_loss_weighted = dense_aux["dense_loss_weighted"]
            dense_ade_m = dense_aux["dense_ade_m"]
            dense_base_ade_m = dense_aux["dense_base_ade_m"]
            dense_ade_gain_m = dense_aux["dense_ade_gain_m"]
            dense_residual_mean_abs_m = dense_aux["dense_residual_mean_abs_m"]
            dense_residual_max_abs_m = dense_aux["dense_residual_max_abs_m"]
            dense_residual_mean_l2_m = dense_aux["dense_residual_mean_l2_m"]
            # RESIDUAL_XY_DIAGNOSTICS_V1
            dense_residual_mean_abs_x_m = dense_aux["dense_residual_mean_abs_x_m"]
            dense_residual_mean_abs_y_m = dense_aux["dense_residual_mean_abs_y_m"]
            dense_residual_max_abs_x_m = dense_aux["dense_residual_max_abs_x_m"]
            dense_residual_max_abs_y_m = dense_aux["dense_residual_max_abs_y_m"]
            dense_residual_x_saturation_ratio = dense_aux["dense_residual_x_saturation_ratio"]
            dense_residual_y_saturation_ratio = dense_aux["dense_residual_y_saturation_ratio"]
            dense_weight_t = dense_aux["dense_weight_t"]
            dense_terminal_fraction = dense_aux["dense_terminal_fraction"]

        # DENSE_RESIDUAL_HEAD_V2
        base_loss = regression_loss + classification_loss
        loss = base_loss + dense_loss_weighted

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

            "base_loss":
                base_loss,

            "trajectory_regression_loss":
                regression_loss,

            "trajectory_classification_loss":
                classification_loss,

            # STAGED_DENSE_SUPERVISION_V1
            "dense_loss_raw":
                dense_loss_raw,

            "dense_loss_weighted":
                dense_loss_weighted,

            "dense_ade_m":
                dense_ade_m,

            "dense_base_ade_m":
                dense_base_ade_m,

            "dense_ade_gain_m":
                dense_ade_gain_m,

            "dense_residual_mean_abs_m":
                dense_residual_mean_abs_m,

            "dense_residual_max_abs_m":
                dense_residual_max_abs_m,

            "dense_residual_mean_l2_m":
                dense_residual_mean_l2_m,

            # RESIDUAL_XY_DIAGNOSTICS_V1
            "dense_residual_mean_abs_x_m":
                dense_residual_mean_abs_x_m,

            "dense_residual_mean_abs_y_m":
                dense_residual_mean_abs_y_m,

            "dense_residual_max_abs_x_m":
                dense_residual_max_abs_x_m,

            "dense_residual_max_abs_y_m":
                dense_residual_max_abs_y_m,

            "dense_residual_x_saturation_ratio":
                dense_residual_x_saturation_ratio,

            "dense_residual_y_saturation_ratio":
                dense_residual_y_saturation_ratio,

            "dense_weight_t":
                dense_weight_t,

            "dense_terminal_fraction":
                dense_terminal_fraction,

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
