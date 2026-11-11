from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TruncatedDDIMSchedule(nn.Module):
    """
    Minimal scheduler for prediction_type="sample" (predict clean x0).

    The original Diffusion-metadrive BEV branch uses DDIMScheduler with
    scaled-linear betas. This implementation keeps the same basic process
    but avoids adding a new runtime dependency on `diffusers`.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ) -> None:
        super().__init__()

        beta_start_sqrt = beta_start ** 0.5
        beta_end_sqrt = beta_end ** 0.5

        betas = torch.linspace(
            beta_start_sqrt,
            beta_end_sqrt,
            num_train_timesteps,
            dtype=torch.float64,
        ) ** 2

        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(
            alphas,
            dim=0,
        ).float()

        self.register_buffer(
            "alpha_cumprod",
            alpha_cumprod,
            persistent=False,
        )

        self.num_train_timesteps = (
            int(num_train_timesteps)
        )

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.alpha_cumprod[
            timesteps
        ].to(
            dtype=clean.dtype,
            device=clean.device,
        )

        while alpha.ndim < clean.ndim:
            alpha = alpha.unsqueeze(-1)

        return (
            alpha.sqrt() * clean
            + (1.0 - alpha).sqrt()
            * noise
        )

    def step_predict_x0(
        self,
        sample: torch.Tensor,
        predicted_x0: torch.Tensor,
        timestep: int,
        prev_timestep: Optional[int],
    ) -> torch.Tensor:
        """
        Deterministic DDIM step (eta=0).

        predicted_x0 is the clean-sample prediction.
        """
        t = int(timestep)

        alpha_t = self.alpha_cumprod[
            t
        ].to(
            dtype=sample.dtype,
            device=sample.device,
        )

        if prev_timestep is None or prev_timestep < 0:
            return predicted_x0

        alpha_prev = self.alpha_cumprod[
            int(prev_timestep)
        ].to(
            dtype=sample.dtype,
            device=sample.device,
        )

        sqrt_one_minus_alpha_t = (
            1.0 - alpha_t
        ).sqrt().clamp_min(1e-6)

        epsilon = (
            sample
            - alpha_t.sqrt()
            * predicted_x0
        ) / sqrt_one_minus_alpha_t

        prev = (
            alpha_prev.sqrt()
            * predicted_x0
            + (1.0 - alpha_prev).sqrt()
            * epsilon
        )

        return prev
