from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TransitionResult:
    prev_sample: torch.Tensor
    log_prob: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


class StochasticDDIMTransition:
    """DDIM transition with replayable Gaussian log-probability.

    This is intentionally a thin wrapper around AllMerge's existing
    ``TruncatedDDIMSchedule.alpha_cumprod``.  It does not duplicate the beta
    schedule or model prediction logic.  The model keeps ``prediction_type``
    semantics of clean-sample/x0 prediction.
    """

    def __init__(self, schedule, *, eta: float = 0.02, min_std: float = 1e-5):
        if eta <= 0.0:
            raise ValueError("GRPO requires eta > 0 so the diffusion policy has non-zero variance")
        if min_std <= 0.0:
            raise ValueError("min_std must be positive")
        self.schedule = schedule
        self.eta = float(eta)
        self.min_std = float(min_std)

    def step(
        self,
        sample: torch.Tensor,
        predicted_x0: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        *,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> TransitionResult:
        t = int(timestep)
        prev_t = int(prev_timestep)
        if prev_t < 0:
            raise ValueError("prev_timestep must be >= 0 for a stochastic GRPO transition")
        if prev_t >= t:
            raise ValueError(f"DDIM reverse step requires prev_timestep < timestep, got {prev_t} >= {t}")

        alpha_t = self.schedule.alpha_cumprod[t].to(sample.device, sample.dtype)
        alpha_prev = self.schedule.alpha_cumprod[prev_t].to(sample.device, sample.dtype)
        one = torch.ones((), device=sample.device, dtype=sample.dtype)
        eps = torch.finfo(sample.dtype).eps

        beta_t = (one - alpha_t).clamp_min(eps)
        pred_epsilon = (sample - alpha_t.sqrt() * predicted_x0) / beta_t.sqrt()

        variance = (
            ((one - alpha_prev) / beta_t)
            * (one - alpha_t / alpha_prev.clamp_min(eps))
        ).clamp_min(0.0)
        std = (self.eta * variance.sqrt()).clamp_min(self.min_std)
        direction_scale = (one - alpha_prev - std.square()).clamp_min(0.0).sqrt()
        mean = alpha_prev.sqrt() * predicted_x0 + direction_scale * pred_epsilon

        if prev_sample is not None and generator is not None:
            raise ValueError("Pass either prev_sample for replay or generator for sampling, not both")
        if prev_sample is None:
            noise = torch.randn(
                sample.shape,
                device=sample.device,
                dtype=sample.dtype,
                generator=generator,
            )
            prev_sample = mean + std * noise

        log_prob = self._log_prob(prev_sample.detach(), mean, std)
        return TransitionResult(prev_sample=prev_sample, log_prob=log_prob, mean=mean, std=std)

    @staticmethod
    def _log_prob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        log_two_pi = sample.new_tensor(1.8378770664093453)  # log(2*pi)
        per_dim = -0.5 * ((sample - mean) / std).square() - torch.log(std) - 0.5 * log_two_pi
        # One trajectory candidate is one diffusion action: sum over [T, xy].
        return per_dim.sum(dim=(-2, -1))
