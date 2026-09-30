from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from highway_env.planner.diffusion.grpo.sampling import _expand_group, _sampling_timesteps


@dataclass(frozen=True)
class MultiplicativeProbeTrace:
    """Probe-only diffusion trace.

    This intentionally stores only what the noise-ablation evaluator needs.
    It is not a GRPO training replay object and must not be used for policy
    ratio / log-prob updates.
    """

    candidates: torch.Tensor  # [B,G,M,T,2] metric-space x0
    logits: torch.Tensor      # [B,G,M]
    timesteps: Tuple[int, ...]


@dataclass(frozen=True)
class MultiplicativeTransitionResult:
    prev_sample: torch.Tensor
    mean: torch.Tensor
    multiplier: torch.Tensor
    ddim_std: torch.Tensor


class MultiplicativeDDIMProbeTransition:
    """DiffusionDriveV2-style multiplicative exploration for probe runs only.

    The deterministic DDIM mean is computed from the existing AllMerge schedule
    exactly as in the current GRPO transition. Exploration then uses

        prev_sample = prev_sample_mean * (1 + eps_xy)

    where eps_xy is sampled once per [batch-group, mode, x/y] and broadcast
    across the whole trajectory horizon. This preserves a coherent trajectory
    shape and expands absolute exploration with trajectory scale.

    Important:
      * the coarse anchor / conditioning input is never modified;
      * the initial forward noising from the anchor is unchanged;
      * this class deliberately does NOT define GRPO log-probability.
    """

    def __init__(
        self,
        schedule,
        *,
        eta: float = 0.02,
        multiplicative_std: float = 0.04,
        min_std: float = 1e-5,
    ):
        if eta <= 0.0:
            raise ValueError("eta must be > 0")
        if multiplicative_std <= 0.0:
            raise ValueError("multiplicative_std must be > 0")
        if min_std <= 0.0:
            raise ValueError("min_std must be > 0")
        self.schedule = schedule
        self.eta = float(eta)
        self.multiplicative_std = float(multiplicative_std)
        self.min_std = float(min_std)

    def step(
        self,
        sample: torch.Tensor,
        predicted_x0: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        *,
        generator: torch.Generator | None = None,
    ) -> MultiplicativeTransitionResult:
        if sample.ndim < 4 or sample.shape[-1] != 2:
            raise ValueError(
                "multiplicative probe expects trajectory tensor [...,mode,horizon,2], "
                f"got {tuple(sample.shape)}"
            )

        t = int(timestep)
        prev_t = int(prev_timestep)
        if prev_t < 0:
            raise ValueError("prev_timestep must be >= 0")
        if prev_t >= t:
            raise ValueError(
                f"DDIM reverse step requires prev_timestep < timestep, got {prev_t} >= {t}"
            )

        alpha_t = self.schedule.alpha_cumprod[t].to(sample.device, sample.dtype)
        alpha_prev = self.schedule.alpha_cumprod[prev_t].to(sample.device, sample.dtype)
        one = torch.ones((), device=sample.device, dtype=sample.dtype)
        eps = torch.finfo(sample.dtype).eps

        beta_t = (one - alpha_t).clamp_min(eps)
        pred_epsilon = (
            sample - alpha_t.sqrt() * predicted_x0
        ) / beta_t.sqrt()

        variance = (
            ((one - alpha_prev) / beta_t)
            * (one - alpha_t / alpha_prev.clamp_min(eps))
        ).clamp_min(0.0)

        # Keep the current AllMerge DDIM mean semantics. eta only controls the
        # direction term here; exploration amplitude is independently swept via
        # multiplicative_std so the ablation has one interpretable variable.
        ddim_std = (self.eta * variance.sqrt()).clamp_min(self.min_std)
        direction_scale = (
            one - alpha_prev - ddim_std.square()
        ).clamp_min(0.0).sqrt()
        mean = alpha_prev.sqrt() * predicted_x0 + direction_scale * pred_epsilon

        # One longitudinal and one lateral multiplier per group/mode, shared
        # across all horizon points, matching the key DiffusionDriveV2 design.
        noise_shape = (*mean.shape[:-2], 1, 2)
        eps_xy = torch.randn(
            noise_shape,
            device=mean.device,
            dtype=mean.dtype,
            generator=generator,
        )
        multiplier = one + self.multiplicative_std * eps_xy
        prev_sample = mean * multiplier

        return MultiplicativeTransitionResult(
            prev_sample=prev_sample,
            mean=mean,
            multiplier=multiplier,
            ddim_std=ddim_std,
        )


class MultiplicativeProbeGroupDiffusionSampler:
    """Probe-only group sampler with multiplicative reverse-transition noise.

    Initial anchor noising stays identical to ``GroupDiffusionSampler`` so runs
    with the same ``noise_seed`` are paired at x_t before the reverse step.
    """

    def __init__(
        self,
        model,
        *,
        group_size: int,
        eta: float = 0.02,
        multiplicative_std: float = 0.04,
        min_std: float = 1e-5,
    ):
        if group_size < 2:
            raise ValueError("group_size must be >= 2")
        self.model = model
        self.group_size = int(group_size)
        self.transition = MultiplicativeDDIMProbeTransition(
            model.schedule,
            eta=eta,
            multiplicative_std=multiplicative_std,
            min_std=min_std,
        )

    def sample(
        self,
        features: Dict[str, torch.Tensor],
        *,
        generator: torch.Generator | None = None,
    ) -> MultiplicativeProbeTrace:
        model = self.model
        group_size = self.group_size
        batch = features["coarse_trajectories"].shape[0]

        expanded = _expand_group(features, group_size)
        scene = model.scene_encoder(expanded)

        # EXACTLY the same anchor and forward-noise construction used by the
        # existing additive GroupDiffusionSampler.
        anchor = model.adapter.normalize_trajectory(
            expanded["coarse_trajectories"]
        )
        timesteps = _sampling_timesteps(model)
        start_t = timesteps[0]
        t_batch = torch.full(
            (anchor.shape[0],),
            start_t,
            dtype=torch.long,
            device=anchor.device,
        )
        initial_noise = torch.randn(
            anchor.shape,
            dtype=anchor.dtype,
            device=anchor.device,
            generator=generator,
        ) * float(model.config.inference_noise_scale)
        sample = model.schedule.add_noise(anchor, initial_noise, t_batch)

        final_x0 = None
        final_logits = None

        for index, timestep in enumerate(timesteps):
            current_t = torch.full(
                (sample.shape[0],),
                int(timestep),
                dtype=torch.long,
                device=sample.device,
            )
            predicted_x0, logits = model.denoiser(sample, current_t, scene)
            final_x0, final_logits = predicted_x0, logits

            if index + 1 >= len(timesteps):
                break

            prev_timestep = timesteps[index + 1]
            result = self.transition.step(
                sample,
                predicted_x0,
                int(timestep),
                int(prev_timestep),
                generator=generator,
            )
            sample = result.prev_sample

        if final_x0 is None or final_logits is None:
            raise RuntimeError("multiplicative probe produced no model output")

        candidates = model.adapter.denormalize_trajectory(final_x0)
        candidates = candidates.reshape(
            batch,
            group_size,
            *candidates.shape[1:],
        )
        logits = final_logits.reshape(batch, group_size, -1)

        return MultiplicativeProbeTrace(
            candidates=candidates.detach(),
            logits=logits.detach(),
            timesteps=timesteps,
        )
