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


@dataclass(frozen=True)
class HybridTransitionResult:
    prev_sample: torch.Tensor
    mean: torch.Tensor
    x_multiplier: torch.Tensor
    y_offset_normalized: torch.Tensor
    y_offset_meters: torch.Tensor
    ddim_std: torch.Tensor


def _ddim_mean(
    schedule,
    sample: torch.Tensor,
    predicted_x0: torch.Tensor,
    timestep: int,
    prev_timestep: int,
    *,
    eta: float,
    min_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the same deterministic DDIM mean used by the probe transitions."""
    t = int(timestep)
    prev_t = int(prev_timestep)
    if prev_t < 0:
        raise ValueError("prev_timestep must be >= 0")
    if prev_t >= t:
        raise ValueError(
            f"DDIM reverse step requires prev_timestep < timestep, got {prev_t} >= {t}"
        )

    alpha_t = schedule.alpha_cumprod[t].to(sample.device, sample.dtype)
    alpha_prev = schedule.alpha_cumprod[prev_t].to(sample.device, sample.dtype)
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

    ddim_std = (float(eta) * variance.sqrt()).clamp_min(float(min_std))
    direction_scale = (
        one - alpha_prev - ddim_std.square()
    ).clamp_min(0.0).sqrt()
    mean = alpha_prev.sqrt() * predicted_x0 + direction_scale * pred_epsilon
    return mean, ddim_std


class MultiplicativeDDIMProbeTransition:
    """DiffusionDriveV2-style multiplicative exploration for probe runs only.

    Exploration uses

        prev_sample = prev_sample_mean * (1 + eps_xy)

    where eps_xy is sampled once per [batch-group, mode, x/y] and broadcast
    across the whole trajectory horizon.
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

        mean, ddim_std = _ddim_mean(
            self.schedule,
            sample,
            predicted_x0,
            timestep,
            prev_timestep,
            eta=self.eta,
            min_std=self.min_std,
        )

        one = torch.ones((), device=mean.device, dtype=mean.dtype)
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


class HybridDDIMProbeTransition:
    """Hybrid x-multiplicative / y-additive reverse exploration.

    Longitudinal exploration:
        x'_i = x_i * (1 + sigma_x * eps_x)

    Lateral exploration:
        y'_i = y_i + ramp_i * sigma_y[m] * eps_y

    where ``ramp_i = (i+1)/H``.  One eps_x and one eps_y are sampled per
    [batch-group, mode] and shared coherently along the horizon.

    ``sigma_y`` is specified in physical meters.  The implementation converts
    it back to normalized trajectory space using the planner trajectory y scale,
    so ``--hybrid-additive-y-std-m 0.50`` means a 0.50 m standard deviation at
    the final (4 s) waypoint, not 0.50 normalized units.

    Anchor trajectory and initial forward noising are unchanged.
    """

    def __init__(
        self,
        schedule,
        *,
        trajectory_scale_y: float,
        eta: float = 0.02,
        multiplicative_x_std: float = 0.12,
        additive_y_std_m: float = 0.50,
        min_std: float = 1e-5,
    ):
        if eta <= 0.0:
            raise ValueError("eta must be > 0")
        if multiplicative_x_std <= 0.0:
            raise ValueError("multiplicative_x_std must be > 0")
        if additive_y_std_m < 0.0:
            raise ValueError("additive_y_std_m must be >= 0")
        if trajectory_scale_y <= 0.0:
            raise ValueError("trajectory_scale_y must be > 0")
        if min_std <= 0.0:
            raise ValueError("min_std must be > 0")

        self.schedule = schedule
        self.trajectory_scale_y = float(trajectory_scale_y)
        self.eta = float(eta)
        self.multiplicative_x_std = float(multiplicative_x_std)
        self.additive_y_std_m = float(additive_y_std_m)
        self.min_std = float(min_std)

    def step(
        self,
        sample: torch.Tensor,
        predicted_x0: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        *,
        generator: torch.Generator | None = None,
    ) -> HybridTransitionResult:
        if sample.ndim < 4 or sample.shape[-1] != 2:
            raise ValueError(
                "hybrid probe expects trajectory tensor [...,mode,horizon,2], "
                f"got {tuple(sample.shape)}"
            )

        mean, ddim_std = _ddim_mean(
            self.schedule,
            sample,
            predicted_x0,
            timestep,
            prev_timestep,
            eta=self.eta,
            min_std=self.min_std,
        )

        # One scalar longitudinal and one scalar lateral perturbation for each
        # batch-group-mode.  Sampling both even when sigma_y=0 keeps generator
        # consumption paired across the lateral-noise sweep.
        scalar_shape = (*mean.shape[:-2], 1)
        eps_x = torch.randn(
            scalar_shape,
            device=mean.device,
            dtype=mean.dtype,
            generator=generator,
        )
        eps_y = torch.randn(
            scalar_shape,
            device=mean.device,
            dtype=mean.dtype,
            generator=generator,
        )

        one = torch.ones((), device=mean.device, dtype=mean.dtype)
        x_multiplier = one + self.multiplicative_x_std * eps_x

        horizon = int(mean.shape[-2])
        ramp = torch.arange(
            1,
            horizon + 1,
            device=mean.device,
            dtype=mean.dtype,
        ) / float(horizon)
        # Broadcast [H] to [..., H].
        y_offset_meters = (
            self.additive_y_std_m
            * eps_y
            * ramp
        )
        y_offset_normalized = y_offset_meters / self.trajectory_scale_y

        prev_sample = mean.clone()
        prev_sample[..., 0] = mean[..., 0] * x_multiplier
        prev_sample[..., 1] = mean[..., 1] + y_offset_normalized

        return HybridTransitionResult(
            prev_sample=prev_sample,
            mean=mean,
            x_multiplier=x_multiplier,
            y_offset_normalized=y_offset_normalized,
            y_offset_meters=y_offset_meters,
            ddim_std=ddim_std,
        )


class MultiplicativeProbeGroupDiffusionSampler:
    """Probe-only group sampler with multiplicative reverse-transition noise."""

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
        return _sample_with_probe_transition(
            self.model,
            self.group_size,
            features,
            self.transition,
            generator=generator,
        )


class HybridProbeGroupDiffusionSampler:
    """Probe-only sampler for x-multiplicative / y-additive exploration."""

    def __init__(
        self,
        model,
        *,
        group_size: int,
        eta: float = 0.02,
        multiplicative_x_std: float = 0.12,
        additive_y_std_m: float = 0.50,
        min_std: float = 1e-5,
    ):
        if group_size < 2:
            raise ValueError("group_size must be >= 2")
        self.model = model
        self.group_size = int(group_size)

        trajectory_scale_y = float(
            model.adapter.trajectory_scale[1].detach().cpu().item()
        )
        self.transition = HybridDDIMProbeTransition(
            model.schedule,
            trajectory_scale_y=trajectory_scale_y,
            eta=eta,
            multiplicative_x_std=multiplicative_x_std,
            additive_y_std_m=additive_y_std_m,
            min_std=min_std,
        )

    def sample(
        self,
        features: Dict[str, torch.Tensor],
        *,
        generator: torch.Generator | None = None,
    ) -> MultiplicativeProbeTrace:
        return _sample_with_probe_transition(
            self.model,
            self.group_size,
            features,
            self.transition,
            generator=generator,
        )


def _sample_with_probe_transition(
    model,
    group_size: int,
    features: Dict[str, torch.Tensor],
    transition,
    *,
    generator: torch.Generator | None = None,
) -> MultiplicativeProbeTrace:
    """Shared probe sampler; anchor and forward noising match core GRPO."""
    batch = features["coarse_trajectories"].shape[0]

    expanded = _expand_group(features, group_size)
    scene = model.scene_encoder(expanded)

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
        result = transition.step(
            sample,
            predicted_x0,
            int(timestep),
            int(prev_timestep),
            generator=generator,
        )
        sample = result.prev_sample

    if final_x0 is None or final_logits is None:
        raise RuntimeError("probe sampler produced no model output")

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
