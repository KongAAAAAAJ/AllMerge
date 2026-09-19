from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch

from .scheduler import StochasticDDIMTransition


@dataclass
class DiffusionTrace:
    """Replay buffer for one GRPO group sample."""

    features: Dict[str, torch.Tensor]
    timesteps: Tuple[int, ...]
    states: Tuple[torch.Tensor, ...]      # each [B,G,M,T,2], x_t
    next_states: Tuple[torch.Tensor, ...] # each [B,G,M,T,2], x_prev
    old_log_prob: torch.Tensor            # [B,G,S,M]
    candidates: torch.Tensor              # [B,G,M,T,2], metric-space x0
    logits: torch.Tensor                  # [B,G,M]


def _expand_group(features: Dict[str, torch.Tensor], group_size: int) -> Dict[str, torch.Tensor]:
    expanded: Dict[str, torch.Tensor] = {}
    for key, value in features.items():
        if not torch.is_tensor(value):
            continue
        shape = value.shape
        expanded[key] = (
            value[:, None]
            .expand(shape[0], group_size, *shape[1:])
            .reshape(shape[0] * group_size, *shape[1:])
        )
    return expanded


def _sampling_timesteps(model) -> Tuple[int, ...]:
    start_t = int(model.config.inference_start_timestep)
    timesteps = tuple(int(x) for x in model.config.inference_timesteps)
    if not timesteps:
        timesteps = (start_t, 0)
    elif timesteps[0] != start_t:
        timesteps = (start_t, *timesteps)
    if len(timesteps) < 2:
        raise ValueError("GRPO sampling needs at least two inference timesteps")
    if any(a <= b for a, b in zip(timesteps, timesteps[1:])):
        raise ValueError(f"inference_timesteps must be strictly decreasing, got {timesteps}")
    return timesteps


class GroupDiffusionSampler:
    """Group sampler that reuses AllMerge scene encoder, denoiser and schedule.

    Only the reverse transition is stochastic (eta > 0), because GRPO needs a
    replayable policy density.  The planner's clean-x0 prediction semantics are
    preserved; no second diffusion implementation is introduced.
    """

    def __init__(self, model, *, group_size: int, eta: float = 0.02, min_std: float = 1e-5):
        if group_size < 2:
            raise ValueError("group_size must be >= 2 for relative advantages")
        self.model = model
        self.group_size = int(group_size)
        self.transition = StochasticDDIMTransition(model.schedule, eta=eta, min_std=min_std)

    def sample(
        self,
        features: Dict[str, torch.Tensor],
        *,
        generator: torch.Generator | None = None,
    ) -> DiffusionTrace:
        model = self.model
        group_size = self.group_size
        batch = features["coarse_trajectories"].shape[0]
        expanded = _expand_group(features, group_size)
        scene = model.scene_encoder(expanded)

        anchor = model.adapter.normalize_trajectory(expanded["coarse_trajectories"])
        timesteps = _sampling_timesteps(model)
        start_t = timesteps[0]
        t_batch = torch.full(
            (anchor.shape[0],), start_t, dtype=torch.long, device=anchor.device
        )
        noise = torch.randn(
            anchor.shape, dtype=anchor.dtype, device=anchor.device, generator=generator
        ) * float(model.config.inference_noise_scale)
        sample = model.schedule.add_noise(anchor, noise, t_batch)

        states = []
        next_states = []
        log_probs = []
        final_x0 = None
        final_logits = None

        for index, timestep in enumerate(timesteps):
            current_t = torch.full(
                (sample.shape[0],), int(timestep), dtype=torch.long, device=sample.device
            )
            predicted_x0, logits = model.denoiser(sample, current_t, scene)
            final_x0, final_logits = predicted_x0, logits
            if index + 1 >= len(timesteps):
                break

            prev_timestep = timesteps[index + 1]
            transition = self.transition.step(
                sample,
                predicted_x0,
                int(timestep),
                int(prev_timestep),
                generator=generator,
            )
            states.append(sample.reshape(batch, group_size, *sample.shape[1:]).detach())
            next_states.append(
                transition.prev_sample.reshape(batch, group_size, *sample.shape[1:]).detach()
            )
            log_probs.append(
                transition.log_prob.reshape(batch, group_size, -1).detach()
            )
            sample = transition.prev_sample

        if final_x0 is None or final_logits is None:
            raise RuntimeError("Diffusion sampling produced no model output")

        candidates = model.adapter.denormalize_trajectory(final_x0)
        candidates = candidates.reshape(batch, group_size, *candidates.shape[1:])
        logits = final_logits.reshape(batch, group_size, -1)
        old_log_prob = torch.stack(log_probs, dim=2)
        detached_features = {k: v.detach() for k, v in features.items() if torch.is_tensor(v)}
        return DiffusionTrace(
            features=detached_features,
            timesteps=timesteps,
            states=tuple(states),
            next_states=tuple(next_states),
            old_log_prob=old_log_prob,
            candidates=candidates.detach(),
            logits=logits.detach(),
        )

    def replay(self, trace: DiffusionTrace, *, model=None) -> torch.Tensor:
        replay_model = self.model if model is None else model
        if replay_model.schedule is not self.model.schedule:
            transition = StochasticDDIMTransition(
                replay_model.schedule,
                eta=self.transition.eta,
                min_std=self.transition.min_std,
            )
        else:
            transition = self.transition

        batch, group_size = trace.old_log_prob.shape[:2]
        expanded = _expand_group(trace.features, group_size)
        scene = replay_model.scene_encoder(expanded)
        log_probs = []
        for index, (state, next_state) in enumerate(zip(trace.states, trace.next_states)):
            timestep = trace.timesteps[index]
            prev_timestep = trace.timesteps[index + 1]
            flat_state = state.reshape(batch * group_size, *state.shape[2:])
            flat_next = next_state.reshape(batch * group_size, *next_state.shape[2:])
            t_batch = torch.full(
                (flat_state.shape[0],), int(timestep), dtype=torch.long, device=flat_state.device
            )
            predicted_x0, _ = replay_model.denoiser(flat_state, t_batch, scene)
            replayed = transition.step(
                flat_state,
                predicted_x0,
                int(timestep),
                int(prev_timestep),
                prev_sample=flat_next,
            )
            log_probs.append(replayed.log_prob.reshape(batch, group_size, -1))
        return torch.stack(log_probs, dim=2)
