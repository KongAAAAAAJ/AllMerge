"""K0: unconditional neural Koopman predictor for the closed-loop TruckSim system.

Input: 5D SI state history and 2D SI reference-control history. The history is
encoded once, then a fixed linear latent transition A z + B u is rolled out
under future controls. Normalization is stored in model buffers and gradients
from future states/poses to future controls are preserved.
"""
from __future__ import annotations

import torch
from torch import nn

STATE_NAMES = ("vx_mps", "vy_mps", "body_yaw_rate_radps", "roll_rad", "euler_roll_rate_radps")
CONTROL_NAMES = ("ax_ref_mps2", "front_wheel_delta_ref_rad")


class NeuralKoopmanK0(nn.Module):
    def __init__(self, history_steps=5, latent_dim=32, hidden_dim=128,
                 state_mean=None, state_std=None, control_mean=None, control_std=None):
        super().__init__()
        self.history_steps = int(history_steps)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        if self.history_steps < 1 or self.latent_dim < 5 or self.hidden_dim < 8:
            raise ValueError("history_steps >=1, latent_dim >=5, hidden_dim >=8 required")
        def _buffer(v, dim, default):
            if v is None: v = [default] * dim
            t = torch.as_tensor(v, dtype=torch.float32).reshape(-1)
            if t.numel() != dim or not torch.isfinite(t).all():
                raise ValueError("Invalid normalization buffer")
            return t
        self.register_buffer("state_mean", _buffer(state_mean, 5, 0.0))
        self.register_buffer("state_std", _buffer(state_std, 5, 1.0).clamp_min(1e-5))
        self.register_buffer("control_mean", _buffer(control_mean, 2, 0.0))
        self.register_buffer("control_std", _buffer(control_std, 2, 1.0).clamp_min(1e-5))
        in_dim = self.history_steps * 5 + (self.history_steps - 1) * 2
        self.lift_net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, latent_dim - 5)) if latent_dim > 5 else None
        self.A = nn.Parameter(torch.eye(latent_dim) * 0.995)
        self.B = nn.Parameter(torch.randn(latent_dim, 2) * 0.005)
        self.decode_correction = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
                                               nn.Linear(hidden_dim, 5))
        nn.init.zeros_(self.decode_correction[-1].weight)
        nn.init.zeros_(self.decode_correction[-1].bias)
        # Independently supervised LTR head. No ground-truth LTR is passed at inference.
        self.ltr_head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, 1))

    def normalize_state(self, states):
        return (states - self.state_mean) / self.state_std

    def normalize_control(self, controls):
        return (controls - self.control_mean) / self.control_std

    def lift(self, state_hist, control_hist):
        if state_hist.ndim != 3 or state_hist.shape[1:] != (self.history_steps, 5):
            raise ValueError(f"state_hist must be [B,{self.history_steps},5]")
        if control_hist.shape != (state_hist.shape[0], self.history_steps - 1, 2):
            raise ValueError(f"control_hist must be [B,{self.history_steps-1},2]")
        xs = self.normalize_state(state_hist)
        us = self.normalize_control(control_hist)
        features = torch.cat((xs.flatten(1), us.flatten(1)), dim=-1)
        extra = self.lift_net(features) if self.lift_net is not None else xs.new_zeros((len(xs), 0))
        return torch.cat((xs[:, -1], extra), dim=-1)

    def decode_normalized(self, z):
        return z[..., :5] + self.decode_correction(z)

    def decode(self, z):
        return self.decode_normalized(z) * self.state_std + self.state_mean

    def forward(self, state_hist, control_hist, future_controls):
        """Return (predicted_states [B, H+1, 5], latent [B,H+1,D]).

        prediction[:,0] corresponds to the state at the last history timestamp.
        Each future_controls[:,j] acts on [t_j,t_{j+1}).
        """
        if future_controls.ndim != 3 or future_controls.shape[0] != state_hist.shape[0] or future_controls.shape[-1] != 2:
            raise ValueError("future_controls must be [B,horizon,2]")
        z = self.lift(state_hist, control_hist)
        zs = [z]
        for j in range(future_controls.shape[1]):
            u = self.normalize_control(future_controls[:, j])
            z = z @ self.A.T + u @ self.B.T
            zs.append(z)
        stack = torch.stack(zs, dim=1)
        return self.decode(stack), stack

    def decode_ltr(self, latent):
        """LTR prediction [B,T]. Requires a supervised LTR head."""
        return torch.tanh(self.ltr_head(latent).squeeze(-1))

    def predict_all(self, state_hist, control_hist, future_controls, initial_pose=None, dt=0.1):
        """Unified K0/K1/K2-ready inference contract: [B,H+1,5], [B,H+1], [B,H+1,3].

        `pose` is returned only when an initial world pose is supplied; no
        TruckSim measurements beyond the initial observation are used.
        """
        states, latent = self.forward(state_hist, control_hist, future_controls)
        result = {"states":states, "ltr":self.decode_ltr(latent), "latent":latent}
        if initial_pose is not None:
            result["pose"] = rollout_pose(states, initial_pose, dt)
        return result

    def settings(self):
        return {"history_steps": self.history_steps, "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "output_ltr": True}


def rollout_pose(states, initial_pose, dt):
    """Differentiable planar integration with body-frame velocities and yaw rate.

    Uses body yaw rate r as approximate Euler heading rate, justified for
    small roll/pitch only; metric results must disclose this approximation.
    states [B,T,5], initial_pose [B,3] = world X,Y,yaw (unwrapped).
    """
    if states.ndim != 3 or states.shape[-1] != 5 or initial_pose.shape != (states.shape[0], 3):
        raise ValueError("Incorrect states or initial_pose shape")
    if dt <= 0: raise ValueError("dt must be positive")
    if states.shape[1] == 1:
        return initial_pose[:, None, :]
    yaw_inc = 0.5 * (states[:, :-1, 2] + states[:, 1:, 2]) * dt
    yaw = torch.cat((initial_pose[:, 2:3], initial_pose[:, 2:3] + torch.cumsum(yaw_inc, dim=1)), dim=1)
    # Trapezoidal integration of world-frame velocity at two adjacent endpoints.
    vx = states[..., 0]
    vy = states[..., 1]
    world_x_dot = vx * torch.cos(yaw) - vy * torch.sin(yaw)
    world_y_dot = vx * torch.sin(yaw) + vy * torch.cos(yaw)
    dx = .5 * (world_x_dot[:, :-1] + world_x_dot[:, 1:]) * dt
    dy = .5 * (world_y_dot[:, :-1] + world_y_dot[:, 1:]) * dt
    x = torch.cat((initial_pose[:, 0:1], initial_pose[:, 0:1] + torch.cumsum(dx, dim=1)), dim=1)
    y = torch.cat((initial_pose[:, 1:2], initial_pose[:, 1:2] + torch.cumsum(dy, dim=1)), dim=1)
    return torch.stack((x, y, yaw), dim=-1)
