"""Native MATP projection adapter; no sibling worktree imports."""
from __future__ import annotations
import torch
from . import matp_w1


def load_matp(guidance_root=None):
    """Compatibility of the CALL signature, not of legacy W1 numerical outputs."""
    return matp_w1


def project_w1(xy:torch.Tensor, *, module=None, reward_config, initial_velocity_xy=None):
    return matp_w1.project_w1_local(xy,reward_config,
                                    initial_velocity_xy=initial_velocity_xy)
