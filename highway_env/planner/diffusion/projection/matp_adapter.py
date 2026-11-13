"""Native MATP W1 adapter with the GuideDistill call signature."""
from __future__ import annotations

from . import matp_w1


def load_matp(guidance_root=None):
    """Return the in-repository W1 module; no sibling worktree import occurs."""
    return matp_w1


def project_w1(
    xy,
    *,
    module=None,
    reward_config,
    initial_velocity_xy=None,
    projector_config=None,
):
    """Compatibility wrapper for the source branch's project_w1 API."""
    config = {} if projector_config is None else {"projector_config": projector_config}
    return matp_w1.project_w1_local(
        xy,
        reward_config,
        initial_velocity_xy=initial_velocity_xy,
        **config,
    )
