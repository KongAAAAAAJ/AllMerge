"""Adapters from the existing frozen K2 execution model to the JTP contract."""
from __future__ import annotations

import torch


class KoopmanExecutionAdapter:
    """Expose VehicleExecution as a world-frame differentiable rollout.

    Input controls are [N,H,2]. VehicleExecution returns ego-local poses; this
    adapter converts them to world-frame poses using the synchronized K2 pose0.
    The Koopman parameters remain frozen while gradients w.r.t. controls pass.
    """

    def __init__(self, vehicle_execution):
        self.vehicle_execution = vehicle_execution
        if getattr(vehicle_execution, "kind", None) != "koopman_k2":
            raise ValueError("Joint Trajectory Projection requires execution.model=koopman_k2")
        self.vehicle_execution.requires_grad_(False)
        self.vehicle_execution.eval()

    def __call__(self, controls: torch.Tensor, context: dict) -> dict:
        if controls.ndim != 3 or controls.shape[-1] != 2:
            raise ValueError("JTP controls must be [N,H,2]")
        n = controls.shape[0]
        pose0 = torch.as_tensor(context["koopman_pose0_si"], device=controls.device,
                                dtype=controls.dtype)
        if pose0.shape != (n, 3):
            raise ValueError("koopman_pose0_si must be [N,3]")
        # The existing execution API uses [B,M,H,2]. M=1 denotes one active
        # trajectory per jointly planned truck.
        output = self.vehicle_execution(controls[:, None], features={}, context=context)
        local = output["poses"][:, 0]
        states = output.get("states")
        ltr = output.get("ltr")
        if states is None or ltr is None:
            raise ValueError("K2 execution must return chassis states and LTR")
        states = states[:, 0]
        ltr = ltr[:, 0]
        psi0 = pose0[:, 2:3]
        c, s = torch.cos(psi0), torch.sin(psi0)
        xw = pose0[:, None, 0] + c * local[..., 0] - s * local[..., 1]
        yw = pose0[:, None, 1] + s * local[..., 0] + c * local[..., 1]
        yaw = psi0 + local[..., 2]
        poses_world = torch.stack((xw, yw, yaw), dim=-1)
        return {"trajectory": poses_world, "states": states, "ltr": ltr}
