from __future__ import annotations

import numpy as np
import torch

from highway_env.planner.diffusion.projection import JointTrajectoryProjector, RoadGeometry, load_jtp_config
from highway_env.planner.diffusion.projection.constraints import stability_residual
from highway_env.planner.diffusion.projection.geometry import curvature_residual, road_corridor_residual


class StraightLineExecution:
    def __call__(self, controls, context):
        n, h, _ = controls.shape
        dt = 0.1
        vx0 = context["koopman_state_hist_si"][:, -1, 0]
        vy0 = context["koopman_state_hist_si"][:, -1, 1]
        r0 = context["koopman_state_hist_si"][:, -1, 2]
        roll0 = context["koopman_state_hist_si"][:, -1, 3]
        roll_rate0 = context["koopman_state_hist_si"][:, -1, 4]
        ax, delta = controls[..., 0], controls[..., 1]
        vx_future = vx0[:, None] + torch.cumsum(ax * dt, dim=1)
        vx = torch.cat((vx0[:, None], vx_future), dim=1)
        yaw_rate_future = r0[:, None] + delta * 2.0
        yaw_rate = torch.cat((r0[:, None], yaw_rate_future), dim=1)
        yaw = torch.cat((torch.zeros(n, 1, device=controls.device, dtype=controls.dtype),
                         torch.cumsum(yaw_rate_future * dt, dim=1)), dim=1)
        speed = vx[:, 1:]
        x_future = torch.cumsum(speed * torch.cos(yaw[:, 1:]) * dt, dim=1)
        y_future = torch.cumsum(speed * torch.sin(yaw[:, 1:]) * dt, dim=1)
        x = torch.cat((torch.zeros(n, 1, device=controls.device, dtype=controls.dtype), x_future), dim=1)
        y = torch.cat((torch.zeros(n, 1, device=controls.device, dtype=controls.dtype), y_future), dim=1)
        pose0 = context["koopman_pose0_si"]
        c, s = torch.cos(pose0[:, 2]), torch.sin(pose0[:, 2])
        world_x = pose0[:, None, 0] + c[:, None] * x - s[:, None] * y
        world_y = pose0[:, None, 1] + s[:, None] * x + c[:, None] * y
        poses = torch.stack((world_x, world_y, pose0[:, None, 2] + yaw), dim=-1)
        vy = torch.zeros_like(vx) + vy0[:, None]
        roll = torch.zeros_like(vx) + roll0[:, None]
        roll_rate = torch.zeros_like(vx) + roll_rate0[:, None]
        states = torch.stack((vx, vy, yaw_rate, roll, roll_rate), dim=-1)
        return {"trajectory": poses, "states": states, "ltr": torch.zeros_like(vx)}


def _context(n=1, speed=15.0, yaw_rate=0.0):
    state = torch.zeros(n, 5, 5)
    state[:, :, 0] = speed
    state[:, -1, 2] = yaw_rate
    return {
        "koopman_state_hist_si": state,
        "koopman_control_hist_si": torch.zeros(n, 4, 2),
        "koopman_pose0_si": torch.stack((torch.arange(n) * 30.0, torch.zeros(n), torch.zeros(n)), dim=-1),
        "koopman_payload_kg": torch.full((n,), 5000.0),
        "koopman_road_mu": torch.full((n,), 0.8),
    }


def _road(width=30.0):
    x = torch.linspace(-30.0, 150.0, 100)
    center = torch.stack((x, torch.zeros_like(x)), dim=-1)
    return RoadGeometry(center, width, width)


def test_jtp_config_has_grouped_projection_and_reserved_qp_backend():
    cfg = load_jtp_config()
    assert cfg["execution"]["horizon_steps"] == 40
    assert cfg["projection"]["max_active_per_group"]["stability"] > 0
    assert cfg["projection"]["solver_method"] == "damped_least_squares"
    assert cfg["projection"]["group_count"] in (2, 3)
    assert cfg["collision"]["clearance_m"] > 0
    assert cfg["control"]["acceleration_rate_max_mps3"] > 0
    assert cfg["control"]["steering_rate_max_rad_s"] > 0
    assert cfg["projection"]["max_active_per_group"]["external_collision"] > 0
    assert cfg["projection"]["max_active_per_group"]["control_rate"] > 0


def test_road_and_curvature_constraints_are_differentiable():
    road = _road(6.0)
    t = torch.arange(6, dtype=torch.float32) * 0.1
    poses = torch.stack((15.0 * t, torch.zeros_like(t), torch.zeros_like(t)), dim=-1).requires_grad_(True)
    residual = road_corridor_residual(poses, road, circle_offsets_m=[0.0],
                                      circle_radius_m=1.75, clearance_m=0.35)
    grad = torch.autograd.grad(residual.sum(), poses)[0]
    assert torch.isfinite(grad).all()
    assert (residual < 0).all()
    assert (curvature_residual(poses, 0.025) < 0).all()


def test_stability_residual_uses_full_horizon_four_indicators():
    cfg = load_jtp_config()
    states = torch.zeros(5, 5)
    states[:, 0] = 20.0
    states[2, 2] = 0.8
    residual = stability_residual(states, torch.zeros(5), torch.tensor(5000.0), torch.tensor(0.8), cfg)
    assert residual.shape == (2 * 4 * 4,)
    assert (residual > 0).any()


def test_jtp_returns_nominal_when_all_constraints_are_feasible():
    cfg = load_jtp_config()
    cfg["geometry"]["footprint"] = {"circle_offsets_m": [0.0], "circle_radius_m": 0.5}
    projector = JointTrajectoryProjector(StraightLineExecution(), cfg)
    controls = torch.zeros(1, 40, 2)
    result = projector.project(controls, _context(), _road())
    assert result.projector_constraints_satisfied
    assert result.status == "nominal_already_feasible"
    assert result.iterations == 0
    torch.testing.assert_close(result.controls, controls)
    assert result.trajectories.shape == (1, 41, 3)


def test_grouped_projection_corrects_yaw_stability_without_osqp():
    cfg = load_jtp_config()
    cfg["geometry"]["footprint"] = {"circle_offsets_m": [0.0], "circle_radius_m": 0.5}
    cfg["geometry"]["curvature_limit_per_m"] = 0.2
    cfg["projection"]["max_iterations"] = 8
    cfg["projection"]["max_active_per_group"]["stability"] = 40
    cfg["projection"]["max_step_norm"] = 0.5
    projector = JointTrajectoryProjector(StraightLineExecution(), cfg)
    controls = torch.zeros(1, 40, 2)
    result = projector.project(controls, _context(yaw_rate=0.60), _road(100.0))
    assert result.iterations > 0
    assert result.diagnostics["projection"]
    assert result.diagnostics["projection"][0]["jacobian_shape"][0] > 0
    assert result.diagnostics["final_metrics"]["max_violation"] < result.diagnostics["initial_metrics"]["max_violation"]
    assert not result.diagnostics.get("qp")
    assert torch.isfinite(result.controls).all()
    assert (result.controls[..., 1] >= cfg["control"]["steering_min_rad"] - 1e-6).all()
    assert (result.controls[..., 1] <= cfg["control"]["steering_max_rad"] + 1e-6).all()
    assert result.diagnostics["collision_handled_by"] == "joint_projection"
    assert result.diagnostics["control_rate_constraints_enabled"] is True


def test_control_box_is_enforced_as_hard_guard():
    cfg = load_jtp_config()
    cfg["geometry"]["footprint"] = {"circle_offsets_m": [0.0], "circle_radius_m": 0.5}
    projector = JointTrajectoryProjector(StraightLineExecution(), cfg)
    controls = torch.zeros(1, 40, 2)
    controls[..., 0] = 10.0
    result = projector.project(controls, _context(), _road())
    assert (result.controls[..., 0] <= cfg["control"]["acceleration_max_mps2"] + 1e-6).all()
    assert (result.controls[..., 0] >= cfg["control"]["acceleration_min_mps2"] - 1e-6).all()


def test_projector_does_not_import_osqp_or_scipy_sparse():
    import inspect
    import highway_env.planner.diffusion.projection.joint_projector as solver_module
    source = inspect.getsource(solver_module)
    assert "import osqp" not in source
    assert "scipy" not in source.lower()


def test_frozen_k2_execution_preserves_control_autograd():
    from highway_env.planner.diffusion.common.execution import VehicleExecution
    from highway_env.planner.diffusion.projection.execution_adapter import KoopmanExecutionAdapter
    metadata = {
        "model_type": "K2",
        "condition_hidden": 16,
        "model_kwargs": {"history_steps": 5, "latent_dim": 8, "hidden_dim": 16},
        "normalization": {
            "state_mean": [0.0] * 5, "state_std": [1.0] * 5,
            "control_mean": [0.0, 0.0], "control_std": [1.0, 1.0],
        },
        "condition_normalization": {
            "condition_mean": [5000.0, 0.8], "condition_std": [10000.0, 0.1],
        },
    }
    execution = VehicleExecution({"model": "koopman_k2"}, dt=0.1, koopman_metadata=metadata)
    adapter = KoopmanExecutionAdapter(execution)
    context = _context()
    controls = torch.zeros(1, 40, 2, requires_grad=True)
    output = adapter(controls, context)
    loss = output["trajectory"].square().mean() + output["states"].square().mean()
    gradient = torch.autograd.grad(loss, controls)[0]
    assert not any(parameter.requires_grad for parameter in execution.parameters())
    assert gradient.shape == controls.shape
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
