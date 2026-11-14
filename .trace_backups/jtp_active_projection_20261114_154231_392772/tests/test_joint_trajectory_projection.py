from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from highway_env.planner.diffusion.projection import (
    ExternalTraffic,
    JointTrajectoryProjector,
    RoadGeometry,
    constant_velocity_poses,
    load_jtp_config,
)
from highway_env.planner.diffusion.projection.constraints import stability_residual
from highway_env.planner.diffusion.projection.geometry import (
    collision_residual,
    curvature_residual,
    geometry_residual,
    road_corridor_residual,
)
from highway_env.planner.diffusion.projection.qp_builder import build_joint_qp
from highway_env.planner.diffusion.common.execution import VehicleExecution
from highway_env.planner.diffusion.projection.execution_adapter import KoopmanExecutionAdapter


class StraightLineExecution:
    """Small differentiable test double implementing the JTP rollout contract."""

    def __call__(self, controls, context):
        n, h, _ = controls.shape
        dt = 0.1
        vx0 = context["koopman_state_hist_si"][:, -1, 0]
        vy0 = context["koopman_state_hist_si"][:, -1, 1]
        r0 = context["koopman_state_hist_si"][:, -1, 2]
        roll0 = context["koopman_state_hist_si"][:, -1, 3]
        roll_rate0 = context["koopman_state_hist_si"][:, -1, 4]
        ax = controls[..., 0]
        delta = controls[..., 1]
        vx_future = vx0[:, None] + torch.cumsum(ax * dt, dim=1)
        vx = torch.cat((vx0[:, None], vx_future), dim=1)
        # A deliberately simple differentiable lateral response for Jacobian tests.
        yaw_rate_future = r0[:, None] + torch.cumsum(vx_future * delta * 0.2, dim=1)
        yaw_rate = torch.cat((r0[:, None], yaw_rate_future), dim=1)
        yaw = torch.cat((torch.zeros(n, 1, device=controls.device, dtype=controls.dtype),
                         torch.cumsum(0.5 * (yaw_rate[:, :-1] + yaw_rate[:, 1:]) * dt, dim=1)), dim=1)
        speed = vx[:, 1:]
        x_future = torch.cumsum(speed * torch.cos(yaw[:, 1:]) * dt, dim=1)
        y_future = torch.cumsum(speed * torch.sin(yaw[:, 1:]) * dt, dim=1)
        x = torch.cat((torch.zeros(n, 1, device=controls.device, dtype=controls.dtype), x_future), dim=1)
        y = torch.cat((torch.zeros(n, 1, device=controls.device, dtype=controls.dtype), y_future), dim=1)
        pose0 = context["koopman_pose0_si"]
        c, s = torch.cos(pose0[:, 2]), torch.sin(pose0[:, 2])
        world_x = pose0[:, None, 0] + c[:, None] * x - s[:, None] * y
        world_y = pose0[:, None, 1] + s[:, None] * x + c[:, None] * y
        world_yaw = pose0[:, None, 2] + yaw
        poses = torch.stack((world_x, world_y, world_yaw), dim=-1)
        vy = torch.zeros_like(vx) + vy0[:, None]
        roll = torch.zeros_like(vx) + roll0[:, None]
        roll_rate = torch.zeros_like(vx) + roll_rate0[:, None]
        states = torch.stack((vx, vy, yaw_rate, roll, roll_rate), dim=-1)
        ltr = torch.zeros_like(vx)
        return {"trajectory": poses, "states": states, "ltr": ltr}


def _context(n=1, *, speed=15.0, device="cpu"):
    state_hist = torch.zeros(n, 5, 5, device=device)
    state_hist[:, :, 0] = speed
    return {
        "koopman_state_hist_si": state_hist,
        "koopman_control_hist_si": torch.zeros(n, 4, 2, device=device),
        "koopman_pose0_si": torch.stack((torch.arange(n, device=device, dtype=torch.float32) * 30,
                                          torch.zeros(n, device=device),
                                          torch.zeros(n, device=device)), dim=-1),
        "koopman_payload_kg": torch.full((n,), 5000.0, device=device),
        "koopman_road_mu": torch.full((n,), 0.8, device=device),
    }


def _straight_road():
    x = torch.linspace(-30.0, 100.0, 80)
    center = torch.stack((x, torch.zeros_like(x)), dim=-1)
    return RoadGeometry(center, 6.0, 6.0)


def test_jtp_config_loads_and_contains_provisional_values():
    cfg = load_jtp_config()
    assert cfg["execution"]["horizon_steps"] == 40
    assert cfg["geometry"]["curvature_limit_per_m"] > 0
    assert cfg["stability"]["ltr_limit"] < 1.0
    assert cfg["control"]["acceleration_min_mps2"] < cfg["control"]["acceleration_max_mps2"]


def test_constant_velocity_rollout_shape_and_motion():
    initial = torch.tensor([[1.0, 2.0, 10.0, -2.0]])
    poses = constant_velocity_poses(initial, horizon_steps=4, dt_s=0.5)
    assert poses.shape == (1, 5, 3)
    assert torch.allclose(poses[0, -1, :2], torch.tensor([21.0, -2.0]))
    assert torch.allclose(poses[0, 0, :2], initial[0, :2])


def test_road_and_curvature_constraints_are_differentiable():
    road = _straight_road()
    t = torch.arange(6, dtype=torch.float32) * 0.1
    poses = torch.stack((15.0 * t, torch.zeros_like(t), torch.zeros_like(t)), dim=-1)
    poses.requires_grad_(True)
    g = road_corridor_residual(poses, road, circle_offsets_m=[0.0],
                               circle_radius_m=1.75, clearance_m=0.35)
    assert (g < 0).all()
    grad = torch.autograd.grad(g.sum(), poses)[0]
    assert torch.isfinite(grad).all()
    assert (curvature_residual(poses, 0.025) < 0).all()


def test_curvature_constraint_detects_excessive_turning():
    x = torch.tensor([0.0, 1.0, 1.8, 2.2, 2.3])
    y = torch.tensor([0.0, 0.0, 0.5, 1.2, 2.0])
    yaw = torch.atan2(torch.gradient(y)[0], torch.gradient(x)[0])
    poses = torch.stack((x, y, yaw), dim=-1)
    residual = curvature_residual(poses, 0.01)
    assert (residual > 0).any()


def test_collision_residual_detects_inter_vehicle_overlap():
    trajectories = torch.zeros(2, 3, 3)
    trajectories[0, :, 0] = torch.tensor([0.0, 0.0, 0.0])
    trajectories[1, :, 0] = torch.tensor([5.0, 2.0, 5.0])
    cfg = load_jtp_config()
    cfg["geometry"]["footprint"] = {"circle_offsets_m": [0.0], "circle_radius_m": 1.0}
    cfg["collision"]["clearance_m"] = 0.5
    cfg["collision"]["qp_compact_footprints"] = False
    residual = collision_residual(trajectories, cfg)
    assert residual.numel() == 2
    assert residual[0] > 0
    assert residual[1] < 0




def test_collision_residual_supports_external_traffic_and_actor_specific_footprints():
    trajectories = torch.zeros(1, 3, 3)
    external_poses = torch.zeros(2, 3, 3)
    external_poses[0, :, 0] = 1.0
    external_poses[1, :, 0] = 10.0
    external = ExternalTraffic(
        external_poses,
        circle_offsets_m=torch.tensor([[0.0], [0.0]]),
        circle_radius_m=torch.tensor([[1.0], [0.5]]),
    )
    cfg = load_jtp_config()
    cfg["geometry"]["footprint"] = {"circle_offsets_m": [0.0], "circle_radius_m": 1.0}
    cfg["collision"]["clearance_m"] = 0.5
    cfg["collision"]["qp_compact_footprints"] = False
    residual = collision_residual(trajectories, cfg, external)
    assert residual.shape == (4,)
    assert (residual[:2] > 0).all()
    assert (residual[2:] < 0).all()


def test_stability_residual_uses_all_four_indicators_and_full_horizon():
    cfg = load_jtp_config()
    states = torch.zeros(5, 5)
    states[:, 0] = 20.0
    states[2, 2] = 0.8  # excessive yaw rate at one future step
    ltr = torch.zeros(5)
    residual = stability_residual(states, ltr, torch.tensor(5000.0), torch.tensor(0.8), cfg)
    assert residual.shape == (2 * 4 * 4,)
    assert (residual > 0).any()


def test_jtp_projector_returns_nominal_when_all_constraints_are_feasible():
    cfg = load_jtp_config()
    projector = JointTrajectoryProjector(StraightLineExecution(), cfg)
    controls = torch.zeros(1, 40, 2)
    result = projector.project(controls, _context(), _straight_road())
    assert result.success
    assert result.status == "nominal_already_feasible"
    assert result.iterations == 0
    assert torch.allclose(result.controls, controls)
    assert result.trajectories.shape == (1, 41, 3)




def test_frozen_k2_execution_preserves_control_autograd():
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
    vehicle_execution = VehicleExecution(
        {"model": "koopman_k2"}, dt=0.1, koopman_metadata=metadata
    )
    adapter = KoopmanExecutionAdapter(vehicle_execution)
    context = _context()
    controls = torch.zeros(1, 40, 2, requires_grad=True)
    output = adapter(controls, context)
    loss = output["trajectory"].square().mean() + output["states"].square().mean()
    gradient = torch.autograd.grad(loss, controls)[0]
    assert not any(parameter.requires_grad for parameter in vehicle_execution.parameters())
    assert gradient.shape == controls.shape
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0




def test_k2_full_horizon_linearization_shapes():
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
    vehicle_execution = VehicleExecution(
        {"model": "koopman_k2"}, dt=0.1, koopman_metadata=metadata
    )
    projector = JointTrajectoryProjector(vehicle_execution, load_jtp_config())
    controls = torch.zeros(1, 40, 2)
    context = _context()
    roads = [ _straight_road() ]
    evaluation = projector._evaluate_constraints(controls, context, roads, None)
    jt, jstab, jgeo, jcol = projector._linearize(controls, context, roads, None, evaluation)
    assert jt[0].shape == (41 * 3, 40 * 2)
    assert jstab[0].shape == (0, 40 * 2)  # disabled in the lightweight QP profile
    assert jgeo[0].shape[1] == 41 * 3
    assert jcol.shape == (0, 41 * 3)
    assert all(torch.isfinite(value).all() for value in (jt[0], jstab[0], jgeo[0]))


def test_joint_qp_builder_uses_delta_tau_and_delta_u_variables():
    cfg = load_jtp_config()
    n, k, t = 1, 2, 3
    controls = np.zeros((n, k, 2), dtype=np.float64)
    controls_nominal = controls.copy()
    tau = np.zeros((n, t, 3), dtype=np.float64)
    jt = [np.zeros((t * 3, k * 2), dtype=np.float64)]
    gs = [np.array([0.1, -0.1], dtype=np.float64)]
    jgs = [np.zeros((2, k * 2), dtype=np.float64)]
    gg = [np.array([0.2, -0.2], dtype=np.float64)]
    jgg = [np.zeros((2, t * 3), dtype=np.float64)]
    qp = build_joint_qp(
        controls=controls,
        controls_nominal=controls_nominal,
        tau_current=tau,
        tau_nominal=tau,
        jac_tau_u=jt,
        g_stability=gs,
        jac_stability_u=jgs,
        g_geometry=gg,
        jac_geometry_tau=jgg,
        g_collision=np.empty(0),
        jac_collision_tau=np.empty((0, t * 3)),
        previous_controls=np.zeros((n, 2)),
        trust_radius=0.2,
        cfg=cfg,
    )
    assert qp.n_tau == n * t * 3
    assert qp.n_control == n * k * 2
    assert qp.P.shape == (qp.n_tau + qp.n_control, qp.n_tau + qp.n_control)
    assert qp.A.shape[1] == qp.n_tau + qp.n_control
    assert qp.diagnostics["equality_rows"] == qp.n_tau
    assert qp.diagnostics["stability_rows"] == 0
    assert qp.diagnostics["geometry_rows"] == 2
    assert qp.diagnostics["control_rate_rows"] == 0
    assert qp.diagnostics["control_box_rows"] == n * k * 2


def test_jtp_autograd_jacobian_preserves_input_gradient():
    cfg = load_jtp_config()
    projector = JointTrajectoryProjector(StraightLineExecution(), cfg)
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    jac = projector._jacobian(lambda z: torch.stack((z[0] ** 2, z[0] * z[1])), x)
    assert torch.allclose(jac, torch.tensor([[2.0, 0.0], [2.0, 1.0]]))


@pytest.mark.skipif(__import__("importlib").util.find_spec("osqp") is None,
                    reason="OSQP is an optional manually installed JTP dependency")
def test_osqp_solves_a_small_feasible_qp():
    from scipy import sparse
    import osqp

    solver = osqp.OSQP()
    solver.setup(P=sparse.eye(2, format="csc"), q=np.array([-1.0, -2.0]),
                 A=sparse.eye(2, format="csc"), l=np.zeros(2), u=np.ones(2),
                 verbose=False)
    result = solver.solve()
    assert str(result.info.status).lower().startswith("solved")
    assert np.allclose(result.x, [1.0, 1.0], atol=1e-3)


def test_compact_external_collision_footprint_reduces_constraint_count():
    cfg = load_jtp_config()
    trajectories = torch.zeros(1, 41, 3)
    external_poses = torch.zeros(8, 41, 3)
    for actor in range(8):
        external_poses[actor, :, 0] = 20.0 + actor * 10.0
    external = ExternalTraffic(external_poses)
    residual = collision_residual(trajectories, cfg, external)
    assert residual.numel() == 8 * 40 * 3
    assert cfg["constraint_profile"]["collision"] is True

def test_lightweight_profile_keeps_only_core_physical_constraint_groups():
    cfg = load_jtp_config()
    profile = cfg["constraint_profile"]
    assert profile["stability"] is False
    assert profile["road_geometry"] is True
    assert profile["collision"] is True
    assert profile["control_box"] is True
    assert profile["control_rate"] is False
    assert profile["trust_region"] is True
    assert cfg["geometry"]["enforce_curvature"] is False


def test_compact_collision_footprint_reduces_controlled_vehicle_pair_rows():
    cfg = load_jtp_config()
    trajectories = torch.zeros(2, 41, 3)
    trajectories[1, :, 0] = 30.0
    residual = collision_residual(trajectories, cfg)
    assert residual.numel() == 40 * 3 * 3


def test_lightweight_qp_row_count_is_reduced_to_core_constraints():
    cfg = load_jtp_config()
    n, k, t, actors = 1, 40, 41, 8
    controls = np.zeros((n, k, 2), dtype=np.float64)
    tau = np.zeros((n, t, 3), dtype=np.float64)
    qp = build_joint_qp(
        controls=controls,
        controls_nominal=controls,
        tau_current=tau,
        tau_nominal=tau,
        jac_tau_u=[np.zeros((t * 3, k * 2))],
        g_stability=[np.zeros(2 * 40 * 4)],
        jac_stability_u=[np.zeros((2 * 40 * 4, k * 2))],
        g_geometry=[np.zeros(5 * 2 * 40)],
        jac_geometry_tau=[np.zeros((5 * 2 * 40, t * 3))],
        g_collision=np.zeros(actors * 40 * 3),
        jac_collision_tau=np.zeros((actors * 40 * 3, t * 3)),
        previous_controls=np.zeros((n, 2)),
        trust_radius=0.25,
        cfg=cfg,
    )
    assert qp.diagnostics["stability_rows"] == 0
    assert qp.diagnostics["geometry_rows"] == 400
    assert qp.diagnostics["collision_rows"] == 960
    assert qp.diagnostics["control_box_rows"] == 80
    assert qp.diagnostics["control_rate_rows"] == 0
    assert qp.diagnostics["trust_region_rows"] == 80
    assert qp.diagnostics["total_constraint_rows"] == 1643
