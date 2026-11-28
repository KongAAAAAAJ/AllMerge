from __future__ import annotations

import numpy as np
import torch

from highway_env.planner.diffusion.projection import (
    JointTrajectoryProjector, RoadGeometry, load_jtp_config,
)
from highway_env.planner.diffusion.projection.constraints import stability_residual
from highway_env.planner.diffusion.projection.geometry import (
    curvature_residual, road_corridor_residual,
)


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
    if isinstance(yaw_rate, (list, tuple, np.ndarray, torch.Tensor)):
        state[:, -1, 2] = torch.as_tensor(yaw_rate)
    else:
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



def test_feasible_nominal_skips_qp(monkeypatch):
    from highway_env.planner.diffusion.projection import joint_projector as module
    monkeypatch.setattr(module, 'solve_qp', lambda *args: (_ for _ in ()).throw(AssertionError('QP not needed')))
    projector = JointTrajectoryProjector(StraightLineExecution())
    u = torch.zeros(3, 40, 2)
    result = projector.project(u, _context(3))
    assert result.projector_constraints_satisfied
    assert result.iterations == 0
    torch.testing.assert_close(result.controls, u)


def test_each_outer_iteration_calls_one_qp_and_bounds_are_included(monkeypatch):
    from highway_env.planner.diffusion.projection import joint_projector as module
    from highway_env.planner.diffusion.projection.qp_solver import solve_qp
    calls = []
    def solve(*args):
        calls.append(args)
        return solve_qp(*args)
    monkeypatch.setattr(module, 'solve_qp', solve)
    cfg = load_jtp_config()
    projector = JointTrajectoryProjector(StraightLineExecution(), cfg)
    u = torch.zeros(1, 40, 2)
    u[...,0] = 3.
    result = projector.project(u, _context(yaw_rate=.6))
    steps = result.diagnostics['per_vehicle'][0]['projection']
    assert len(calls) == len(steps) <= cfg['projection']['max_iterations']
    assert all('solver_status' in step for step in steps)
    assert set(result.diagnostics['per_vehicle'][0]['final_metrics']) == {
        'max_violation','stability_violation_count','control_box_feasible','projector_constraints_satisfied'}


def test_control_restoration_is_the_same_qp(monkeypatch):
    from highway_env.planner.diffusion.projection import qp_solver
    calls = []
    original = qp_solver._solve_osqp
    def solve(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(qp_solver, '_solve_osqp', solve)
    projector = JointTrajectoryProjector(StraightLineExecution())
    u = torch.zeros(1,40,2); u[...,0] = 3.
    result = projector.project(u, _context())
    assert len(calls) == 1
    assert result.projector_constraints_satisfied
    assert float(result.controls[...,0].max()) <= projector.cfg['control']['acceleration_max_mps2']


def test_timeout_retains_last_accepted_iterate(monkeypatch):
    from highway_env.planner.diffusion.projection import joint_projector as module
    def timeout(*args):
        return None, dict(solver_status='run time limit reached', solver_solve_time_s=.05)
    monkeypatch.setattr(module, 'solve_qp', timeout)
    u = torch.zeros(1,40,2)
    result = JointTrajectoryProjector(StraightLineExecution()).project(u, _context(yaw_rate=.6))
    assert result.status == 'solver_failed:run time limit reached'
    assert result.diagnostics['optimization_time_ms_by_vehicle'] == [50.]
    torch.testing.assert_close(result.controls,u)


def test_line_search_rejects_stability_regression(monkeypatch):
    from highway_env.planner.diffusion.projection import joint_projector as module
    def worsening(u, *args):
        step = torch.zeros_like(u); step[...,1] = .1
        return step, dict(solver_status='solved', solver_solve_time_s=.001)
    monkeypatch.setattr(module,'solve_qp',worsening)
    u = torch.zeros(1,40,2)
    result = JointTrajectoryProjector(StraightLineExecution()).project(u,_context(yaw_rate=.6))
    assert not any(s['accepted'] for s in result.diagnostics['per_vehicle'][0]['projection'])
    torch.testing.assert_close(result.controls,u)


def test_stability_linearization_matches_direct_autograd():
    from highway_env.planner.diffusion.projection.linearization import shared_linearization, jacobian
    projector = JointTrajectoryProjector(StraightLineExecution())
    controls = torch.zeros(1,40,2)
    context = _context(yaw_rate=.6)
    current = projector._evaluate(controls,context)
    selected = torch.arange(320)
    actual = shared_linearization(projector,controls,context,current,selected,None)
    flat = controls.flatten().requires_grad_(True)
    expected = jacobian(lambda v: projector._stability(projector._rollout(v.reshape_as(controls),context),context),flat)
    torch.testing.assert_close(actual,expected)
