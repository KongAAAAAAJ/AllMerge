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
from highway_env.planner.diffusion.projection.grouped_solver import build_priority_groups


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


def _ls_config(stride=2):
    cfg = load_jtp_config()
    cfg["projection"]["solver_method"] = "damped_least_squares"
    cfg["projection"]["constraint_sampling_stride"] = stride
    return cfg


def test_config_has_two_solver_backends_configurable_qp_priority_and_sparse_sampling():
    cfg = load_jtp_config()
    assert cfg["execution"]["horizon_steps"] == 40
    assert cfg["projection"]["solver_method"] in {"damped_least_squares", "constrained_qp"}
    assert 1 <= cfg["projection"]["constraint_sampling_stride"] <= cfg["execution"]["horizon_steps"]
    assert set(cfg["projection"]["max_active_per_group"]) == {
        "stability", "road", "curvature", "control_rate"}
    assert "collision" not in cfg
    assert "collision" not in cfg["runtime"]
    assert "priority_weights" not in cfg["projection"]
    assert "group_count" not in cfg["projection"]
    assert cfg["projection"]["qp"]["backend"] == "osqp"
    assert cfg["projection"]["qp_group_order"] == [
        "road_curvature", "stability", "control_rate"]
    order, _ = build_priority_groups({}, cfg["projection"], {})
    assert order == [["road", "curvature"], ["stability"], ["control_rate"]]

    permutations = [
        ["stability", "road_curvature", "control_rate"],
        ["stability", "control_rate", "road_curvature"],
        ["road_curvature", "stability", "control_rate"],
        ["road_curvature", "control_rate", "stability"],
        ["control_rate", "stability", "road_curvature"],
        ["control_rate", "road_curvature", "stability"],
    ]
    family_map = {
        "stability": ["stability"],
        "road_curvature": ["road", "curvature"],
        "control_rate": ["control_rate"],
    }
    for names in permutations:
        qp_projection_cfg = dict(cfg["projection"])
        qp_projection_cfg["qp_group_order"] = names
        qp_order, _ = build_priority_groups({}, qp_projection_cfg, {})
        assert qp_order == [family_map[name] for name in names]

    ls_projection_cfg = dict(cfg["projection"])
    ls_projection_cfg["solver_method"] = "damped_least_squares"
    ls_projection_cfg["qp_group_order"] = [
        "control_rate", "road_curvature", "stability"]
    ls_order, _ = build_priority_groups({}, ls_projection_cfg, {})
    assert ls_order == [["stability"], ["road", "curvature"], ["control_rate"]]


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


def test_stability_residual_uses_four_indicators_over_full_horizon():
    cfg = load_jtp_config()
    states = torch.zeros(5, 5)
    states[:, 0] = 20.0
    states[2, 2] = 0.8
    residual = stability_residual(states, torch.zeros(5), torch.tensor(5000.0), torch.tensor(0.8), cfg)
    assert residual.shape == (2 * 4 * 4,)
    assert (residual > 0).any()


def test_sparse_temporal_indices_have_expected_row_counts():
    for stride, stability_rows, road_rows, curvature_rows in (
            (1, 320, 400, 39), (2, 160, 200, 20), (4, 80, 100, 10)):
        cfg = _ls_config(stride)
        projector = JointTrajectoryProjector(StraightLineExecution(), cfg)
        controls = torch.zeros(1, 40, 2)
        with torch.no_grad():
            evaluated = projector._evaluate(controls, _context(), [_road()])
        indices = projector._sample_constraint_indices(
            evaluated.groups, projector._initial_sampled_times())
        assert indices["stability"].numel() == stability_rows
        assert indices["road"].numel() == road_rows
        assert indices["curvature"].numel() == curvature_rows
        assert indices["control_rate"].numel() == 160
        assert indices["road"].numel() == len(cfg["geometry"]["footprint"]["circle_offsets_m"]) * 2 * (40 // stride)


def test_ls_topk_and_qp_full_group_use_distinct_selection_rules():
    cfg = _ls_config(2)
    cfg["projection"]["max_active_per_group"]["stability"] = 2
    projector = JointTrajectoryProjector(StraightLineExecution(), cfg)
    with torch.no_grad():
        evaluated = projector._evaluate(torch.zeros(1, 40, 2), _context(), [_road()])
    groups = {name: value.clone() for name, value in evaluated.groups.items()}
    thresholds = projector._constraint_thresholds(groups)
    sampled_times = projector._initial_sampled_times()
    candidate_indices = projector._sample_constraint_indices(groups, sampled_times)
    groups["stability"][candidate_indices["stability"][:5]] = 1.0
    refined_times = {name: set() for name in ("stability", "road", "curvature")}

    ls_selected, ls_diag = projector._active_indices(groups, candidate_indices, refined_times)
    qp_selected, qp_diag = projector._full_group_indices(groups, candidate_indices)
    assert ls_selected["stability"].numel() == 2
    assert ls_diag["active_topk_truncated_counts"]["stability"] == 3
    assert qp_selected["stability"].numel() == candidate_indices["stability"].numel()
    assert qp_diag["solver_constraint_rows_by_family"]["stability"] == 160
    assert qp_diag["active_topk_truncated_counts"]["stability"] == 0


def test_full_horizon_validation_adds_unsampled_g1_g2_times():
    projector = JointTrajectoryProjector(StraightLineExecution(), _ls_config(2))
    with torch.no_grad():
        evaluated = projector._evaluate(torch.zeros(1, 40, 2), _context(), [_road()])
    groups = {name: value.clone() for name, value in evaluated.groups.items()}
    thresholds = projector._constraint_thresholds(groups)
    sampled_times = projector._initial_sampled_times()
    refined_times = {name: set() for name in ("stability", "road", "curvature")}
    groups["stability"][8] = thresholds["stability"][8] + 1.0  # t=3, first stability indicator
    groups["road"][20] = thresholds["road"][20] + 1.0        # t=3, first circle/side
    groups["curvature"][1] = thresholds["curvature"][1] + 1.0 # curvature ending at t=3
    additions = projector._refine_constraint_indices(
        groups, thresholds, sampled_times, refined_times)
    assert additions == {"stability": [3], "road": [3], "curvature": [3]}
    assert refined_times["stability"] == {3}
    assert refined_times["road"] == {3}
    assert refined_times["curvature"] == {3}


def test_jtp_returns_nominal_when_single_vehicle_is_feasible():
    cfg = _ls_config()
    cfg["geometry"]["footprint"] = {"circle_offsets_m": [0.0], "circle_radius_m": 0.5}
    projector = JointTrajectoryProjector(StraightLineExecution(), cfg)
    controls = torch.zeros(1, 40, 2)
    result = projector.project(controls, _context(), _road())
    assert result.projector_constraints_satisfied
    assert result.status == "nominal_already_feasible"
    assert result.iterations == 0
    torch.testing.assert_close(result.controls, controls)
    assert result.trajectories.shape == (1, 41, 3)
    assert result.diagnostics["iterations_by_vehicle"] == [0]
    assert result.diagnostics["constraint_sampling_stride"] == 2


def test_three_vehicles_are_projected_independently_in_80d_spaces():
    cfg = _ls_config()
    cfg["geometry"]["footprint"] = {"circle_offsets_m": [0.0], "circle_radius_m": 0.5}
    cfg["geometry"]["curvature_limit_per_m"] = 0.2
    cfg["projection"]["max_iterations"] = 8
    cfg["projection"]["max_active_per_group"]["stability"] = 40
    cfg["projection"]["max_step_norm"] = 0.5
    projector = JointTrajectoryProjector(StraightLineExecution(), cfg)
    controls = torch.zeros(3, 40, 2)
    context = _context(3, yaw_rate=[0.60, 0.0, 0.0])
    roads = [_road(100.0), _road(100.0), _road(100.0)]
    batch_result = projector.project(controls, context, roads)
    single_result = projector.project(controls[:1], {key: value[:1] for key, value in context.items()}, roads[:1])
    assert batch_result.controls.shape == (3, 40, 2)
    assert len(batch_result.diagnostics["per_vehicle"]) == 3
    assert len(batch_result.diagnostics["projection_latency_ms_by_vehicle"]) == 3
    assert all(value >= 0.0 for value in batch_result.diagnostics["projection_latency_ms_by_vehicle"])
    assert all("projection_latency_ms" in item for item in batch_result.diagnostics["per_vehicle"])
    assert batch_result.diagnostics["iterations_by_vehicle"][0] > 0
    for item in batch_result.diagnostics["per_vehicle"]:
        for step in item["projection"]:
            if "jacobian_shape" in step:
                assert step["jacobian_shape"][1] == 80
    torch.testing.assert_close(batch_result.controls[0], single_result.controls[0])


def test_grouped_ls_corrects_yaw_stability_without_qp():
    cfg = _ls_config()
    cfg["geometry"]["footprint"] = {"circle_offsets_m": [0.0], "circle_radius_m": 0.5}
    cfg["geometry"]["curvature_limit_per_m"] = 0.2
    cfg["projection"]["max_iterations"] = 8
    cfg["projection"]["max_active_per_group"]["stability"] = 40
    cfg["projection"]["max_step_norm"] = 0.5
    projector = JointTrajectoryProjector(StraightLineExecution(), cfg)
    controls = torch.zeros(1, 40, 2)
    result = projector.project(controls, _context(yaw_rate=0.60), _road(100.0))
    assert result.iterations > 0
    assert result.diagnostics["accepted_iterations"] > 0
    assert result.diagnostics["projection"]
    assert result.diagnostics["projection"][0]["jacobian_shape"][1] == 80
    assert result.diagnostics["final_metrics"]["max_violation"] < result.diagnostics["initial_metrics"]["max_violation"]
    assert torch.isfinite(result.controls).all()
    assert (result.controls[..., 1] >= cfg["control"]["steering_min_rad"] - 1e-6).all()
    assert (result.controls[..., 1] <= cfg["control"]["steering_max_rad"] + 1e-6).all()
    assert result.diagnostics["collision_constraints_enabled"] is False


def test_qp_max_iterations_one_runs_one_linearization_and_one_group_pass_per_vehicle(monkeypatch):
    from highway_env.planner.diffusion.projection import joint_projector as projector_module

    cfg = load_jtp_config()
    cfg["projection"]["solver_method"] = "constrained_qp"
    cfg["projection"]["max_iterations"] = 1
    cfg["geometry"]["footprint"] = {"circle_offsets_m": [0.0], "circle_radius_m": 0.5}
    calls = []

    def fake_solve_groups(controls, groups, selected, rows, thresholds, metric, order, cfg,
                          scale, trust, damping_multiplier, lower, upper,
                          control_weights=None):
        calls.append({"shape": tuple(controls.shape), "order": order})
        return torch.zeros_like(controls), {"groups": [], "solver_method": "constrained_qp"}

    monkeypatch.setattr(projector_module, "solve_groups", fake_solve_groups)
    projector = JointTrajectoryProjector(StraightLineExecution(), cfg)
    controls = torch.zeros(3, 40, 2)
    context = _context(3, yaw_rate=[0.60, 0.0, 0.0])
    roads = [_road(100.0), _road(100.0), _road(100.0)]

    result = projector.project(controls, context, roads)

    assert len(calls) == 1
    assert calls[0]["shape"] == (1, 40, 2)
    assert calls[0]["order"] == [["road", "curvature"], ["stability"], ["control_rate"]]
    per_vehicle = result.diagnostics["per_vehicle"]
    assert per_vehicle[0]["linearization_count"] == 1
    assert len(per_vehicle[0]["projection"]) == 1
    assert per_vehicle[1]["linearization_count"] == 0
    assert per_vehicle[2]["linearization_count"] == 0


def test_control_box_is_enforced_by_bounded_group_solver():
    cfg = _ls_config()
    cfg["geometry"]["footprint"] = {"circle_offsets_m": [0.0], "circle_radius_m": 0.5}
    projector = JointTrajectoryProjector(StraightLineExecution(), cfg)
    controls = torch.zeros(1, 40, 2)
    controls[..., 0] = 10.0
    result = projector.project(controls, _context(), _road())
    assert (result.controls[..., 0] <= cfg["control"]["acceleration_max_mps2"]).all()
    assert (result.controls[..., 0] >= cfg["control"]["acceleration_min_mps2"]).all()
    assert result.diagnostics["projection"][0]["hard_box_restoration_norm"] > 0


def test_qp_backend_uses_all_sparse_rows_without_active_set_truncation(monkeypatch):
    from highway_env.planner.diffusion.projection import qp_solver

    cfg = load_jtp_config()["projection"]
    controls = torch.zeros(1, 40, 2)
    groups = {
        "stability": torch.zeros(8),
        "road": torch.zeros(10),
        "curvature": torch.zeros(6),
        "control_rate": torch.zeros(12),
    }
    selected = {
        "stability": torch.tensor([0, 3, 7]),
        "road": torch.tensor([1, 4, 8, 9]),
        "curvature": torch.tensor([0, 5]),
        "control_rate": torch.tensor([0, 2, 4, 6, 8]),
    }
    rows = {name: torch.zeros(len(indices), 80) for name, indices in selected.items()}
    thresholds = {name: torch.zeros_like(value) for name, value in groups.items()}
    captured = []

    def fake_solve(problem, solver_cfg):
        P, _, A, _, _ = problem
        captured.append((P.shape[0], A.shape[0]))
        return np.zeros(P.shape[0], dtype=np.float64), {
            "backend": "mock_osqp", "solver_status": "solved", "solver_status_val": 1,
            "solver_iterations": 1, "primal_residual": 0.0, "dual_residual": 0.0,
            "solver_objective": 0.0, "solver_setup_time_s": 0.0, "solver_solve_time_s": 0.0,
        }

    monkeypatch.setattr(qp_solver, "_solve_osqp", fake_solve)
    delta, info = qp_solver.solve_groups_qp(
        controls, groups, selected, rows, thresholds,
        [["road", "curvature"], ["stability"], ["control_rate"]], cfg,
        torch.tensor([-4.0, -0.08]), torch.tensor([2.5, 0.08]), [0.16, 156.25])
    assert delta.shape == controls.shape
    assert info["constraint_rows_by_family"] == {
        "stability": 3, "road": 4, "curvature": 2, "control_rate": 5}
    assert [group["families"] for group in info["groups"]] == [["road", "curvature"], ["stability"], ["control_rate"]]
    assert [group["constraint_rows"] for group in info["groups"]] == [6, 3, 5]
    assert len(captured) == 3


def test_projector_does_not_import_osqp_or_scipy_sparse():
    import inspect
    import highway_env.planner.diffusion.projection.joint_projector as solver_module
    source = inspect.getsource(solver_module)
    assert "import osqp" not in source
    assert "scipy" not in source.lower()


def test_projector_constraint_groups_do_not_contain_collision_families():
    projector = JointTrajectoryProjector(StraightLineExecution(), _ls_config())
    with torch.no_grad():
        evaluated = projector._evaluate(torch.zeros(1, 40, 2), _context(), [_road()])
    assert set(evaluated.groups) == {
        "stability", "road", "curvature", "control", "control_rate"}
