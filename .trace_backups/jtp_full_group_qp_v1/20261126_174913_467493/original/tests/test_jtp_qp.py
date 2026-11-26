from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")

from highway_env.planner.diffusion.projection.config import load_jtp_config
from highway_env.planner.diffusion.projection.qp_solver import _build_qp_problem


def _qp_config():
    return load_jtp_config(overrides={
        "projection": {"solver_method": "constrained_qp"}
    })["projection"]


def test_qp_problem_contains_control_box_state_slack_and_protection_slack():
    cfg = _qp_config()
    control_diagonal = np.array([0.16, 156.25], dtype=np.float64)
    control_lower = np.array([-1.0, -0.08], dtype=np.float64)
    control_upper = np.array([1.0, 0.08], dtype=np.float64)
    state_jacobian = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    state_residual = np.array([0.4, 0.8], dtype=np.float64)
    state_weights = np.array([0.5, 0.5], dtype=np.float64)
    protection_jacobian = np.array([[1.0, -1.0]], dtype=np.float64)
    protection_weights = np.array([1.0], dtype=np.float64)

    P, q, A, lower, upper = _build_qp_problem(
        control_diagonal, control_lower, control_upper,
        state_jacobian, state_residual, state_weights,
        protection_jacobian, protection_weights, cfg)

    assert P.shape == (5, 5)
    assert A.shape == (8, 5)
    np.testing.assert_allclose(P.diagonal(), [0.16, 156.25, 0.5, 0.5, 100.0])
    np.testing.assert_allclose(q, [0.0, 0.0, 5.0, 5.0, 1000.0])
    np.testing.assert_allclose(lower[:2], control_lower)
    np.testing.assert_allclose(upper[:2], control_upper)
    np.testing.assert_allclose(upper[2:4], [-0.4, -0.8])
    np.testing.assert_allclose(upper[4:6], [0.4, 0.8])
    np.testing.assert_allclose(upper[6], 0.0)
    np.testing.assert_allclose(upper[7], cfg["protection_tolerance"])
    assert np.isneginf(lower[2:4]).all()
    assert np.isneginf(lower[6])
    assert (A.toarray()[2, :2] == state_jacobian[0]).all()
    assert (A.toarray()[6, :2] == protection_jacobian[0]).all()


def test_qp_config_validates_penalties_and_solver_settings():
    cfg = load_jtp_config(overrides={
        "projection": {"solver_method": "constrained_qp"}
    })
    assert cfg["projection"]["qp"]["backend"] == "osqp"
    broken = load_jtp_config()
    broken["projection"]["solver_method"] = "constrained_qp"
    broken["projection"]["qp"]["state_slack"]["linear_penalty"] = 0.0
    from highway_env.planner.diffusion.projection.config import validate_jtp_config
    with pytest.raises(ValueError, match="state_slack penalties"):
        validate_jtp_config(broken)


@pytest.mark.skipif(importlib.util.find_spec("osqp") is None, reason="OSQP optional dependency is not installed")
def test_qp_three_groups_protect_previous_groups_and_have_no_trust_region():
    from highway_env.planner.diffusion.projection.qp_solver import solve_groups_qp

    cfg = _qp_config()
    controls = torch.zeros(1, 1, 2, dtype=torch.float64)
    groups = {
        "g1": torch.tensor([2.0], dtype=torch.float64),
        "g2": torch.tensor([1.0], dtype=torch.float64),
        "g3": torch.tensor([0.2], dtype=torch.float64),
    }
    selected = {name: torch.tensor([0]) for name in groups}
    rows = {
        "g1": torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        "g2": torch.tensor([[-1.0, 1.0]], dtype=torch.float64),
        "g3": torch.tensor([[0.0, 1.0]], dtype=torch.float64),
    }
    thresholds = {name: torch.zeros_like(value) for name, value in groups.items()}
    delta, info = solve_groups_qp(
        controls, groups, selected, rows, thresholds,
        [["g1"], ["g2"], ["g3"]], cfg,
        torch.tensor([-4.0, -0.08], dtype=torch.float64),
        torch.tensor([2.5, 0.08], dtype=torch.float64),
        [0.16, 156.25],
    )
    assert delta is not None
    assert info["trust_region_enabled"] is False
    assert len(info["groups"]) == 3
    assert abs(float(delta[0, 0, 0])) > 0.25 * 2.5
    assert info["groups"][1]["protection_linearized_increase_max"] <= cfg["protection_tolerance"] + 1.0e-7
    assert info["groups"][2]["protection_linearized_increase_max"] <= cfg["protection_tolerance"] + 1.0e-7
    assert all(group["solver_status"] == "solved" for group in info["groups"])
