from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECTOR = ROOT / "highway_env/planner/diffusion/projection/joint_projector.py"
GROUPED_SOLVER = ROOT / "highway_env/planner/diffusion/projection/grouped_solver.py"


def _project_method_source() -> str:
    source = PROJECTOR.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PROJECTOR))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "JointTrajectoryProjector":
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "project":
                    return ast.get_source_segment(source, member) or ""
    raise AssertionError("JointTrajectoryProjector.project() was not found")


def test_global_box_cap_is_diagnostic_only_and_does_not_change_alpha():
    method = _project_method_source()
    tree = ast.parse(method)
    alpha_assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "alpha"
            for target in node.targets
        ):
            alpha_assignments.append(ast.get_source_segment(method, node) or "")
    assert any('alpha=float(proj_cfg["alpha_initial"])' in item for item in alpha_assignments)
    assert not any("cap.min" in item or "min(alpha" in item for item in alpha_assignments)
    assert 'diag["global_box_cap_disabled"] = True' in method
    assert 'diag["would_be_alpha_box_cap"] = would_be_alpha_box_cap' in method
    assert 'while alpha>=float(proj_cfg["alpha_minimum"])' in method


def test_elementwise_control_box_guard_is_preserved_and_diagnosed():
    method = _project_method_source()
    assert "guarded=torch.maximum(torch.minimum(candidate_controls,upper),lower)" in method
    assert '"box_guard_changed_count":guard_changed_count' in method
    assert '"box_guard_max_correction":guard_max_correction' in method


def test_grouped_solver_trust_bound_is_preserved():
    source = GROUPED_SOLVER.read_text(encoding="utf-8")
    assert "bound=scale.repeat(controls.numel()//2)*min(trust,float(cfg['max_step_norm']))" in source


def test_projector_source_is_valid_python():
    ast.parse(PROJECTOR.read_text(encoding="utf-8"), filename=str(PROJECTOR))
