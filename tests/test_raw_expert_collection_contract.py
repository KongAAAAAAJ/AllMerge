from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ACTION_PATH = ROOT / "highway_env/envs/common/action.py"


def _load_expert_trajectory_method():
    tree = ast.parse(ACTION_PATH.read_text(encoding="utf-8"))
    method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_build_expert_trajectory":
            method = node
            break
    assert method is not None
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(ACTION_PATH), "exec"), namespace)
    return namespace[method.name]


class _Vehicle:
    def __init__(self, trajectory):
        self.latest_planner_trajectory = trajectory


class _Env:
    def __init__(self, trajectories):
        self.controlled_vehicles = [_Vehicle(x) for x in trajectories]
        self.latest_expert_trajectory = None


class _Action:
    pass


def _make_trajectory(offset=0.0):
    dense_t = np.arange(1, 41, dtype=np.float32) * 0.1
    dense_xy = np.stack([dense_t + offset, 0.2 * dense_t], axis=-1).astype(np.float32)
    sparse_t = np.arange(1, 9, dtype=np.float32) * 0.5
    sparse_idx = np.rint(sparse_t / 0.1).astype(int) - 1
    return {
        "time_s": sparse_t,
        "xy": dense_xy[sparse_idx],
        "dense_time_s": dense_t,
        "future_trajectory_dense": dense_xy,
        "dense_dt": 0.1,
        "trajectory_horizon_s": 4.0,
    }


def test_raw_expert_trajectory_export_does_not_require_anchor_fields():
    method = _load_expert_trajectory_method()
    action = _Action()
    action.env = _Env([_make_trajectory(), _make_trajectory(1.0)])
    result = method(action, {
        "state": True,
        "type": "Polynomial",
        "Polynomial": {"mode": "aligned"},
        "ExpertTrajectory": {"enabled": True},
    })
    assert result is action.env.latest_expert_trajectory
    assert result["expert_trajectory_xy"].shape == (2, 8, 2)
    assert result["future_trajectory_dense"].shape == (2, 40, 2)
    assert result["dense_time_s"].shape == (40,)
    assert not hasattr(action.env, "expert_trajectory_aligner")


def test_raw_expert_trajectory_export_rejects_sparse_dense_mismatch():
    method = _load_expert_trajectory_method()
    trajectory = _make_trajectory()
    trajectory["xy"] = trajectory["xy"].copy()
    trajectory["xy"][0, 0] += 1.0
    action = _Action()
    action.env = _Env([trajectory])
    with pytest.raises(RuntimeError, match="Sparse/dense expert target inconsistency"):
        method(action, {
            "state": True,
            "type": "Polynomial",
            "Polynomial": {"mode": "aligned"},
            "ExpertTrajectory": {"enabled": True},
        })


def test_raw_collector_uses_root_relative_paths_and_new_contract():
    collector = (ROOT / "data_pipeline/expert_collection/worker.py").read_text(encoding="utf-8")
    assert "def _resolve_project_path" in collector
    assert "os.chdir(ROOT)" in collector
    assert "os.chdir(WORKSPACE)" not in collector
    assert "latest_expert_trajectory" in collector
    assert "latest_expert_alignment" not in collector
    assert "ExpertTrajectory" in collector


def test_raw_feature_build_does_not_instantiate_anchor_builder_when_disabled():
    source = ACTION_PATH.read_text(encoding="utf-8")
    start = source.index("    def _build_planner_features(")
    end = source.index("    def _run_diffusion_shadow(", start)
    method = source[start:end]
    import_pos = method.index("from highway_env.planner.anchor_builder import AllMergeAnchorBuilder")
    guard_pos = method.index('if bool(anchors_cfg.get("enabled", True))')
    assert import_pos > guard_pos
    assert "planner_anchor_builder = anchor_builder" in method


def test_project_path_helper_anchors_relative_paths_at_repository_root(tmp_path):
    source = (ROOT / "data_pipeline/expert_collection/worker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_project_path"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {"Path": Path, "ROOT": tmp_path}
    exec(compile(module, "collect_raw_expert_dataset.py", "exec"), namespace)
    assert namespace[fn.name](Path("configs/koopman/collect.yaml")) == (
        tmp_path / "configs/koopman/collect.yaml"
    ).resolve()
    absolute = (tmp_path / "absolute.yaml").resolve()
    assert namespace[fn.name](absolute) == absolute
