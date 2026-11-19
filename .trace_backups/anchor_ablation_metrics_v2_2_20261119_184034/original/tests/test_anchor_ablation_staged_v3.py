from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from evaluation.ablation.anchor_ablation.paper_metrics import (
    compute_anchor_metrics,
    compute_planning_minade,
    geometry_valid_mask,
)
from evaluation.ablation.anchor_ablation.reporting import summarize, write_latex_table
from evaluation.ablation.anchor_ablation.runner import _filter_a1_and_sample_frames
from highway_env.planner.diffusion.projection import JointTrajectoryProjector, RoadGeometry


def test_a1_geometry_mask_ignores_traffic_mode_valid_mask():
    xy = torch.zeros(2, 10, 8, 2)
    xy[:, :, :, 0] = torch.arange(8, dtype=torch.float32)[None, None]
    xy[0, 0] = 0.0
    features = {"mode_valid_mask": torch.zeros(2, 10, dtype=torch.bool)}
    valid = geometry_valid_mask("dynamic", features, {}, xy)
    assert valid.shape == (2, 10)
    assert not valid[0, 0]
    assert valid[0, 1:].all()
    assert valid[1].all()


def test_anchor_metrics_return_four_quality_metrics():
    xy = torch.randn(2, 10, 8, 2)
    controls = torch.randn(2, 10, 40, 2)
    expert = torch.randn(2, 8, 2)
    valid = torch.ones(2, 10, dtype=torch.bool)
    safe = torch.ones(2, 10, dtype=torch.bool)
    safe[0, 0] = False
    result = compute_anchor_metrics(
        xy=xy, controls=controls, expert_xy=expert, valid=valid,
        safety={"hard_safe": safe}, expected_anchor_count=10,
    )
    assert result["anchor_minade_m"].shape == (2,)
    assert torch.allclose(result["safe_ratio"], torch.tensor([0.9, 1.0]))
    assert result["trajectory_diversity_m"].shape == (2,)
    assert result["control_diversity_normalized"].shape == (2,)


def test_planning_minade_ignores_invalid_modes():
    candidates = torch.zeros(1, 10, 8, 2)
    candidates[:, 0] = 100.0
    candidates[:, 1, :, 0] = 2.0
    expert = torch.zeros(1, 8, 2)
    valid = torch.zeros(1, 10, dtype=torch.bool)
    valid[:, 1] = True
    assert compute_planning_minade(candidates, expert, valid).item() == 2.0


class _ToyDataset(Dataset):
    def __init__(self):
        self.rows = []
        for frame in range(3):
            for ego in range(2):
                xy = np.zeros((10, 8, 2), dtype=np.float32)
                xy[:, :, 0] = np.arange(8, dtype=np.float32)[None, :]
                if frame == 1:
                    xy[0] = 0.0
                features = {"coarse_trajectories": torch.from_numpy(xy)}
                self.rows.append((features, {}, {
                    "episode_index": 0, "frame_index": frame, "ego_index": ego,
                }))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def test_a1_common_filter_keeps_only_complete_geometric_frames():
    dataset = _ToyDataset()
    subset = Subset(dataset, list(range(len(dataset))))
    filtered, manifest = _filter_a1_and_sample_frames(
        subset, expected_anchor_count=10, num_frames=0, seed=1,
        prepare_anchors=lambda features: features,
    )
    assert len(filtered) == 4
    assert manifest["a1_full_anchor_frames_retained_before_sampling"] == 2
    assert manifest["a1_full_anchor_frames_skipped"] == 1


def test_reporting_keeps_projection_unavailable_without_counting_failures(tmp_path):
    anchor_rows = [{
        "method": "A0", "anchor_minade_m": 1.0, "safe_ratio": 0.8,
        "trajectory_diversity_m": 5.0, "control_diversity_normalized": 1.2,
    }]
    planning_rows = [{
        "method": "A0", "planning_minade_m": 0.9,
        "preprojection_feasible": True, "projection_success": None,
        "projection_correction_m": None,
    }]
    summary = summarize(anchor_rows, planning_rows, methods=["A0"])
    row = summary[0]
    assert row["anchor_minade_m"] == 1.0
    assert row["planning_minade_m"] == 0.9
    assert row['preprojection_feasibility_pct'] == 100.0
    assert row['projection_correction_m'] is None
    assert row['projection_success_rate_pct'] is None
    assert row['num_projection_runs'] == 0
    assert row['num_projection_success'] == 0
    assert row['num_projection_failures'] == 0
    expected = {
        "anchor_minade_m", "safe_ratio_pct", "trajectory_diversity_m",
        "control_diversity_normalized", "planning_minade_m",
        "preprojection_feasibility_pct", "projection_correction_m",
    }
    assert expected.issubset(row)
    partial = summarize(anchor_rows, [], methods=["A0"])[0]
    assert partial["anchor_minade_m"] == 1.0
    assert partial["planning_minade_m"] is None
    path = tmp_path / "table.tex"
    write_latex_table(path, summary)
    table = path.read_text()
    assert "Anchor Quality" in table and "Planning Performance" in table
    assert "Control Div." in table and "Proj. Corr." in table
    assert r'Fixed Anchors & 1.000 & 80.000 & 5.000 & 1.200 & 0.900 & 100.000 & -- \\' in table


def test_jtp_preprojection_api_evaluates_without_projecting():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    config = project_root / "configs" / "projection" / "jtp.yaml"

    calls = []

    def rollout(controls, context):
        calls.append(controls.detach().clone())
        n, horizon, _ = controls.shape
        t = torch.arange(horizon + 1, device=controls.device, dtype=controls.dtype)
        trajectory = torch.zeros(n, horizon + 1, 3, device=controls.device, dtype=controls.dtype)
        trajectory[:, :, 0] = t[None]
        states = torch.zeros(n, horizon + 1, 5, device=controls.device, dtype=controls.dtype)
        states[:, :, 0] = 10.0
        ltr = torch.zeros(n, horizon + 1, device=controls.device, dtype=controls.dtype)
        return {"trajectory": trajectory, "states": states, "ltr": ltr}

    projector = JointTrajectoryProjector(rollout, config=config)
    context = {
        "koopman_state_hist_si": torch.zeros(1, 5, 5),
        "koopman_control_hist_si": torch.zeros(1, 4, 2),
        "koopman_pose0_si": torch.zeros(1, 3),
        "koopman_payload_kg": torch.tensor([10000.0]),
        "koopman_road_mu": torch.tensor([0.8]),
    }
    center = torch.stack((torch.linspace(-5, 50, 32), torch.zeros(32)), dim=-1)
    road = RoadGeometry(centerlines_xy=center, left_width_m=5.0, right_width_m=5.0)
    nominal = torch.zeros(1, 40, 2)
    result = projector.evaluate_preprojection(nominal, context, road)
    assert result["projector_constraints_satisfied"]
    assert result["trajectories"].shape == (1, 41, 3)
    assert result["states"].shape == (1, 41, 5)
    assert result["ltr"].shape == (1, 41)
    calls.clear()
    projected = projector.project(nominal, context, road)
    assert projected.status == 'nominal_already_feasible'
    assert len(calls) == 1
    assert torch.equal(projected.trajectories, result['trajectories'])
    assert torch.equal(projected.controls, nominal)


def test_planning_pass_checks_constraints_and_collisions_without_projecting(monkeypatch):
    from evaluation.ablation.anchor_ablation import runner
    from evaluation.ablation.anchor_ablation.data import collate_anchor_batch
    from torch.utils.data import DataLoader

    poses = torch.zeros(2, 2, 41, 3)
    poses[:, 0, 1:, 0] = 4.0
    poses[:, 1, 1:, 0] = 2.0

    class Model:
        anchors_config = {'strategy': 'fixed'}

        def eval(self):
            return self

        def _prepare_anchors(self, features):
            return dict(features, coarse_trajectories=poses[:, :, 5::5, :2],
                        mode_valid_mask=torch.ones(2, 2, dtype=torch.bool))

        def _sample_controls(self, features, generator):
            return torch.zeros(2, 2, 40, 2)

        def _all_predictions(self, controls, features, context):
            return {'poses': poses}

        def _select_goal_endpoint_mode(self, prediction, features):
            return torch.zeros(2, dtype=torch.long), None

    class ConstraintChecker:
        calls = 0

        def evaluate_preprojection(self, nominal, context, road):
            self.calls += 1
            return {'trajectories': poses[:1, 0],
                    'projector_constraints_satisfied': self.calls == 1,
                    'metrics': {'max_violation': 0.0 if self.calls == 1 else 1.0}}

        def project(self, *args, **kwargs):
            raise AssertionError('Anchor Ablation must never run iterative projection')

    features = {'ego_state': torch.zeros(14), 'agent_states': torch.zeros(0, 11),
                'agent_valid_mask': torch.zeros(0, dtype=torch.bool),
                'map_polylines': torch.zeros(1, 2, 10),
                'map_valid_mask': torch.ones(1, dtype=torch.bool), 'target_point': torch.zeros(2)}
    targets = {'trajectory': torch.zeros(8, 2),
               'koopman_state_hist_si': torch.zeros(5, 5),
               'koopman_control_hist_si': torch.zeros(4, 2),
               'koopman_pose0_si': torch.zeros(3),
               'koopman_payload_kg': torch.tensor(10000.0), 'koopman_road_mu': torch.tensor(0.8)}
    loader = DataLoader([(features, targets, {'episode_index': 0, 'frame_index': 1, 'ego_index': i})
                         for i in range(2)], batch_size=2, collate_fn=collate_anchor_batch)
    monkeypatch.setattr(runner, 'road_geometry_from_features', lambda *args: None)
    cfg = {'geometry': {'footprint': {'circle_offsets_m': [0.0], 'circle_radius_m': 1.0}},
           'execution': {'dt_s': 0.1}}
    checker = ConstraintChecker()
    rows, figure = runner._planning_seed_pass(
        'A0', Model(), checker, loader, 'cpu', 0, cfg, (0, 1, 0))
    assert checker.calls == 2
    assert [r['planning_minade_m'] for r in rows] == [2.0, 2.0]
    assert [r['preprojection_feasible'] for r in rows] == [True, False]
    assert all(r['projection_status'] == 'not_run' for r in rows)
    assert all(r['projection_success'] is None and r['projection_correction_m'] is None for r in rows)
    assert 'selected_projected_xy' not in figure
    np.testing.assert_array_equal(figure['selected_raw_xy'], poses[0, 0, 5::5, :2])
    runner._apply_synchronized_intertruck(rows, cfg)
    assert all(not r['preprojection_feasible'] for r in rows)
    assert all(not r['intertruck_collision_feasible'] for r in rows)
