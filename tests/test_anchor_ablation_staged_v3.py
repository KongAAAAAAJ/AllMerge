from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from evaluation.ablation.anchor_ablation.paper_metrics import (
    compute_anchor_metrics,
    compute_planning_minade,
    geometry_valid_mask,
)
from evaluation.ablation.anchor_ablation.paper_safety import (
    drivable_area_compliance,
    external_collision_free,
    external_min_ttc_s,
    expert_local_poses,
    intertruck_group_collision_free,
    intertruck_group_min_ttc_s,
    local_to_world_poses,
    road_geometry_from_features,
)
from evaluation.ablation.anchor_ablation.reporting import summarize, write_latex_table
from evaluation.ablation.anchor_ablation.runner import _filter_a1_and_sample_frames


def _vehicle_cfg():
    return {
        "geometry": {"footprint": {"circle_offsets_m": [-3.0, 0.0, 3.0],
                                   "circle_radius_m": 1.45}},
        "execution": {"dt_s": 0.1},
        "evaluation": {"collision_margin_m": 0.25},
    }


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


def test_reporting_writes_exact_eight_metric_table(tmp_path):
    anchor_rows = [{
        "method": "A0", "anchor_minade_m": 1.0, "safe_ratio": 0.8,
        "trajectory_diversity_m": 5.0, "control_diversity_normalized": 1.2,
    }]
    planning_rows = [{
        "method": "A0", "planning_minade_m": 0.9,
        "collision": True, "drivable_area_compliance_pct": 95.0,
        "ttc_compliant": False, "min_ttc_s": 0.8, "external_min_ttc_s": 1.2,
    }]
    summary = summarize(anchor_rows, planning_rows, methods=["A0"])
    row = summary[0]
    assert row["anchor_minade_m"] == 1.0
    assert row["safe_ratio_pct"] == 80.0
    assert row["planning_minade_m"] == 0.9
    assert row["collision_rate_pct"] == 100.0
    assert row["drivable_area_compliance_pct"] == 95.0
    assert row["ttc_compliance_pct"] == 0.0
    assert row["mean_min_ttc_s"] == 0.8
    partial = summarize(anchor_rows, [], methods=["A0"])[0]
    assert partial["planning_minade_m"] is None
    assert partial["collision_rate_pct"] is None
    path = tmp_path / "table.tex"
    write_latex_table(path, summary)
    table = path.read_text(encoding="utf-8")
    assert "Anchor Quality" in table and "Planning Performance" in table
    assert "Control Div." in table
    assert r"\textbf{CR}" in table and r"\textbf{DAC}" in table and r"\textbf{TTC}" in table
    assert "Pre-proj." not in table and "Proj. Corr." not in table
    assert r"Fixed Anchors & 1.000 & 80.000 & 5.000 & 1.200 & 0.900 & 100.000 & 95.000 & 0.000 \\" in table


def test_expert_heading_uses_path_tangent_and_stays_finite_at_stops():
    poses = expert_local_poses(np.array([[1., 1.], [2., 2.], [2., 2.], [2., 2.]]))
    np.testing.assert_array_equal(poses[0], [0., 0., 0.])
    np.testing.assert_allclose(poses[1:, 2], np.pi / 4)
    assert np.isfinite(expert_local_poses(np.zeros((40, 2)))).all()
    with np.testing.assert_raises(ValueError):
        expert_local_poses(np.array([[np.nan, 0.]]))


def test_expert_reference_reuses_safety_checks_and_keeps_anchor_columns_empty(tmp_path):
    from evaluation.ablation.anchor_ablation import runner
    from evaluation.ablation.anchor_ablation.data import collate_anchor_batch

    features = {key: torch.as_tensor(value) for key, value in _road_features(width=40.).items()}
    features.update(ego_state=torch.zeros(14), agent_states=torch.zeros(1, 11),
                    agent_valid_mask=torch.zeros(1, dtype=torch.bool), target_point=torch.zeros(2))
    dense = torch.stack((torch.arange(1, 41, dtype=torch.float32), torch.zeros(40)), dim=-1)
    samples = []
    for ego in range(2):
        samples.append((features, {'trajectory_dense': dense,
                                  'koopman_pose0_si': torch.tensor([float(ego), 0., 0.])},
                        {'episode_index': 3, 'frame_index': 7, 'ego_index': ego, 'seed': 12}))
    loader = DataLoader(samples, batch_size=1, collate_fn=collate_anchor_batch)
    rows = runner._expert_pass(loader, _vehicle_cfg(), {'ttc_threshold_s': 1.0})
    assert len(rows) == 2
    assert all(row['seed'] is None for row in rows)
    assert all(row['planning_minade_m'] == 0.0 for row in rows)
    assert all(row['external_collision_free'] for row in rows)
    # Synchronized expert trucks one metre apart must collide, even when
    # their samples arrive in separate batches and background traffic is absent.
    assert all(not row['intertruck_collision_free'] and row['collision'] for row in rows)
    assert all(not row['ttc_compliant'] and row['min_ttc_s'] == 0.0 for row in rows)
    assert all(row['drivable_area_compliance_pct'] == 100.0 for row in rows)
    assert all('_raw_world_poses' not in row for row in rows)
    summary = summarize([], rows, methods=['Expert'])
    expert = summary[0]
    assert all(expert[key] is None for key in ('anchor_minade_m', 'safe_ratio_pct',
                                               'trajectory_diversity_m', 'control_diversity_normalized'))
    assert expert['collision_rate_pct'] == 100.0
    assert expert['num_anchor_scenes'] == 0 and expert['num_planning_runs'] == 2
    path = tmp_path / 'expert_table.tex'
    write_latex_table(path, summary)
    assert r'Expert trajectory & - & - & - & - & 0.000 & 100.000 & 100.000 & 0.000 \\' in path.read_text()


def _road_features(width=8.0):
    maps = np.zeros((1, 32, 10), dtype=np.float32)
    maps[0, :, 0] = np.linspace(-10.0, 50.0, 32)
    maps[0, :, 4] = width
    maps[0, :, 6] = 1.0  # is_current_lane / recognized drivable lane
    return {
        "map_polylines": maps,
        "map_valid_mask": np.asarray([True]),
    }


def test_drivable_area_compliance_uses_full_vehicle_footprint():
    features = _road_features(width=8.0)
    road = road_geometry_from_features(features, np.zeros(3))
    poses_inside = np.zeros((41, 3), dtype=np.float64)
    poses_inside[:, 0] = np.linspace(0.0, 40.0, 41)
    assert drivable_area_compliance(
        poses_inside, road, _vehicle_cfg()
    ) == 100.0

    poses_outside = poses_inside.copy()
    poses_outside[:, 1] = 3.0
    assert drivable_area_compliance(
        poses_outside, road, _vehicle_cfg()
    ) == 0.0


def test_external_collision_and_ttc_use_constant_velocity_background_actor():
    features = {
        "ego_state": np.zeros(14, dtype=np.float64),
        "agent_states": np.zeros((1, 11), dtype=np.float64),
        "agent_valid_mask": np.asarray([True]),
    }
    features["ego_state"][0] = 10.0
    features["agent_states"][0, 0] = 20.0
    features["agent_states"][0, 2] = -10.0  # relative actor velocity; actor is stationary in world
    features["agent_states"][0, 7] = 1.0
    features["agent_states"][0, 8] = 4.0
    features["agent_states"][0, 9] = 2.0

    ego_poses = np.zeros((41, 3), dtype=np.float64)
    ego_poses[:, 0] = np.arange(41, dtype=np.float64)  # 10 m/s at 0.1 s intervals
    cfg = _vehicle_cfg()
    assert not external_collision_free(ego_poses, features, cfg)
    assert external_min_ttc_s(ego_poses, features, cfg) == 0.0

    features["agent_valid_mask"][:] = False
    assert np.isinf(external_min_ttc_s(ego_poses, features, cfg))
    assert external_collision_free(ego_poses, features, cfg)


def test_local_to_world_pose_transform():
    local = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.2]])
    pose0 = np.asarray([100.0, 20.0, np.pi / 2])
    world = local_to_world_poses(local, pose0)
    np.testing.assert_allclose(world[0], [100.0, 20.0, np.pi / 2])
    np.testing.assert_allclose(world[1, :2], [100.0, 21.0], atol=1e-8)
    np.testing.assert_allclose(world[1, 2], np.pi / 2 + 0.2)


def test_intertruck_collision_and_ttc_use_synchronized_raw_k2_paths():
    cfg = _vehicle_cfg()
    first = np.zeros((41, 3), dtype=np.float64)
    second = np.zeros((41, 3), dtype=np.float64)
    second[:, 0] = 1.0
    records = [{"raw_world_poses": first}, {"raw_world_poses": second}]
    assert not intertruck_group_collision_free(records, cfg)
    ttc = intertruck_group_min_ttc_s(records, cfg)
    assert ttc == [0.0, 0.0]


def test_planning_pass_uses_raw_k2_and_never_invokes_jtp(monkeypatch):
    from evaluation.ablation.anchor_ablation import runner
    from evaluation.ablation.anchor_ablation.data import collate_anchor_batch

    poses = torch.zeros(2, 2, 41, 3)
    poses[:, 0, 1:, 0] = 4.0
    poses[:, 1, 1:, 0] = 2.0

    class Model:
        anchors_config = {"strategy": "fixed"}

        def eval(self):
            return self

        def _prepare_anchors(self, features):
            return dict(
                features,
                coarse_trajectories=poses[:, :, 5::5, :2],
                mode_valid_mask=torch.ones(2, 2, dtype=torch.bool),
            )

        def _sample_controls(self, features, generator):
            return torch.zeros(2, 2, 40, 2)

        def _all_predictions(self, controls, features, context):
            return {"poses": poses}

        def _select_goal_endpoint_mode(self, prediction, features):
            return torch.zeros(2, dtype=torch.long), None

    features = {
        "ego_state": torch.zeros(14),
        "agent_states": torch.zeros(0, 11),
        "agent_valid_mask": torch.zeros(0, dtype=torch.bool),
        "map_polylines": torch.zeros(1, 2, 10),
        "map_valid_mask": torch.ones(1, dtype=torch.bool),
        "target_point": torch.zeros(2),
    }
    targets = {
        "trajectory": torch.zeros(8, 2),
        "koopman_state_hist_si": torch.zeros(5, 5),
        "koopman_control_hist_si": torch.zeros(4, 2),
        "koopman_pose0_si": torch.zeros(3),
        "koopman_payload_kg": torch.tensor(10000.0),
        "koopman_road_mu": torch.tensor(0.8),
    }
    loader = DataLoader(
        [(features, targets, {"episode_index": 0, "frame_index": 1, "ego_index": i})
         for i in range(2)],
        batch_size=2,
        collate_fn=collate_anchor_batch,
    )
    monkeypatch.setattr(runner, "road_geometry_from_features", lambda *args: {})
    monkeypatch.setattr(runner, "external_collision_free", lambda *args: True)
    monkeypatch.setattr(runner, "external_min_ttc_s", lambda *args, **kwargs: 2.0)
    monkeypatch.setattr(runner, "drivable_area_compliance", lambda *args, **kwargs: 98.0)

    cfg = _vehicle_cfg()
    safety_cfg = {
        "ttc_threshold_s": 1.0,
        "closing_speed_epsilon_mps": 0.1,
        "drivable_area_tolerance_m": 1e-6,
    }
    rows, figure = runner._planning_seed_pass(
        "A0", Model(), loader, "cpu", 0, cfg, safety_cfg, (0, 1, 0)
    )
    assert [row["planning_minade_m"] for row in rows] == [2.0, 2.0]
    assert all(row["external_collision_free"] for row in rows)
    assert all(row["drivable_area_compliance_pct"] == 98.0 for row in rows)
    assert all(row["ttc_compliant"] for row in rows)
    assert "selected_projected_xy" not in figure
    np.testing.assert_array_equal(figure["selected_raw_xy"], poses[0, 0, 5::5, :2])
    assert not hasattr(runner, "JointTrajectoryProjector")

    runner._apply_synchronized_intertruck(rows, cfg, safety_cfg)
    # The two synchronized controlled trucks have identical raw K2 paths.
    assert all(not row["intertruck_collision_free"] for row in rows)
    assert all(row["collision"] for row in rows)
    assert all(row["min_ttc_s"] == 0.0 for row in rows)
    assert all(not row["ttc_compliant"] for row in rows)
