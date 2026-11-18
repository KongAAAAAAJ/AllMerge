from __future__ import annotations

import numpy as np
import pytest
import torch

# The A1 geometry builder itself does not depend on Gymnasium. Keep this unit
# test runnable in lightweight analysis environments that omit the simulator.
try:
    import gymnasium  # noqa: F401
except ImportError:
    import sys
    import types
    gym = types.ModuleType("gymnasium")
    envs = types.ModuleType("gymnasium.envs")
    registration = types.ModuleType("gymnasium.envs.registration")
    registration.register = lambda *args, **kwargs: None
    gym.envs = envs
    envs.registration = registration
    sys.modules.setdefault("gymnasium", gym)
    sys.modules.setdefault("gymnasium.envs", envs)
    sys.modules.setdefault("gymnasium.envs.registration", registration)

from highway_env.planner.a1_anchor_builder import A1AnchorBuilder
from highway_env.planner.a1_anchor_manager import A1AnchorManager, A1_ROAD_GEOMETRY_KEYS


def _live_geometry(batch=2):
    lanes, points = 3, 200
    x = np.linspace(-30.0, 250.0, points, dtype=np.float32)
    s = x + 30.0
    geom = np.zeros((batch, lanes, points, 9), np.float32)
    point_mask = np.ones((batch, lanes, points), bool)
    lane_mask = np.ones((batch, lanes), bool)
    meta = np.zeros((batch, lanes, 7), np.float32)
    connections = np.zeros((batch, lanes, lanes), bool)
    for b in range(batch):
        for lane_i, y in enumerate((-3.7, 0.0, 3.7)):
            geom[b, lane_i, :, 0] = x
            geom[b, lane_i, :, 1] = y
            geom[b, lane_i, :, 2] = x
            geom[b, lane_i, :, 3] = y + 1.85
            geom[b, lane_i, :, 4] = x
            geom[b, lane_i, :, 5] = y - 1.85
            geom[b, lane_i, :, 6] = 0.0
            geom[b, lane_i, :, 7] = 3.7
            geom[b, lane_i, :, 8] = s
            meta[b, lane_i, 0] = 33.0
            meta[b, lane_i, 1] = 300.0
            meta[b, lane_i, 2] = 30.0
        connections[b, 1, :] = True
        connections[b, :, 1] = True
    return {
        "road_lane_geometry": geom,
        "road_lane_point_mask": point_mask,
        "road_lane_mask": lane_mask,
        "road_current_lane_slot": np.ones((batch,), np.int16),
        "road_target_lane_slot": np.ones((batch,), np.int16),
        "road_lane_connections": connections,
        "road_lane_meta": meta,
    }


def _manager():
    return A1AnchorManager(
        {"a1_config": "configs/anchors/a1.yaml"},
        control_steps=40, dt=0.1, wheelbase_m=5.5,
        max_accel_mps2=2.5, max_steer_rad=0.08,
    )


def test_a1_online_generates_ten_current_scene_anchors():
    manager = _manager()
    geometry = _live_geometry(batch=2)
    features = {
        "ego_state": torch.tensor([[20.0] + [0.0] * 13] * 2),
    }
    features.update({k: torch.as_tensor(v) for k, v in geometry.items()})
    result = manager.select(features)
    assert result["coarse_controls"].shape == (2, 10, 40, 2)
    assert result["coarse_trajectories"].shape == (2, 10, 8, 2)
    assert result["mode_valid_mask"].shape == (2, 10)
    assert result["mode_valid_mask"].any(dim=1).all()
    assert torch.isfinite(result["coarse_controls"]).all()
    assert torch.isfinite(result["coarse_trajectories"]).all()
    assert manager.last_latency_ms >= 0


def test_a1_online_matches_offline_builder_for_identical_input():
    manager = _manager()
    geometry = _live_geometry(batch=1)
    ego = np.asarray([[20.0] + [0.0] * 13], np.float32)
    features = {"ego_state": torch.from_numpy(ego)}
    features.update({k: torch.as_tensor(v) for k, v in geometry.items()})
    online = manager.select(features)

    builder = A1AnchorBuilder(manager.builder_config)
    sample = {"ego_state": ego[0]}
    sample.update({k: v[0] for k, v in geometry.items()})
    offline = builder.build(sample)
    np.testing.assert_allclose(
        online["coarse_controls"][0].cpu().numpy(),
        offline["a1_coarse_controls"], atol=1e-6,
    )
    np.testing.assert_allclose(
        online["coarse_trajectories"][0].cpu().numpy(),
        offline["a1_coarse_trajectories"], atol=1e-6,
    )


def test_stored_anchors_do_not_replace_missing_scene_geometry():
    manager = _manager()
    with pytest.raises(KeyError, match='live road geometry'):
        manager.select({'ego_state': torch.zeros(1,14),
                        'coarse_controls': torch.zeros(1,10,40,2)})


def test_a1_online_fails_closed_without_live_road_geometry():
    manager = _manager()
    with pytest.raises(KeyError, match="live road geometry"):
        manager.select({
            "ego_state": torch.zeros((1, 14)),
            })


def test_control_diffusion_dispatches_a1_strategy_for_raw_scene():
    from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner

    model = ControlDiffusionPlanner(
        model_config={"num_modes": 10},
        control_config={
            "horizon_steps": 40, "dt": 0.1, "wheelbase_m": 5.5,
            "max_accel_mps2": 2.5, "max_steer_rad": 0.08,
        },
        execution_config={"model": "kinematic"},
        loss_config={"mode_selection": "goal_endpoint"},
        anchors_config={
            "strategy": "a1", "a1_config": "configs/anchors/a1.yaml",
            "num_selected": 10,
        },
    )
    assert model.anchor_manager.runtime_kind == "a1"
    features = {"ego_state": torch.tensor([[20.] + [0.]*13]),
                **{k: torch.as_tensor(v) for k,v in _live_geometry(batch=1).items()}}
    result = model._prepare_anchors(features)
    assert result["coarse_controls"].shape == (1, 10, 40, 2)


def test_control_runtime_injects_live_road_geometry_and_runs_a1_online():
    from highway_env.planner.diffusion.common.tensor_adapter import PlannerTensorAdapter
    from highway_env.planner.diffusion.inference.control import ControlDiffusionRuntime
    from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner

    model = ControlDiffusionPlanner(
        model_config={"num_modes": 10},
        control_config={
            "horizon_steps": 40, "dt": 0.1, "wheelbase_m": 5.5,
            "max_accel_mps2": 2.5, "max_steer_rad": 0.08,
        },
        execution_config={"model": "kinematic"},
        loss_config={"mode_selection": "goal_endpoint"},
        anchors_config={
            "strategy": "a1", "a1_config": "configs/anchors/a1.yaml",
            "num_selected": 10,
        },
    ).eval()
    runtime = ControlDiffusionRuntime.__new__(ControlDiffusionRuntime)
    runtime.device = torch.device("cpu")
    runtime.model = model
    runtime.adapter = PlannerTensorAdapter(model.scene_config, runtime.device)
    runtime.deterministic_seed = 0
    runtime.inference_steps = 2
    runtime.jtp_enabled = False
    runtime.jtp_projector = None
    runtime.jtp_config_path = None
    runtime.last_latency_ms = None
    runtime.last_diffusion_latency_ms = None
    runtime.last_jtp_latency_ms = 0.0

    features = {
        "ego_state": np.asarray([[20.0] + [0.0] * 13], np.float32),
        "agent_states": np.zeros((1, 16, 11), np.float32),
        "agent_valid_mask": np.zeros((1, 16), bool),
        "map_polylines": np.zeros((1, 8, 32, 10), np.float32),
        "map_valid_mask": np.zeros((1, 8), bool),
        "target_point": np.asarray([[60.0, 0.0]], np.float32),
        "target_lane_polyline": np.zeros((1, 32, 10), np.float32),
    }
    features["map_valid_mask"][0, 0] = True
    features["map_polylines"][0, 0, :, 0] = np.linspace(0, 100, 32)
    features["map_polylines"][0, 0, :, 4] = 3.7
    features["map_polylines"][0, 0, :, 6] = 1.0

    result = runtime.infer(features, a1_road_geometry=_live_geometry(batch=1))
    assert result["anchor_runtime_kind"] == "a1"
    assert result["control"].shape == (1, 40, 2)
    assert result["trajectory"].shape == (1, 8, 2)
    assert result["a1_valid_counts"].shape == (1,)
    assert int(result["a1_valid_counts"][0]) > 0


def test_a0_fixed_bank_is_loaded_and_rolled_out_online(tmp_path):
    import json
    from highway_env.planner.fixed_anchor_manager import FixedAnchorManager

    anchors = np.zeros((10, 40, 2), np.float32)
    metadata = {
        "implementation": "A0_FIXED_CONTROL_KMEANS_EXEMPLARS_V4",
        "num_anchors": 10,
        "source_split": "train",
        "cluster_space": "flattened_40_step_physical_control_sequence",
        "prototype_type": "nearest_expert_to_cluster_centroid",
        "medoid_selection_used": False,
        "trajectory_to_control_matching_used": False,
        "mode_valid_mask_used": False,
        "safety_scoring_used": False,
        "adaptive_selection_used": False,
    }
    bank = tmp_path / "a0_bank.npz"
    np.savez_compressed(
        bank, anchors_controls=anchors, cluster_centers_controls=anchors,
        control_scale=np.asarray([2.5, 0.08], np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    manager = FixedAnchorManager(
        {"bank_path": str(bank), "num_selected": 10}, num_modes=10,
    )
    result = manager.select({"ego_state": torch.tensor([[20.0] + [0.0] * 13])})
    assert manager.runtime_kind == "a0"
    assert result["coarse_controls"].shape == (1, 10, 40, 2)
    assert result["coarse_trajectories"].shape == (1, 10, 8, 2)


def test_a2_scorer_generates_and_scores_live_candidates(tmp_path):
    from highway_env.planner.anchor_scoring.model import AnchorSafetyScorer
    from highway_env.planner.anchor_scoring.manager import AdaptiveAnchorManager

    from highway_env.planner.adaptive_anchors import AdaptiveCandidateBuilder

    scorer = AnchorSafetyScorer({})
    checkpoint = tmp_path / "a2_scorer.pt"
    # Save the resolved runtime candidate configuration. The manager also
    # accepts older checkpoints that contain only the sparse YAML fragment.
    candidate_builder = AdaptiveCandidateBuilder({})
    candidate_config = dict(getattr(
        candidate_builder, "resolved_config",
        getattr(candidate_builder.base, "config", {}),
    ))
    torch.save({
        "architecture": "TRACE_Adaptive_Anchor_Safety_Scorer_V1",
        "model_config": {},
        "candidate_generation": candidate_config,
        "model_state": scorer.state_dict(),
    }, checkpoint)
    manager = AdaptiveAnchorManager({
        "scorer_checkpoint": str(checkpoint),
        "candidate_generation": {},
        "num_selected": 10,
        "top_r": 20,
    })
    features = {
        "ego_state": torch.tensor([[20.0] + [0.0] * 13]),
        "agent_states": torch.zeros((1, 16, 11)),
        "agent_valid_mask": torch.zeros((1, 16), dtype=torch.bool),
        "map_polylines": torch.zeros((1, 8, 32, 10)),
        "map_valid_mask": torch.zeros((1, 8), dtype=torch.bool),
        "target_point": torch.tensor([[60.0, 0.0]]),
        "target_lane_polyline": torch.zeros((1, 32, 10)),
    }
    for index, y in enumerate((0.0, 3.7, -3.7)):
        features["map_valid_mask"][0, index] = True
        features["map_polylines"][0, index, :, 0] = torch.linspace(0, 120, 32)
        features["map_polylines"][0, index, :, 1] = y
        features["map_polylines"][0, index, :, 3] = 1.0
        features["map_polylines"][0, index, :, 4] = 3.7
        features["map_polylines"][0, index, :, 5] = 33.0
    features["map_polylines"][0, 0, :, 6] = 1.0
    features["map_polylines"][0, 0, :, 9] = 1.0
    features["map_polylines"][0, 1, :, 7] = 1.0
    features["map_polylines"][0, 2, :, 8] = 1.0

    result = manager.select(features)
    assert manager.runtime_kind == "a2"
    assert result["coarse_controls"].shape == (1, 10, 40, 2)
    assert result["coarse_trajectories"].shape == (1, 10, 8, 2)
    assert manager.last_valid_counts == [36]
    assert manager.last_latency_ms >= 0


def test_a2_rejects_genuinely_different_candidate_generator(tmp_path):
    from highway_env.planner.anchor_scoring.model import AnchorSafetyScorer
    from highway_env.planner.anchor_scoring.manager import AdaptiveAnchorManager
    from highway_env.planner.adaptive_anchors import AdaptiveCandidateBuilder

    builder = AdaptiveCandidateBuilder({})
    # Compare resolved physical candidate-builder parameters. Do not assume
    # that the wrapper exposes a `resolved_config` attribute: in this project
    # the canonical resolved mapping is AllMergeAnchorBuilder.config.
    trained_config = dict(builder.base.config)
    scorer = AnchorSafetyScorer({})
    checkpoint = tmp_path / "a2_scorer_mismatch.pt"
    torch.save({
        "architecture": "TRACE_Adaptive_Anchor_Safety_Scorer_V1",
        "model_config": {},
        "candidate_generation": trained_config,
        "model_state": scorer.state_dict(),
    }, checkpoint)

    changed_config = dict(trained_config)
    assert "max_steer_rad" in changed_config
    changed_config["max_steer_rad"] = float(changed_config["max_steer_rad"]) + 0.01
    with pytest.raises(ValueError, match="candidate generator differs"):
        AdaptiveAnchorManager({
            "scorer_checkpoint": str(checkpoint),
            "candidate_generation": changed_config,
            "num_selected": 10,
            "top_r": 20,
        })
