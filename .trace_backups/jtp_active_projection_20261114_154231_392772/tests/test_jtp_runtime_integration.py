from __future__ import annotations

import math
import numpy as np
import torch

from highway_env.planner.diffusion.inference.jtp_context import LiveK2ContextBuilder
from highway_env.planner.diffusion.inference.control import ControlDiffusionRuntime
from highway_env.planner.diffusion.projection.types import JTPResult


class _Lane:
    def __init__(self, y, width=4.0, length=500.0):
        self.y = float(y)
        self.width = float(width)
        self.length = float(length)
        self.forbidden = False

    def position(self, longitudinal, lateral):
        return np.asarray([float(longitudinal), self.y + float(lateral)], dtype=np.float64)

    def local_coordinates(self, position):
        return float(position[0]), float(position[1] - self.y)

    def heading_at(self, longitudinal):
        return 0.0

    def width_at(self, longitudinal):
        return self.width


class _Network:
    def __init__(self):
        self._lanes = {
            ("a", "b", 0): _Lane(0.0),
            ("a", "b", 1): _Lane(4.0),
            ("a", "b", 2): _Lane(8.0),
        }

    def lanes_dict(self):
        return dict(self._lanes)

    def get_lane(self, index):
        return self._lanes[tuple(index)]


class _Vehicle:
    def __init__(self, x, y, *, controlled=False):
        self.position = np.asarray([x, y], dtype=np.float64)
        self.heading = 0.0
        self.lane_index = ("a", "b", 1)
        self.action = {"acceleration": 0.0, "steering": 0.0}
        self.export_array = np.zeros(15, dtype=np.float64)
        self.export_array[2] = 36.0
        self.export_array[3] = 3.6
        self.export_array[6] = 1.0
        self.export_array[7] = 0.5
        self.export_array[8] = 2.0
        self.export_array[9] = 0.2
        self.export_array[11] = 3.0
        self.LENGTH = 16.5 if controlled else 4.5
        self.WIDTH = 2.55 if controlled else 1.8
        self.speed = 10.0


class _FeatureBuilder:
    @staticmethod
    def _vehicle_world_motion(vehicle):
        return np.asarray([vehicle.speed, 0.0]), np.zeros(2)


class _Road:
    def __init__(self, vehicles):
        self.network = _Network()
        self.vehicles = list(vehicles)


class _Env:
    def __init__(self, ego, background):
        self.controlled_vehicles = [ego]
        self.road = _Road([ego, background])
        self.config = {"TruckSim": {}}
        self.time = 0.1


def _config():
    return {
        "execution": {"dt_s": 0.1, "horizon_steps": 40, "control_steps": 40},
        "runtime": {
            "context": {
                "sample_dt_s": 0.1,
                "max_gap_tolerance_s": 0.02,
                "history_state_steps": 5,
                "payload_kg": 1200.0,
                "road_mu": 0.7,
            },
            "road": {"sample_points": 32, "backward_range_m": 10.0, "forward_range_m": 150.0},
            "external_traffic": {"enabled": True, "max_distance_m": 120.0, "max_actors": 8},
        },
    }


def test_live_context_warmup_history_road_and_external_traffic():
    ego = _Vehicle(10.0, 4.0, controlled=True)
    background = _Vehicle(35.0, 8.0, controlled=False)
    env = _Env(ego, background)
    builder = LiveK2ContextBuilder(_config())
    feature_builder = _FeatureBuilder()

    live = None
    for step in range(5):
        env.time = 0.1 * (step + 1)
        live = builder.build(
            env, feature_builder, [("a", "b", 2)], include_projection_inputs=True)
        if step < 4:
            assert live["ready"] is False
            assert live["status"] == "history_warmup"
        # This command is applied over the next 0.1-s interval and is observed
        # as vehicle.action at the next call.
        ego.action = {"acceleration": 0.1 * step, "steering": 0.01 * step}

    assert live["ready"] is True
    context = live["context"]
    assert context["koopman_state_hist_si"].shape == (1, 5, 5)
    assert context["koopman_control_hist_si"].shape == (1, 4, 2)
    assert context["koopman_pose0_si"].shape == (1, 3)
    assert context["koopman_payload_kg"].tolist() == [1200.0]
    np.testing.assert_allclose(context["koopman_road_mu"], [0.7], atol=1e-6)
    np.testing.assert_allclose(
        context["koopman_control_hist_si"][0, :, 0], [0.0, 0.1, 0.2, 0.3], atol=1e-6)

    roads = live["road_geometry"]
    assert len(roads) == 1
    center, left, right = roads[0].tensors(device="cpu", dtype=torch.float32)
    assert center.shape == (1, 32, 2)
    # Three 4-m lanes form an 12-m carriageway, hence 6-m half-width.
    torch.testing.assert_close(left, torch.full_like(left, 6.0))
    torch.testing.assert_close(right, torch.full_like(right, 6.0))

    external = live["external_traffic"]
    assert external is not None
    poses, offsets, radii = external.tensors(
        device="cpu", dtype=torch.float32, expected_steps=41)
    assert poses.shape == (1, 41, 3)
    assert offsets.shape == (1, 3)
    assert radii.shape == (1, 3)
    torch.testing.assert_close(poses[0, 10, 0], torch.tensor(45.0))


def test_live_context_resets_after_non_ten_hz_gap():
    ego = _Vehicle(10.0, 4.0, controlled=True)
    background = _Vehicle(35.0, 8.0)
    env = _Env(ego, background)
    builder = LiveK2ContextBuilder(_config())
    feature_builder = _FeatureBuilder()
    for step in range(5):
        env.time = 0.1 * (step + 1)
        live = builder.build(env, feature_builder, [ego.lane_index], include_projection_inputs=False)
        ego.action = {"acceleration": 0.0, "steering": 0.0}
    assert live["ready"]

    env.time += 0.35
    reset = builder.build(env, feature_builder, [ego.lane_index], include_projection_inputs=False)
    assert reset["ready"] is False
    assert reset["history_state_counts"] == [1]
    assert reset["history_control_counts"] == [0]


def test_jtp_result_is_converted_from_world_to_ego_local_and_replaces_controls():
    batch, horizon = 1, 40
    pose0 = torch.tensor([[100.0, 200.0, math.pi / 2]], dtype=torch.float32)
    world_pose = torch.zeros((batch, horizon + 1, 3), dtype=torch.float32)
    world_pose[..., 0] = 100.0
    world_pose[..., 1] = 200.0 + torch.arange(horizon + 1) * 0.1
    world_pose[..., 2] = math.pi / 2
    states = torch.zeros((batch, horizon + 1, 5), dtype=torch.float32)
    states[..., 0] = 10.0
    ltr = torch.zeros((batch, horizon + 1), dtype=torch.float32)
    controls = torch.ones((batch, horizon, 2), dtype=torch.float32)
    result = JTPResult(
        controls=controls,
        trajectories=world_pose,
        states=states,
        ltr=ltr,
        success=True,
        status="converged_feasible",
        iterations=2,
        diagnostics={},
    )
    prediction = {
        "control": torch.zeros_like(controls),
        "pose": torch.zeros_like(world_pose),
        "trajectory_dense": torch.zeros((batch, horizon, 2)),
        "trajectory": torch.zeros((batch, 8, 2)),
    }
    updated = ControlDiffusionRuntime._apply_jtp_result(
        prediction, result,
        {"koopman_pose0_si": pose0}, dt_s=0.1)
    torch.testing.assert_close(updated["control"], controls)
    torch.testing.assert_close(updated["trajectory_dense"][0, 9, 0], torch.tensor(1.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(updated["trajectory_dense"][0, 9, 1], torch.tensor(0.0), atol=1e-5, rtol=0)
    assert updated["trajectory"].shape == (batch, 8, 2)


class _FakeAdapter:
    def to_torch(self, values):
        return {key: torch.as_tensor(value) for key, value in values.items()
                if isinstance(value, (np.ndarray, list, tuple))}


class _FakeVehicleModel:
    kind = "koopman_k2"


class _FakeControlConfig:
    dt = 0.1
    horizon_steps = 40


class _FakeDiffusionModel:
    vehicle_model = _FakeVehicleModel()
    ctrl_cfg = _FakeControlConfig()

    def infer_multimodal(self, features, generator=None, num_steps=None, context=None):
        batch = 1
        controls = torch.zeros((batch, 40, 2), dtype=torch.float32)
        pose = torch.zeros((batch, 41, 3), dtype=torch.float32)
        states = torch.zeros((batch, 41, 5), dtype=torch.float32)
        states[..., 0] = 10.0
        return {
            "control": controls,
            "pose": pose,
            "trajectory_dense": torch.zeros((batch, 40, 2)),
            "trajectory": torch.zeros((batch, 8, 2)),
            "speeds": states[..., 0],
            "states": states,
            "ltr": torch.zeros((batch, 41)),
        }


class _FakeProjector:
    def __init__(self, success=True):
        self.success = success

    def project(self, nominal_controls, context, road_geometry, external_traffic):
        batch = nominal_controls.shape[0]
        controls = nominal_controls + 0.01
        pose0 = context["koopman_pose0_si"]
        poses = pose0[:, None, :].repeat(1, 41, 1)
        poses[:, :, 1] += torch.arange(41, dtype=torch.float32)[None] * 0.1
        states = torch.zeros((batch, 41, 5), dtype=torch.float32)
        states[..., 0] = 10.0
        return JTPResult(
            controls=controls,
            trajectories=poses,
            states=states,
            ltr=torch.zeros((batch, 41)),
            success=self.success,
            status="converged_feasible" if self.success else "stalled_infeasible",
            iterations=3,
            diagnostics={"final_metrics": {"max_violation": 0.0 if self.success else 0.2}},
        )


def _fake_runtime(success=True):
    runtime = ControlDiffusionRuntime.__new__(ControlDiffusionRuntime)
    runtime.device = torch.device("cpu")
    runtime.model = _FakeDiffusionModel()
    runtime.adapter = _FakeAdapter()
    runtime.deterministic_seed = None
    runtime.inference_steps = 2
    runtime.jtp_enabled = True
    runtime.jtp_projector = _FakeProjector(success=success)
    runtime.last_latency_ms = None
    runtime.last_diffusion_latency_ms = None
    runtime.last_jtp_latency_ms = 0.0
    return runtime


def _fake_k2_features():
    return {
        "ego_state": np.zeros((1, 14), dtype=np.float32),
        "koopman_state_hist_si": np.zeros((1, 5, 5), dtype=np.float32),
        "koopman_control_hist_si": np.zeros((1, 4, 2), dtype=np.float32),
        "koopman_pose0_si": np.asarray([[100.0, 200.0, math.pi / 2]], dtype=np.float32),
        "koopman_payload_kg": np.asarray([1200.0], dtype=np.float32),
        "koopman_road_mu": np.asarray([0.7], dtype=np.float32),
    }


def test_control_runtime_calls_jtp_and_publishes_projected_control():
    runtime = _fake_runtime(success=True)
    output = runtime.infer(
        _fake_k2_features(),
        jtp_inputs={"ready": True, "road_geometry": [object()], "external_traffic": None},
    )
    assert bool(output["jtp_success"])
    assert output["jtp_status"] == "converged_feasible"
    assert output["control"].shape == (1, 40, 2)
    np.testing.assert_allclose(output["control"][0, 0], [0.01, 0.01], atol=1e-6)
    # World +Y is ego-local +X for a pi/2 initial heading.
    np.testing.assert_allclose(output["trajectory_dense"][0, 9], [1.0, 0.0], atol=1e-5)
    assert output["jtp_max_violation"] == np.float32(0.0)


def test_control_runtime_keeps_nominal_output_when_jtp_fails():
    runtime = _fake_runtime(success=False)
    output = runtime.infer(
        _fake_k2_features(),
        jtp_inputs={"ready": True, "road_geometry": [object()], "external_traffic": None},
    )
    assert not bool(output["jtp_success"])
    assert output["jtp_status"] == "stalled_infeasible"
    np.testing.assert_allclose(output["control"], output["control_nominal"], atol=0.0)
    np.testing.assert_allclose(output["trajectory_dense"], output["trajectory_dense_nominal"], atol=0.0)
