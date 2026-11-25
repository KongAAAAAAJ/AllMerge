"""Focused tests for TRACE Anchor-assigned multimodal training."""
from __future__ import annotations

import importlib.machinery
import sys
import types

import numpy as np
import pytest
import torch

# Keep these planner tests runnable when the optional simulator package is absent.
try:
    import gymnasium  # noqa: F401
except ImportError:
    gym = types.ModuleType('gymnasium')
    gym.__spec__ = importlib.machinery.ModuleSpec('gymnasium', loader=None)
    envs = types.ModuleType('gymnasium.envs')
    envs.__spec__ = importlib.machinery.ModuleSpec('gymnasium.envs', loader=None)
    registration = types.ModuleType('gymnasium.envs.registration')
    registration.__spec__ = importlib.machinery.ModuleSpec(
        'gymnasium.envs.registration', loader=None
    )
    registration.register = lambda *args, **kwargs: None
    gym.envs = envs
    envs.registration = registration
    sys.modules.setdefault('gymnasium', gym)
    sys.modules.setdefault('gymnasium.envs', envs)
    sys.modules.setdefault('gymnasium.envs.registration', registration)

from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner
from highway_env.vehicle.koopman.models.conditional_model import ConditionalKoopman


SMALL = dict(d_model=32, d_ffn=64, num_heads=4, num_modes=10,
             num_scene_layers=0, num_denoiser_layers=1, dropout=0.0)
CTRL = dict(horizon_steps=40, dt=0.1, wheelbase_m=5.5,
            max_accel_mps2=2.5, max_steer_rad=0.08,
            num_train_timesteps=12, inference_steps=2,
            inference_start_timestep=8, inference_noise_scale=1.0)
FUEL = dict(empty_mass_kg=4455., rolling_resistance=.007,
            drag_coefficient=.6, frontal_area_m2=6.8,
            drivetrain_efficiency=.86, alpha0=.00156,
            alpha1=.00008, alpha2=1e-8,
            air_density_kg_m3=1.225, gravity_m_s2=9.81)


def _live_geometry(batch=1):
    lanes, points = 3, 200
    x = np.linspace(-30.0, 250.0, points, dtype=np.float32)
    geom = np.zeros((batch, lanes, points, 9), np.float32)
    point_mask = np.ones((batch, lanes, points), bool)
    lane_mask = np.ones((batch, lanes), bool)
    connections = np.zeros((batch, lanes, lanes), bool)
    meta = np.zeros((batch, lanes, 7), np.float32)
    for b in range(batch):
        for lane_i, y in enumerate((-3.7, 0.0, 3.7)):
            geom[b, lane_i, :, 0] = x
            geom[b, lane_i, :, 1] = y
            geom[b, lane_i, :, 2] = x
            geom[b, lane_i, :, 3] = y + 1.85
            geom[b, lane_i, :, 4] = x
            geom[b, lane_i, :, 5] = y - 1.85
            geom[b, lane_i, :, 7] = 3.7
            geom[b, lane_i, :, 8] = x + 30.0
            meta[b, lane_i, 0] = 33.0
            meta[b, lane_i, 1] = 300.0
            meta[b, lane_i, 2] = 30.0
        connections[b, 1, :] = True
        connections[b, :, 1] = True
    return {
        'road_lane_geometry': geom,
        'road_lane_point_mask': point_mask,
        'road_lane_mask': lane_mask,
        'road_current_lane_slot': np.ones((batch,), np.int16),
        'road_target_lane_slot': np.ones((batch,), np.int16),
        'road_lane_connections': connections,
        'road_lane_meta': meta,
    }


def _loss_config():
    return {
        'mode_selection': 'goal_endpoint',
        'training_scheme': 'anchor_consistent',
        'trajectory': {'weight': 1.0},
        'background_collision': {'weight': 1.0},
        'intertruck_collision': {'weight': 1.0},
        'ttc': {'weight': 1.0},
        'road_boundary': {'weight': 1.0},
        'safety': {
            'vehicle_length_m': 8.0, 'vehicle_width_m': 2.6,
            'collision_margin_m': .2, 'ttc_threshold_s': 1.0,
            'softplus_temperature_m': .2, 'collision_distance_scale_m': 1.0,
            'ttc_distance_scale_m': 5.0, 'road_distance_scale_m': 1.0,
        },
        'anchor_consistency': {'weight': .1},
        'ltr': {'weight': 0.0, 'limit': .6},
        'fuel': {'weight': 0.0, 'relative_weight': .1,
                 'relative_epsilon': 1e-8, 'parameters': FUEL},
        'progress': {'weight': 0.0, 'minimum_progress_m': 60.0},
    }


def _make_planner():
    norm = dict(state_mean=[0.] * 5, state_std=[1.] * 5,
                control_mean=[0.] * 2, control_std=[1.] * 2)
    cond = dict(condition_mean=[3500., .6], condition_std=[1000., .1])
    metadata = {
        'model_type': 'K2',
        'model_kwargs': dict(history_steps=5, latent_dim=8, hidden_dim=16),
        'normalization': norm,
        'condition_normalization': cond,
        'condition_hidden': 12,
    }
    model = ControlDiffusionPlanner(
        SMALL, CTRL,
        execution_config={'model': 'koopman_k2'},
        loss_config=_loss_config(), koopman_metadata=metadata,
        anchors_config={'strategy': 'a1', 'a1_config': 'configs/anchors/a1.yaml'},
    )
    model._validate_objectives()
    return model


def _features():
    features = {
        'ego_state': torch.zeros((1, 14), dtype=torch.float32),
        'agent_states': torch.zeros((1, 16, 11), dtype=torch.float32),
        'agent_valid_mask': torch.zeros((1, 16), dtype=torch.bool),
        'map_polylines': torch.zeros((1, 8, 32, 10), dtype=torch.float32),
        'map_valid_mask': torch.zeros((1, 8), dtype=torch.bool),
        'target_point': torch.tensor([[40.0, 0.0]], dtype=torch.float32),
        'target_lane_polyline': torch.zeros((1, 32, 10), dtype=torch.float32),
    }
    features['ego_state'][:, 0] = 20.0
    features['target_lane_polyline'][:, :, 0] = torch.linspace(-20, 250, 32)
    features['target_lane_polyline'][:, :, 1] = 0.0
    features['target_lane_polyline'][:, :, 7] = 3.7
    for key, value in _live_geometry().items():
        features[key] = torch.as_tensor(value)
    return features


def test_anchor_expert_assignment_uses_full_horizon_k2_ade_and_masks_invalid():
    batch, modes, horizon = 2, 10, 40
    anchor_poses = torch.zeros((batch, modes, horizon + 1, 3))
    target = torch.zeros((batch, horizon, 2))
    target[:, :, 0] = torch.arange(1, horizon + 1)
    for mode in range(modes):
        anchor_poses[:, mode, 1:, 0] = torch.arange(1, horizon + 1)
        anchor_poses[:, mode, 1:, 1] = float(mode)
    target[0, :, 1] = 7.0
    target[1, :, 1] = 3.0
    valid = torch.ones((batch, modes), dtype=torch.bool)
    valid[0, 7] = False
    valid[1, 3] = False
    selected, ade, per_mode = ControlDiffusionPlanner._select_anchor_expert_mode(
        anchor_poses, target, valid,
    )
    assert selected.tolist() == [6, 2]
    assert ade.tolist() == pytest.approx([1.0, 1.0])
    assert torch.isinf(per_mode[0, 7]) and torch.isinf(per_mode[1, 3])


def test_anchor_assigned_training_uses_anchor_gt_mode_and_auxiliary_losses_for_all_modes():
    torch.manual_seed(5)
    model = _make_planner()
    features = _features()
    modes = 10
    anchor_controls = torch.zeros((1, modes, 40, 2), dtype=torch.float32)
    generated_controls = torch.full((1, modes, 40, 2), 0.2, dtype=torch.float32,
                                    requires_grad=True)
    features['coarse_controls'] = anchor_controls
    features['coarse_trajectories'] = torch.zeros((1, modes, 8, 2), dtype=torch.float32)
    features['coarse_trajectories'][:, :, :, 0] = torch.arange(5, 41, 5)
    features['coarse_trajectories'][0, :, :, 1] = torch.arange(modes).float()[:, None]
    features['mode_valid_mask'] = torch.ones((1, modes), dtype=torch.bool)

    # The expert follows Anchor mode 8; the stubbed shared candidate
    # selector chooses mode 0 so the test isolates expert assignment.
    def prepare(self, value):
        return value

    def sample(self, value, **kwargs):
        return generated_controls

    def predictions(self, controls, value, context=None):
        is_anchor = controls is anchor_controls
        b, m, h, _ = controls.shape
        t = torch.arange(1, h + 1, dtype=controls.dtype, device=controls.device)
        x = t.view(1, 1, h).expand(b, m, h)
        mode_y = torch.arange(m, dtype=controls.dtype, device=controls.device).view(1, m)
        offset = controls[:, :, 0, 0]
        y0 = mode_y if is_anchor else mode_y + offset
        y = y0[:, :, None].expand(b, m, h)
        future = torch.stack((x, y), dim=-1)
        origin = torch.zeros((b, m, 1, 2), dtype=controls.dtype, device=controls.device)
        xy = torch.cat((origin, future), dim=2)
        yaw = torch.zeros((b, m, h + 1, 1), dtype=controls.dtype, device=controls.device)
        poses = torch.cat((xy, yaw), dim=-1)
        speed0 = 20.0 + offset
        speeds = speed0[:, :, None].expand(b, m, h + 1)
        return {'poses': poses, 'speeds': speeds, 'states': None, 'ltr': None}

    model._prepare_anchors = types.MethodType(prepare, model)
    model._sample_controls = types.MethodType(sample, model)
    model._all_predictions = types.MethodType(predictions, model)

    def select_candidates(self, controls, prediction, value, context=None):
        selected = torch.zeros((controls.shape[0],), dtype=torch.long, device=controls.device)
        scores = torch.zeros(controls.shape[:2], dtype=controls.dtype, device=controls.device)
        return {'selected_mode': selected, 'candidate_scores': scores}

    model._select_candidate_mode = types.MethodType(select_candidates, model)

    expert = torch.zeros((1, 40, 2), dtype=torch.float32)
    expert[0, :, 0] = torch.arange(1, 41, dtype=torch.float32)
    expert[0, :, 1] = 8.0
    platoon = torch.zeros((1, 2, 41, 3), dtype=torch.float32)
    platoon[:, :, :, 0] = 500.0
    context = {'koopman_payload_kg': torch.tensor([3500.0])}

    result = model.forward_train(
        features, expert, {'platoon_reference_poses': platoon}, context=context,
    )
    assert result['expert_mode'].tolist() == [8]
    assert result['goal_selected_mode'].tolist() == [0]
    assert result['selected_mode'].tolist() == [0]  # inference selector retained
    assert result['expert_goal_mode_disagreement'].item() == pytest.approx(1.0)
    assert result['expert_anchor_ade'].item() == pytest.approx(0.0, abs=1e-6)
    assert result['trajectory_loss'].item() == pytest.approx(0.04, abs=1e-4)
    assert result['goal_selected_ade'].item() > 7.0
    assert result['safety_mode_count'].item() == pytest.approx(10.0)
    assert result['anchor_consistency_loss'].item() == pytest.approx(0.04, abs=1e-4)
    assert torch.isfinite(result['loss'])

    result['loss'].backward()
    per_mode_grad = generated_controls.grad[:, :, :, 0].abs().sum(dim=-1)
    assert torch.all(per_mode_grad > 0), per_mode_grad



def test_anchor_consistent_real_k2_execution_is_frozen_and_differentiable(tmp_path):
    norm = dict(state_mean=[0.] * 5, state_std=[1.] * 5,
                control_mean=[0.] * 2, control_std=[1.] * 2)
    cond = dict(condition_mean=[3500., .6], condition_std=[1000., .1])
    kwargs = dict(history_steps=5, latent_dim=8, hidden_dim=16)
    k2 = ConditionalKoopman('K2', condition_hidden=12, **norm, **cond, **kwargs)
    checkpoint = tmp_path / 'random_k2.pt'
    torch.save({
        'model_type': 'K2', 'model_kwargs': kwargs,
        'normalization': norm, 'condition_normalization': cond,
        'condition_hidden': 12, 'model_state': k2.state_dict(),
    }, checkpoint)
    model = ControlDiffusionPlanner(
        SMALL, CTRL,
        execution_config={'model': 'koopman_k2', 'checkpoint': str(checkpoint)},
        loss_config=_loss_config(),
        anchors_config={'strategy': 'a1', 'a1_config': 'configs/anchors/a1.yaml'},
    )
    model._validate_objectives()
    features = _features()
    expert = torch.zeros((1, 40, 2), dtype=torch.float32)
    expert[0, :, 0] = torch.arange(1, 41, dtype=torch.float32) * 2.0
    platoon = torch.zeros((1, 2, 41, 3), dtype=torch.float32)
    platoon[:, :, :, 0] = 500.0
    state = torch.zeros((1, 5, 5), dtype=torch.float32)
    state[:, :, 0] = 20.0
    context = {
        'koopman_state_hist_si': state,
        'koopman_control_hist_si': torch.zeros((1, 4, 2), dtype=torch.float32),
        'koopman_pose0_si': torch.zeros((1, 3), dtype=torch.float32),
        'koopman_payload_kg': torch.tensor([3500.0]),
        'koopman_road_mu': torch.tensor([.6]),
    }
    result = model.forward_train(
        features, expert, {'platoon_reference_poses': platoon}, context=context,
        generator=torch.Generator(device='cpu').manual_seed(21),
    )
    for key in ('loss', 'trajectory_loss', 'anchor_consistency_loss',
                'anchor_execution_gap_m', 'fuel_relative_loss',
                'fuel_relative_ratio', 'expert_anchor_ade',
                'goal_selected_ade', 'background_collision_loss',
                'intertruck_collision_loss', 'ttc_loss', 'road_boundary_loss'):
        assert torch.isfinite(result[key]), key
    assert result['safety_mode_count'].item() == pytest.approx(10.0)
    assert result['anchor_consistency_loss'].item() >= 0.0
    assert all(parameter.grad is None for parameter in model.vehicle_model.koopman.parameters())
    result['loss'].backward()
    grads = [parameter.grad for parameter in model.denoiser.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert sum(float(grad.abs().sum()) for grad in grads) > 0.0
    assert all(parameter.grad is None for parameter in model.vehicle_model.koopman.parameters())

def test_relative_fuel_objective_is_zero_at_anchor_and_negative_when_improved():
    model = _make_planner()
    anchor = torch.linspace(20.0, 22.0, 41).view(1, 1, 41).expand(1, 3, 41).clone()
    valid = torch.tensor([[True, True, False]])
    payload = torch.tensor([3500.0])
    parity = model._relative_fuel_objective(anchor, anchor, payload, valid)
    assert parity['loss'].item() == pytest.approx(0.0, abs=1e-7)
    assert parity['ratio'].item() == pytest.approx(1.0, abs=1e-6)
    assert parity['improvement_pct'].item() == pytest.approx(0.0, abs=1e-5)

    generated = (anchor - 1.0).detach().requires_grad_(True)
    improved = model._relative_fuel_objective(generated, anchor, payload, valid)
    assert improved['loss'].item() < 0.0
    assert improved['improvement_pct'].item() > 0.0
    improved['loss'].backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()
    assert torch.all(generated.grad[:, 2] == 0)  # invalid mode is masked


def test_yaml_switch_preserves_supported_shared_scheme_and_anchor_assigned_config():
    from highway_env.planner.diffusion.config.experiment import load_mode_yaml

    baseline = load_mode_yaml('configs/train/diffusion.yaml')
    anchor = load_mode_yaml('configs/train/ablation/diffusion_anchor_consistent.yaml')
    assert baseline['losses'].get('training_scheme', 'goal_endpoint') in ('goal_endpoint', 'anchor_consistent')
    assert anchor['losses']['training_scheme'] == 'anchor_consistent'
    assert baseline['model'] == anchor['model']
    assert baseline['control'] == anchor['control']
    assert baseline['execution'] == anchor['execution']
    assert baseline['anchors'] == anchor['anchors']
    assert baseline['training']['output_dir'] != anchor['training']['output_dir']
    assert 'selected_mode_scale' not in anchor['losses']['anchor_consistency']
