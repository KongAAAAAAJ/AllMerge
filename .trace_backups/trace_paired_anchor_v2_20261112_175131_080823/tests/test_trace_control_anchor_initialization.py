"""Regression checks for shared trajectory/control anchor-noise initialization."""
from __future__ import annotations

# The module-level planners do not need an active gym environment; allow this
# focused test to run on research machines with torch but without gymnasium.
import importlib.util
import sys
import types
from pathlib import Path

if importlib.util.find_spec('gymnasium') is None and 'highway_env' not in sys.modules:
    base = Path(__file__).resolve().parents[1]
    for name, path in (
        ('highway_env', base / 'highway_env'),
        ('highway_env.planner', base / 'highway_env' / 'planner'),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

import torch

from highway_env.planner.diffusion.models.control import ControlDiffusionPlanner
from highway_env.planner.diffusion.common.anchor_controls import xy_anchors_to_controls


def _coarse(batch=1, modes=2, vx=20.):
    t = torch.arange(1, 9, dtype=torch.float32) * .5
    straight = torch.stack((vx * t, torch.zeros_like(t)), dim=-1)
    return straight[None, None].expand(batch, modes, -1, -1).clone()


def _planner():
    return ControlDiffusionPlanner(
        {'d_model': 32, 'd_ffn': 64, 'num_heads': 4,
         'num_scene_layers': 1, 'num_denoiser_layers': 1,
         'dropout': 0., 'num_modes': 2},
        {'horizon_steps': 40, 'dt': .1, 'inference_steps': 2,
         'inference_start_timestep': 8, 'inference_noise_scale': 1.},
        execution_config={'model': 'kinematic'},
    )


def _features(planner):
    c = planner.scene_config
    f = {
        'ego_state': torch.zeros(1, c.ego_dim),
        'agent_states': torch.zeros(1, c.max_agents, c.agent_dim),
        'agent_valid_mask': torch.zeros(1, c.max_agents, dtype=torch.bool),
        'map_polylines': torch.zeros(1, c.max_map_polylines, c.map_points, c.map_dim),
        'map_valid_mask': torch.zeros(1, c.max_map_polylines, dtype=torch.bool),
        'target_point': torch.tensor([[80., 0.]]),
        'target_lane_polyline': torch.zeros(1, c.map_points, c.map_dim),
        'coarse_trajectories': _coarse(),
        'mode_valid_mask': torch.ones(1, 2, dtype=torch.bool),
    }
    f['ego_state'][:, 0] = 20.
    f['coarse_trajectories'][:, 1, :, 1] = torch.linspace(0, 2, 8)
    return f


def test_straight_constant_speed_inverse_control_is_zero():
    anchor = _coarse()
    u = xy_anchors_to_controls(anchor, torch.tensor([20.]))
    assert u.shape == (1, 2, 40, 2)
    torch.testing.assert_close(u, torch.zeros_like(u), atol=2e-4, rtol=0)


def test_corner_cases_are_bounded_and_finite():
    a = _coarse()
    a[:, 0, :, 1] = torch.linspace(0., 6., 8)
    a[:, 1, :, 0] = torch.tensor([9., 16., 21., 24., 26., 26., 26., 26.])
    u = xy_anchors_to_controls(a, torch.tensor([20.]))
    assert torch.isfinite(u).all()
    assert u[..., 0].abs().amax() <= 2.5 + 1e-6
    assert u[..., 1].abs().amax() <= .08 + 1e-6
    assert u[0, 0, :, 1].abs().max() > 1e-5


def test_control_uses_exact_anchor_noising_and_t8_to_t0():
    torch.manual_seed(42)
    m = _planner().eval()
    features = _features(m)
    recorded = {}
    original_add = m.schedule.add_noise
    original_step = m.schedule.step_predict_x0

    def add_noise(clean, noise, timestep):
        recorded['clean'] = clean.detach().clone()
        recorded['noise'] = noise.detach().clone()
        recorded['timestep'] = timestep.detach().clone()
        return original_add(clean, noise, timestep)

    def step(sample, x0, t, next_t):
        recorded.setdefault('schedule', []).append((t, next_t))
        return original_step(sample, x0, t, next_t)

    m.schedule.add_noise = add_noise
    m.schedule.step_predict_x0 = step
    g = torch.Generator().manual_seed(5)
    out = m._sample_controls(features, generator=g)
    torch.testing.assert_close(recorded['clean'], m._anchor_controls(features) / m.control_limits)
    assert recorded['timestep'].tolist() == [8]
    assert recorded['schedule'] == [(8, 0), (0, None)]
    assert recorded['clean'][0, 0].abs().amax() < 1e-4  # constant-speed straight
    assert recorded['clean'][0, 1].abs().amax() > 1e-4  # second anchor affects seed
    assert out.shape == (1, 2, 40, 2)
    assert torch.isfinite(out).all()


def test_control_train_inference_and_backward_smoke():
    torch.manual_seed(9)
    m = _planner().train()
    f = _features(m)
    dense_t = torch.arange(1, 41, dtype=torch.float32) * .1
    dense = torch.stack((20. * dense_t, torch.zeros_like(dense_t)), dim=-1)[None]
    out = m.forward_train(f, dense, generator=torch.Generator().manual_seed(6))
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert m.denoiser.control_head[-1].weight.grad is not None
    assert torch.isfinite(m.denoiser.control_head[-1].weight.grad).all()
    with torch.no_grad():
        inference = m.eval().infer_multimodal(f, generator=torch.Generator().manual_seed(6))
    assert inference['control_candidates'].shape == (1, 2, 40, 2)
    assert inference['trajectory'].shape == (1, 8, 2)
    assert torch.isfinite(inference['trajectory']).all()


def test_absent_mode_does_not_generate_reverse_control_seeds():
    m = _planner()
    features = _features(m)
    features['coarse_trajectories'][:, 1] = 0
    features['mode_valid_mask'][:, 1] = False
    controls = m._anchor_controls(features)
    torch.testing.assert_close(controls[:, 1], controls[:, 0])
