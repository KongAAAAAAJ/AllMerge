"""Safety-aware TRACE diffusion checkpoint continuation regression tests."""
from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

if importlib.util.find_spec('gymnasium') is None:
    pkg = types.ModuleType('highway_env')
    pkg.__path__ = [str(Path(__file__).resolve().parents[1] / 'highway_env')]
    sys.modules.setdefault('highway_env', pkg)

from highway_env.planner.diffusion.training import control as training


class _DummyVehicle(nn.Module):
    def __init__(self):
        super().__init__()
        self.kind = 'koopman_k2'
        self.koopman_metadata = {
            'model_type': 'K2',
            'model_kwargs': {'history_steps': 5, 'latent_dim': 8},
            'normalization': {'state_mean': [0.0] * 5},
        }
        self.koopman = nn.Linear(2, 2)
        self.koopman.requires_grad_(False)


class _DummySafetyPlanner(nn.Module):
    def __init__(self, model_config, control_config, *, execution_config,
                 loss_config, anchors_config):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.vehicle_model = _DummyVehicle()
        self.execution_config = copy.deepcopy(execution_config)
        self.loss_config = copy.deepcopy(loss_config)
        self.anchors_config = copy.deepcopy(anchors_config)

    def _validate_objectives(self):
        required = (
            'trajectory', 'background_collision', 'intertruck_collision',
            'ttc', 'road_boundary', 'safety', 'ltr', 'fuel', 'progress',
        )
        if any(key not in self.loss_config for key in required):
            raise ValueError('Incomplete Safety-aware Loss configuration')


class _Dataset:
    def __getitem__(self, index):
        features = {'road_lane_geometry': torch.zeros(3, 40, 9)}
        targets = {
            'koopman_state_hist_si': torch.zeros(5, 5),
            'koopman_control_hist_si': torch.zeros(4, 2),
            'koopman_pose0_si': torch.zeros(3),
            'koopman_payload_kg': torch.tensor(3500.0),
            'koopman_road_mu': torch.tensor(0.6),
            'platoon_reference_poses': torch.zeros(2, 41, 3),
        }
        return features, targets, {}


class _Loader:
    def __init__(self):
        self.dataset = _Dataset()


def _config():
    return {
        'output_space': 'control',
        'model': {'d_model': 32, 'num_modes': 3},
        'control': {'horizon_steps': 40, 'dt': 0.1},
        'execution': {'model': 'koopman_k2', 'checkpoint': 'frozen_k2.pt'},
        'anchors': {'strategy': 'fixed', 'bank_sha256': 'fixed-bank'},
        'losses': {
            'mode_selection': 'goal_endpoint',
            'trajectory': {'weight': 1.0},
            'background_collision': {'weight': 1.0},
            'intertruck_collision': {'weight': 1.0},
            'ttc': {'weight': 1.0},
            'road_boundary': {'weight': 1.0},
            'safety': {'ttc_threshold_s': 1.0},
            'ltr': {'weight': 0.0},
            'fuel': {'weight': 0.0},
            'progress': {'weight': 0.0},
        },
        'training': {
            'batch_size': 3,
            'num_workers': 0,
            'seed': 0,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'grad_clip': 1.0,
            'patience': 0,
        },
    }


def _metrics(loss):
    values = {key: 0.0 for key in training.METRIC_KEYS}
    values['loss'] = float(loss)
    values['trajectory_loss'] = float(loss)
    return values


def test_safety_checkpoint_resume_restores_optimizer_history_and_best(
        tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'ControlDiffusionPlanner', _DummySafetyPlanner)
    monkeypatch.setattr(training, '_make_loader', lambda *args, **kwargs: _Loader())

    validation_losses = iter([0.5, 0.8, 0.9])

    def run_pass(model, loader, optimizer, device, max_batches=None, clip_norm=1.0):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            model.weight.square().backward()
            optimizer.step()
            return _metrics(1.0)
        return _metrics(next(validation_losses))

    monkeypatch.setattr(training, '_run_pass', run_pass)
    cfg = _config()
    original = tmp_path / 'original'
    continued = tmp_path / 'continued'

    training.train(
        cfg, root=tmp_path / 'dataset', output_dir=original,
        max_epochs=2, max_steps=1, device='cpu',
    )
    first_last = torch.load(original / 'last.pt', map_location='cpu', weights_only=True)
    first_best_bytes = (original / 'best.pt').read_bytes()

    assert first_last['architecture'] == training.SAFETY_ARCHITECTURE
    assert first_last['epoch'] == 2
    assert first_last['best_epoch'] == 1
    assert first_last['best_val_loss'] == pytest.approx(0.5)
    assert [row['epoch'] for row in first_last['history']] == [1, 2]

    training.train(
        cfg, root=tmp_path / 'dataset', output_dir=continued,
        max_epochs=3, max_steps=1, device='cpu',
        resume=original / 'last.pt',
    )
    resumed = torch.load(continued / 'last.pt', map_location='cpu', weights_only=True)

    assert resumed['epoch'] == 3
    assert resumed['best_epoch'] == 1
    assert resumed['best_val_loss'] == pytest.approx(0.5)
    assert [row['epoch'] for row in resumed['history']] == [1, 2, 3]
    assert max(int(state['step']) for state in resumed['optimizer_state']['state'].values()) == 3
    assert (continued / 'best.pt').read_bytes() == first_best_bytes
    assert len((continued / 'loss.csv').read_text(encoding='utf-8').splitlines()) == 4


def test_resume_rejects_changed_safety_loss_and_old_architecture(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'ControlDiffusionPlanner', _DummySafetyPlanner)
    monkeypatch.setattr(training, '_make_loader', lambda *args, **kwargs: _Loader())

    def run_pass(model, loader, optimizer, device, max_batches=None, clip_norm=1.0):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            model.weight.square().backward()
            optimizer.step()
        return _metrics(0.5)

    monkeypatch.setattr(training, '_run_pass', run_pass)
    cfg = _config()
    output = tmp_path / 'run'
    training.train(cfg, root=tmp_path / 'dataset', output_dir=output,
                   max_epochs=1, max_steps=1, device='cpu')

    changed = copy.deepcopy(cfg)
    changed['losses']['ttc']['weight'] = 2.0
    with pytest.raises(ValueError, match='loss_config differs'):
        training.train(
            changed, root=tmp_path / 'dataset', output_dir=tmp_path / 'changed',
            max_epochs=2, max_steps=1, device='cpu', resume=output / 'last.pt',
        )

    old = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
    old['architecture'] = 'TRACE_control_diffusion_D2'
    old_path = tmp_path / 'old.pt'
    torch.save(old, old_path)
    model = _DummySafetyPlanner(
        cfg['model'], cfg['control'], execution_config=cfg['execution'],
        loss_config=cfg['losses'], anchors_config=cfg['anchors'],
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg['training']['lr'],
    )
    with pytest.raises(ValueError, match='Safety-aware Diffusion Training V1'):
        training._restore_training_state(old_path, model, optimizer, cfg)


def test_resume_uses_total_epoch_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'ControlDiffusionPlanner', _DummySafetyPlanner)
    monkeypatch.setattr(training, '_make_loader', lambda *args, **kwargs: _Loader())

    def run_pass(model, loader, optimizer, device, max_batches=None, clip_norm=1.0):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            model.weight.square().backward()
            optimizer.step()
        return _metrics(0.5)

    monkeypatch.setattr(training, '_run_pass', run_pass)
    cfg = _config()
    output = tmp_path / 'run'
    training.train(cfg, root=tmp_path / 'dataset', output_dir=output,
                   max_epochs=1, max_steps=1, device='cpu')
    with pytest.raises(ValueError, match='total epoch limit'):
        training.train(
            cfg, root=tmp_path / 'dataset', output_dir=output,
            max_epochs=1, max_steps=1, device='cpu', resume=output / 'last.pt',
        )


def test_best_through_epoch_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'ControlDiffusionPlanner', _DummySafetyPlanner)
    monkeypatch.setattr(training, '_make_loader', lambda *args, **kwargs: _Loader())
    validation_losses = iter([0.5, 0.8, 0.3])

    def run_pass(model, loader, optimizer, device, max_batches=None, clip_norm=1.0):
        if optimizer is not None:
            with torch.no_grad():
                model.weight.add_(1.0)
            return _metrics(1.0)
        return _metrics(next(validation_losses))

    monkeypatch.setattr(training, '_run_pass', run_pass)
    output = tmp_path / 'snapshots'
    training.train(_config(), root=tmp_path / 'dataset', output_dir=output,
                   max_epochs=3, max_steps=1, device='cpu')
    snapshots = [torch.load(output / f'best_through_epoch_{epoch:04d}.pt',
                            map_location='cpu', weights_only=True)
                 for epoch in (1, 2, 3)]
    assert [saved['epoch'] for saved in snapshots] == [1, 1, 3]
    assert [saved['best_val_loss'] for saved in snapshots] == [0.5, 0.5, 0.3]
    assert [saved['model_state']['weight'].item() for saved in snapshots] == [2.0, 2.0, 4.0]
    assert (output / 'best_through_epoch_0001.pt').read_bytes() == (output / 'best_through_epoch_0002.pt').read_bytes()
    assert (output / 'best_through_epoch_0003.pt').read_bytes() == (output / 'best.pt').read_bytes()
    latest = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
    assert latest['epoch'] == 3
    assert latest['model_state']['weight'].item() == 4.0
