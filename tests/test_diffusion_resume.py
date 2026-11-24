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


def test_auto_resume_fresh_processes_and_frozen_parameters(tmp_path, monkeypatch):
    import json
    import yaml
    from scripts.training import run_diffusion as entry
    cfg = _config()
    cfg['training']['patience'] = 15
    cfg['training']['max_epochs'] = 60
    config_path = tmp_path / 'experiment.yaml'
    config_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    child = tmp_path / 'fake_training.py'
    child.write_text("""import argparse, json, sys
from pathlib import Path
import torch
p=argparse.ArgumentParser()
p.add_argument('--_auto-worker')
p.add_argument('--resume')
a=p.parse_args()
run=json.loads(Path(a._auto_worker).read_text(encoding='utf-8'))
out=Path(run['train_args']['output_dir'])
seen_path=out/'seen.json'
seen=json.loads(seen_path.read_text()) if seen_path.exists() else []
seen.append({'resume':a.resume, 'config':run['config'], 'args':run['train_args']})
seen_path.write_text(json.dumps(seen))
epoch=len(seen)
torch.save({'epoch':epoch},out/'last.pt')
sys.exit(1 if epoch==1 else 0)
""", encoding='utf-8')
    actual_run = entry._run_attempt
    commands = []
    def run(command):
        commands.append(command[:])
        replaced = command[:]
        replaced[2] = str(child)
        result = actual_run(replaced)
        config_path.write_text('output_space: invalid', encoding='utf-8')
        return result
    monkeypatch.setattr(entry, '_run_attempt', run)
    sleeps = []
    monkeypatch.setattr(entry.time, 'sleep', sleeps.append)
    output = tmp_path / 'output'
    assert entry.main(['--auto-resume', '--config', str(config_path),
                       '--dataset-root', str(tmp_path / 'raw'), '--output-dir', str(output),
                       '--anchor-cache-dir', str(tmp_path / 'cache'), '--device', 'cpu',
                       '--max-epochs', '2', '--max-steps', '7', '--limit-samples', '5']) == 0
    seen = json.loads((output / 'seen.json').read_text())
    assert len(commands) == len(seen) == 2 and sleeps == [10]
    assert '--auto-resume' not in commands[0]
    assert seen[0]['resume'] is None
    assert seen[1]['resume'] == str(output / 'last.pt')
    assert seen[0]['args'] == seen[1]['args']
    assert seen[0]['config'] == seen[1]['config']
    assert seen[0]['config']['training']['patience'] == 15
    assert seen[0]['args'] == dict(root=str(tmp_path / 'raw'), output_dir=str(output),
                                  anchor_cache_dir=str(tmp_path / 'cache'), device='cpu',
                                  max_epochs=2, max_steps=7, limit_samples=5)


@pytest.mark.parametrize('scenario, exit_code, expected_calls', [
    ('no_progress', 1, 3), ('progress_reset', 1, 6),
    ('patience', 0, 1), ('target_after_error', 0, 1), ('interrupt', 130, 1)])
def test_auto_resume_exit_rules(tmp_path, monkeypatch, scenario, exit_code, expected_calls):
    from scripts.training import run_diffusion as entry
    calls = []
    sleeps = []
    monkeypatch.setattr(entry.time, 'sleep', sleeps.append)
    output = tmp_path / 'output'
    def run(command):
        calls.append(command)
        if scenario == 'interrupt':
            raise KeyboardInterrupt
        if scenario == 'patience':
            torch.save({'epoch': 1}, output / 'last.pt')
            return 0
        if scenario == 'target_after_error':
            torch.save({'epoch': 10}, output / 'last.pt')
        if scenario == 'progress_reset' and len(calls) == 3:
            torch.save({'epoch': 1}, output / 'last.pt')
        return 1
    monkeypatch.setattr(entry, '_run_attempt', run)
    assert entry._auto_resume(_config(), dict(output_dir=str(output), max_epochs=10), None) == exit_code
    assert len(calls) == expected_calls
    assert len(sleeps) == max(expected_calls - 1, 0)


def test_auto_resume_initial_checkpoint_and_existing_output(tmp_path, monkeypatch):
    from scripts.training import run_diffusion as entry
    initial = tmp_path / 'initial.pt'
    torch.save({'epoch': 4}, initial)
    output = tmp_path / 'output'
    output.mkdir()
    torch.save({'epoch': 9}, output / 'last.pt')
    with pytest.raises(ValueError, match='without checkpoints'):
        entry._auto_resume(_config(), dict(output_dir=str(output), max_epochs=10), None)
    calls = []
    monkeypatch.setattr(entry.time, 'sleep', lambda seconds: None)
    def fail(command):
        calls.append(command)
        return 1
    monkeypatch.setattr(entry, '_run_attempt', fail)
    assert entry._auto_resume(_config(), dict(output_dir=str(output), max_epochs=10), str(initial)) == 1
    assert len(calls) == 3
    assert all(command[-2:] == ['--resume', str(initial)] for command in calls)
    calls.clear()
    assert entry._auto_resume(_config(), dict(output_dir=str(output), max_epochs=4), str(initial)) == 0
    assert calls == []


def test_auto_worker_and_original_entry_do_not_change_training_call(tmp_path, monkeypatch):
    import json
    from scripts.training import run_diffusion as entry
    cfg = _config()
    cfg['training'].update(dataset_root=str(tmp_path / 'raw'), output_dir=str(tmp_path / 'output'))
    calls = []
    monkeypatch.setattr(training, 'train', lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(entry, 'load_mode_yaml', lambda path: cfg)
    assert entry.main(['--config', 'unused', '--max-steps', '3']) == 0
    assert calls[-1][0] == (cfg,)
    assert calls[-1][1] == dict(root=tmp_path / 'raw', output_dir=tmp_path / 'output',
                               max_steps=3, max_epochs=None, limit_samples=0,
                               device=None, resume=None, anchor_cache_dir=None)
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps({'config': cfg, 'train_args': {'max_epochs': 8}}))
    def forbidden(path):
        raise AssertionError('Worker must never reload experiment YAML')
    monkeypatch.setattr(entry, 'load_mode_yaml', forbidden)
    assert entry.main(['--_auto-worker', str(snapshot), '--resume', 'fixed.pt']) == 0
    assert calls[-1] == ((cfg,), {'max_epochs': 8, 'resume': 'fixed.pt'})


def test_auto_resume_interrupt_stops_child_tree(monkeypatch):
    from scripts.training import run_diffusion as entry
    waits = iter([KeyboardInterrupt(), 0])
    class Child:
        pid = 12345
        def wait(self):
            value = next(waits)
            if isinstance(value, BaseException):
                raise value
            return value
    monkeypatch.setattr(entry.subprocess, 'Popen', lambda *a, **k: Child())
    stopped = []
    monkeypatch.setattr(entry.subprocess, 'run', lambda command, **kwargs: stopped.append(command))
    monkeypatch.setattr(entry.os, 'killpg', lambda pid, sig: stopped.append(pid), raising=False)
    with pytest.raises(KeyboardInterrupt):
        entry._run_attempt(['fake'])
    assert len(stopped) == 1
    if entry.os.name == 'nt':
        assert stopped[0] == ['taskkill', '/PID', '12345', '/T', '/F']
    else:
        assert stopped[0] == 12345


def test_atomic_checkpoint_failure_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / 'last.pt'
    training._atomic_save_checkpoint({'epoch': 1}, path)
    original = path.read_bytes()
    def broken_save(value, temporary):
        temporary.write_bytes(b'incomplete')
        raise OSError('simulated write failure')
    monkeypatch.setattr(training.torch, 'save', broken_save)
    with pytest.raises(OSError, match='simulated write failure'):
        training._atomic_save_checkpoint({'epoch': 2}, path)
    assert path.read_bytes() == original
    assert not path.with_name('last.pt.tmp').exists()


def test_auto_resume_later_uses_first_training_snapshot(tmp_path, monkeypatch):
    import json
    from scripts.training import run_diffusion as entry
    cfg = _config()
    cfg['training'].update(dataset_root=str(tmp_path / 'raw'), output_dir=str(tmp_path / 'output'))
    calls = []
    monkeypatch.setattr(training, 'train', lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(entry, 'load_mode_yaml', lambda path: cfg)
    output = tmp_path / 'output'
    assert entry.main(['--config', 'original.yaml', '--device', 'cpu', '--max-epochs', '60',
                       '--max-steps', '3', '--limit-samples', '5',
                       '--anchor-cache-dir', str(tmp_path / 'cache')]) == 0
    fixed = json.loads((output / 'training_run.json').read_text())
    torch.save({'epoch': 12}, output / 'last.pt')
    cfg['losses']['training_scheme'] = 'different'
    monkeypatch.setattr(entry, 'load_mode_yaml', lambda path: cfg)
    captured = []
    monkeypatch.setattr(entry, '_auto_resume', lambda *args: captured.append(args) or 0)
    with pytest.warns(RuntimeWarning, match='Current YAML differs'):
        assert entry.main(['--auto-resume', '--resume', str(output / 'last.pt'),
                           '--config', 'edited.yaml', '--max-epochs', '100', '--limit-samples', '999']) == 0
    assert captured == [(fixed['config'], fixed['train_args'], str(output / 'last.pt'))]
    assert captured[0][1]['max_epochs'] == 60
    assert captured[0][1]['max_steps'] == 3
    assert captured[0][1]['limit_samples'] == 5
    assert captured[0][1]['anchor_cache_dir'] == str(tmp_path / 'cache')
    assert 'training_scheme' not in captured[0][0]['losses']
    assert json.loads((output / 'training_run.json').read_text()) == fixed


@pytest.mark.parametrize('automatic', [False, True])
@pytest.mark.parametrize('new_output', [False, True])
def test_legacy_resume_without_snapshot_uses_current_options(tmp_path, monkeypatch, automatic, new_output):
    import json
    from scripts.training import run_diffusion as entry
    checkpoint = tmp_path / 'legacy' / 'last.pt'
    checkpoint.parent.mkdir()
    torch.save({'epoch': 12}, checkpoint)
    cfg = _config()
    cfg['training'].update(dataset_root='unused', output_dir=str(tmp_path / 'wrong_yaml_output'), max_epochs=50)
    monkeypatch.setattr(entry, 'load_mode_yaml', lambda path: cfg)
    calls = []
    monkeypatch.setattr(training, 'train', lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(entry, '_auto_resume', lambda *a: calls.append(a) or 0)
    output = tmp_path / 'new_output' if new_output else checkpoint.parent
    options = ['--resume', str(checkpoint), '--device', 'cpu', '--max-epochs', '60',
               '--dataset-root', str(tmp_path / 'data'), '--anchor-cache-dir', str(tmp_path / 'cache'),
               '--max-steps', '7', '--limit-samples', '5']
    if automatic:
        options.append('--auto-resume')
    if new_output:
        options.extend(['--output-dir', str(output)])
    with pytest.warns(RuntimeWarning, match='Missing training_run.json'):
        assert entry.main(options) == 0
    run = json.loads((output / 'training_run.json').read_text())
    expected = dict(root=str(tmp_path / 'data'), output_dir=str(output), max_epochs=60,
                    max_steps=7, limit_samples=5, device='cpu', anchor_cache_dir=str(tmp_path / 'cache'))
    assert run['train_args'] == expected
    assert run['yaml_config'] == cfg
    if automatic:
        assert calls == [(run['config'], expected, str(checkpoint))]
    else:
        assert calls == [((run['config'],), {**expected, 'resume': str(checkpoint)})]
    if new_output:
        assert not (checkpoint.parent / 'training_run.json').exists()
    # Subsequent resumes consume the new output snapshot, not the edited YAML or overrides.
    cfg['losses']['trajectory']['weight'] = 99.
    next_checkpoint = output / 'last.pt'
    calls.clear()
    with pytest.warns(RuntimeWarning, match='Current YAML differs'):
        assert entry.main(['--resume', str(next_checkpoint), '--max-epochs', '100']) == 0
    assert calls == [((run['config'],), {**expected, 'resume': str(next_checkpoint)})]
    assert json.loads((output / 'training_run.json').read_text()) == run


def test_legacy_resume_still_rejects_incompatible_checkpoint(tmp_path, monkeypatch):
    import json
    from scripts.training import run_diffusion as entry
    monkeypatch.setattr(training, 'ControlDiffusionPlanner', _DummySafetyPlanner)
    monkeypatch.setattr(training, '_make_loader', lambda *a, **k: _Loader())
    monkeypatch.setattr(training, '_run_pass', lambda *a, **k: _metrics(.5))
    cfg = _config()
    output = tmp_path / 'legacy'
    training.train(cfg, root=tmp_path, output_dir=output, max_epochs=1, device='cpu')
    current = copy.deepcopy(cfg)
    current['training']['dataset_root'] = str(tmp_path)
    current['losses']['trajectory']['weight'] = 2.
    monkeypatch.setattr(entry, 'load_mode_yaml', lambda path: current)
    with pytest.warns(RuntimeWarning, match='Missing training_run.json'):
        with pytest.raises(ValueError, match='loss_config differs'):
            entry.main(['--resume', str(output / 'last.pt'), '--device', 'cpu', '--max-epochs', '2'])
    snapshot = output / 'training_run.json'
    assert json.loads(snapshot.read_text())['config']['losses']['trajectory']['weight'] == 2.
    saved_bytes = snapshot.read_bytes()
    current['losses']['trajectory']['weight'] = 1.
    with pytest.warns(RuntimeWarning, match='Current YAML differs'):
        with pytest.raises(ValueError, match='loss_config differs'):
            entry.main(['--resume', str(output / 'last.pt')])
    assert snapshot.read_bytes() == saved_bytes


def test_manual_resume_uses_first_snapshot_and_warns_for_yaml_change(tmp_path, monkeypatch):
    import json
    from scripts.training import run_diffusion as entry
    cfg = _config()
    output = tmp_path / 'run'
    cfg['training'].update(dataset_root=str(tmp_path / 'raw'), output_dir=str(output))
    calls = []
    monkeypatch.setattr(training, 'train', lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(entry, 'load_mode_yaml', lambda path: cfg)
    assert entry.main(['--device', 'cpu', '--max-epochs', '60', '--max-steps', '2']) == 0
    fixed = json.loads((output / 'training_run.json').read_text())
    checkpoint = output / 'last.pt'
    torch.save({'epoch': 10}, checkpoint)
    cfg['losses']['training_scheme'] = 'anchor_consistent'
    with pytest.warns(RuntimeWarning, match='Current YAML differs'):
        assert entry.main(['--resume', str(checkpoint), '--max-epochs', '100', '--max-steps', '9']) == 0
    assert calls[-1] == ((fixed['config'],), {**fixed['train_args'], 'resume': str(checkpoint)})
    assert calls[-1][1]['max_epochs'] == 60
    assert calls[-1][1]['max_steps'] == 2
    assert calls[-1][0][0]['training']['patience'] == fixed['config']['training']['patience']


def test_legacy_resume_does_not_overwrite_destination_snapshot(tmp_path, monkeypatch):
    from scripts.training import run_diffusion as entry
    checkpoint = tmp_path / 'source' / 'last.pt'
    checkpoint.parent.mkdir()
    torch.save({'epoch': 1}, checkpoint)
    output = tmp_path / 'destination'
    output.mkdir()
    snapshot = output / 'training_run.json'
    snapshot.write_text('existing snapshot', encoding='utf-8')
    cfg = _config()
    cfg['training']['dataset_root'] = str(tmp_path)
    monkeypatch.setattr(entry, 'load_mode_yaml', lambda path: cfg)
    with pytest.warns(RuntimeWarning, match='Missing training_run.json'):
        with pytest.raises(FileExistsError, match='already exists'):
            entry.main(['--resume', str(checkpoint), '--output-dir', str(output), '--device', 'cpu'])
    assert snapshot.read_text(encoding='utf-8') == 'existing snapshot'
    assert not (checkpoint.parent / 'training_run.json').exists()
