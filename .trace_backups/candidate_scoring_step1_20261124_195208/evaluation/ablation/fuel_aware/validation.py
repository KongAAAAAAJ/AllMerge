"""Frozen Koopman and calibrated VT-CPFM validation contracts."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import torch
from highway_env.planner.diffusion.common.fuel import FuelParameters
from highway_env.planner.diffusion.config.experiment import load_mode_yaml

REQUIRED_LABELS = ('F0','F1','F2','F3')

def resolve_device(name: str) -> torch.device:
    return torch.device('cuda' if name == 'auto' and torch.cuda.is_available()
                        else 'cpu' if name == 'auto' else name)


def fuel_parameters_from_yaml(path: str | Path) -> FuelParameters:
    cfg = load_mode_yaml(path)
    if cfg['output_space'] != 'control':
        raise ValueError('Fuel evaluation requires output_space: control YAML')
    values = cfg.get('losses', {}).get('fuel', {}).get('parameters')
    if not isinstance(values, dict):
        raise ValueError('Set calibrated control.losses.fuel.parameters in YAML; '
                         'these must use litres/second at power in kW.')
    try:
        params = FuelParameters(**values)
        params.validate()
    except (TypeError, ValueError) as err:
        raise ValueError(f'Invalid calibrated fuel.parameters: {err}') from err
    return params


def checkpoint_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for part in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def k2_signature(model) -> str:
    if model.vehicle_model.kind != 'koopman_k2':
        raise ValueError('Fuel ablation requires koopman_k2 execution for every method')
    h = hashlib.sha256()
    for key, tensor in sorted(model.vehicle_model.koopman.state_dict().items()):
        h.update(key.encode('utf-8'))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_ablation_objectives(label: str, loss_cfg: dict, params: FuelParameters):
    """Guard against mislabeled checkpoints in the fuel-only ablation."""
    def w(name):
        return float(loss_cfg.get(name, {}).get('weight',
                     1.0 if name == 'trajectory' else 0.0))
    if not math.isclose(w('trajectory'), 1., abs_tol=1e-8):
        raise ValueError(f'{label}: expected trajectory.weight=1.0')
    if not math.isclose(w('ltr'), 0., abs_tol=1e-8):
        raise ValueError(f'{label}: expected ltr.weight=0 for this fuel-only ablation')
    if label == 'F0':
        if w('fuel') != 0. or w('progress') != 0.:
            raise ValueError('F0 must use trajectory imitation only (fuel=progress=0)')
    else:
        if w('fuel') <= 0 or w('progress') <= 0:
            raise ValueError(f'{label} needs both positive fuel and progress weights')
        used = loss_cfg.get('fuel', {}).get('parameters')
        if not isinstance(used, dict):
            raise ValueError(f'{label} must store fuel calibration in checkpoint')
        try:
            used_params = FuelParameters(**used)
        except TypeError as err:
            raise ValueError(f'{label} has invalid fuel calibration: {err}') from err
        for field, value in vars(params).items():
            if not math.isclose(float(getattr(used_params, field)), float(value), rel_tol=1e-7, abs_tol=1e-10):
                raise ValueError(f'{label} fuel calibration {field} differs from evaluation YAML')


def check_checkpoint_compatibility(reference, current, label):
    """Fail before evaluating incomparable planner structures/K2 surrogates."""
    if current['model_config'] != reference['model_config']:
        raise ValueError(f'{label}: model_config differs from F0')
    if current['control_config'] != reference['control_config']:
        raise ValueError(f'{label}: control_config differs from F0')
    if current['k2_signature'] != reference['k2_signature']:
        raise ValueError(f'{label}: frozen K2 weights differ from F0')
    if current['mode_selection'] != 'goal_endpoint':
        raise ValueError(f'{label}: expected goal_endpoint training mode')
