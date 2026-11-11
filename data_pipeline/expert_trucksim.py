"""TruckSim operating-condition mapping and time-aligned K2 context for expert data.

This module does not change the Polynomial expert or any road scenario.  A
separate 1-episode child process must own each native TruckSim session.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
from typing import Dict, Sequence

import numpy as np
import yaml


RAW_EXPORTS = 19
STATE_HISTORY = 5
CONTROL_HISTORY = STATE_HISTORY - 1


def load_collection_config(path: str | Path) -> dict:
    with Path(path).open('r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict) or not isinstance(cfg.get('trucksim'), dict):
        raise ValueError('Expected an existing Koopman collection YAML with trucksim mapping')
    return cfg


def physical_cases(koopman_cfg: dict):
    """The 63 physical plants are indexed by payload and friction, not speed."""
    truck = koopman_cfg['trucksim']
    specs = truck['models']
    payloads = truck['payload_tiers_kg']
    frictions = truck['friction_tiers']
    for payload_id, payload in payloads.items():
        for mu_id, mu in frictions.items():
            model_id = specs['id_template'].format(payload_tier=payload_id, friction_tier=mu_id)
            overrides = specs.get('overrides', {}).get(model_id, {})
            fields = dict(payload_tier=payload_id, friction_tier=mu_id, id=model_id)
            settings = {**specs.get('defaults', {}), **overrides}
            if not settings.get('enabled', True):
                continue
            if not settings.get('verified_in_trucksim', False):
                raise RuntimeError(f'{model_id} has not been verified_in_trucksim')
            simfile = settings.get('simfile', specs['simfile_template'].format(**fields))
            dll = settings.get('dll', specs['dll_template'].format(**fields))
            yield dict(model_id=model_id, payload_tier=payload_id,
                       friction_tier=mu_id, payload_kg=float(payload),
                       road_mu=float(mu), simfile=str(simfile), dll=str(dll))


def select_case(cfg: dict, episode_index: int) -> dict:
    cases = list(physical_cases(cfg))
    if not cases:
        raise ValueError('No enabled verified TruckSim models in Koopman config')
    # A full 63-episode sweep visits each payload/friction pair once.
    return deepcopy(cases[int(episode_index) % len(cases)])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_independent_dlls(case: dict, count: int, *, copy_instances: bool) -> list[str]:
    """Three cars must never share the same native DLL image in one process.

    Copy beside the original DLL to preserve its relative dependency lookup.
    Copies are stable on disk and verified before reuse.  Validity of multiple
    VehicleSim instances still requires a Windows/TruckSim smoke test.
    """
    simfile = Path(case['simfile'])
    dll = Path(case['dll'])
    if not simfile.is_file() or not dll.is_file():
        raise FileNotFoundError(f'TruckSim model not found: {simfile}, {dll}')
    original_hash = _sha256(dll)
    results = []
    for index in range(count):
        distinct = dll.with_name(dll.stem + f'__expert_ego{index}' + dll.suffix)
        if not distinct.exists():
            if not copy_instances:
                raise FileNotFoundError(
                    f'Missing independent DLL {distinct}. Enable copy_instances in '
                    'expert config, or prepare separate DLLs manually.')
            shutil.copy2(dll, distinct)
        if _sha256(distinct) != original_hash:
            raise RuntimeError(f'Stale/different TruckSim solver instance: {distinct}')
        results.append(str(distinct))
    if len({Path(p).resolve() for p in results}) != count:
        raise ValueError('Each controlled vehicle requires a distinct DLL image')
    return results


def scenario_trucksim_config(koopman_cfg: dict, case: dict, num_egos: int,
                             *, copy_instances: bool = False) -> dict:
    """Only the controlled-vehicle plant differs; planner/scenes remain intact."""
    dlls = prepare_independent_dlls(case, num_egos, copy_instances=copy_instances)
    r = koopman_cfg.get('reference_controller', {})
    lat = r.get('lateral', {})
    # Existing TrucksimVehicle requires per-wheel 19-channel outputs.
    native = dict(
        control_mode='longitudinal_lateral',
        simfiles=[case['simfile']] * num_egos,
        dlls=dlls,
        steering_ratio=float(lat.get('steering_ratio', 25.0)),
        max_rpm=float(r.get('max_rpm', 3000)),
        engine_map_path=r.get('engine_map_path'),
        # Match Koopman collection's closed-loop controller timing and units.
        expert_koopman_aligned=True,
        actuator_update_dt_s=float(koopman_cfg.get('excitation', {}).get('controller_dt_s', 0.1)),
        longitudinal_pid=deepcopy(r.get('longitudinal_pid', {})),
        throttle_limit_pct=float(r.get('throttle_limit_pct', 35.0)),
        brake_limit_mpa=float(r.get('brake_limit_mpa', 2.0)),
        signal_map={
            'imports': {'throttle': 0, 'brake': 1, 'steering_wheel_deg': 2},
            'exports': {
                'ax_g': 0, 'ay_g': 1, 'vx_kmh': 2, 'vy_kmh': 3,
                'x_m': 4, 'y_m': 5, 'roll_deg': 6,
                'yaw_deg': 10, 'yaw_rate_deg_s': 11,
                'engine_rpm': 12, 'steer_left_deg': 13, 'steer_right_deg': 14,
                # LTR is COMPUTED from wheel loads 15:19, never mislabel Fz as LTR.
            }
        },
        lateral_controller={
            'kp': float(lat.get('kp', 0.3)),
            'ki': float(lat.get('ki', 0.0)),
            'kd': float(lat.get('kd', 0.0)),
            'max_wheel_deg': float(lat.get('max_wheel_deg', 100.0)),
            'max_rate_deg_s': float(lat.get('max_rate_deg_s', 180.0)),
            'steering_tau_s': float(lat.get('steering_tau_s', 0.0)),
        },
        wheel_compression_sign=float(koopman_cfg['units'].get('wheel_compression_sign', 1)),
        operating_condition=deepcopy(case),
    )
    return {
        'Controller': {'type': 'lqr', 'vehicle_model': 'trucksim'},
        'TruckSim': native,
    }


def attach_scenario_trucksim(env):
    """Replace only controlled platoon vehicles after scenario road/platoon creation."""
    from highway_env.vehicle.trucksim_dynamics import TrucksimVehicle, load_engine_map
    from highway_env.vehicle.trucksim_simulation.Trucksim_Simulation_ltr import TrucksimSimulationLTR

    setting = env.config['TruckSim']
    if setting.get('control_mode') != 'longitudinal_lateral':
        raise ValueError('Koopman-aligned expert data require longitudinal_lateral TruckSim')
    if len(setting['dlls']) != len(env.controlled_vehicles):
        raise ValueError('One unique TruckSim DLL is required for each controlled vehicle')
    cache_path = setting.get('engine_map_cache_npy')
    if cache_path:
        map_data = np.asarray(np.load(cache_path, allow_pickle=False), dtype=np.float64)
        if (map_data.ndim != 2 or map_data.shape[1] != 12 or
                not np.isfinite(map_data).all() or
                np.any(np.diff(map_data[:,0]) <= 0)):
            raise ValueError('Invalid cached engine map')
    else:
        map_data = load_engine_map(setting)
    env.trucksim_models = []
    for index, original in enumerate(tuple(env.controlled_vehicles)):
        sim = TrucksimSimulationLTR(setting['simfiles'][index], setting['dlls'][index])
        env.trucksim_models.append(sim)
        # start() sets SV_VXS (km/h) after vs_setdef_and_read, BEFORE Initialize.
        sim.start({'vx_init': float(original.speed) * 3.6})
        if int(sim.configuration['n_import']) != 3 or int(sim.configuration['n_export']) != RAW_EXPORTS:
            raise RuntimeError(f'Expected 3 inputs and 19 exports; got {sim.configuration}')
        replacement = TrucksimVehicle.from_vehicle(original, sim, setting, map_data)
        location = env.road.vehicles.index(original)
        env.road.vehicles[location] = replacement
        env.controlled_vehicles[index] = replacement
    env._expert_trucksim_active = True


def stop_scenario_trucksim(env):
    """One-shot process: never reuse this native DLL session afterwards."""
    for model in getattr(env, 'trucksim_models', []):
        if getattr(model, 'is_running', False):
            model.stop(native_terminate=False)
    env._expert_trucksim_active = False


def snapshot(vehicle, fz_compression_sign=1.0):
    """Match Koopman/data/processing.py SI conversion and highway world pose."""
    from highway_env.vehicle.koopman.data.processing import to_si, ltr_from_fz
    raw = np.asarray(vehicle.export_array, dtype=np.float64).reshape(1, -1)
    if raw.shape[1] != RAW_EXPORTS:
        raise ValueError('TruckSim expert history requires all 19 channels')
    states, _, extra = to_si(raw, {'velocity': 'km/h', 'position': 'm', 'engine_speed': 'rpm'})
    ltr = ltr_from_fz(raw, sign=fz_compression_sign)[0]
    if not np.isfinite(ltr):
        raise ValueError('Invalid TruckSim wheel loads; cannot construct LTR label')
    pose = np.array([*np.asarray(vehicle.position, dtype=float), float(vehicle.heading)], dtype=np.float64)
    state = states[0].copy()
    if not np.isfinite(state).all() or not np.isfinite(pose).all():
        raise ValueError('Nonfinite TruckSim state/pose')
    return state, pose, float(ltr)


class HistoricalContexts:
    """Five plant states, four prior commanded controls, same 10-Hz grid.

    At planning time k, the last state is s[k], and the four controls in the
    buffer moved s[k-4] -> ... -> s[k].  The polynomial planner's label and
    feature batch are computed at time k before the subsequent env.step update.
    """
    def __init__(self, vehicles: Sequence, *, compression_sign: float):
        self.sign = float(compression_sign)
        self.states = [deque(maxlen=STATE_HISTORY) for _ in vehicles]
        self.controls = [deque(maxlen=CONTROL_HISTORY) for _ in vehicles]
        self.poses = [deque(maxlen=STATE_HISTORY) for _ in vehicles]
        self.ltrs = [deque(maxlen=STATE_HISTORY) for _ in vehicles]
        self.initial_speed_kmh = []
        for index, v in enumerate(vehicles):
            s, p, ltr = snapshot(v, self.sign)
            self.initial_speed_kmh.append(float(s[0])*3.6)
            self.states[index].append(s)
            self.poses[index].append(p)
            self.ltrs[index].append(ltr)

    def ready(self) -> bool:
        return all(len(s) == STATE_HISTORY for s in self.states) and all(
            len(u) == CONTROL_HISTORY for u in self.controls)

    def at_planning_time(self, index: int, case: dict):
        if not self.ready():
            return None
        return {
            'koopman_state_hist_si': np.stack(self.states[index]).astype(np.float32),
            'koopman_control_hist_si': np.stack(self.controls[index]).astype(np.float32),
            'koopman_pose0_si': np.asarray(self.poses[index][-1], np.float32),
            'koopman_ltr0': np.asarray(self.ltrs[index][-1], np.float32),
            'koopman_payload_kg': np.asarray(case['payload_kg'], np.float32),
            'koopman_road_mu': np.asarray(case['road_mu'], np.float32),
            'koopman_model_id': np.asarray(case['model_id']),
            'koopman_vx_init_kmh': np.asarray(self.initial_speed_kmh[index], np.float32),
        }

    def append_executed_transition(self, vehicles: Sequence):
        for index, v in enumerate(vehicles):
            # Vehicle.action is the requested control just applied by Road.step.
            u = np.asarray([v.action['acceleration'], v.action['steering']], dtype=np.float64)
            if u.shape != (2,) or not np.isfinite(u).all():
                raise ValueError('Nonfinite or incorrectly shaped applied reference control')
            state, pose, ltr = snapshot(v, self.sign)
            self.controls[index].append(u)
            self.states[index].append(state)
            self.poses[index].append(pose)
            self.ltrs[index].append(ltr)


def check_context(sample: dict):
    assert np.asarray(sample['koopman_state_hist_si']).shape == (5, 5)
    assert np.asarray(sample['koopman_control_hist_si']).shape == (4, 2)
    assert np.asarray(sample['koopman_pose0_si']).shape == (3,)
    assert np.asarray(sample['future_trajectory_dense']).shape == (40, 2)
    for field in ('koopman_state_hist_si', 'koopman_control_hist_si',
                  'koopman_pose0_si', 'koopman_ltr0', 'koopman_payload_kg', 'koopman_road_mu'):
        if not np.isfinite(np.asarray(sample[field], dtype=float)).all():
            raise ValueError(f'Nonfinite {field}')
