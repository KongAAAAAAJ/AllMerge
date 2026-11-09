"""No-DLL compatibility smoke test. Run from project root: python scripts/verify_trucksim_modes.py"""

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def install_package(name, directory):
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(directory)]
    sys.modules[name] = pkg


install_package('highway_env', ROOT / 'highway_env')
install_package('highway_env.vehicle', ROOT / 'highway_env' / 'vehicle')
simpath = ROOT / 'highway_env' / 'vehicle' / 'trucksim_simulation'
install_package('highway_env.vehicle.trucksim_simulation', simpath)
controller_stub = types.ModuleType('highway_env.vehicle.controller')


class FOLLOWVehicle:
    def on_state_update(self):
        pass

    def clip_actions(self):
        pass


controller_stub.FOLLOWVehicle = FOLLOWVehicle
sys.modules['highway_env.vehicle.controller'] = controller_stub
inverse = module('highway_env.vehicle.trucksim_simulation.inverse_dynamics', simpath / 'inverse_dynamics.py')
low = module('highway_env.vehicle.trucksim_simulation.low_controller', simpath / 'low_controller.py')
raw = module('highway_env.vehicle.trucksim_simulation.Simulation', simpath / 'Simulation.py')
ltr = module('highway_env.vehicle.trucksim_simulation.Trucksim_Simulation_ltr', simpath / 'Trucksim_Simulation_ltr.py')
plant = module('highway_env.vehicle.trucksim_dynamics', ROOT / 'highway_env' / 'vehicle' / 'trucksim_dynamics.py')


class FakeModel:
    def __init__(self, n_export):
        self.configuration = {'n_import': 3, 'n_export': n_export}
        self.current_time = 0.0
        self.dt = 0.01
        self.inputs_history = []
        self.export = np.zeros(n_export)
        self.export[2] = 60
        self.export[12] = 1200

    def get_export_array(self):
        return self.export.copy()

    def get_time_step(self):
        return self.dt

    def run(self, t, inputs, exports):
        self.current_time = float(t)
        self.inputs_history.append(np.asarray(inputs).copy())
        return 0, exports


class FakeVehicle:
    def __init__(self):
        self.position = np.array([0.0, 0.0])
        self.heading = 0.0
        self.speed = 10.0
        self.action = {'acceleration': 0.3, 'steering': 0.03}
        self.timer = 0.0
        self.impact = None


mapcfg = {
    'imports': {'brake': 0, 'steering_wheel_deg': 1, 'throttle': 2},
    'exports': {
        'ax_g': 0, 'ay_g': 1, 'vx_kmh': 2, 'vy_kmh': 3,
        'x_m': 4, 'y_m': 5, 'roll_deg': 6, 'roll_rate_deg_s': 7,
        'yaw_deg': 10, 'yaw_rate_deg_s': 11, 'engine_rpm': 12,
        'steer_left_deg': 13, 'steer_right_deg': 14, 'ltr': 15,
    },
}
baseconfig = {'steering_ratio': 25.0, 'max_rpm': 3000}
engine_map = np.column_stack([np.array([600, 2000]), np.tile(np.linspace(0, 2000, 11), (2, 1))])

# Inverse steering: consistent physical units and static transmission.
assert math.isclose(inverse.inverse_steering_dynamics(0.1, 25), math.degrees(0.1) * 25)
assert math.isclose(inverse.inverse_steering_dynamics(0.1, 25,
                          previous_ref_rad=0.0, dt=0.1, steering_tau_s=0.1), math.degrees(0.2) * 25)
latctrl = low.LateralSteeringController(steering_ratio=25, dt=0.01, kp=0.1, ki=0,
                                      kd=0, max_wheel_deg=100, max_rate_deg_s=1000)
first = latctrl.control(0.1, 0)
second = latctrl.control(-0.1, 0)
assert abs(first) <= 100 and abs(second) <= 100
assert abs(second - first) <= 1000 * 0.01 + 1e-8

# Backward-compatible default mode, import layout and direct steering unchanged.
old_model = FakeModel(15)
old = plant.TrucksimVehicle.from_vehicle(FakeVehicle(), old_model, baseconfig, engine_map)
assert old.control_mode == 'longitudinal' and old.lateral_controller is None
old.step(0.03)
assert old_model.inputs_history
assert np.isclose(old_model.inputs_history[-1][2], math.degrees(0.03) * 25)
assert old_model.inputs_history[-1].shape == (3,)

# New model has 16 exported values and explicitly remapped imports.
new_model = FakeModel(16)
new_model.export[15] = 0.25
new_cfg = dict(baseconfig, control_mode='longitudinal_lateral', signal_map=mapcfg,
               lateral_controller={'kp': 0.1, 'ki': 0, 'kd': 0,
                                   'max_wheel_deg': 100, 'max_rate_deg_s': 1000})
new = plant.TrucksimVehicle.from_vehicle(FakeVehicle(), new_model, new_cfg, engine_map)
assert np.isclose(new.ltr, 0.25)
new.step(0.03)
assert new_model.inputs_history
assert abs(new_model.inputs_history[-1][1]) <= 100
assert np.isclose(new_model.inputs_history[-1][0], 0.0)  # brake
assert np.isclose(new_model.inputs_history[-1][2], old_model.inputs_history[-1][0])  # throttle
assert new.front_wheel_angle_deg == 0
assert new.roll_rad == 0.0 and new.yaw_rate_rad_s == 0.0

# Fail safely rather than implicitly treating 16 exports as legacy 15.
try:
    plant.TrucksimVehicle.from_vehicle(FakeVehicle(), new_model, baseconfig, engine_map)
except ValueError:
    pass
else:
    raise AssertionError('Legacy mode silently accepted the wrong export layout')

# Raw LTR wrapper must not make DLL assumptions until start().
bridge = ltr.TrucksimSimulationLTR('x.sim', 'y.dll')
assert bridge.configuration is None
print('[PASS] no-DLL longitudinal legacy, lon+lat controller, mapped I/O and inverse steering')
