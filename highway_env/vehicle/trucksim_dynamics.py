"""TruckSim plant for the existing FOLLOWVehicle decision/control interface."""
import copy
from pathlib import Path

import numpy as np

from highway_env.vehicle.controller import FOLLOWVehicle
from highway_env.vehicle.trucksim_simulation.low_controller import (
    PIDController, acceleration_control, LateralSteeringController,
)


class TrucksimVehicle(FOLLOWVehicle):
    @classmethod
    def create_from(cls, vehicle):
        # History/constant-speed snapshots do not own a native solver.
        return FOLLOWVehicle.create_from(vehicle)

    @classmethod
    def from_vehicle(cls, vehicle, model, config, engine_map):
        # Preserve all planner, group and observation fields of the scene vehicle.
        result = cls.__new__(cls)
        result.__dict__ = vehicle.__dict__.copy()
        result.trucksim_model = model
        result.export_array = np.asarray(model.get_export_array(), dtype=float)
        result.solver_dt = float(model.get_time_step())
        if not np.isfinite(result.solver_dt) or result.solver_dt <= 0:
            raise ValueError("TruckSim solver time step must be positive")
        result.control_mode = config.get('control_mode', 'longitudinal')
        if result.control_mode not in {'longitudinal', 'longitudinal_lateral'}:
            raise ValueError(f"Unknown TruckSim control_mode: {result.control_mode!r}")
        result.signal_map = None
        if result.control_mode == 'longitudinal':
            # Preserve the existing 3-import/15-export dataset contract exactly.
            if (model.configuration['n_import'] != 3 or model.configuration['n_export'] != 15
                    or len(result.export_array) != 15):
                raise ValueError('Expected legacy TruckSim 3 imports and 15 exports')
        else:
            if len(result.export_array) != model.configuration['n_export']:
                raise ValueError('TruckSim LTR export array length disagrees with configuration')
            # The new .sim model may export a different number and order of signals.
            # Explicit, validated indices prevent silent unit/field mismatches.
            mapping = config.get('signal_map')
            if not isinstance(mapping, dict):
                raise ValueError('longitudinal_lateral mode requires TruckSim.signal_map')
            import_map = mapping.get('imports', {})
            export_map = mapping.get('exports', {})
            required_import = {'throttle', 'brake', 'steering_wheel_deg'}
            required_export = {'ax_g', 'ay_g', 'vx_kmh', 'vy_kmh', 'x_m', 'y_m',
                               'yaw_deg', 'engine_rpm', 'steer_left_deg', 'steer_right_deg'}
            if not required_import.issubset(import_map) or not required_export.issubset(export_map):
                raise ValueError('TruckSim.signal_map missing required input/output names')
            if model.configuration['n_import'] != 3 or set(import_map) != required_import:
                raise ValueError('Current longitudinal_lateral controller requires exactly '
                                 'three mapped imports: throttle, brake, steering_wheel_deg')
            if not all(type(v) is int and 0 <= v < model.configuration['n_import']
                       for v in import_map.values()):
                raise ValueError('TruckSim import indices are out of range')
            if not all(type(v) is int and 0 <= v < model.configuration['n_export']
                       for v in export_map.values()):
                raise ValueError('TruckSim export indices are out of range')
            if len(set(import_map.values())) != len(import_map.values()):
                raise ValueError('TruckSim import indices must be distinct')
            if len(set(export_map.values())) != len(export_map.values()):
                raise ValueError('TruckSim export indices must be distinct')
            result.signal_map = {'imports': dict(import_map), 'exports': dict(export_map)}
        result.solver_target_time = model.current_time
        # Preserve both coordinate-frame origins.  A heading offset requires a
        # matching rotation of the native displacement; translating native XY
        # alone is incorrect for curved scenes with a nonzero initial heading.
        result.scene_origin_xy = np.asarray(vehicle.position, dtype=float)[:2].copy()
        if result.control_mode == 'longitudinal':
            result.native_origin_xy = result.export_array[4:6].copy()
            native_heading = np.deg2rad(result.export_array[10])
        else:
            e = result.signal_map['exports']
            result.native_origin_xy = result.export_array[[e['x_m'], e['y_m']]].copy()
            native_heading = np.deg2rad(result.export_array[e['yaw_deg']])
        result.heading_offset = float(vehicle.heading - native_heading)
        result.frame_cos = float(np.cos(result.heading_offset))
        result.frame_sin = float(np.sin(result.heading_offset))
        result.steering_ratio = float(config['steering_ratio'])
        if not np.isfinite(result.steering_ratio) or result.steering_ratio <= 0:
            raise ValueError('TruckSim steering_ratio must be positive')
        result.engine_map = engine_map
        result.max_rpm = float(config['max_rpm'])
        # Optional alignment with the calibrated 10-Hz Koopman data collector.
        # Legacy TruckSim controllers retain their original solver-step gains.
        result.expert_koopman_aligned = bool(config.get('expert_koopman_aligned', False))
        result.actuator_update_dt_s = (float(config.get('actuator_update_dt_s', 0.1))
                                       if result.expert_koopman_aligned else result.solver_dt)
        if result.actuator_update_dt_s < result.solver_dt - 1e-9:
            raise ValueError('actuator_update_dt_s cannot be smaller than solver dt')
        result._next_actuator_update_time = None
        result.throttle_limit_pct = float(config.get('throttle_limit_pct', 35.0))
        result.brake_limit_mpa = float(config.get('brake_limit_mpa', 2.0))
        result.lower_ctrl_state = 'throttle'
        gains = config.get('longitudinal_pid', {}) if result.expert_koopman_aligned else {}
        result.low_controller = PIDController(
            kp=float(gains.get('kp',12)), ki=float(gains.get('ki',12)),
            kd=float(gains.get('kd',0.15)), dt=result.actuator_update_dt_s)
        result.lateral_controller = None
        if result.control_mode == 'longitudinal_lateral':
            lat = config.get('lateral_controller', {})
            required = {'kp', 'ki', 'kd', 'max_wheel_deg', 'max_rate_deg_s'}
            if not required.issubset(lat):
                raise ValueError(f'TruckSim.lateral_controller requires {sorted(required)}')
            result.lateral_controller = LateralSteeringController(
                steering_ratio=result.steering_ratio,
                dt=result.actuator_update_dt_s,
                kp=lat['kp'], ki=lat['ki'], kd=lat['kd'],
                max_wheel_deg=lat['max_wheel_deg'],
                max_rate_deg_s=lat['max_rate_deg_s'],
                steering_tau_s=lat.get('steering_tau_s', 0.0),
            )
        result.measured_acceleration = np.zeros(2)
        result.trucksim_inputs = np.zeros(model.configuration['n_import'])
        result._sync_state()
        return result

    def __deepcopy__(self, memo):
        # Planning snapshots are ordinary vehicles. Never copy or share native
        # solver handles with prediction, including copies reached via Road.
        result = FOLLOWVehicle.__new__(FOLLOWVehicle)
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            if name not in {'trucksim_model', 'low_controller', 'lateral_controller', 'engine_map'}:
                setattr(result, name, copy.deepcopy(value, memo))
        return result

    def _native_to_scene_position(self, native_xy):
        """Rigid-body SE(2) map using the native and scene initial poses."""
        delta = np.asarray(native_xy, dtype=float) - self.native_origin_xy
        c, si = self.frame_cos, self.frame_sin
        return self.scene_origin_xy + np.asarray(
            [c * delta[0] - si * delta[1],
             si * delta[0] + c * delta[1]], dtype=float)

    def _sync_state(self):
        export = np.asarray(self.export_array, dtype=float)
        if not np.all(np.isfinite(export)):
            raise RuntimeError("TruckSim returned non-finite state")
        if self.control_mode == 'longitudinal':
            self.position = self._native_to_scene_position(export[4:6])
            self.speed = float(np.hypot(export[2], export[3]) / 3.6)
            # Body sideslip (Vy/Vx) is not the vehicle yaw angle.
            self.heading = float(np.deg2rad(export[10]) + self.heading_offset)
            self.measured_acceleration = export[:2] * 9.8
            self.front_wheel_angle_deg = float((export[13] + export[14]) / 2)
            self.ltr = None
        else:
            e = self.signal_map['exports']
            self.position = self._native_to_scene_position(export[[e['x_m'], e['y_m']]])
            self.speed = float(np.hypot(export[e['vx_kmh']], export[e['vy_kmh']]) / 3.6)
            self.heading = float(np.deg2rad(export[e['yaw_deg']]) + self.heading_offset)
            self.measured_acceleration = np.array([export[e['ax_g']], export[e['ay_g']]]) * 9.8
            self.front_wheel_angle_deg = float((export[e['steer_left_deg']] +
                                                export[e['steer_right_deg']]) / 2)
            # LTR must be an actual exported signal for Koopman collection.
            self.ltr = float(export[e['ltr']]) if 'ltr' in e else None
            self.roll_rad = float(np.deg2rad(export[e['roll_deg']])) if 'roll_deg' in e else None
            self.roll_rate_rad_s = (float(np.deg2rad(export[e['roll_rate_deg_s']]))
                                    if 'roll_rate_deg_s' in e else None)
            self.yaw_rate_rad_s = (float(np.deg2rad(export[e['yaw_rate_deg_s']]))
                                   if 'yaw_rate_deg_s' in e else None)
        self.on_state_update()

    def step(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Vehicle time step must be positive')
        self.clip_actions()
        desired_acceleration = float(self.action['acceleration'])
        desired_steering_rad = float(self.action['steering'])
        steering_wheel_deg = float(np.rad2deg(desired_steering_rad) * self.steering_ratio)
        if not np.all(np.isfinite([desired_acceleration, steering_wheel_deg])):
            raise ValueError('TruckSim control inputs must be finite')
        # Accumulate absolute target time instead of truncating dt/solver_dt on
        # every call. At 15 Hz with 0.0025 s steps this yields 26,27,27,... steps.
        self.solver_target_time += dt
        while self.trucksim_model.current_time + self.solver_dt <= self.solver_target_time + 1e-10:
            # Koopman collection holds native inputs for 0.1 s.  Apply exactly
            # the same update cadence in expert scenes, including PID dt and
            # steering rate limits.  Original mode still updates every step.
            update_native = (not self.expert_koopman_aligned or
                             self._next_actuator_update_time is None or
                             self.trucksim_model.current_time >= self._next_actuator_update_time - 1e-9)
            if update_native:
                self._next_actuator_update_time = (
                    self.trucksim_model.current_time + self.actuator_update_dt_s)
            if update_native:
                export = self.export_array
                if self.control_mode == 'longitudinal':
                    vx_mps = export[2] / 3.6
                    engine_rpm = export[12]
                    ax_mps2 = export[0] * 9.8
                else:
                    e = self.signal_map['exports']
                    vx_mps = export[e['vx_kmh']] / 3.6
                    engine_rpm = export[e['engine_rpm']]
                    ax_mps2 = export[e['ax_g']] * 9.8
                throttle, brake, self.lower_ctrl_state = acceleration_control(
                    vx=vx_mps, ax_ref=desired_acceleration,
                    engine_rpm=engine_rpm, ax_actual=ax_mps2,
                    state=self.lower_ctrl_state, low_controller=self.low_controller,
                    engine_map=self.engine_map, max_rpm=self.max_rpm,
                )
                if self.control_mode == 'longitudinal_lateral':
                    e = self.signal_map['exports']
                    measured_delta_rad = np.deg2rad((export[e['steer_left_deg']] +
                                                      export[e['steer_right_deg']]) / 2)
                    steering_wheel_deg = self.lateral_controller.control(
                        desired_steering_rad, measured_delta_rad)
                    self.trucksim_inputs = np.zeros(self.trucksim_model.configuration['n_import'])
                    u = self.signal_map['imports']
                    self.trucksim_inputs[u['throttle']] = (
                        float(np.clip(100.0 * throttle, 0.0, self.throttle_limit_pct))
                        if self.expert_koopman_aligned else throttle)
                    self.trucksim_inputs[u['brake']] = (
                        float(np.clip(brake, 0.0, self.brake_limit_mpa))
                        if self.expert_koopman_aligned else brake)
                    self.trucksim_inputs[u['steering_wheel_deg']] = steering_wheel_deg
                else:
                    # Exact legacy I/O mapping and open-loop steering path.
                    self.trucksim_inputs = np.array([throttle, brake, steering_wheel_deg])
            status, self.export_array = self.trucksim_model.run(
                self.trucksim_model.current_time + self.solver_dt,
                self.trucksim_inputs, self.export_array,
            )
            if status:
                self.trucksim_model.stop(native_terminate=not self.expert_koopman_aligned)
                raise RuntimeError(f"TruckSim solver stopped with status {status}")
        # Keep command and measurement separate: action is the requested input.
        self._sync_state()
        self.timer += dt
        # Highway collision detection still determines episode termination.
        if self.impact is not None:
            if np.any(self.impact):
                self.crashed = True
            self.impact = None


def load_engine_map(config):
    path = config.get('engine_map_path')
    if path is None:
        path = Path(__file__).parent / 'trucksim_simulation' / 'engine_map_4455kg.csv'
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    if data.ndim != 2 or data.shape[1] != 12 or not np.all(np.isfinite(data)):
        raise ValueError('Engine map requires RPM and 11 throttle columns, all finite')
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError('Engine map RPM rows must be increasing')
    return data
