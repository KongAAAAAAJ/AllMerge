"""TruckSim plant for the existing FOLLOWVehicle decision/control interface."""
import copy
from pathlib import Path

import numpy as np

from highway_env.vehicle.controller import FOLLOWVehicle
from highway_env.vehicle.trucksim_simulation.low_controller import PIDController, acceleration_control


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
        if model.configuration['n_import'] != 3 or model.configuration['n_export'] != 15 or len(result.export_array) != 15:
            raise ValueError("Expected TruckSim 3 imports and 15 exports (including steer_l1, steer_r1); see README")
        result.solver_target_time = model.current_time
        result.position_offset = vehicle.position - result.export_array[4:6]
        result.heading_offset = vehicle.heading - np.deg2rad(result.export_array[10])
        result.steering_ratio = float(config['steering_ratio'])
        if not np.isfinite(result.steering_ratio) or result.steering_ratio <= 0:
            raise ValueError('TruckSim steering_ratio must be positive')
        result.engine_map = engine_map
        result.max_rpm = float(config['max_rpm'])
        result.lower_ctrl_state = 'throttle'
        result.low_controller = PIDController(kp=12, ki=12, kd=0.15, dt=result.solver_dt)
        result.measured_acceleration = np.zeros(2)
        result.trucksim_inputs = np.zeros(3)
        result._sync_state()
        return result

    def __deepcopy__(self, memo):
        # Planning snapshots are ordinary vehicles. Never copy or share native
        # solver handles with prediction, including copies reached via Road.
        result = FOLLOWVehicle.__new__(FOLLOWVehicle)
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            if name not in {'trucksim_model', 'low_controller', 'engine_map'}:
                setattr(result, name, copy.deepcopy(value, memo))
        return result

    def _sync_state(self):
        export = np.asarray(self.export_array, dtype=float)
        if not np.all(np.isfinite(export)):
            raise RuntimeError("TruckSim returned non-finite state")
        self.position = export[4:6] + self.position_offset
        self.speed = float(np.hypot(export[2], export[3]) / 3.6)
        # Body sideslip (Vy/Vx) is not the vehicle yaw angle.
        self.heading = float(np.deg2rad(export[10]) + self.heading_offset)
        self.measured_acceleration = export[:2] * 9.8
        self.front_wheel_angle_deg = float((export[13] + export[14]) / 2)
        self.on_state_update()

    def step(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Vehicle time step must be positive')
        self.clip_actions()
        desired_acceleration = float(self.action['acceleration'])
        steering_wheel_deg = float(np.rad2deg(self.action['steering']) * self.steering_ratio)
        if not np.all(np.isfinite([desired_acceleration, steering_wheel_deg])):
            raise ValueError('TruckSim control inputs must be finite')
        # Accumulate absolute target time instead of truncating dt/solver_dt on
        # every call. At 15 Hz with 0.0025 s steps this yields 26,27,27,... steps.
        self.solver_target_time += dt
        while self.trucksim_model.current_time + self.solver_dt <= self.solver_target_time + 1e-10:
            export = self.export_array
            throttle, brake, self.lower_ctrl_state = acceleration_control(
                vx=export[2] / 3.6, ax_ref=desired_acceleration,
                engine_rpm=export[12], ax_actual=export[0] * 9.8,
                state=self.lower_ctrl_state, low_controller=self.low_controller,
                engine_map=self.engine_map, max_rpm=self.max_rpm,
            )
            self.trucksim_inputs = np.array([throttle, brake, steering_wheel_deg])
            status, self.export_array = self.trucksim_model.run(
                self.trucksim_model.current_time + self.solver_dt,
                self.trucksim_inputs, self.export_array,
            )
            if status:
                self.trucksim_model.stop()
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
