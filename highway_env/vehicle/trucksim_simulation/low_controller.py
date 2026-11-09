"""Adapted from multi_truck_lon_control/Data_driven_model; local to all_merge."""
import numpy as np
from .inverse_dynamics import inverse_brake_model, inverse_engine_model, find_brake_edge


Throttle_limits = (0, 1)  # [-]
Brake_limits = (0, 10)  # [Mpa]
Acceleration_limits = (-7.5, 1.5)  # m/s^2
"""Apply_limits: Apply limits of actuator"""


class PIDController:
    def __init__(self, kp, ki, kd, dt=0.001, output_limits=(-7.5, 1.5)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.output_min, self.output_max = output_limits

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def control(self, error):
        # PID计算
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = np.clip(output, self.output_min, self.output_max)  # 限幅

        return output


def acceleration_control(vx, engine_rpm, ax_ref, ax_actual, state, low_controller, engine_map, max_rpm):
    """
    输入：
        ax_ref:     参考加速度 (m/s²)
        ax_actual:  当前车辆实际加速度 (m/s²)

    输出：
        throttle: 油门开度 (0~1)
        brake:    主缸制动压力 (0~10Mpa)
    """
    error = ax_ref - ax_actual
    a_ctrl = low_controller.control(error)
    a_b_max = find_brake_edge(v=vx)

    if ax_ref >= a_b_max + 0.1:
        state = 'throttle'
    elif ax_ref <= a_b_max - 0.1:
        state = 'brake'

    # feedforward control: inverse engine koop_model and inverse brake koop_model
    # t1 = time.time()
    throttle = inverse_engine_model(v=vx, ax=a_ctrl, N_e_rpm=engine_rpm, engine_map=engine_map, max_rpm=max_rpm) if state == 'throttle' else 0.0
    # t2 = time.time()
    # print(f"inverse t = {t2 - t1}s")
    brake = inverse_brake_model(v=vx, ax=a_ctrl) if state == 'brake' else 0.0

    return throttle, brake, state




class LateralSteeringController:
    """Road-wheel angle tracking with steering-wheel feedforward and PID correction.

    Use only in longitudinal_lateral mode. No dependence on TruckSim exports or
    the neural Koopman model. Gain/limit calibration belongs to the .sim model.
    """

    def __init__(self, *, steering_ratio, dt, kp, ki, kd,
                 max_wheel_deg, max_rate_deg_s, steering_tau_s=0.0):
        from .inverse_dynamics import inverse_steering_dynamics
        self._inverse = inverse_steering_dynamics
        vals = [steering_ratio, dt, kp, ki, kd, max_wheel_deg, max_rate_deg_s, steering_tau_s]
        if not np.all(np.isfinite(vals)) or steering_ratio <= 0 or dt <= 0:
            raise ValueError('Invalid lateral steering controller parameters')
        if max_wheel_deg <= 0 or max_rate_deg_s <= 0 or steering_tau_s < 0:
            raise ValueError('Steering limits must be positive; tau must be >= 0')
        self.ratio = float(steering_ratio)
        self.dt = float(dt)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.max_wheel_deg = float(max_wheel_deg)
        self.max_rate_deg_s = float(max_rate_deg_s)
        self.steering_tau_s = float(steering_tau_s)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_ref = None
        self.prev_command = None

    def control(self, delta_ref_rad, delta_measured_rad):
        if not np.all(np.isfinite([delta_ref_rad, delta_measured_rad])):
            raise ValueError('Steering reference/feedback must be finite')
        error = float(delta_ref_rad - delta_measured_rad)
        # Derivative kick at the first sample is avoided.
        derivative = 0.0 if self.prev_ref is None else (error - self.prev_error) / self.dt
        candidate_integral = self.integral + error * self.dt
        correction_rad = (self.kp * error + self.ki * candidate_integral
                          + self.kd * derivative)
        command_ff_deg = self._inverse(
            delta_ref_rad, self.ratio, previous_ref_rad=self.prev_ref,
            dt=self.dt, steering_tau_s=self.steering_tau_s if self.prev_ref is not None else 0.0,
        )
        raw_deg = command_ff_deg + np.rad2deg(correction_rad) * self.ratio
        bounded_deg = float(np.clip(raw_deg, -self.max_wheel_deg, self.max_wheel_deg))
        if self.prev_command is not None:
            rate_limit = self.max_rate_deg_s * self.dt
            bounded_deg = float(np.clip(
                bounded_deg, self.prev_command - rate_limit, self.prev_command + rate_limit,
            ))
        # Conditional integration: avoid accumulating error when saturated.
        if abs(bounded_deg - raw_deg) < 1e-9 or np.sign(error) != np.sign(raw_deg - bounded_deg):
            self.integral = candidate_integral
        self.prev_error = error
        self.prev_ref = float(delta_ref_rad)
        self.prev_command = bounded_deg
        return bounded_deg
