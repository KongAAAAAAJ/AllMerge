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


