import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class Truck5DoFDynamics:
    def __init__(self, dt=0.01, drivetrain_delay=0.5):
        self.dt = dt  # 仿真步长

        # 车辆参数
        self.m = 4455
        self.lf = 1.25
        self.lr = 4.0
        self.Izz = 34802
        self.Iyy = 35402.8
        self.Ixx = 2283.9
        self.h = 1.175
        self.C_alpha_f = 6557.83
        self.C_alpha_r = 6557.83
        self.roll_stiffness = 8500
        self.roll_damping = 15000
        self.pitch_stiffness = 8500
        self.pitch_damping = 15000

        # 驱动/制动参数
        self.max_engine_force = 11052
        self.max_brake_force = 19607
        self.brake_efficiency = 0.95

        # 空气阻力参数
        self.air_drag_coefficient = 0.3
        self.frontal_area = 6.8
        self.air_density = 1.206

        # 延迟设置（仅作用于油门）
        self.delay_steps = int(drivetrain_delay / dt)
        self.throttle_buffer = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)

        self.reset()

    def reset(self):
        self.vx = 5
        self.vy = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.yaw_dot = 0.0
        self.pitch = 0.0
        self.pitch_dot = 0.0
        self.roll = 0.0
        self.roll_dot = 0.0
        self.stopped = False  # 车辆是否处于停止状态

    def tire_forces(self, vx, vy, psi_dot, delta):
        if abs(vx) < 0.1:
            vx = 0.1
        alpha_f = np.arctan2((vy + self.lf * psi_dot), vx) - delta
        alpha_r = np.arctan2((vy - self.lr * psi_dot), vx)
        Fy_f = -self.C_alpha_f * alpha_f
        Fy_r = -self.C_alpha_r * alpha_r
        return Fy_f, Fy_r

    def compute_air_drag(self, vx):
        return 0.5 * self.air_density * self.frontal_area * self.air_drag_coefficient * vx ** 2

    def step(self, throttle, delta, brake):
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        self.throttle_buffer.append(throttle)
        throttle_delayed = self.throttle_buffer[0]

        Fx_drive = throttle_delayed * self.max_engine_force
        Fx_brake = brake * self.max_brake_force * self.brake_efficiency
        air_drag = self.compute_air_drag(self.vx)
        Fx = Fx_drive - Fx_brake - air_drag

        Fy_f, Fy_r = self.tire_forces(self.vx, self.vy, self.yaw_dot, delta)

        dvx = (Fx / self.m + self.vy * self.yaw_dot)
        dvy = (Fy_f + Fy_r) / self.m - self.vx * self.yaw_dot
        dyaw_dot = (self.lf * Fy_f - self.lr * Fy_r) / self.Izz

        pitch_torque = self.h * Fx - self.pitch_damping * self.pitch_dot - self.pitch_stiffness * np.tanh(self.pitch)
        dpitch_dot = pitch_torque / self.Iyy

        lateral_force = Fy_f + Fy_r
        roll_torque = self.h * lateral_force - self.roll_damping * self.roll_dot - self.roll_stiffness * np.tanh(self.roll)
        droll_dot = roll_torque / self.Ixx

        # 停车保持与起步逻辑
        if self.stopped:
            if throttle > 0.02:
                self.stopped = False
        if not self.stopped:
            self.vx += dvx * self.dt
            if self.vx < 0.01:
                self.vx = 0.0
                self.stopped = True
            self.vy += dvy * self.dt
            self.x += (self.vx * np.cos(self.yaw) - self.vy * np.sin(self.yaw)) * self.dt
            self.y += (self.vx * np.sin(self.yaw) + self.vy * np.cos(self.yaw)) * self.dt
        else:
            dvx = 0.0
            dvy = 0.0

        self.yaw_dot += dyaw_dot * self.dt
        self.yaw += self.yaw_dot * self.dt
        self.pitch_dot += dpitch_dot * self.dt
        self.pitch += self.pitch_dot * self.dt
        self.roll_dot += droll_dot * self.dt
        self.roll += self.roll_dot * self.dt

        ax = dvx
        ay = dvy + self.vx * self.yaw_dot
        return [ax, ay, self.vx, self.vy, self.x, self.y, self.roll, self.roll_dot,
                self.pitch, self.pitch_dot, self.yaw, self.yaw_dot]

if __name__ == '__main__':
    model = Truck5DoFDynamics(dt=0.01, drivetrain_delay=0.3)
    T = 10.0
    steps = int(T / model.dt)

    delta_profile = 0.5 * np.ones(steps)
    throttle_profile = 0.0 * np.ones(steps)
    brake_profile = 0 * np.ones(steps)

    history = []
    for i in range(steps):
        out = model.step(throttle_profile[i], delta_profile[i], brake_profile[i])
        history.append(out)
    data = np.array(history)
    t = np.arange(steps) * model.dt

    labels = ["dvx","dvy","vx","vy","x","y","roll","roll_dot","pitch","pitch_dot","yaw","yaw_dot"]
    fig, axs = plt.subplots(6, 2, figsize=(14, 12))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.plot(t, data[:, i])
        ax.set_title(labels[i])
        ax.grid(True)
    plt.tight_layout()
    plt.show()
