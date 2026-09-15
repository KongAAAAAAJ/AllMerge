import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.interpolate import interp1d

class TruckHighFidelityDynamics:
    """
    高保真卡车动力学模型（接受油门开度 throttle_cmd 输入）:
    - 多体悬挂
    - 发动机-变速箱动力链
    - 轮胎纵横向耦合
    - 制动 & 传动延迟
    """

    def __init__(self, dt=0.005, drivetrain_delay=0.3):
        self.m = 30000.0
        self.Izz = 4e6
        self.Iyy = 2.5e6
        self.Ixx = 3e5
        self.lf = 3.5
        self.lr = 4.5
        self.h = 1.2
        self.g = 9.81
        self.dt = dt

        rpm = np.array([800,1200,1600,2000,2400,2800,3200])
        torque = np.array([2000,2400,2600,2500,2300,2100,1800])  # Nm
        self.engine_map = interp1d(rpm, torque, fill_value="extrapolate")
        self.gear_ratios = [12,8,5,3,1.5,1.0]
        self.trans_eff = 0.9
        self.wheel_radius = 0.6

        self.kf = 3e6
        self.kr = 4e6
        self.cf = 4e4
        self.cr = 5e4

        self.pacejka = {
            'B': np.array([10,12]),
            'C': np.array([1.9,1.9]),
            'D': np.array([1.2e5,1.2e5]),
            'E': np.array([-1,-1])
        }

        self.delay_steps = int(drivetrain_delay / dt)
        self.throttle_buf = deque([0.0]*self.delay_steps, maxlen=self.delay_steps)

        self.max_brake = 3e5
        self.brake_eff = 0.9

        self.reset()

    def engine_torque(self, throttle, engine_speed):
        rpm = engine_speed * 60 / (2*np.pi)
        return self.engine_map(rpm) * throttle

    def shift_logic(self, engine_speed, gear):
        up = [2000,2200,2400,2600,2800]
        if gear < len(self.gear_ratios) and engine_speed > up[gear-1]:
            gear += 1
        return gear

    def pacejka_F(self, alpha, Fz, idx):
        B = self.pacejka['B'][idx]
        C = self.pacejka['C'][idx]
        D = self.pacejka['D'][idx]
        E = self.pacejka['E'][idx]
        return D * np.sin(C * np.arctan(B*alpha - E*(B*alpha - np.arctan(B*alpha)))) * (Fz/(self.m*self.g))

    def compute_air_drag(self, vx):
        rho = 1.225; Cd = 0.6; A = 8.0
        return 0.5 * rho * A * Cd * vx**2

    def step(self, throttle_cmd, delta, brake_cmd):
        self.throttle_buf.append(throttle_cmd)
        throttle = self.throttle_buf[0]

        engine_speed = self.vx/self.wheel_radius * self.gear_ratios[self.gear-1]
        torque = self.engine_torque(throttle, engine_speed)
        Fx_prop = torque * self.trans_eff / self.wheel_radius

        brake_force = self.max_brake * brake_cmd * self.brake_eff

        Fz_static = self.m*self.g/2
        dFz = self.roll * self.h * self.m * self.g / (self.lf + self.lr)
        Fz_f = Fz_static - dFz; Fz_r = Fz_static + dFz

        alpha_f = (self.vy + self.lf * self.yaw_dot) / max(self.vx, 0.1) - delta
        alpha_r = (self.vy - self.lr * self.yaw_dot) / max(self.vx, 0.1)
        Fy_f = self.pacejka_F(alpha_f, Fz_f, 0)
        Fy_r = self.pacejka_F(alpha_r, Fz_r, 1)

        Fzf = self.kf * (-self.wheel_defl_front) + self.cf * (-self.pitch_dot)
        Fzr = self.kr * (-self.wheel_defl_rear) + self.cr * (-self.pitch_dot)

        air_drag = self.compute_air_drag(self.vx)

        Fx = Fx_prop - brake_force - air_drag

        dvx = Fx / self.m + self.vy * self.yaw_dot + Fy_f * np.sin(delta) / self.m
        dvy = (Fy_f*np.cos(delta) + Fy_r + Fzf + Fzr)/self.m - self.vx*self.yaw_dot
        dpsi_dot = (self.lf*Fy_f*np.cos(delta) - self.lr*Fy_r)/self.Izz

        pitch_torque = self.h*(Fx - (Fzf+Fzr)) - 5e5*self.pitch_dot - 1e6*np.tanh(self.pitch)
        dpitch_dot = pitch_torque/self.Iyy
        roll_torque = self.h*(Fy_f+Fy_r) - 3e4*self.roll_dot - 5e5*np.tanh(self.roll)
        droll_dot = roll_torque/self.Ixx

        self.vx += dvx * self.dt; self.vy += dvy * self.dt
        self.x += (self.vx * np.cos(self.yaw) - self.vy * np.sin(self.yaw)) * self.dt
        self.y += (self.vx * np.sin(self.yaw) + self.vy * np.cos(self.yaw)) * self.dt
        self.yaw_dot += dpsi_dot * self.dt; self.yaw += self.yaw_dot * self.dt
        self.pitch_dot += dpitch_dot * self.dt; self.pitch += self.pitch_dot * self.dt
        self.roll_dot += droll_dot * self.dt; self.roll += self.roll_dot * self.dt

        self.gear = self.shift_logic(engine_speed, self.gear)

        return [dvx, dvy, self.vx, self.vy, self.x, self.y, self.roll, self.roll_dot,
                self.pitch, self.pitch_dot, self.yaw, self.yaw_dot]

    def reset(self):
        self.vx = self.vy = self.x = self.y = self.yaw = self.yaw_dot = 0.0
        self.pitch = self.pitch_dot = self.roll = self.roll_dot = 0.0
        self.wheel_defl_front = self.wheel_defl_rear = 0.0
        self.gear = 1
        self.throttle_buf.clear()

if __name__ == '__main__':
    model = TruckHighFidelityDynamics(dt=0.01, drivetrain_delay=0.3)
    T = 10.0
    steps = int(T / model.dt)

    throttle_profile = np.concatenate([np.ones(steps//3)*0.7,
                                       np.zeros(steps//3),
                                       np.zeros(steps - 2*(steps//3))])
    delta_profile = 0.05 * np.sin(np.linspace(0, 2*np.pi, steps))
    brake_profile = np.concatenate([np.zeros(2*steps//3), np.ones(steps - 2*steps//3)])

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
