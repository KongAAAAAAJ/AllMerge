"""Adapted from multi_truck_lon_control/Data_driven_model; local to all_merge."""
import numpy as np

# 定义车辆和环境参数
params = {
    'm': 4455,  # kg
    'r_w': 0.51,  # m
    'Cd': 0.6,
    'A': 6.8,  # m²
    'rho': 1.206,  # kg/m³
    'Crr': 0.015,
    'gear_ratio': [7.59, 5.06, 3.38, 2.25, 1.5, 1.0, 0.75],
    'diff_ratio': 5.0,
    'eta': 0.9,
    'k_p': 1503.76,  # MPa/Nm
    'p_max': 10,  # MPa
}


def build_inverse_engine_model(rpm_grid, torque_grid, throttle_grid):
    """
    构建一个反查油门的插值器。
    rpm_grid      : 1D 数组 (N_rpm,)——发动机转速档位，例如 [1000, 1500, …, 6000]
    torque_grid   : 2D 数组 (N_rpm, N_throttle)——在各转速下，不同油门开度时的最大扭矩
    throttle_grid : 1D 数组 (N_throttle,)——油门开度档位，例如 [0.0, 0.1, …, 1.0]
    返回：一个函数 f(omega, T_req) → throttle
    """

    # 简化思路：对每个 rpm 用一维反插值（严格可用 root find）
    def inverse_lookup(rpm, T_req):
        # 对该 rpm 行做 1D 插值：throttle_grid → torque_grid_row
        T_row = torque_grid[np.searchsorted(rpm_grid, rpm).clip(0, len(rpm_grid) - 1), :]
        # 如果超出表格范围，进行截断
        T_row = np.clip(T_row, T_row.min(), T_row.max())
        # 一维反插：throttle = f⁻¹(T_req)
        return np.interp(T_req, T_row, throttle_grid)
    return inverse_lookup


def estimate_gear(N_e_rpm, v, params):
    """
    N_e_rpm : 发动机转速 (rpm)
    v       : 车速 (m/s)
    params  : {
      'r_w'           : 车轮半径 (m),
      'diff_ratio'    : 主减速比,
      'gear_ratios'   : list 各挡位传动比 [i1,i2,...],
      'error_tol'     : 匹配容差 (可选)
    }
    返回：gear_index (1~N), 若匹配失败返回 0
    """
    # 换算角速度
    omega_e = 2*np.pi/60 * N_e_rpm
    omega_w = v / params['r_w']
    if omega_w < 1e-3:
        return -1
    # 估算总传动比
    i_total = omega_e / (omega_w * params['diff_ratio'])
    # 与标定挡位比匹配
    ratios = np.array(params['gear_ratio'])
    errs   = np.abs(ratios - i_total)
    idx    = np.argmin(errs)
    if 'error_tol' in params and errs[idx] > params['error_tol']:
        return -1
    return int(idx)


def inverse_engine_model(v, ax, N_e_rpm, engine_map, max_rpm):
    """
    v      : 车辆速度 (m/s)
    a_ref  : 参考加速度 (m/s²)
    N_e_rpm : 发动机转速（rpm）
    返回：
        throttle (0.0~1.0)
    """

    # 1. 载入engine map 表

    rpm_grid = engine_map[:, 0]
    torque_grid = engine_map[:, 1:]
    throttle_grid = np.linspace(0, 1, 11)

    # 2. 构建发动机查表函数
    model = build_inverse_engine_model(rpm_grid, torque_grid, throttle_grid)

    # 3. 估算挡位
    gear = estimate_gear(N_e_rpm, v, params)
    if gear < 0:
        # 无效挡位时，返回保守值
        return 0.0

    m        = params['m']
    r_w      = params['r_w']
    Cd       = params['Cd']
    A        = params['A']
    rho      = params['rho']
    Crr      = params['Crr']
    i_g      = params['gear_ratio'][gear]
    i_d      = params['diff_ratio']
    eta      = params['eta']

    # 阻力加速度
    a_roll  = Crr * 9.8
    a_aero  = 0.5 * rho * Cd * A * v**2 / m

    # 车轮扭矩要求
    T_w = m * (ax + a_roll + a_aero) * r_w

    # 发动机扭矩要求
    T_e = T_w / (i_g * i_d * eta)

    # 发动机转速 (rad/s → RPM)
    omega_e = v / r_w * i_g * i_d    # rad/s
    rpm = omega_e * 60.0 / (2*np.pi)
    rpm = np.clip(rpm, 0, max_rpm)  # 超过max_rpm会产生拖滞转矩，用于安全限速

    # 反查油门
    throttle = model(rpm, T_e)

    # 截断到 [0,1]
    return float(np.clip(throttle, 0.0, 1.0))


def inverse_brake_model(v, ax):
    """
    输入：
        v      : 当前车速 (m/s)
        a_ref  : 参考加速度 (负值，m/s^2)
    输出：
        P_brake : 主缸压力 (MPa)，范围限制在 [0, p_max]
    """

    # 参数提取
    m    = params['m']
    R_w  = params['r_w']
    Crr  = params['Crr']
    rho  = params['rho']
    Cd   = params['Cd']
    A    = params['A']
    k_p  = params['k_p']
    p_max = params['p_max']

    # 1. 滚阻 + 空气阻力
    F_roll = Crr * m * 9.81
    F_aero = 0.5 * rho * Cd * A * v**2
    F_resist = F_roll + F_aero

    # 2. 计算所需总制动力（含克服阻力）
    F_brake = -m * ax + F_resist  # a_ref 为负值，符号相加
    F_brake = max(F_brake, 0.0)  # 不可能是负的

    # 3. 制动器力矩
    T_b = F_brake * R_w

    # 4. 主缸压力（反查）
    P_brake = T_b / k_p

    # 限制最大压力
    P_brake = min(P_brake, p_max)
    return P_brake


def find_brake_edge(v):
    """无油门刹车输入时的最大制动减速度 数据从Trucksim中获取"""
    # [km/h] [g]
    a_b_max_table = np.array([[5, -0.01342],
                              [10, -0.0075],
                              [15, -0.0075],
                              [20, -0.015],
                              [25, -0.0114],
                              [30, -0.01425],
                              [35, -0.0165],
                              [40, -0.01885],
                              [45, -0.02134],
                              [50, -0.02396],
                              [55, -0.02670],
                              [60, -0.03960],
                              [65, -0.03268],
                              [70, -0.03589],
                              [75, -0.03925],
                              [80, -0.04230],
                              [85, -0.06270],
                              [90, -0.08291],
                              [95, -0.1026],
                              [100, -0.1223],
                              [105, -0.14085],
                              [110, -0.1581]])
    a_b_max = np.interp(v * 3.6, a_b_max_table[:, 0], a_b_max_table[:, 1])
    return a_b_max * 9.8  # [m/s^2]


