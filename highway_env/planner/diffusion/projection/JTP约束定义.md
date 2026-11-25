# JTP 约束汇总

当前 JTP 包含 7 个物理约束族；优先级组是独立的求解分组。下表对应 `configs/projection/jtp.yaml` 的默认配置；运行时若覆盖配置，以实际加载值为准。

记 `mu_e = max(road_mu, 0.05)`，`m = max(payload_kg, 0)`（kg），`clip(x, lo, hi)` 为区间截断。`H = 40`、`dt = 0.1 s`，默认预测时域为 4 s。

| 约束组 | 约束量 | 下限 | 上限 | 默认参数、计算方式及适用范围 | 实现位置 |
| --- | --- | --- | --- | --- | --- |
| 稳定性 `stability` | 横摆角速度 `r`（rad/s） | `-r_max` | `+r_max` | `r_max = 0.85 * mu_e * 9.81 / max(abs(vx), 3.0)`；随各预测步纵向速度及道路附着系数变化；检查未来 `t = 1…H`。例如 `mu = 0.8, vx = 20 m/s` 时为约 `±0.334 rad/s`。 | [constraints.py](constraints.py)：`stability_limits`、`stability_residual` |
| 稳定性 `stability` | 侧偏角 `beta = atan2(vy, vx)` | `-beta_max` | `+beta_max` | `beta_max = 5° * clip(sqrt(mu_e / 0.8), 0.65, 1.15)`；限值幅度为 `3.25°…5.75°`；`mu = 0.8` 时为 `±5°`；检查未来 `t = 1…H`。代码内部使用 rad。 | [constraints.py](constraints.py)：`stability_limits`、`stability_residual` |
| 稳定性 `stability` | 侧倾角 `phi` | `-phi_max` | `+phi_max` | `phi_max = 5° * clip(sqrt(mu_e / 0.8), 0.55, 1.10) / (1 + 0.20 * m / 10000)`；载荷越大限值越小。例如 `mu = 0.8, m = 10000 kg` 时为约 `±4.17°`；检查未来 `t = 1…H`。代码内部使用 rad。 | [constraints.py](constraints.py)：`stability_limits`、`stability_residual` |
| 稳定性 `stability` | 载荷转移率 `LTR`（无量纲） | `-0.80` | `+0.80` | 固定双侧限值；检查未来 `t = 1…H` 的预测 LTR。 | [constraints.py](constraints.py)：`stability_residual` |
| 道路 `road` | 每个车辆轮廓圆心相对关联道路中心线的有符号横向偏移 `d`（m，左正右负） | `-w_R + 1.80` | `w_L - 1.80` | `w_L`、`w_R` 为在线道路几何提供、沿关联线段插值得到的左右半宽。车身纵向圆心偏移为 `[-3, -1.5, 0, 1.5, 3] m`，各圆半径 `1.45 m`，额外净空 `0.35 m`，合计边界距离 `1.80 m`；检查未来 `t = 1…H` 的全部 5 个圆心。 | [geometry.py](geometry.py)：`footprint_circle_centers`、`road_corridor_residual` |
| 曲率 `curvature` | 离散轨迹曲率 `kappa`（m⁻¹） | `-0.025` | `+0.025` | `kappa = wrap_angle(psi[t+1] - psi[t]) / max(norm(p[t+1] - p[t]), 1e-4 m)`，`t = 1…H-1`；实现残差为 `kappa² - 0.025²`。对应曲率半径至少 `40 m`（非零曲率时）。 | [geometry.py](geometry.py)：`curvature_residual` |
| 控制 `control` | 纵向加速度 `a_x`（m/s²） | `-4.0` | `+2.5` | 对所有车辆、所有控制节点施加幅值约束；默认控制节点数 `K = H = 40`。 | [joint_projector.py](joint_projector.py)：`_constraint_groups` |
| 控制 `control` | 前轮转向角 `delta_f`（rad） | `-0.08`（约 `-4.58°`） | `+0.08`（约 `+4.58°`） | 对所有车辆、所有控制节点施加幅值约束；约束的是前轮转向角。 | [joint_projector.py](joint_projector.py)：`_constraint_groups` |

| 编队内部碰撞 `intertruck_collision` | 任意两辆车的覆盖圆圆心距离 | `2*1.45+0.35 m` | 无 | 对同步未来时刻全部跨车圆对检查。 | `geometry.py` |
| 外部碰撞 `external_collision` | 主车覆盖圆与外部车辆覆盖圆距离 | `1.45+r_actor+0.35 m` | 无 | 外部车辆预测轨迹固定；圆半径依外车宽度确定。 | `geometry.py` |
| 控制变化率 `control_rate` | 加速度变化率、前轮转角变化率 | `-2.5 m/s³`, `-0.12 rad/s` | `2.5 m/s³`, `0.12 rad/s` | 0.1 s 差分包含上一时刻实际控制。 | `constraints.py` |

物理阈值不因 Stage 1 求解器改变。稳定性和道路等内部约束使用归一化残差容差；控制变化率保留 SI 容差。论文独立评价器仍使用其原配置。
