# JTP 约束汇总

当前 JTP 包含六类参与优先级分组的状态约束，以及一类固定控制范围硬约束。控制范围不参与优先级评分或活动集截断，并在每个 LS 子问题中共同施加。下表对应 `configs/projection/jtp.yaml` 的默认配置；运行时若覆盖配置，以实际加载值为准。

记 `mu_e = max(road_mu, 0.05)`，`m = max(payload_kg, 0)`（kg），`clip(x, lo, hi)` 为区间截断。`H = 40`、`dt = 0.1 s`，默认预测时域为 4 s。

| 约束族 / 类型 | 约束量 | 下限 | 上限 | 默认参数、计算方式及适用范围 | 实现位置 |
| --- | --- | --- | --- | --- | --- |
| 稳定性 `stability` | 横摆角速度 `r`（rad/s） | `-r_max` | `+r_max` | `r_max = 0.85 * mu_e * 9.81 / max(abs(vx), 3.0)`；随各预测步纵向速度及道路附着系数变化；检查未来 `t = 1…H`。例如 `mu = 0.8, vx = 20 m/s` 时为约 `±0.334 rad/s`。 | [constraints.py](constraints.py)：`stability_limits`、`stability_residual` |
| 稳定性 `stability` | 侧偏角 `beta = atan2(vy, vx)` | `-beta_max` | `+beta_max` | `beta_max = 5° * clip(sqrt(mu_e / 0.8), 0.65, 1.15)`；限值幅度为 `3.25°…5.75°`；`mu = 0.8` 时为 `±5°`；检查未来 `t = 1…H`。代码内部使用 rad。 | [constraints.py](constraints.py)：`stability_limits`、`stability_residual` |
| 稳定性 `stability` | 侧倾角 `phi` | `-phi_max` | `+phi_max` | `phi_max = 5° * clip(sqrt(mu_e / 0.8), 0.55, 1.10) / (1 + 0.20 * m / 10000)`；载荷越大限值越小。例如 `mu = 0.8, m = 10000 kg` 时为约 `±4.17°`；检查未来 `t = 1…H`。代码内部使用 rad。 | [constraints.py](constraints.py)：`stability_limits`、`stability_residual` |
| 稳定性 `stability` | 载荷转移率 `LTR`（无量纲） | `-0.80` | `+0.80` | 固定双侧限值；检查未来 `t = 1…H` 的预测 LTR。 | [constraints.py](constraints.py)：`stability_residual` |
| 道路 `road` | 每个车辆轮廓圆心相对关联道路中心线的有符号横向偏移 `d`（m，左正右负） | `-w_R + 1.80` | `w_L - 1.80` | `w_L`、`w_R` 为在线道路几何提供、沿关联线段插值得到的左右半宽。车身纵向圆心偏移为 `[-3, -1.5, 0, 1.5, 3] m`，各圆半径 `1.45 m`，额外净空 `0.35 m`，合计边界距离 `1.80 m`；检查未来 `t = 1…H` 的全部 5 个圆心。 | [geometry.py](geometry.py)：`footprint_circle_centers`、`road_corridor_residual` |
| 曲率 `curvature` | 离散轨迹曲率 `kappa`（m⁻¹） | `-0.025` | `+0.025` | `kappa = wrap_angle(psi[t+1] - psi[t]) / max(norm(p[t+1] - p[t]), 1e-4 m)`，`t = 1…H-1`；实现残差为 `kappa² - 0.025²`。对应曲率半径至少 `40 m`（非零曲率时）。 | [geometry.py](geometry.py)：`curvature_residual` |
| 固定硬约束 `control` | 纵向加速度 `a_x`（m/s²） | `-4.0` | `+2.5` | 对所有车辆、所有控制节点施加幅值硬约束；默认控制节点数 `K = H = 40`。每个优先级组 LS 子问题均满足该范围。 | [joint_projector.py](joint_projector.py)：`_constraint_groups`；[grouped_solver.py](grouped_solver.py)：`solve_groups` |
| 固定硬约束 `control` | 前轮转向角 `delta_f`（rad） | `-0.08`（约 `-4.58°`） | `+0.08`（约 `+4.58°`） | 对所有车辆、所有控制节点施加幅值硬约束；约束的是前轮转向角，不参与状态约束优先级排序。 | [joint_projector.py](joint_projector.py)：`_constraint_groups`；[grouped_solver.py](grouped_solver.py)：`solve_groups` |

| 编队内部碰撞 `intertruck_collision` | 任意两辆车的覆盖圆圆心距离 | `2*1.45+0.35 m` | 无 | 对同步未来时刻全部跨车圆对检查。 | `geometry.py` |
| 外部碰撞 `external_collision` | 主车覆盖圆与外部车辆覆盖圆距离 | `1.45+r_actor+0.35 m` | 无 | 外部车辆预测轨迹固定；圆半径依外车宽度确定。 | `geometry.py` |
| 控制变化率 `control_rate` | 加速度变化率、前轮转角变化率 | `-2.5 m/s³`, `-0.12 rad/s` | `2.5 m/s³`, `0.12 rad/s` | 0.1 s 差分包含上一时刻实际控制。 | `constraints.py` |

物理阈值不因 Stage 1 求解器改变。稳定性、道路、曲率和碰撞等状态约束使用统一的归一化残差容差；控制变化率保留 SI 容差，并在活动约束筛选、LS 残差构造和非线性评价中使用同一容差映射。控制范围硬约束不使用正容差。Stage 1 保持道路约束五圆心、40步的完整非线性验证，LS 道路活动集采样及上限暂不改变。论文独立评价器仍使用其原配置。

## Stage 2 非线性接受规则

对每个状态约束族 `f`，使用统一容差张量 `tau_f` 定义逐行超容差违反程度：

`v_f = relu(g_f - tau_f)`。

对每个优先级组 `G_k`，候选控制序列必须同时满足：

1. 该组所有约束行均不退化：`v_new[f,j] <= v_old[f,j] + nondegradation_tolerance`；
2. 该组违反约束数量不增加：`count(v_new[f] > 0) <= count(v_old[f] > 0)`，其中数量在该优先级组内跨全部约束族累计；
3. 至少一个优先级组出现单条约束的有效改善，或违反约束数量减少。

固定控制范围硬约束始终单独判定。若名义控制量原本超出控制范围，允许接受硬约束恢复，但仍要求所有状态约束行不退化且各组违反数量不增加。

一次 Backtracking line search 内冻结最近道路关联，保证投影前后道路约束行逐一对应。候选被接受后，使用已计算的 K2 rollout 重新计算最近道路关联，供下一次外层迭代使用。
