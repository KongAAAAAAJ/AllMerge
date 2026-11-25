# JTP 约束汇总

当前 JTP 包含 4 个约束组、8 类约束。下表对应 `configs/projection/jtp.yaml` 的默认配置；运行时若覆盖配置，以实际加载值为准。

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

## JTP 求解伪代码（旧）

**算法 1：基于活跃集加权 Jacobian 的联合轨迹投影**

**输入：** 名义控制序列 $U_0$，车辆上下文 $C$，道路几何 $R$，冻结的 K2 预测模型 $F$。  
**输出：** 修正控制 $U^\star$、对应预测结果 $F(U^\star;C)$ 及约束满足标志。

令 $c(U)$ 为全部归一化约束残差，$v(U)=\|[c(U)]_+\|_\infty$ 为最大违反程度，其中 $[\cdot]_+=\max(\cdot,0)$。以名义轨迹 $\tau_0=\tau(U_0)$ 定义评价函数：

$$
\Phi(U)=\frac12\|\tau(U)-\tau_0\|_{W_\tau}^2
+\frac{\lambda_u}{2}\|U-U_0\|_{W_u}^2
+\rho\|[c(U)]_+\|_2^2.
$$

| 步骤 | 伪代码 |
| --- | --- |
| 1 | **初始化** $U\leftarrow U_0$，信赖域半径 $\Delta$ 和阻尼 $\lambda$。 |
| 2 | **for** $k=1,\ldots,K_{\max}$ **do** |
| 3 | 　 使用 K2 预测轨迹与车辆状态，更新道路关联并计算全部约束 $c(U)$；若 $v(U)\leq\varepsilon$，则终止。 |
| 4 | 　 按组选择违反最严重的约束，构成活跃集 $\mathcal A$；固定本轮道路关联，计算 $r=c_{\mathcal A}(U)$ 与 $J=\partial r/\partial U$。 |
| 5 | 　 构造违反程度权重 $D$ 和局部度量 $M$，令 $A=DJ$、$b=D[r]_+$。 |
| 6 | 　 **联合修正：** $\delta U\leftarrow-M^{-1}A^\top(AM^{-1}A^\top+\lambda I)^{-1}b$。 |
| 7 | 　 按控制尺度逐节点限制 $\delta U$ 的幅度，使其位于信赖域内。 |
| 8 | 　 **回溯线搜索：** 构造 $\widetilde U=\Pi_{\mathcal U}(U+\alpha\delta U)$，重新进行 K2 预测并检查全部约束。 |
| 9 | 　 若 $\Phi$ 充分下降、$v$ 充分下降或候选满足全部约束，则接受 $U\leftarrow\widetilde U$；否则减小 $\alpha$ 重试，直至步长下限。 |
| 10 | 　 根据接受情况调整信赖域和阻尼；若修正求解失败，则收缩信赖域、增大阻尼并重试；若持续停滞或信赖域过小，则终止。 |
| 11 | **end for** |
| 12 | **返回** $U^\star\leftarrow U$、最终 K2 预测结果及 $v(U^\star)\leq\varepsilon$。 |

其中，$\Pi_{\mathcal U}$ 表示加速度和转向角幅值截断；$D$ 提高严重违反约束的权重；$M=J_\tau^\top W_\tau J_\tau+\lambda_uW_u+\eta S^{-2}$ 综合轨迹变化、控制变化与数值正则化，$S$ 为控制尺度对角矩阵。上述逆矩阵表示线性系统求解，实际实现不显式求逆。默认 $K_{\max}=12$、$\varepsilon=10^{-4}$。

该算法对应 [joint_projector.py](joint_projector.py)。活跃集用于计算修正方向，候选接受与最终可行性判定均检查全部已实现约束。若未收敛，返回最后接受的控制；若没有接受任何修正，则保留 $U_0$，约束满足标志可能为假。

## 适用范围与数值容差

稳定性与道路约束不检查初始时刻 `t = 0`；曲率使用未来预测点之间的航向差与位移。状态和轨迹由冻结的 K2/Koopman 执行模型根据候选控制序列预测。

以上为约束目标边界。实现先对残差归一化，再以 `projection.constraint_tolerance = 1e-4` 判定是否满足；该容差不是所有物理量统一增加 `1e-4`。求解最多迭代 12 次，返回结果是否满足全部已实现约束由 `projector_constraints_satisfied` 报告。

当前 JTP 未单独加入车速上下限、侧倾角速度、加速度变化率（jerk）、转向变化率、车间碰撞或外部车辆碰撞约束。模块 [README.md](README.md) 说明：碰撞规避由上游 Adaptive Anchor 评分/选择处理，碰撞与控制变化率还由独立评估器检查。信赖域、步长、阻尼和线搜索参数属于投影求解设置，不属于上表中的车辆物理约束。
