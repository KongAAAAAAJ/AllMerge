# JTP 算法流程：单车独立 LS/QP 双后端

1. 对输入批次中的每辆车分别建立单车投影问题。每辆车仅优化自己的 40 个加速度和前轮转角控制点，优化变量维度为 80。三辆车之间不存在联合优化变量或碰撞耦合约束。
2. 使用冻结 K2 执行模型预测完整 40 步轨迹和底盘响应，计算稳定性、道路边界、曲率、控制变化率及控制幅值残差。
3. 按 `projection.constraint_sampling_stride` 建立 G1/G2 稀疏时域候选。G3 控制变化率和加速度/前轮转角硬边界始终覆盖完整时域。
4. `constrained_qp` 按 `projection.qp_group_order` 指定的顺序执行稳定性、道路与曲率、控制连续性三组约束，支持六种排列；默认顺序为 G1 道路与曲率、G2 稳定性、G3 控制连续性。`damped_least_squares` 始终保持原有 G1 稳定性、G2 道路与曲率、G3 控制连续性的顺序。根据 `projection.solver_method` 选择后端：
   - `damped_least_squares`：对稀疏候选执行每类约束 Top-K 活跃集筛选，保留阻尼自适应、有界活动集控制边界求解、Trust Region 和度量半空间跨组保护。
   - `constrained_qp`：对每个优先级组中的全部稀疏候选行构建 Full-Group QP，保留逐行松弛、统一容差、跨组保护及控制幅值硬边界，不使用活动集截断或 Trust Region。
5. 两个后端均通过 Backtracking 对候选控制修正进行完整非线性验收。验收检查全部 40 步的 G1/G2/G3 约束、控制幅值硬约束、逐约束非退化和组内违反数量不增加。
6. 若完整验收发现未采样 G1/G2 违反，则将对应时刻加入下一轮采样集合。若被拒绝的候选在未采样时刻引入违反，也将相应时刻加入下一轮采样集合。新增采样点不做 Top-K 截断。
7. 每辆车返回最后接受的控制序列及对应 K2 执行轨迹。批量接口按车辆索引合并结果。碰撞评价在投影器外部独立执行。

## 约束采样

`constraint_sampling_stride=1/2/4` 对应初始 G1/G2 采样时刻数 40/20/10。采样时刻为 `k, 2k, ...`，并包含预测终点。曲率采样按照离散曲率区间对应的终点时刻映射。控制变化率和控制幅值不进行时域稀疏化。

## Solver 切换

```yaml
projection:
  solver_method: constrained_qp  # damped_least_squares | constrained_qp
  qp_group_order: [road_curvature, stability, control_rate]
  # 可调整为上述三个组的任意排列
  constraint_sampling_stride: 2  # 1 | 2 | 4
```

LS 保留原有活动集、Top-K、阻尼、Trust Region 和半空间保护。QP 保留 OSQP Full-Group、松弛变量和跨组保护。QP 的 Full-Group 指当前稀疏采样集合中的全部约束行，不包含未采样时刻的行；完整时域验收及自适应加密负责检查未采样状态。
