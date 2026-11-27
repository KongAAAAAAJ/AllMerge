# JTP 独立碰撞评价

编队内部碰撞和外部交通参与者碰撞只属于 JTP 离线评价指标，不进入 `JointTrajectoryProjector` 的约束构建、Jacobian、LS 活跃集或 QP 求解。

## 评价范围

- **编队内部碰撞：** 使用每辆受控卡车的覆盖圆，在同步未来时刻检查不同受控车辆之间的圆对距离。
- **外部碰撞：** 使用受控车辆覆盖圆与数据集提供的外部交通参与者预测轨迹进行距离检查。
- **统计范围：** 对完整预测时域计算碰撞违反率及碰撞穿透程度。外部车辆数量、筛选距离、碰撞净空和评价容差由 `configs/evaluation/jtp_ablation.yaml` 独立管理。

## 与投影器的接口关系

`JointTrajectoryProjector.project()` 不接收 `external_actors`，也不计算编队内部碰撞。离线评价程序仍在投影前后分别调用 `evaluate_group()`，使用相同的受控车辆、外部交通参与者和评价配置统计碰撞指标。碰撞指标不参与 JTP 的 G1/G2/G3 优先级、活动集选择、LS/QP 约束或 Stage 2 投影验收。

## 评价实现

- `evaluation/ablation/jtp/metrics.py`：完整时域碰撞评价。
- `evaluation/ablation/jtp/data.py`：离线外部交通参与者数据构建。
- `evaluation/ablation/jtp/runner.py`：投影前后独立统计碰撞指标。
- `configs/evaluation/jtp_ablation.yaml`：碰撞净空、外部交通参与者数量和筛选距离。
