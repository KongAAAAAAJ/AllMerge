# MyHighwayEnv 的 TruckSim 模型切换

## 启用

在 `all_merge/highway_env/envs/my_highway_env.py` 的默认配置中设置：

```python
"Controller": {
    "type": "lqr",                   # 本次不分派控制器类型
    "vehicle_model": "trucksim",     # kinematics / trucksim
},
```

然后从工作区根目录运行 `python all_merge/run_test.py`。
默认仍为 `kinematics`；该分支无需加载 DLL。仅 MyHighwayEnv 的受控车辆切换，
背景车辆、编队动作、规则决策器和轨迹规划器保持原逻辑。
`8-DoF` 暂不实现车辆状态积分，选择时抛出 NotImplementedError。

也可以直接构造：

```python
from highway_env.envs.my_highway_env import MyHighwayEnv

env = MyHighwayEnv(config={"Controller": {"vehicle_model": "trucksim"}})
try:
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(3)
finally:
    env.close()
```

代码运行时必须能导入 `all_merge` 下的本地 `highway_env`。

## 数据集和单位

`TruckSim` 配置默认指向：
- `D:/Users/Public/Documents/TruckSim2016.1_Data_now_using/truck_{1,2,3}_lon_lat.sim`
- `D:/Users/Public/Documents/TruckSim2016.1_Data_now_using/Extensions/Multi_vehicle_lon_lat/s_s_{1,2,3}.dll`

每辆车独占一个 DLL。配置项 `simfiles`、`dlls` 按受控车辆顺序排列。
`steering_ratio=25` 来自当前数据集的转向传动比；
`engine_map_path=None` 使用随代码复制的参考 4455kg 发动机映射，
`max_rpm=3000`。加速度跟踪采用参考项目的 PID 和执行器逆模型。
更换车辆数据集后，应同步标定这些参数及执行器逆模型，不要只替换文件名。

底层环境配置采用浅层更新；覆盖整个 `TruckSim` 字典时先复制
`MyHighwayEnv.default_config()["TruckSim"]` 再修改所需项。

| 输入 | 单位/含义 |
|---|---|
| throttle | 油门开度，0～1 |
| brake | 制动主缸压力，MPa |
| steer | 方向盘角，度；degrees(action["steering"]) × steering_ratio |

输出顺序固定为：
`Ax, Ay, Vx, Vy, Xo, Yo, Roll, AVx, Pitch, AVy, Yaw, AVz, AV_Eng, steer_l1, steer_r1`。
加速度由 g 乘 9.8 转为 m/s²，速度由 km/h 除 3.6 转为 m/s，
Yaw 由度转为弧度。航向使用 Yaw，不使用 Vy/Vx。
状态位置用二维平移对齐当前场景；当前场景和数据集初始航向均为零。
保留原场景车辆的几何尺寸、初始间距和分组，以兼容现有决策与观测。

## 调用和状态

```text
MyHighwayEnv.reset()
  → 创建原场景和背景车
  → _create_trucksim_vehicles()
  → 每车 TrucksimSimulation.reset(vx_init)
  → 用 TrucksimVehicle 替换受控车辆并重新绑定动作/观测

env.step(group_action)
  → MultiAgentAction → RULE_MAKER → FOLLOWVehicle.act()
  → action = {acceleration, steering}
  → Road.step(dt) → TrucksimVehicle.step(dt)
  → PID + 逆模型 → [throttle, brake, steering_wheel_deg]
  → TrucksimSimulation.run()
  → position / speed / heading / measured_acceleration
  → on_state_update() 更新实际车道
```

`action["acceleration"]` 是控制指令，实测值为
`vehicle.measured_acceleration = [ax, ay]`，不会覆盖已有
`FOLLOWVehicle.acceleration()` 方法。
`info["trucksim"]` 按车辆顺序提供实测加速度、三输入和求解器时间；
原有 `info["acceleration"]` 仍保留指令含义。

环境每次策略步推进 1 秒，内含 15 次道路更新。
求解器使用自身步长，累计未满一个求解步的余量。
例如低层步长为 0.0025 秒时，三次道路更新分别推进 26、27、27 个求解步；
状态时间最多落后一个求解步，不会逐步累计时间损失。

预测深拷贝和历史快照生成普通 FOLLOWVehicle，规划器不会复制或推进真实 DLL。
回合结束、reset、close 和步进异常会停止求解器；reset 复用已加载的模型。
不在求解器状态外额外做自行车模型积分；道路碰撞仍决定回合结束。
规划阶段原有八自由度侧翻校核仍保留，它与 vehicle_model 选择是不同环节。

## 验证

从工作区根目录、使用具有项目依赖的 Python 执行：

```powershell
python -m pytest all_merge/tests/test_trucksim.py -q -o cache_dir=all_merge/tests/.pytest_cache --basetemp=all_merge/tests/_pytest_temp
python all_merge/tests/trucksim_smoke.py --native --full --policy
```

第一条不调用真实 DLL。第二条需要本机 TruckSim 求解器及有效许可证，
使用当前 PPO 模型与录像包装器，并将测试 sim 副本及仿真输出写入
`all_merge/tests/_trucksim_output/`，外部源数据集只读。

已验证：14 项自动化检查；真实三车初始速度 25 m/s；
加速、制动、转向、连续重置；保持→拆分→合并；
PPO + VecVideoRecorder + DummyVecEnv 的回合结束及自动重置。
真实联仿为短时功能验证，不代表完成长时间控制性能或车辆参数标定。


### Episode data and steering feedback (15 exports)

The adapter requires 3 inputs and 15 outputs. Outputs 13 and 14 (zero-based)
are steer_l1 and steer_r1 in degrees. Their arithmetic mean is the actual
front-wheel feedback; it is not divided by steering_ratio. Kinematics uses
the applied steering action converted from radians to degrees.

DataSaver samples at the environment simulation frequency (currently 15 Hz),
including initial and terminal state, and writes the following at episode end:
- infos/all_data/substeps_episode_*.csv: time, environment X/Y, speed,
  longitudinal acceleration and front-wheel angle for all three vehicles.
- infos/trucksim_data/trucksim_episode_*.csv: native solver times and all 15
  raw outputs, retaining native units and coordinates (TruckSim only).
- infos/result_plots/episode_*.png: 2x2 trajectory, speed, acceleration, steering.
- infos/result_plots/episode_*_trucksim.png: 5x3 raw output plots (TruckSim only).

The existing policy-step CSV adds three front-wheel angle columns in degrees.
DataSaver(output_root=..., plot_dir=...) isolates CSV and image output for tests.
finish_episode returns the artifact paths before clearing episode sample buffers.
