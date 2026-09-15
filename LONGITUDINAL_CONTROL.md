# Longitudinal controller integration

## Selecting the controller

Both highway-platoon-v0 and merge-platoon-v0 default to LQR. Pass the
Controller section when constructing the environment, or in reset options:

~~~python
env = gym.make(
    "highway-platoon-v0",
    config={"Controller": {"type": "lmpc", "vehicle_model": "kinematics"}},
)

env.unwrapped.reset(options={
    "config": {"Controller": {"type": "lqr", "vehicle_model": "kinematics"}}
})
~~~

A Controller containing only type is supported; vehicle_model then defaults
to kinematics. Type is applied on creation/reset, not by editing config while
an episode is running. Unknown types (including the old misleading pid label)
raise ValueError. The PPO observation and high-level action spaces are unchanged.
The existing env_reset() factory uses environment defaults; it does not expose a
new configuration argument. To use its recording workflow, set Controller.type
in the selected environment's default_config, or supply the config in its gym.make call.

Optional LMPC settings under Controller:

~~~python
"lmpc": {
    "horizon": 10,
    "Q": [1.0, 0.475, 0.01],
    "Qf": [1.0, 0.475, 0.01],
    "R": 0.1,
    "tau": 0.05,
    "slack_penalty": 1e6,
}
~~~

Q and Qf accept three diagonal weights or symmetric positive-semidefinite 3x3
matrices. horizon must be a positive integer; R, tau and slack_penalty must be
positive finite values. No communication delay compensation is enabled.
CasADi is loaded only for lmpc; IPOPT is the optimization backend. LQR retains
its original implementation, including its original fixed model timestep.

## Actual longitudinal/lateral interface

The flow is: platoon decision -> per-vehicle acceleration and target lane ->
vehicle lateral tracking -> vehicle dynamics.

| Layer | Input | Output |
| --- | --- | --- |
| Longitudinal policy | Ego/reference position (m), speed (m/s), acceleration (m/s2), reference gap/speed, dt (s), acceleration bounds | Desired acceleration (m/s2), future acceleration sequence, solver status |
| Vehicle execution | Existing lateral action, target lane, desired acceleration, planner/group context | action dictionary containing acceleration (m/s2) and steering (rad) |
| Kinematics | action dictionary | Updated position, speed and heading |
| TruckSim | Same action dictionary | Existing acceleration PID produces throttle/brake; steering is converted to steering-wheel degrees using the existing ratio |

With polynomial planning enabled, FOLLOWVehicle.steering_pid_control computes
0.05*(reference_y-y) + 1.15*(reference_heading-heading), with steering-angle and
increment limits. Despite its name, this contains only proportional feedback;
there are no integral/derivative terms. Without the planner, lateral position
and heading use cascaded proportional control. Controller.type selects only
longitudinal control. All steering functions and their limits are unchanged.
Different longitudinal speeds can still change the planned trajectory and
therefore the steering reference.

## LMPC model and adaptation

The optimization core is adapted from
multi_truck_lon_control/algorithms/linear_mpc_policy.py without importing that
module or modifying any source outside all_merge. Its observation decoder,
cwd/sys.path mutations, unrelated CBF/Koopman imports, and delay predictor are
not used.

The state is [reference_x - ego_x - desired_gap, reference_v - ego_v, ego_a].
Positions/gaps retain the existing environment's longitudinal x/center-distance
convention, not bumper-to-bumper distance. The continuous model is:

~~~text
gap_error_dot = speed_error
speed_error_dot = reference_acceleration - ego_acceleration
ego_acceleration_dot = (command - ego_acceleration) / tau
~~~

A matrix exponential supplies exact zero-order-hold discretization. Actual
control uses 1/simulation_frequency (the high-level action is re-evaluated at
each simulation substep), while polynomial speed prediction uses PolyPlanner.DT.
Free cruising, speed holding and slowing use velocity tracking with zero gap
weight and constant target speed.

The model uses measured longitudinal acceleration for TruckSim and the current
acceleration command for kinematics. Each execution frame snapshots feedback
before decision code updates acceleration commands. Temporary leaders copy
the source vehicle's identity and feedback. Speculative planning uses independent
copied roads and reconstructed solver caches; native solver handles are never
deep-copied. Planning advances its own kinematic references and feedback.

Following considers both the physical front vehicle and the group leader,
retaining the existing minimum-command composition. Leaders retain current-lane,
target-lane, slowing and speed-holding composition. Existing branch-specific
acceleration limits apply. The diagnostic is_limited=False leader path uses
wide bounds +/-1e6 rather than the normal vehicle limits.

Only a selected, published command's prediction can be used by another car.
Current-frame predictions are used when the reference is in the same group
and has the same prediction timestep. Otherwise a constant reference
acceleration is used. Group changes and new frames clear communication candidates;
no stale historical sequence or fixed predecessor index is used. Soft
string-error constraints use a same-group reference's available spacing/velocity
errors; they are inactive when such feedback is unavailable. These soft
constraints are not a collision guarantee or a proof of string stability.

## Failure handling and diagnostics

Invalid configuration or missing CasADi fails at initialization. If a numerical
solve fails or returns non-finite values, the corresponding original LQR
calculation supplies the command. No silent zero-acceleration fallback is used.

Both environments add info["longitudinal_control"]:

- type: lqr or lmpc.
- successes: successful LMPC candidate solves on the execution road.
- failures/fallbacks: failed candidate solves and successful LQR fallbacks.
- last_error: most recent numerical failure, or None.

Counters reset each episode. They count candidate solves (including execution
road decision checks), not vehicles or environment steps. Independent planning
copies do not alter execution counters.

## Other source algorithms

| Family | Requirements for a future adapter |
| --- | --- |
| LQR / H-infinity | Map actual state/error conventions and per-vehicle roles; existing all_merge LQR remains the baseline |
| Nonlinear MPC variants | Align plant parameters, acceleration/actuator input units and reference predictions |
| MPC + CBF | Port barrier constraints and validate their state and gap conventions |
| Koopman / delay MPC | Provide trained model artifacts, model-specific feature normalization, and timestamped communication history |
| Sliding-mode control | Separate the controller/observer from its standalone PlatoonSimulator and align stateful update intervals |

Only lqr and lmpc are registered in this implementation.

## Validation

Run from the workspace root:

~~~powershell
& 'D:\ProgramData\Miniconda3\envs\offpolicyEnv\python.exe' -m pytest all_merge/tests/test_longitudinal.py all_merge/tests/test_trucksim.py all_merge/tests/test_seeding.py all_merge/tests/test_episode_recording.py all_merge/tests/test_episode_plots.py -q --basetemp=all_merge/tests/_lmpc_temp -o cache_dir=all_merge/tests/.pytest_cache
~~~

Tests exercise real LMPC/IPOPT solves, LQR equivalence, state signs/bounds,
configuration and reset, branching/group changes, steering invariance, isolated
planning, forced failure recovery, both environment classes, and TruckSim via
the existing native-solver substitute.

The installed Gurobi license expired on 2025-06-02, preventing a full live Game
optimization run in this environment. Game integration tests substitute only
the discrete candidate selector with an argmax over supplied rewards; the
downstream Game decisions and longitudinal solves still execute. Rule tests use
the actual decision implementation. TruckSim tests do not establish native DLL
performance or closed-loop tracking quality.
