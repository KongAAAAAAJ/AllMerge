# TRACE TruckSim longitudinal + lateral adapter

## What this changes

The original `Truck_Simulation.py` is **unchanged**. `Trucksim_Simulation_ltr.py`
is a separate raw interface using the shared `Simulation.py` ctypes API. Existing
`trucksim_dynamics.TrucksimVehicle` retains the old 3-input / 15-output contract
in the default `control_mode="longitudinal"` (longitudinal acceleration PID, direct
open-loop front-wheel angle to steering-wheel angle conversion).

`control_mode="longitudinal_lateral"` selects the new DLL wrapper and activates
steering angle feedback via `LateralSteeringController`. This is still the same
`TrucksimVehicle` class, so existing simulation/observation/planner entry points
are retained. No change to the Diffusion planner is required for this patch.

## Step 1: verify native TruckSim I/O before integrating into the environment

Use the new `.sim` dataset and `.dll` selected by the researcher. **These must
be switched in TruckSim locally.** Do not point this probe to the old model.

```bash
python -m highway_env.vehicle.trucksim_simulation.Trucksim_Simulation_ltr ^
  --simfile "D:\...\new_vehicle.sim" ^
  --dll "D:\...\new_solver_64.dll" ^
  --seconds 2 ^
  --steer-wheel-deg 0 ^
  --csv "outputs\\koopman\\trucksim_ltr_probe.csv"
```

The probe prints the actual `n_import`, `n_export`, timestep and time bounds,
then saves **raw** `export_0`, `export_1`, ... columns. The probe currently
requires exactly three import channels **in the order throttle, brake,
steering-wheel angle (degrees)**. Validate these channels in the TruckSim
Procedure before running. Begin at zero steering and a safe initial speed.
Never infer signal names by position from the old model.

The DLL is only available on the user's Windows/TruckSim installation;
no native simulation was run while preparing this patch.

## Step 2: configure the new model

In `MyHighwayEnv.default_config()` / experiment override under `TruckSim`,
set `control_mode` and your new `simfiles`, `dlls`. Use explicit **zero-based**
index mapping from the new TruckSim Export Variables list:

```python
"TruckSim": {
    "control_mode": "longitudinal_lateral",
    "simfiles": ["...new_1.sim", "...new_2.sim", "...new_3.sim"],
    "dlls": ["...new_1.dll", "...new_2.dll", "...new_3.dll"],
    "steering_ratio": 25.0,  # calibration needed for the new model
    "max_rpm": 3000,
    "engine_map_path": None,
    "signal_map": {
        "imports": {
            "throttle": 0, "brake": 1, "steering_wheel_deg": 2
        },
        "exports": {
            # Every 'INDEX' must be replaced with its REAL export index.
            "ax_g": INDEX, "ay_g": INDEX,
            "vx_kmh": INDEX, "vy_kmh": INDEX,
            "x_m": INDEX, "y_m": INDEX,
            "yaw_deg": INDEX, "engine_rpm": INDEX,
            "steer_left_deg": INDEX, "steer_right_deg": INDEX,
            # Optional to run coupled controls, needed for later Koopman data:
            "roll_deg": INDEX, "roll_rate_deg_s": INDEX,
            "yaw_rate_deg_s": INDEX, "ltr": INDEX,
        },
    },
    "lateral_controller": {
        # Example layout ONLY. Enter gains and calibrated limits of the new model.
        "kp": 0.0, "ki": 0.0, "kd": 0.0,
        "max_wheel_deg": MAX_STEER_WHEEL_DEG,
        "max_rate_deg_s": MAX_STEER_RATE_DEG_S,
        "steering_tau_s": 0.0,
    },
}
```

The placeholders `INDEX`, `MAX_STEER_WHEEL_DEG`, etc. are intentionally
undefined and must be replaced before use. For the initial interface test,
use `kp=ki=kd=0` and `steering_tau_s=0`, which applies static steering-ratio
feedforward with rate and amplitude limiting. Tune feedback gains and any
actuator time constant only after confirming steering angle signs and units.
The exported `ltr` value is passed through unchanged; the TruckSim dataset
must define its sign, axle scope and normalization. No LTR formula is assumed.

### I/O units

- `action['acceleration']`: reference longitudinal acceleration in m/s^2.
- `action['steering']`: reference **front road-wheel angle** in radians.
- new model imports: throttle [0,1], brake [model-specific pressure],
  **steering-wheel angle in degrees** (not front wheel degrees).
- exports: longitudinal/lateral accelerations in g, vehicle body velocities
  in km/h, position in m, yaw/roll angles in degrees, angular rates in deg/s,
  left/right steer angles in deg, engine speed in rpm, LTR dimensionless.
- `steering_ratio`: steering wheel angle / road wheel angle (both in same unit).

The inverse steering function currently implements the static steering gear
inverse with an optional first-order actuator compensation term. It does **not**
claim to identify a physical steering actuator without calibration.

## Step 3: backward compatibility

Leave `control_mode` absent or set it to `longitudinal`. This preserves the
original `Truck_Simulation.py`, exact legacy 3-import/15-export mapping,
original longitudinal `PIDController` and `acceleration_control`, and original
road-wheel angle -> steering-wheel angle direct command. The two modes can be
switched by configuration and user-managed `.sim`/DLL paths; no global code
rewrites are required.

## Step 4: local checks

```bash
python scripts/verify_trucksim_modes.py
```

This runs **without a native DLL** and checks basic mode selection, legacy
steering commands, mapped I/O, static inverse and feedback limits. It is not
a replacement for the physical/TruckSim integration test.

### Limits of the present patch

- It provides the interface/controller adaptation, **not** the Koopman
  collection/training scripts or an identified steering model.
- `inverse_dynamics.py` still uses the existing reference 4455 kg powertrain
  and brake calibration. Before collecting **variable-payload** data, ensure
  the low-level inverse model and engine map are appropriate to each payload,
  or record actual actuator signals to diagnose tracking bias.
- The new `signal_map` must match the locally configured TruckSim dataset; a
  raw-index probe alone cannot identify a signal's physical meaning.
