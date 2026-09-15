"""Longitudinal control in SI units, independent of observations and steering.

The LMPC objective and soft string-error constraints are adapted from
multi_truck_lon_control/algorithms/linear_mpc_policy.py. No source module is
imported: those modules change cwd and depend on a different highway_env.
"""
import copy
from dataclasses import dataclass
from functools import wraps

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class LongitudinalState:
    position: float
    speed: float
    acceleration: float


@dataclass(frozen=True)
class ControlResult:
    acceleration: float
    future_accelerations: np.ndarray
    status: str


def vehicle_state(vehicle):
    measured = getattr(vehicle, "measured_acceleration", None)
    acceleration = (float(np.asarray(measured).flat[0]) if measured is not None
                    else float(getattr(vehicle, "action", {}).get("acceleration", 0)))
    return LongitudinalState(float(vehicle.position[0]), float(vehicle.speed), acceleration)


def bind_longitudinal_source(target, source):
    """Bind temporary decision vehicles without changing the legacy LQR path."""
    context = getattr(target.road, "longitudinal_control", None)
    if context is not None and context.kind == "lmpc":
        target._lon_id = getattr(source, "_lon_id", None)
        target.action = source.action.copy()
        if hasattr(source, "measured_acceleration"):
            target.measured_acceleration = np.array(source.measured_acceleration, copy=True)
        if hasattr(source, "_lon_dt"):
            target._lon_dt = source._lon_dt
    return target


def publish_longitudinal_result(method):
    """Publish the selected command after existing min-combination logic."""
    @wraps(method)
    def wrapped(vehicle, *args, **kwargs):
        context = getattr(vehicle.road, "longitudinal_control", None)
        if context is None or context.kind == "lqr":
            return method(vehicle, *args, **kwargs)
        vehicle._lon_candidates = []
        acceleration = method(vehicle, *args, **kwargs)
        context.publish(vehicle, acceleration)
        return acceleration
    return wrapped


class LinearMPC:
    """A reusable CasADi problem; only numerical parameters change per solve."""

    def __init__(self, settings, dt, speed_only=False):
        import casadi as ca

        self.ca = ca
        self.N = settings["horizon"]
        self.dt = dt
        self.speed_only = speed_only
        tau = settings["tau"]
        # x = [gap - reference_gap, reference_v - ego_v, ego_a].
        # Exact ZOH discretization of dx = A*x + B*u + G*reference_a.
        continuous = np.zeros((5, 5))
        continuous[:3, :3] = [[0, 1, 0], [0, 0, -1], [0, 0, -1 / tau]]
        continuous[2, 3] = 1 / tau
        continuous[1, 4] = 1
        discrete = expm(continuous * dt)
        self.A, self.B, self.G = discrete[:3, :3], discrete[:3, 3], discrete[:3, 4]

        opti = ca.Opti()
        x = opti.variable(3, self.N + 1)
        u = opti.variable(1, self.N)
        initial = opti.parameter(3)
        disturbance = opti.parameter(1, self.N)
        lower, upper = opti.parameter(), opti.parameter()
        previous_errors = opti.parameter(2)
        q, qf = settings["Q"].copy(), settings["Qf"].copy()
        if speed_only:
            q[0, :], q[:, 0] = 0, 0
            qf[0, :], qf[:, 0] = 0, 0
        opti.subject_to(x[:, 0] == initial)
        objective = 0
        for k in range(self.N):
            objective += ca.mtimes([x[:, k].T, q, x[:, k]]) + settings["R"] * u[0, k] ** 2
            opti.subject_to(x[:, k + 1] == self.A @ x[:, k] + self.B * u[0, k]
                            + self.G * disturbance[0, k])
        objective += ca.mtimes([x[:, -1].T, qf, x[:, -1]])
        opti.subject_to(opti.bounded(lower, u, upper))
        if not speed_only:
            slack = opti.variable(2)
            opti.subject_to(slack >= 0)
            opti.subject_to(x[:2, 1] <= previous_errors + slack)
            opti.subject_to(-x[:2, 1] <= previous_errors + slack)
            objective += settings["slack_penalty"] * ca.sumsqr(slack)
        else:
            # Keep this parameter used in speed-only problems as well.
            objective += 0 * ca.sumsqr(previous_errors)
        opti.minimize(objective)
        opti.solver("ipopt", {"print_time": False},
                    {"print_level": 0, "sb": "yes", "max_iter": 200})
        self.opti, self.x, self.u = opti, x, u
        self.initial, self.disturbance = initial, disturbance
        self.lower, self.upper, self.previous_errors = lower, upper, previous_errors

    def solve(self, state, reference, reference_distance, reference_speed,
              bounds, disturbance=None, predecessor_errors=None):
        if self.speed_only:
            initial = [0, reference_speed - state.speed, state.acceleration]
            disturbance = np.zeros(self.N)
        else:
            initial = [reference.position - state.position - reference_distance,
                       reference.speed - state.speed, state.acceleration]
            if disturbance is None:
                disturbance = np.full(self.N, reference.acceleration)
        values = np.r_[initial, disturbance, bounds]
        if not np.all(np.isfinite(values)) or bounds[0] > bounds[1]:
            raise ValueError("LMPC requires finite states and ordered acceleration bounds")
        self.opti.set_value(self.initial, initial)
        self.opti.set_value(self.disturbance, np.asarray(disturbance).reshape(1, self.N))
        self.opti.set_value(self.lower, bounds[0])
        self.opti.set_value(self.upper, bounds[1])
        self.opti.set_value(self.previous_errors,
                            [1e6, 1e6] if predecessor_errors is None else np.abs(predecessor_errors))
        # A fresh numerical start avoids leaking a speculative solve to another car.
        self.opti.set_initial(self.opti.x, 0)
        solution = self.opti.solve()
        future = np.asarray(solution.value(self.u), dtype=float).reshape(self.N)
        if not np.all(np.isfinite(future)):
            raise RuntimeError("LMPC returned non-finite accelerations")
        future = np.clip(future, *bounds)
        return ControlResult(float(future[0]), future, "success")


class LongitudinalControl:
    """Road-owned configuration/cache. Deep copies never copy solver handles."""

    DEFAULTS = {"horizon": 10, "Q": [1, 0.475, 0.01], "Qf": [1, 0.475, 0.01],
                "R": 0.1, "tau": 0.05, "slack_penalty": 1e6}

    def __init__(self, config, dt):
        self.config = copy.deepcopy(config)
        self.kind = config.get("type", "lqr")
        if self.kind not in {"lqr", "lmpc"}:
            raise ValueError(f"Unknown Controller.type: {self.kind!r}; expected lqr or lmpc")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Longitudinal control dt must be positive and finite")
        self.dt = float(dt)
        self.settings = dict(self.DEFAULTS)
        self.settings.update(config.get("lmpc", {}))
        self.solvers = {}
        self.states, self.predictions, self.errors, self.groups = {}, {}, {}, {}
        self.failures = self.fallbacks = self.successes = 0
        self.last_error = None
        if self.kind == "lmpc":
            self._validate()
            try:
                import casadi  # noqa: F401 -- lazy, fail before scene/native initialization
            except ImportError as error:
                raise ImportError("Controller.type='lmpc' requires CasADi in the active Python environment") from error
            self._solver(self.dt, False)
            self._solver(self.dt, True)

    def _validate(self):
        settings = self.settings
        unknown = set(settings) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Controller.lmpc options: {sorted(unknown)}")
        horizon = settings["horizon"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Controller.lmpc.horizon must be a positive integer")
        for name in ("R", "tau", "slack_penalty"):
            value = float(settings[name])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Controller.lmpc.{name} must be positive and finite")
            settings[name] = value
        for name in ("Q", "Qf"):
            matrix = np.asarray(settings[name], dtype=float)
            if matrix.shape == (3,):
                matrix = np.diag(matrix)
            if (matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))
                    or not np.allclose(matrix, matrix.T) or np.linalg.eigvalsh(matrix).min() < 0):
                raise ValueError(f"Controller.lmpc.{name} must be a positive-semidefinite 3x3 matrix or 3 weights")
            settings[name] = matrix

    def __deepcopy__(self, memo):
        # Rebuild only numerical configuration; solvers are lazily reconstructed.
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            setattr(result, name, {} if name in {"solvers", "states", "predictions", "errors"}
                    else copy.deepcopy(value, memo))
        result.failures = result.fallbacks = result.successes = 0
        result.last_error = None
        return result

    def _solver(self, dt, speed_only):
        key = (float(dt), bool(speed_only))
        if key not in self.solvers:
            self.solvers[key] = LinearMPC(self.settings, *key)
        return self.solvers[key]

    def begin_frame(self, vehicles, index_groups):
        self.states = {}
        self.predictions = {}
        self.errors = {}
        self.groups = {}
        for index, vehicle in enumerate(vehicles):
            vehicle._lon_id = index
            self.states[index] = vehicle_state(vehicle)
        self.set_groups(index_groups)

    def set_groups(self, index_groups):
        groups = {}
        for entry in index_groups:
            indices = tuple(entry) if isinstance(entry, list) else (entry,)
            for index in indices:
                groups[index] = indices
        if groups != self.groups:
            self.predictions.clear()
            self.errors.clear()
        self.groups = groups

    def diagnostics(self):
        return {"type": self.kind, "successes": self.successes,
                "failures": self.failures, "fallbacks": self.fallbacks,
                "last_error": self.last_error}

    def compute(self, vehicle, reference, reference_distance, reference_speed, bounds,
                fallback, speed_only=False, state_vehicle=None):
        dt = getattr(vehicle, "_lon_dt", self.dt)
        ego_id = getattr(vehicle, "_lon_id", None)
        ref_id = getattr(reference, "_lon_id", None)
        # Frame snapshots protect measured feedback from decision-layer command updates.
        state = self.states.get(ego_id, vehicle_state(state_vehicle or vehicle))
        ref_state = None if reference is None else self.states.get(ref_id, vehicle_state(reference))
        same_group = (ego_id is not None and ref_id is not None and ego_id != ref_id
                      and ref_id in self.groups.get(ego_id, ()))
        disturbance, previous_errors = None, None
        if same_group:
            prediction = self.predictions.get(ref_id)
            if prediction is not None and prediction[0] == dt:
                disturbance = prediction[1]
            previous_errors = self.errors.get(ref_id)
        try:
            result = self._solver(dt, speed_only).solve(
                state, ref_state, reference_distance, reference_speed, bounds,
                disturbance, previous_errors)
            if (not np.isfinite(result.acceleration)
                    or not np.all(np.isfinite(result.future_accelerations))):
                raise RuntimeError("LMPC returned non-finite accelerations")
            self.successes += 1
        except (RuntimeError, ValueError) as error:
            self.failures += 1
            self.last_error = str(error)
            acceleration = float(fallback())
            if not np.isfinite(acceleration):
                raise RuntimeError("Both LMPC and LQR fallback failed to produce finite acceleration") from error
            self.fallbacks += 1
            result = ControlResult(acceleration, np.full(self.settings["horizon"], acceleration), "lqr_fallback")
        # These are candidates, published only after the decision has selected its
        # final (possibly min-combined) acceleration via publish().
        errors = None
        if not speed_only and same_group:
            errors = np.array([ref_state.position - state.position - reference_distance,
                               ref_state.speed - state.speed])
        vehicle._lon_candidates = getattr(vehicle, "_lon_candidates", []) + [(result, dt, errors)]
        return result.acceleration

    def publish(self, vehicle, acceleration):
        ego_id = getattr(vehicle, "_lon_id", None)
        if ego_id is None:
            return
        candidates = getattr(vehicle, "_lon_candidates", [])
        match = next(((result, dt, errors) for result, dt, errors in reversed(candidates)
                      if np.isclose(result.acceleration, acceleration, atol=1e-9, rtol=0)), None)
        self.predictions[ego_id] = (match[1], match[0].future_accelerations.copy()) if match else (
            getattr(vehicle, "_lon_dt", self.dt), np.full(self.settings["horizon"], acceleration))
        self.errors.pop(ego_id, None)
        if match is not None and match[2] is not None:
            self.errors[ego_id] = match[2]
        vehicle._lon_candidates = []
