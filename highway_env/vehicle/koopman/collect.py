"""TruckSim reference-command collection with one isolated subprocess per run for TRACE.

Each trajectory runs in a separate Python child process. The batch parent
loads the fixed engine calibration CSV exactly once and passes its numeric
matrix to workers in their job requests. SDK solver termination is mandatory.
No memory-corruption workaround is used.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .data import SIGNALS, UNITS_RAW, audit_npz, ltr_from_fz, resample, to_si


def read_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML in offpolicyEnv (pip install pyyaml)") from exc
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Koopman config must be a YAML mapping")
    return cfg


def json_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def file_signature(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    # Detect accidental model changes. Note .sim may reference additional .par
    # files that must be manually verified by the operator.
    st = p.stat()
    return {"path": str(p.resolve()), "size": st.st_size,
            "mtime_ns": st.st_mtime_ns}


def _tier_table(trucksim, key):
    """Validate explicit tier->physical-value mapping; never infer values from file names."""
    table = trucksim.get(key)
    if not isinstance(table, dict) or not table:
        raise ValueError(f"trucksim.{key} must be a nonempty mapping of tier names to numeric values")
    clean = {}
    for tier, physical_value in table.items():
        if not isinstance(tier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tier):
            raise ValueError(f"Invalid {key} tier name: {tier!r}")
        try:
            v = float(physical_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Tier {key}.{tier} needs a numeric value") from exc
        if not np.isfinite(v):
            raise ValueError(f"Tier {key}.{tier} is not finite")
        if key == "payload_tiers_kg" and v < 0:
            raise ValueError(f"Tier {key}.{tier}: payload must be >= 0 kg")
        if key == "speed_tiers_kmh" and not (0 < v <= 180):
            raise ValueError(f"Tier {key}.{tier}: initial speed must be within (0, 180] km/h")
        if key == "friction_tiers" and not (0 < v <= 2):
            raise ValueError(f"Tier {key}.{tier}: friction must satisfy 0 < mu <= 2")
        clean[tier] = v
    return clean


def mapped_models(cfg, model_id=None, speed_tier=None):
    """Explicit 3D mapping (payload tier, friction tier, initial-speed tier) -> .sim/.dll.

    Operators configure and verify all physics in TruckSim. Python sets SV_VXS
    from the mapped speed tier at startup; it does not change load or friction.
    Every enabled grid cell requires its own .sim and .dll as requested.
    """
    trucksim = cfg.get("trucksim", {})
    payload_levels = _tier_table(trucksim, "payload_tiers_kg")
    friction_levels = _tier_table(trucksim, "friction_tiers")
    speed_levels = _tier_table(trucksim, "speed_tiers_kmh")
    if speed_tier is not None and speed_tier not in speed_levels:
        raise ValueError(f"Unknown speed tier: {speed_tier}")
    models = trucksim.get("models", [])
    if not isinstance(models, list) or not models:
        raise ValueError("trucksim.models must contain at least one .sim/.dll model")
    seen_ids, seen_conditions, seen_sim, seen_dll = set(), set(), set(), set()
    selected = []
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("Every trucksim.models entry must be a mapping")
        mid = item.get("id", "")
        if not isinstance(mid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", mid):
            raise ValueError(f"Invalid model id {mid!r}; use letters/numbers/_/- only")
        if mid in seen_ids:
            raise ValueError(f"Duplicate TruckSim model id: {mid}")
        seen_ids.add(mid)
        p_tier = item.get("payload_tier")
        f_tier = item.get("friction_tier")
        v_tier = item.get("speed_tier")
        if p_tier not in payload_levels or f_tier not in friction_levels or v_tier not in speed_levels:
            raise ValueError(f"Model {mid}: unknown payload/friction/speed tier: "
                             f"{p_tier!r}, {f_tier!r}, {v_tier!r}")
        if not isinstance(item.get("enabled", True), bool):
            raise ValueError(f"Model {mid}: enabled must be boolean")
        if not item.get("enabled", True):
            continue
        combo = (p_tier, f_tier, v_tier)
        if combo in seen_conditions:
            raise ValueError(f"Duplicate enabled payload/friction/speed combination: {combo}")
        seen_conditions.add(combo)
        for key in ("simfile", "dll"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Enabled model {mid} needs nonempty {key}")
        sim_path = str(Path(item["simfile"]).expanduser()).casefold()
        dll_path = str(Path(item["dll"]).expanduser()).casefold()
        if sim_path in seen_sim:
            raise ValueError(f"Multiple enabled models reference the same .sim: {item['simfile']}")
        if dll_path in seen_dll:
            raise ValueError(f"Multiple enabled models reference the same .dll: {item['dll']}")
        seen_sim.add(sim_path)
        seen_dll.add(dll_path)
        normalized = dict(item)
        normalized["payload_kg"] = payload_levels[p_tier]
        normalized["road_mu"] = friction_levels[f_tier]
        normalized["vx_init_kmh"] = speed_levels[v_tier]
        selected.append(normalized)
    if trucksim.get("require_complete_grid", False) and model_id is None:
        missing = sorted((p, f, v) for p in payload_levels for f in friction_levels
                         for v in speed_levels if (p, f, v) not in seen_conditions)
        if missing:
            raise ValueError(f"Missing enabled payload/friction/speed model combinations: {missing}")
    if model_id is not None:
        selected = [item for item in selected if item["id"] == model_id]
        if not selected:
            raise ValueError(f"Unknown or disabled model id: {model_id}")
    if speed_tier is not None:
        selected = [item for item in selected if item["speed_tier"] == speed_tier]
        if not selected:
            raise ValueError(f"No enabled models for speed tier: {speed_tier}")
    if not selected:
        raise ValueError("No enabled TruckSim models available")
    return selected


def model_for_run(cfg, run):
    """Resolve the exact model selected by the parent process."""
    mid = run.get("model_id")
    if not isinstance(mid, str) or not mid:
        raise ValueError("Worker request must specify model_id")
    return mapped_models(cfg, model_id=mid)[0]


def model_signature(model):
    return {key: file_signature(model[key]) for key in ("simfile", "dll")}


def model_condition(model):
    """Validate manually verified ground-truth condition labels."""
    mid = model["id"]
    if not model.get("verified_in_trucksim", False):
        raise ValueError(f"Confirm model {mid} payload/mu in TruckSim and set verified_in_trucksim=true")
    try:
        payload = float(model["payload_kg"])
        mu = float(model["road_mu"])
        vx = float(model["vx_init_kmh"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Model {mid} needs numeric payload_kg, road_mu, vx_init_kmh") from exc
    if not np.isfinite(payload) or payload < 0:
        raise ValueError(f"Model {mid} payload_kg must be finite and >=0")
    if not np.isfinite(mu) or not 0 < mu <= 2:
        raise ValueError(f"Model {mid} road_mu must satisfy 0 < mu <= 2")
    if not np.isfinite(vx) or not 0 < vx <= 180:
        raise ValueError(f"Model {mid} vx_init_kmh must be within (0, 180]")
    return {"model_id": mid, "payload_tier": model["payload_tier"],
            "friction_tier": model["friction_tier"], "speed_tier": model["speed_tier"],
            "payload_kg": payload, "road_mu": mu, "vx_init_kmh": vx,
            "verified_in_trucksim": True,
            "note": model.get("note", "")}


def validate(cfg, require_local_files=False, model_id=None, speed_tier=None):
    models = mapped_models(cfg, model_id=model_id, speed_tier=speed_tier)
    units = cfg["units"]
    excite = cfg["excitation"]
    batch = cfg["batch"]
    if units["velocity"] != "km/h":
        raise ValueError("Confirmed TruckSim exports require units.velocity = km/h")
    if units["engine_speed"] != "rpm":
        raise ValueError("Confirmed TruckSim AV_Eng requires units.engine_speed = rpm")
    if units["position"] != "m":
        raise ValueError("Expected position in metres")
    if excite.get("mode", "reference") != "reference":
        raise ValueError("Koopman collection requires excitation.mode=reference; "
                         "direct throttle/brake/steering tests belong ONLY in Trucksim_Simulation_ltr.py")
    patterns = excite["patterns"]
    if not patterns or any(p not in {"step", "sine", "chirp", "prbs", "combined"} for p in patterns):
        raise ValueError("Bad excitation.patterns")
    seeds = batch["seeds"]
    if (not isinstance(seeds, list) or not seeds or
        any(type(seed) is not int or seed < 0 for seed in seeds) or
        len(set(seeds)) != len(seeds)):
        raise ValueError("batch.seeds must be nonempty, unique nonnegative integers")
    if len(set(patterns)) != len(patterns):
        raise ValueError("excitation.patterns must be unique")
    if float(excite["duration_s"]) <= 0 or float(excite["warmup_s"]) < 0:
        raise ValueError("Invalid excitation duration / warmup")
    if float(excite["warmup_s"]) >= float(excite["duration_s"]):
        raise ValueError("Warmup cannot cover entire trajectory")
    if float(excite["target_dt_s"]) <= 0 or float(excite["controller_dt_s"]) <= 0:
        raise ValueError("Sample intervals must be positive")
    if not np.isclose(float(excite["target_dt_s"]), float(excite["controller_dt_s"]), atol=1e-9):
        raise ValueError("controller_dt_s must equal target_dt_s until interval-averaged controls are implemented")
    validate_excitation_config(excite)
    # Catch duplicated reference sequences BEFORE launching expensive TruckSim workers.
    for pattern in patterns:
        fingerprints = [excitation_fingerprint(excite, pattern, seed) for seed in seeds]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError(f"Duplicate excitation sequence across seeds for {pattern}")
    if not cfg["reference_controller"].get("confirmed_calibrated", False):
        raise ValueError("Reference mode is blocked until reference_controller.confirmed_calibrated=true. "
                         "Confirm reference tracking and actuator units across enabled vehicle loads")
    if require_local_files:
        for model in models:
            model_signature(model)
            model_condition(model)
    return True


def build_runs(cfg, model_id=None, speed_tier=None):
    """One trajectory for each (model, waveform family, distinct seed)."""
    excite = cfg["excitation"]
    signatures = {(p, seed): excitation_fingerprint(excite, p, seed)
                  for p in excite["patterns"] for seed in cfg["batch"]["seeds"]}
    for model in mapped_models(cfg, model_id=model_id, speed_tier=speed_tier):
        for pattern in excite["patterns"]:
            for seed in cfg["batch"]["seeds"]:
                yield {"id": f"{model['id']}__{pattern}_seed{int(seed):03d}",
                       "model_id": model["id"], "pattern": pattern,
                       "seed": int(seed), "excitation_sha256": signatures[(pattern, seed)]}


def _number(seq, name, *, minimum=None, maximum=None):
    try:
        n = float(seq[name])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"excitation.pattern_params.{name} must be numeric") from exc
    if not np.isfinite(n) or (minimum is not None and n <= minimum) or (maximum is not None and n > maximum):
        raise ValueError(f"Invalid excitation parameter {name}={n}")
    return n


def validate_excitation_config(cfg):
    """Validate explicitly configured frequency, timing, and amplitude for all patterns."""
    ref = cfg.get("reference", {})
    for k in ("ax_max_mps2", "front_steer_max_deg"):
        _number(ref, k, minimum=0)
    pars = cfg.get("pattern_params")
    if not isinstance(pars, dict):
        raise ValueError("excitation.pattern_params must be a YAML mapping")
    for pattern in cfg["patterns"]:
        d = pars.get(pattern)
        if not isinstance(d, dict):
            raise ValueError(f"Missing excitation.pattern_params.{pattern}")
        if pattern == "step":
            period = _number(d, "period_s", minimum=0)
            fade = _number(d, "transition_s", minimum=0)
            if fade > period:
                raise ValueError("step.transition_s must not exceed step.period_s")
            jitter = _number(d, "level_jitter", minimum=-1e-12, maximum=0.4)
            lv = np.asarray(d.get("levels", []), dtype=float)
            if lv.ndim != 2 or lv.shape[1] != 2 or len(lv) < 2 or not np.isfinite(lv).all() or np.any(np.abs(lv) > 1):
                raise ValueError("step.levels must be >=2 bounded [a, steer] pairs within [-1,1]")
        elif pattern == "sine":
            for name in ("ax_frequency_hz", "steer_frequency_hz"):
                _number(d, name, minimum=0)
            for name in ("ax_scale", "steer_scale"):
                _number(d, name, minimum=0, maximum=1)
        elif pattern == "chirp":
            _number(d, "start_frequency_hz", minimum=0)
            _number(d, "frequency_slope_hz_per_s", minimum=-1e-12)
            for name in ("ax_scale", "steer_scale"):
                _number(d, name, minimum=0, maximum=1)
        elif pattern == "prbs":
            period = _number(d, "period_s", minimum=0)
            fade = _number(d, "transition_s", minimum=0)
            if fade > period:
                raise ValueError("prbs.transition_s must not exceed prbs.period_s")
            for name in ("ax_scale", "steer_scale"):
                _number(d, name, minimum=0, maximum=1)
        elif pattern == "combined":
            for prefix in ("ax", "steer"):
                freq = d.get(prefix+"_frequencies_hz")
                amp = d.get(prefix+"_scales")
                if (not isinstance(freq, list) or not isinstance(amp, list) or
                    len(freq) == 0 or len(freq) != len(amp) or len(freq) > 4):
                    raise ValueError(f"combined {prefix} frequencies and scales must be matching nonempty lists")
                for f, a in zip(freq, amp):
                    if not np.isfinite(float(f)) or float(f) <= 0 or not np.isfinite(float(a)) or float(a) < 0:
                        raise ValueError(f"Invalid combined {prefix} frequency/amplitude")
            _number(d, "clip", minimum=0, maximum=1)
    return True


class Excitation:
    """Seeded, reproducible [a_ref m/s², front steer rad] with YAML-driven waveforms."""

    def __init__(self, cfg, pattern, seed):
        self.cfg = cfg
        self.pattern = pattern
        self.seed = int(seed)
        d = cfg["pattern_params"][pattern]
        self.params = d
        rng = np.random.default_rng(self.seed)
        self.phases = rng.uniform(0, 2*np.pi, size=8)
        if pattern == "step":
            levels = np.asarray(d["levels"], dtype=float)
            self.step_levels = levels[rng.permutation(len(levels))]
            self.step_jitter = rng.uniform(-float(d["level_jitter"]),
                                          float(d["level_jitter"]), size=(4096, 2))
        if pattern == "prbs":
            n = max(4096, int(np.ceil(float(cfg["duration_s"])/float(d["period_s"]))) + 4)
            self.prbs_long = rng.choice([-1., 1.], size=n)
            self.prbs_lat = rng.choice([-1., 1.], size=n)

    def targets(self, t):
        after = float(t) - float(self.cfg["warmup_s"])
        if after < 0:
            return np.zeros(2, dtype=float)
        amp = self.cfg["reference"]
        wave = np.clip(self._wave(after), -1.0, 1.0)
        return np.array([float(wave[0]) * float(amp["ax_max_mps2"]),
                         np.deg2rad(float(wave[1]) * float(amp["front_steer_max_deg"]))],
                        dtype=float)

    @staticmethod
    def _smooth(x):
        a = np.clip(x, 0., 1.)
        return a*a*(3-2*a)

    def _wave(self, t):
        d = self.params
        ph = self.phases
        if self.pattern == "step":
            period = float(d["period_s"])
            k = int(t // period)
            level = self.step_levels[k % len(self.step_levels)] + self.step_jitter[k % len(self.step_jitter)]
            previous = (np.zeros(2) if k == 0 else
                        self.step_levels[(k-1) % len(self.step_levels)] +
                        self.step_jitter[(k-1) % len(self.step_jitter)])
            alpha = self._smooth((t % period) / float(d["transition_s"]))
            return np.clip((1.-alpha)*previous + alpha*level, -1., 1.)
        if self.pattern == "sine":
            return np.array([float(d["ax_scale"])*np.sin(2*np.pi*float(d["ax_frequency_hz"])*t+ph[0]),
                             float(d["steer_scale"])*np.sin(2*np.pi*float(d["steer_frequency_hz"])*t+ph[1])])
        if self.pattern == "chirp":
            phase = 2*np.pi*(float(d["start_frequency_hz"])*t +
                             .5*float(d["frequency_slope_hz_per_s"])*t*t)
            # Seeded phases ensure chirp does not repeat with different seeds.
            return np.array([float(d["ax_scale"])*np.sin(phase+ph[0]),
                             float(d["steer_scale"])*np.sin(phase+ph[1])])
        if self.pattern == "prbs":
            period = float(d["period_s"])
            k = int(t // period)
            alpha = self._smooth((t % period) / float(d["transition_s"]))
            a = self.prbs_long[min(k, len(self.prbs_long)-2)]
            b = self.prbs_long[min(k+1, len(self.prbs_long)-1)]
            c = self.prbs_lat[min(k, len(self.prbs_lat)-2)]
            e = self.prbs_lat[min(k+1, len(self.prbs_lat)-1)]
            return np.array([float(d["ax_scale"])*((1-alpha)*a + alpha*b),
                             float(d["steer_scale"])*((1-alpha)*c + alpha*e)])
        if self.pattern == "combined":
            v = []
            for j, prefix in enumerate(("ax", "steer")):
                fs = d[prefix+"_frequencies_hz"]
                scales = d[prefix+"_scales"]
                v.append(sum(float(a)*np.sin(2*np.pi*float(f)*t+ph[j*4+i])
                             for i, (f, a) in enumerate(zip(fs, scales))))
            return np.clip(np.asarray(v), -float(d["clip"]), float(d["clip"]))
        raise ValueError(f"Unsupported excitation pattern: {self.pattern}")


def excitation_fingerprint(cfg, pattern, seed):
    """Hash the precise scheduled 10 Hz reference sequence to prevent redundant seeds."""
    dt = float(cfg["controller_dt_s"])
    duration = float(cfg["duration_s"])
    generator = Excitation(cfg, pattern, seed)
    n = int(np.floor(duration/dt+1e-8))
    u = np.stack([generator.targets(k*dt) for k in range(n)])
    return hashlib.sha256(np.round(u, 12).astype('<f8').tobytes()).hexdigest()


def _validated_engine_map(value):
    """A calibrated, finite RPM x 11-throttle map held in one NumPy array."""
    data = np.asarray(value, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] != 12 or not np.isfinite(data).all():
        raise ValueError("Engine map must have >=2 RPM rows and 12 columns, all finite")
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("Engine map RPM rows must be strictly increasing")
    data = np.ascontiguousarray(data)
    data.setflags(write=False)
    return data


def _engine_map_sha256(data):
    """Stable checksum of calibrated numeric values, independent of CSV formatting."""
    canonical = np.asarray(data, dtype='<f8', order='C')
    return hashlib.sha256(canonical.tobytes(order='C')).hexdigest()


def load_engine_map_for_collection(config):
    """Parse the shared engine CSV once in collect_batch, before starting workers.

    A worker launched by collect_batch receives the parsed values in its job JSON
    and therefore does not open the CSV. Direct standalone worker invocations
    without a batch job payload parse the CSV once before loading TruckSim.
    No import from trucksim_dynamics.py (which pulls in the simulator/SciPy).
    """
    path = config.get('engine_map_path')
    if path is None:
        path = Path(__file__).resolve().parent.parent / 'trucksim_simulation' / 'engine_map_4455kg.csv'
    else:
        path = Path(path)
        if not path.is_absolute():
            # Relative paths in configs/koopman.yaml are relative to all_merge/.
            path = Path(__file__).resolve().parents[3] / path
    data = _validated_engine_map(np.loadtxt(path, delimiter=',', skiprows=1))
    return data


class ReferenceActuator:
    """Map [ax_ref, delta_ref] to native TruckSim actuation via existing controllers.

    Calibration must cover vehicle mass, engine map, steering ratio and gains.
    """

    def __init__(self, cfg, units, dt, engine_map=None):
        from highway_env.vehicle.trucksim_simulation.low_controller import (
            PIDController, acceleration_control, LateralSteeringController)
        c = cfg
        lat = c["lateral"]
        self.low = PIDController(**c["longitudinal_pid"], dt=dt)
        self.lat = LateralSteeringController(
            steering_ratio=float(lat["steering_ratio"]), dt=dt,
            kp=lat["kp"], ki=lat["ki"], kd=lat["kd"],
            max_wheel_deg=lat["max_wheel_deg"],
            max_rate_deg_s=lat["max_rate_deg_s"],
            steering_tau_s=lat["steering_tau_s"])
        self.acceleration_control = acceleration_control
        # The batch parent parses the CSV once. Every worker receives in-memory
        # calibration values; the control loop must never do file I/O.
        self.engine_map = _validated_engine_map(
            engine_map if engine_map is not None else
            load_engine_map_for_collection({"engine_map_path": c.get("engine_map_path")})
        )
        self.max_rpm = float(c["max_rpm"])
        self.ctrl_state = "throttle"
        self.units = units
        self.throttle_limit_pct = float(c.get("throttle_limit_pct", 35.0))
        self.brake_limit_mpa = float(c.get("brake_limit_mpa", 2.0))

    def control(self, ref, raw):
        from .data import G0
        vx_mps = float(raw[2]) / 3.6 if self.units["velocity"] == "km/h" else float(raw[2])
        ax_mps2 = float(raw[0]) * G0
        engine_rpm = float(raw[12])/6 if self.units["engine_speed"] == "deg/s" else float(raw[12])
        throttle01, brake, self.ctrl_state = self.acceleration_control(
            vx=vx_mps, engine_rpm=engine_rpm, ax_ref=float(ref[0]),
            ax_actual=ax_mps2, state=self.ctrl_state,
            low_controller=self.low, engine_map=self.engine_map, max_rpm=self.max_rpm)
        # Legacy controller returns [0,1], this TruckSim import is PERCENT [0,100].
        throttle_pct = 100.0*float(throttle01)
        steer_deg = self.lat.control(float(ref[1]),
            np.deg2rad(float(0.5*(raw[13]+raw[14]))))
        return np.array([np.clip(throttle_pct, 0, self.throttle_limit_pct),
                         np.clip(brake, 0, self.brake_limit_mpa), steer_deg], dtype=float)


def run_worker(cfg, run, output_dir):
    """Executed exactly once in a dedicated subprocess. No reuse of native DLL."""
    from highway_env.vehicle.trucksim_simulation.Trucksim_Simulation_ltr import (
        TrucksimSimulationLTR, plot_all_exports)

    model = model_for_run(cfg, run)
    validate(cfg, require_local_files=True, model_id=model["id"])
    conditions = model_condition(model)
    # Bind worker to the exact file signatures/labels chosen by the parent.
    # This prevents a mid-batch YAML edit or .sim/.dll replacement from
    # silently putting a different load under the original run ID.
    if "expected_condition" in run and run["expected_condition"] != conditions:
        raise RuntimeError(f"Run {run['id']} condition differs from parent request")
    if "expected_signature" in run and run["expected_signature"] != model_signature(model):
        raise RuntimeError(f"Run {run['id']} TruckSim model files differ from parent request")
    mode = "reference"
    excite = Excitation(cfg["excitation"], run["pattern"], run["seed"])
    fingerprint = excitation_fingerprint(cfg["excitation"], run["pattern"], run["seed"])
    if run.get("excitation_sha256", fingerprint) != fingerprint:
        raise RuntimeError(f"Run {run['id']} excitation settings changed since scheduling")
    # Decode the calibration sent by the batch parent. No engine CSV reads in
    # batch workers; direct standalone worker calls fall back to one CSV load.
    serialized_map = run.get("engine_map_values")
    if serialized_map is not None:
        engine_map = _validated_engine_map(serialized_map)
        expected_map_sha = run.get("engine_map_sha256")
        if not expected_map_sha or _engine_map_sha256(engine_map) != expected_map_sha:
            raise RuntimeError("Engine map checksum mismatch in worker request")
        map_origin = "batch_parent_once"
    else:
        engine_map = load_engine_map_for_collection(cfg["reference_controller"])
        expected_map_sha = _engine_map_sha256(engine_map)
        map_origin = "standalone_worker_once"
    # Construct all controller dependencies BEFORE loading the TruckSim DLL.
    # Avoid importing trucksim_dynamics -> controller -> SciPy during solver use.
    ctrl_dt = float(cfg["excitation"]["controller_dt_s"])
    controller = ReferenceActuator(cfg["reference_controller"], cfg["units"],
                                   ctrl_dt, engine_map=engine_map)
    print(f"[WORKER] Controller ready; engine map={map_origin}, sha256={expected_map_sha[:12]}; "
          f"native DLL not loaded yet: {run['id']}", flush=True)
    sim = TrucksimSimulationLTR(model["simfile"], model["dll"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0_start = time.monotonic()
    started = False
    times = []
    states = []
    us_native = []
    us_reference = []
    try:
        print(f"[WORKER] Native TruckSim start: {run['id']}", flush=True)
        sim.start({"vx_init": conditions["vx_init_kmh"]})
        started = True
        print(f"[WORKER] Native TruckSim initialized: {run['id']}", flush=True)
        nc = int(sim.configuration["n_import"])
        ne = int(sim.configuration["n_export"])
        if nc != 3 or ne != 19:
            raise ValueError(f"TruckSim expected imports=3, exports=19; got {nc}/{ne}")
        dt = float(sim.get_time_step())
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Invalid solver dt")
        ctrl_dt = float(cfg["excitation"]["controller_dt_s"])
        stride_float = ctrl_dt/dt
        stride = round(stride_float)
        if stride < 1 or not np.isclose(stride, stride_float, atol=1e-7):
            raise ValueError(f"controller_dt_s {ctrl_dt} must be integer multiple of TruckSim dt {dt}")
        t0 = float(sim.current_time)
        tstop = float(sim.configuration["t_stop"])
        duration = float(cfg["excitation"]["duration_s"])
        if tstop >= 0:
            duration = min(duration, tstop-t0)
        nsteps = int(np.floor(duration/dt+1e-7))
        if nsteps <= 0:
            raise ValueError(f"Invalid duration: tstop={tstop}, t0={t0}")
        if nsteps > int(cfg["batch"].get("max_solver_steps_per_run", 100000)):
            raise ValueError("Exceeded configured maximum solver steps")
        x = np.asarray(sim.get_export_array(), dtype=float)
        if x.shape != (19,) or not np.isfinite(x).all():
            raise ValueError("Bad initial TruckSim exports")
        vx_start_actual = float(x[2])
        speed_tol = float(cfg["trucksim"].get("initial_speed_tolerance_kmh", 2.0))
        if not np.isfinite(speed_tol) or speed_tol < 0:
            raise ValueError("trucksim.initial_speed_tolerance_kmh must be >= 0")
        if abs(vx_start_actual-conditions["vx_init_kmh"]) > speed_tol:
            raise ValueError(f"TruckSim initial vx={vx_start_actual:g} km/h does not match "
                             f"mapped {conditions['speed_tier']}={conditions['vx_init_kmh']:g} km/h; "
                             "verify SV_VXS and the selected .sim/.dll model")
        times.append(t0)
        states.append(x.copy())
        last_native = np.zeros(3, dtype=float)
        last_ref = np.zeros(2, dtype=float)
        for k in range(nsteps):
            if k % stride == 0:
                last_ref = excite.targets(k*dt)
                last_native = controller.control(last_ref, x)
                if not np.isfinite(last_native).all() or last_native.shape != (3,):
                    raise ValueError("Nonfinite native control")
                if not (0 <= last_native[0] <= 100 and 0 <= last_native[1] <= 10):
                    raise ValueError("Throttle percent or brake MPa outside safe actuator domain")
            tn = t0 + (k+1)*dt
            status, x = sim.run(tn, last_native, x)
            us_native.append(last_native.copy())
            us_reference.append(last_ref.copy())
            times.append(float(sim.current_time))
            states.append(np.asarray(x, dtype=float).copy())
            if status:
                raise RuntimeError(f"TruckSim solver stopped early at t={tn:g} status={status}")
        times = np.asarray(times)
        raw = np.stack(states)
        native = np.stack(us_native)
        refs = np.stack(us_reference)
        st, pose, extra = to_si(raw, cfg["units"])
        ltr = ltr_from_fz(raw, sign=float(cfg["units"].get("wheel_compression_sign", 1)))
        if not np.isfinite(ltr).all():
            raise RuntimeError("Invalid wheel loads or undefined LTR; verify wheel load sign")
        processed = resample(times, st, pose, ltr, native, refs,
                             dt=float(cfg["excitation"]["target_dt_s"]))
        # IMPORTANT: Shutdown must succeed BEFORE any file is marked train-eligible.
        # Keep the trajectory in memory until the real solver termination returns.
        print(f"[WORKER] Native TruckSim terminate: {run['id']}", flush=True)
        sim.stop(native_terminate=True)
        print(f"[WORKER] Native TruckSim terminated: {run['id']}", flush=True)
        path = out / (run["id"] + ".npz")
        np.savez_compressed(path,
            t_raw_s=times, trucksim_raw=raw, u_native_raw=native, u_ref_raw_si=refs,
            state_raw_si=st, pose_raw_si=pose, ltr_raw=ltr,
            ax_raw_mps2=extra["ax_mps2"], ay_raw_mps2=extra["ay_mps2"],
            delta_front_raw_rad=extra["delta_front_rad"],
            engine_raw_rpm=extra["engine_rpm"],
            **processed, train_eligible=np.array(True),
            model_id=np.array(model["id"]),
            payload_tier=np.array(conditions["payload_tier"]),
            friction_tier=np.array(conditions["friction_tier"]),
            speed_tier=np.array(conditions["speed_tier"]),
            vx_init_kmh=np.array(conditions["vx_init_kmh"], dtype=float),
            excitation_sha256=np.array(fingerprint),
            payload_kg=np.array(conditions["payload_kg"], dtype=float),
            road_mu=np.array(conditions["road_mu"], dtype=float))
        # CSV is human-readable; row i contains state at t[i], and native
        # command applied over [t[i], t[i+1]). Last row has no outgoing input.
        csv_path = out / (run["id"] + ".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", *SIGNALS,
                        "u_to_next_throttle_pct", "u_to_next_brake_mpa", "u_to_next_steer_wheel_deg",
                        "a_ref_to_next_mps2", "delta_ref_to_next_rad", "LTR_from_Fz"])
            for i in range(len(times)):
                u = native[i].tolist() if i < len(native) else [""]*3
                r = refs[i].tolist() if i < len(refs) else [""]*2
                w.writerow([times[i], *raw[i].tolist(), *u, *r, ltr[i]])
        meta = {
            "run_id": run["id"], "model_id": model["id"],
            "pattern": run["pattern"], "seed": run["seed"],
            "excitation_sha256": fingerprint,
            "engine_map_sha256": expected_map_sha,
            "engine_map_source": map_origin,
            "excitation_config": cfg["excitation"],
            "mode": mode, "train_eligible": True,
            "duration_actual_s": float(times[-1]-times[0]),
            "solver_dt_s": dt, "target_dt_s": cfg["excitation"]["target_dt_s"],
            "raw_samples": len(times), "target_samples": len(processed["t10_s"]),
            "trucksim": model_signature(model),
            "model_units": cfg["units"], "raw_channel_units": UNITS_RAW,
            "payload_tier": conditions["payload_tier"],
            "friction_tier": conditions["friction_tier"],
            "speed_tier": conditions["speed_tier"],
            "vx_init_kmh": conditions["vx_init_kmh"],
            "vx_init_actual_kmh": vx_start_actual,
            "native_import_units": ["percent_0_to_100", "MPa", "deg"],
            "state_si_order": ["vx_mps", "vy_mps", "body_yaw_rate_radps", "roll_rad", "euler_roll_rate_radps"],
            "reference_si_order": ["ax_ref_mps2", "front_wheel_delta_ref_rad"],
            "manual_conditions": conditions,
            "notes": "Payload/friction are fixed within selected .sim; initial speed is set from mapped speed tier via SV_VXS. "
                     "Native actuation is generated ONLY via low_controller.py + inverse_dynamics.py; "
                     "direct actuation diagnostic is exclusive to Trucksim_Simulation_ltr.py.",
            "sample_alignment": "raw control[k] acts from raw_state[k] to raw_state[k+1]; similarly at 10Hz",
        }
        json_write(path.with_suffix(".json"), meta)
        if bool(run.get("plot", False)):
            plot_all_exports(times, np.column_stack([raw, ltr]),
                             out / (run["id"]+".png"),
                             title=f"TruckSim TRACE | {run['id']} | {mode}")
        audit = audit_npz(path)
        if not audit["valid"]:
            raise RuntimeError(f"Dataset audit failed: {audit['issues']}")
        result = {"status": "ok", "run_id": run["id"], "npz": str(path),
                  "csv": str(csv_path), "duration_s": meta["duration_actual_s"],
                  "samples": len(times), "train_eligible": True,
                  "wall_time_s": round(time.monotonic()-t0_start,3)}
        json_write(out / (run["id"] + ".result.json"), result)
        print(f"[OK] {run['id']} {len(times)} raw samples; {len(processed['t10_s'])} target samples", flush=True)
        return result
    finally:
        if started and sim.is_running:
            # Attempt SDK shutdown during an abnormal Python path as well.
            # The worker remains failed if this native call raises or faults.
            sim.stop(native_terminate=True)


def collect_batch(cfg, config_path, project_root, dry_run=False, model_id=None, speed_tier=None):
    """Spawn one isolated child per run; select exact .sim/.dll via model map."""
    validate(cfg, require_local_files=not dry_run, model_id=model_id, speed_tier=speed_tier)
    jobs = list(build_runs(cfg, model_id=model_id, speed_tier=speed_tier))
    models = {m["id"]: m for m in mapped_models(cfg, model_id=model_id, speed_tier=speed_tier)}
    output = Path(cfg["batch"]["output_dir"])
    if not output.is_absolute():
        output = Path(project_root) / output
    print(f"[COLLECT] mode={cfg['excitation']['mode']} models={list(models)} "
          f"runs={len(jobs)} output={output}")
    if dry_run:
        for run in jobs:
            m = models[run["model_id"]]
            print("[PLAN]", run["id"], "->", m["simfile"],
                  "payload_tier=", m["payload_tier"], "payload_kg=", m["payload_kg"],
                  "friction_tier=", m["friction_tier"], "mu=", m["road_mu"],
                  "speed_tier=", m["speed_tier"], "vx_init_kmh=", m["vx_init_kmh"],
                  "sha256=", run["excitation_sha256"][:12], "dll=", m["dll"])
        return []
    output.mkdir(parents=True, exist_ok=True)
    # One CSV parse for the ENTIRE sequential batch; make the small 2D table
    # part of each worker request so independent processes do not parse CSV.
    batch_engine_map = load_engine_map_for_collection(cfg["reference_controller"])
    engine_map_values = batch_engine_map.tolist()
    engine_map_sha256 = _engine_map_sha256(batch_engine_map)
    print(f"[ENGINE MAP] CSV parsed once for {len(jobs)} runs; "
          f"shape={batch_engine_map.shape}, sha256={engine_map_sha256[:12]}", flush=True)
    # File signatures alone cannot capture linked TruckSim .par contents;
    # operator verifies every .sim and its referenced parameters manually.
    signatures = {mid: model_signature(m) for mid, m in models.items()}
    batch_name = model_id if model_id is not None else (f"speed_{speed_tier}" if speed_tier else "all")
    json_write(output / f"batch_config_{batch_name}.json", {
        "config": cfg, "model_signatures_before": signatures,
        "engine_map_sha256": engine_map_sha256,
        "engine_map_rows": len(batch_engine_map),
        "resolved_models": {mid: model_condition(m) for mid, m in models.items()},
        "runs": jobs})
    script = Path(project_root) / "scripts" / "run_koopman.py"
    if not script.exists():
        raise FileNotFoundError(script)
    manifest = []
    summary_path = output / f"manifest_{batch_name}.json"
    timeout = float(cfg["batch"].get("timeout_s_per_run", 120))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    for i, job in enumerate(jobs):
        existing = output / (job["id"] + ".npz")
        if existing.exists() and not bool(cfg["batch"].get("allow_overwrite", False)):
            raise FileExistsError(f"Existing run will not be overwritten: {existing}")
        # Validate all configured model files remain unchanged during the batch.
        for mid, m in models.items():
            if model_signature(m) != signatures[mid]:
                raise RuntimeError(f"TruckSim model {mid} file changed mid-batch; stop to relabel data")
        job = dict(job)
        job["expected_signature"] = signatures[job["model_id"]]
        job["expected_condition"] = model_condition(models[job["model_id"]])
        job["engine_map_values"] = engine_map_values
        job["engine_map_sha256"] = engine_map_sha256
        job["plot"] = bool(cfg["batch"].get("plot_first_per_pattern", False)) and (
            int(job["seed"]) == int(cfg["batch"]["seeds"][0]))
        job_file = output / (job["id"] + ".request.json")
        json_write(job_file, job)
        cmd = [sys.executable, str(script), "worker", "--config", str(Path(config_path).resolve()),
               "--job", str(job_file), "--output", str(output)]
        start = time.monotonic()
        print(f"[RUN {i+1}/{len(jobs)}] {job['id']} ({job['model_id']})", flush=True)
        try:
            p = subprocess.run(cmd, cwd=str(project_root), env=env,
                               timeout=timeout, capture_output=True, text=True)
            status = "ok" if p.returncode == 0 else "failed"
            details = {"id": job["id"], "model_id": job["model_id"],
                       "status": status, "returncode": p.returncode,
                       "wall_time_s": round(time.monotonic()-start,2)}
            (output/(job["id"]+".stdout.log")).write_text(p.stdout, encoding="utf-8")
            (output/(job["id"]+".stderr.log")).write_text(p.stderr, encoding="utf-8")
            if p.returncode == 0:
                result_path = output / (job["id"] + ".result.json")
                npz_path = output / (job["id"] + ".npz")
                if not result_path.is_file() or not npz_path.is_file():
                    status = details["status"] = "missing_output"
                else:
                    result_info = json.loads(result_path.read_text(encoding="utf-8"))
                    audit = audit_npz(npz_path)
                    if result_info.get("run_id") != job["id"] or not audit["valid"]:
                        status = details["status"] = "invalid_output"
                        details["issues"] = audit["issues"]
                    else:
                        with np.load(npz_path, allow_pickle=False) as f:
                            if (str(f["model_id"].item()) != job["model_id"] or
                                str(f["payload_tier"].item()) != job["expected_condition"]["payload_tier"] or
                                str(f["friction_tier"].item()) != job["expected_condition"]["friction_tier"] or
                                str(f["speed_tier"].item()) != job["expected_condition"]["speed_tier"] or
                                float(f["vx_init_kmh"].item()) != job["expected_condition"]["vx_init_kmh"] or
                                str(f["excitation_sha256"].item()) != job["excitation_sha256"] or
                                float(f["payload_kg"].item()) != job["expected_condition"]["payload_kg"] or
                                float(f["road_mu"].item()) != job["expected_condition"]["road_mu"]):
                                status = details["status"] = "wrong_model_label"
            if status != "ok":
                print("[FAILED]", job["id"], status, p.stderr[-1500:], flush=True)
        except subprocess.TimeoutExpired:
            details = {"id": job["id"], "model_id": job["model_id"],
                       "status": "timeout", "wall_time_s": round(time.monotonic()-start,2)}
            print("[TIMEOUT]", job["id"], flush=True)
        manifest.append(details)
        json_write(summary_path, {"config": str(config_path), "runs": manifest})
        if details["status"] != "ok" and bool(cfg["batch"].get("stop_on_error", True)):
            raise RuntimeError(f"Worker failure: {job['id']}; see {output} logs")
    print(f"[DONE] {sum(x['status']=='ok' for x in manifest)}/{len(jobs)} runs OK; {summary_path}")
    return manifest
