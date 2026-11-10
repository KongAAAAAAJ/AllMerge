"""TruckSim reference-command collection with one child process per trajectory.

A process may crash during native DLL teardown even after producing all output.
The parent therefore accepts only COMPLETE, independently audited artifacts,
regardless of child exit status. Each child writes to an isolated staging
folder; the parent alone publishes verified files to the training directory.
This does not claim that the native heap defect has been repaired.
"""
from __future__ import annotations

import csv
import hashlib
from itertools import product
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import shutil

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
    """Resolve physical TruckSim models from payload x friction ONLY.

    Each enabled (payload, friction) pair owns one .sim/.dll. Initial speed is
    a per-run VS solver initialization parameter and never selects DLL paths.
    ``speed_tier`` is accepted as a filter validation for the public CLI, but
    filtering by speed happens in :func:`build_runs`.
    """
    trucksim = cfg.get("trucksim", {})
    payload_levels = _tier_table(trucksim, "payload_tiers_kg")
    friction_levels = _tier_table(trucksim, "friction_tiers")
    speed_levels = _tier_table(trucksim, "speed_tiers_kmh")
    if speed_tier is not None and speed_tier not in speed_levels:
        raise ValueError(f"Unknown speed tier: {speed_tier}")
    models = trucksim.get("models", [])
    if isinstance(models, dict):
        defaults = models.get("defaults", {})
        overrides = models.get("overrides", {})
        if not isinstance(defaults, dict) or not isinstance(overrides, dict):
            raise ValueError("Model defaults and overrides must be mappings")
        templates = {}
        for key in ("id", "simfile", "dll"):
            template = models.get(f"{key}_template")
            if not isinstance(template, str) or not template.strip():
                raise ValueError(f"trucksim.models.{key}_template must be a nonempty string")
            if "speed_tier" in template or "vx_init" in template:
                raise ValueError(f"trucksim.models.{key}_template must NOT contain speed: "
                                 "speed is set with SV_VXS before Initialize()")
            templates[key] = template
        expanded = []
        generated_ids = set()
        for p, f in product(payload_levels, friction_levels):
            fields = {"payload_tier": p, "friction_tier": f}
            try:
                mid = templates["id"].format(**fields)
                fields["id"] = mid
                item = dict(defaults)
                item.update(fields)
                for key in ("simfile", "dll"):
                    item[key] = templates[key].format(**fields)
            except (KeyError, IndexError, ValueError, AttributeError) as exc:
                raise ValueError(f"Invalid TruckSim physical model template: {exc}") from exc
            override = overrides.get(mid, {})
            if not isinstance(override, dict):
                raise ValueError(f"Model override {mid} must be a mapping")
            if set(override) & {"id", "payload_tier", "friction_tier", "speed_tier", "vx_init_kmh"}:
                raise ValueError(f"Model override {mid} cannot change model identity/tiers/speed")
            item.update(override)
            generated_ids.add(mid)
            expanded.append(item)
        unknown = set(overrides) - generated_ids
        if unknown:
            raise ValueError(f"Unknown physical model override ids: {sorted(unknown, key=str)}")
        models = expanded
    if not isinstance(models, list) or not models:
        raise ValueError("trucksim.models must contain at least one physical .sim/.dll model")
    seen_ids, seen_pairs, seen_sim, seen_dll = set(), set(), set(), set()
    selected = []
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("Every trucksim.models entry must be a mapping")
        mid = item.get("id", "")
        if not isinstance(mid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", mid):
            raise ValueError(f"Invalid physical model id {mid!r}")
        if mid in seen_ids:
            raise ValueError(f"Duplicate physical model id: {mid}")
        seen_ids.add(mid)
        p_tier = item.get("payload_tier")
        f_tier = item.get("friction_tier")
        if p_tier not in payload_levels or f_tier not in friction_levels:
            raise ValueError(f"Model {mid}: unknown payload/friction tier: {p_tier!r}, {f_tier!r}")
        if "speed_tier" in item or "vx_init_kmh" in item:
            raise ValueError(f"Physical model {mid} must not bind initial speed; "
                             "set speed_tiers_kmh and let build_runs vary SV_VXS")
        if not isinstance(item.get("enabled", True), bool):
            raise ValueError(f"Model {mid}: enabled must be boolean")
        if not item.get("enabled", True):
            continue
        pair = (p_tier, f_tier)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate enabled payload/friction pair: {pair}")
        seen_pairs.add(pair)
        for key in ("simfile", "dll"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Enabled physical model {mid} needs nonempty {key}")
        sim_path = str(Path(item["simfile"]).expanduser()).casefold()
        dll_path = str(Path(item["dll"]).expanduser()).casefold()
        if sim_path in seen_sim:
            raise ValueError(f"Different physical models reference the same .sim: {item['simfile']}")
        if dll_path in seen_dll:
            raise ValueError(f"Different physical models reference the same .dll: {item['dll']}")
        seen_sim.add(sim_path)
        seen_dll.add(dll_path)
        normalized = dict(item)
        normalized["payload_kg"] = payload_levels[p_tier]
        normalized["road_mu"] = friction_levels[f_tier]
        selected.append(normalized)
    if trucksim.get("require_complete_grid", False) and model_id is None:
        missing = sorted((p, f) for p in payload_levels for f in friction_levels
                         if (p, f) not in seen_pairs)
        if missing:
            raise ValueError(f"Missing enabled payload/friction models: {missing}")
    if model_id is not None:
        selected = [item for item in selected if item["id"] == model_id]
        if not selected:
            raise ValueError(f"Unknown or disabled physical model id: {model_id}")
    if not selected:
        raise ValueError("No enabled physical TruckSim models available")
    return selected


def model_for_run(cfg, run):
    """Resolve the exact model selected by the parent process."""
    mid = run.get("model_id")
    if not isinstance(mid, str) or not mid:
        raise ValueError("Worker request must specify model_id")
    return mapped_models(cfg, model_id=mid)[0]


def model_signature(model):
    return {key: file_signature(model[key]) for key in ("simfile", "dll")}


def model_condition(model, cfg, speed_tier=None):
    """Physical labels belong to .sim/.dll; initial speed belongs to a RUN."""
    mid = model["id"]
    if not model.get("verified_in_trucksim", False):
        raise ValueError(f"Confirm model {mid} payload/mu in TruckSim and set verified_in_trucksim=true")
    try:
        payload = float(model["payload_kg"])
        mu = float(model["road_mu"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Model {mid} needs numeric payload_kg and road_mu") from exc
    if not np.isfinite(payload) or payload < 0:
        raise ValueError(f"Model {mid} payload_kg must be finite and >=0")
    if not np.isfinite(mu) or not 0 < mu <= 2:
        raise ValueError(f"Model {mid} road_mu must satisfy 0 < mu <= 2")
    result = {"model_id": mid, "payload_tier": model["payload_tier"],
              "friction_tier": model["friction_tier"],
              "payload_kg": payload, "road_mu": mu,
              "verified_in_trucksim": True, "note": model.get("note", "")}
    if speed_tier is not None:
        speeds = _tier_table(cfg["trucksim"], "speed_tiers_kmh")
        if speed_tier not in speeds:
            raise ValueError(f"Unknown initial speed tier for {mid}: {speed_tier}")
        result["speed_tier"] = speed_tier
        result["vx_init_kmh"] = speeds[speed_tier]
    return result


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
        fingerprints = [excitation_fingerprint(excite, pattern, seed, episode_key=f"validation__{pattern}__seed{seed}") for seed in seeds]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError(f"Duplicate excitation sequence across seeds for {pattern}")
    if not cfg["reference_controller"].get("confirmed_calibrated", False):
        raise ValueError("Reference mode is blocked until reference_controller.confirmed_calibrated=true. "
                         "Confirm reference tracking and actuator units across enabled vehicle loads")
    if require_local_files:
        for model in models:
            model_signature(model)
            model_condition(model, cfg)
    return True


def build_runs(cfg, model_id=None, speed_tier=None):
    """One trajectory per physical model x initial speed x excitation x seed.

    All initial speeds of one (payload, mu) physical model reuse the EXACT SAME
    .sim/.dll, in separate fresh Python worker processes.
    """
    excite = cfg["excitation"]
    speeds = _tier_table(cfg["trucksim"], "speed_tiers_kmh")
    if speed_tier is not None:
        if speed_tier not in speeds:
            raise ValueError(f"Unknown speed tier: {speed_tier}")
        speeds = {speed_tier: speeds[speed_tier]}
    for model in mapped_models(cfg, model_id=model_id, speed_tier=speed_tier):
        for v in speeds:
            for pattern in excite["patterns"]:
                for seed in cfg["batch"]["seeds"]:
                    rid = f"{model['id']}_{v}__{pattern}_seed{int(seed):03d}"
                    yield {"id": rid,
                           "model_id": model["id"], "speed_tier": v,
                           "pattern": pattern, "seed": int(seed),
                           "excitation_sha256": excitation_fingerprint(excite, pattern, seed,
                                                                         episode_key=rid)}


def _number(seq, name, *, minimum=None, maximum=None):
    try:
        n = float(seq[name])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"excitation.pattern_params.{name} must be numeric") from exc
    if not np.isfinite(n) or (minimum is not None and n <= minimum) or (maximum is not None and n > maximum):
        raise ValueError(f"Invalid excitation parameter {name}={n}")
    return n


def _random_interval(random_cfg, pattern, name, default, *, minimum=0., maximum=None):
    """Return range or fixed value; no RNG and no filesystem I/O."""
    chosen = random_cfg.get(pattern, {}).get(name, default)
    values = chosen if isinstance(chosen, (list, tuple)) else [chosen, chosen]
    if len(values) != 2:
        raise ValueError(f"randomized.{pattern}.{name} must be a [lo, hi] interval")
    try:
        lo, hi = (float(values[0]), float(values[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"randomized.{pattern}.{name}: expected numbers") from exc
    if not (np.isfinite(lo) and np.isfinite(hi) and minimum <= lo <= hi):
        raise ValueError(f"randomized.{pattern}.{name}: require {minimum} <= lo <= hi")
    if maximum is not None and hi > maximum:
        raise ValueError(f"randomized.{pattern}.{name}: maximum cannot exceed {maximum}")
    return lo, hi


def _random_scalar(random_cfg, pattern, name, default, rng, *, minimum=0., maximum=None):
    lo, hi = _random_interval(random_cfg, pattern, name, default, minimum=minimum, maximum=maximum)
    return float(lo) if lo == hi else float(rng.uniform(lo, hi))


def _fraction(random_cfg, pattern, name, default=0.0):
    value = float(random_cfg.get(pattern, {}).get(name, default))
    if not np.isfinite(value) or not 0 <= value <= 0.5:
        raise ValueError(f"randomized.{pattern}.{name} must be in [0, 0.5]")
    return value


def _effective_seed(seed, pattern, episode_key, cfg):
    scope = cfg.get('seed_scope', 'shared_across_conditions')
    if scope not in ('per_episode', 'shared_across_conditions'):
        raise ValueError('excitation.seed_scope must be per_episode or shared_across_conditions')
    if scope == 'per_episode' and not episode_key:
        raise ValueError('per_episode excitation requires unique episode_key/run_id')
    # Do not use Python's hash() because it changes across processes.
    token = json.dumps([int(seed), str(pattern), str(episode_key) if scope == 'per_episode' else ''],
                       ensure_ascii=True, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(token.encode('utf-8')).digest()[:8], 'little')


def validate_excitation_config(cfg):
    """Check all fixed waveform values and optional bounded random ranges."""
    ref = cfg.get('reference', {})
    for key in ('ax_max_mps2', 'front_steer_max_deg'):
        _number(ref, key, minimum=0)
    params = cfg.get('pattern_params')
    if not isinstance(params, dict):
        raise ValueError('excitation.pattern_params must be a YAML mapping')
    randomized = cfg.get('randomized', {})
    safe_max_hz = 0.45 / float(cfg['target_dt_s'])
    if not isinstance(randomized, dict):
        raise ValueError('excitation.randomized must be a mapping')
    if 'phase_range_rad' in randomized:
        _random_interval({'global': randomized}, 'global', 'phase_range_rad', [0., 2*np.pi], minimum=-1000., maximum=1000.)
    if cfg.get('seed_scope', 'shared_across_conditions') not in ('per_episode','shared_across_conditions'):
        raise ValueError('excitation.seed_scope must be per_episode or shared_across_conditions')
    allowed_randomized = {
        'step': {'period_s','period_jitter_frac','level_scale'},
        'sine': {'ax_frequency_hz','steer_frequency_hz','ax_scale','steer_scale'},
        'chirp': {'start_frequency_hz','frequency_slope_hz_per_s','ax_scale','steer_scale'},
        'prbs': {'period_s','period_jitter_frac','amplitude_jitter_frac','ax_scale','steer_scale'},
        'combined': {'frequency_jitter_frac','amplitude_jitter_frac'},
    }
    for name, values in randomized.items():
        if name == 'phase_range_rad':
            continue
        if name not in allowed_randomized or not isinstance(values, dict):
            raise ValueError(f'Unknown excitation.randomized section: {name}')
        unknown = set(values) - allowed_randomized[name]
        if unknown:
            raise ValueError(f'Unknown excitation.randomized.{name} keys: {sorted(unknown)}')
    for pattern in cfg['patterns']:
        d = params.get(pattern)
        if not isinstance(d, dict):
            raise ValueError(f'Missing excitation.pattern_params.{pattern}')
        if not isinstance(randomized.get(pattern, {}), dict):
            raise ValueError(f'excitation.randomized.{pattern} must be a mapping')
        if pattern == 'step':
            period = _number(d, 'period_s', minimum=0)
            fade = _number(d, 'transition_s', minimum=0)
            if fade > period:
                raise ValueError('step.transition_s must not exceed period_s')
            _number(d, 'level_jitter', minimum=-1e-12, maximum=0.4)
            levels = np.asarray(d.get('levels', []), dtype=float)
            if (levels.ndim != 2 or levels.shape[1] != 2 or len(levels) < 2 or
                    not np.isfinite(levels).all() or np.any(np.abs(levels) > 1)):
                raise ValueError('step.levels must be >=2 bounded [a, steer] pairs')
            lo, _ = _random_interval(randomized,pattern,'period_s',period,minimum=0.01)
            if lo * (1 - _fraction(randomized,pattern,'period_jitter_frac')) < max(fade,float(cfg['controller_dt_s'])):
                raise ValueError('step minimum randomized period must exceed transition_s')
            _random_interval(randomized,pattern,'level_scale',1.,minimum=0.,maximum=1.)
        elif pattern == 'sine':
            for key in ('ax_frequency_hz','steer_frequency_hz'):
                v=_number(d,key,minimum=0)
                _random_interval(randomized,pattern,key,v,minimum=0.001,maximum=safe_max_hz)
            for key in ('ax_scale','steer_scale'):
                v=_number(d,key,minimum=0,maximum=1)
                _random_interval(randomized,pattern,key,v,minimum=0.,maximum=1.)
        elif pattern == 'chirp':
            f0=_number(d,'start_frequency_hz',minimum=0)
            slope=_number(d,'frequency_slope_hz_per_s',minimum=-1e-12)
            _, fmax = _random_interval(randomized,pattern,'start_frequency_hz',f0,minimum=0.,maximum=safe_max_hz)
            _, smax = _random_interval(randomized,pattern,'frequency_slope_hz_per_s',slope,minimum=0.)
            if fmax + smax * max(0.,float(cfg['duration_s'])-float(cfg['warmup_s'])) > safe_max_hz:
                raise ValueError('chirp randomized peak frequency exceeds 90% of Nyquist')
            for key in ('ax_scale','steer_scale'):
                v=_number(d,key,minimum=0,maximum=1)
                _random_interval(randomized,pattern,key,v,minimum=0.,maximum=1.)
        elif pattern == 'prbs':
            period = _number(d,'period_s',minimum=0)
            fade = _number(d,'transition_s',minimum=0)
            if fade > period:
                raise ValueError('prbs transition_s cannot exceed period_s')
            lo,_ = _random_interval(randomized,pattern,'period_s',period,minimum=0.01)
            if lo * (1 - _fraction(randomized,pattern,'period_jitter_frac')) < max(fade,float(cfg['controller_dt_s'])):
                raise ValueError('PRBS minimum randomized period must exceed transition_s')
            _fraction(randomized,pattern,'amplitude_jitter_frac')
            for key in ('ax_scale','steer_scale'):
                v=_number(d,key,minimum=0,maximum=1)
                _random_interval(randomized,pattern,key,v,minimum=0.,maximum=1.)
        elif pattern == 'combined':
            for prefix in ('ax','steer'):
                freqs=d.get(prefix+'_frequencies_hz')
                scales=d.get(prefix+'_scales')
                if (not isinstance(freqs,list) or not isinstance(scales,list) or
                    not freqs or len(freqs)!=len(scales) or len(freqs)>4):
                    raise ValueError('combined frequencies/scales must be matched 1..4-element lists')
                for f,a in zip(freqs,scales):
                    if not (np.isfinite(float(f)) and float(f)>0 and np.isfinite(float(a)) and float(a)>=0):
                        raise ValueError('combined frequency/amplitude invalid')
            _number(d,'clip',minimum=0,maximum=1)
            freq_jitter = _fraction(randomized,pattern,'frequency_jitter_frac')
            _fraction(randomized,pattern,'amplitude_jitter_frac')
            for prefix in ('ax','steer'):
                if max(map(float,d[prefix+'_frequencies_hz']))*(1+freq_jitter) > safe_max_hz:
                    raise ValueError('combined randomized frequencies exceed 90% of Nyquist')
    return True


class Excitation:
    """Deterministically randomized control reference for ONE physical episode.

    Random parameters and switching schedules are sampled exactly once in
    __init__. No RNG calls or disk access occur during targets() / TruckSim steps.
    """

    def __init__(self, cfg, pattern, seed, *, episode_key=None):
        self.cfg=cfg
        self.pattern=pattern
        self.seed=int(seed)
        self.episode_key=episode_key
        self.effective_seed=_effective_seed(self.seed,pattern,episode_key,cfg)
        self.params=cfg['pattern_params'][pattern]
        random_cfg=cfg.get('randomized',{})
        d=self.params
        rng=np.random.default_rng(self.effective_seed)
        phlo, phhi = _random_interval({'global': random_cfg}, 'global', 'phase_range_rad',
                                     [0., 2*np.pi], minimum=-1000., maximum=1000.)
        self.phases=rng.uniform(phlo,phhi,size=8)
        self.realized_params={'effective_seed':self.effective_seed, 'base_seed':self.seed,
                              'seed_scope':cfg.get('seed_scope','shared_across_conditions'),
                              'phase_range_rad':[float(phlo), float(phhi)]}
        sample=lambda name,default,maximum=None: _random_scalar(random_cfg,pattern,name,default,rng,
                                                                 maximum=maximum)
        active=max(0.,float(cfg['duration_s'])-float(cfg['warmup_s']))
        if pattern=='step':
            period=sample('period_s',d['period_s'])
            jitter=_fraction(random_cfg,pattern,'period_jitter_frac')
            scale=sample('level_scale',1.,maximum=1.)
            n=max(8,int(np.ceil(active/(period*(1-jitter))))+5)
            steps=period*rng.uniform(1-jitter,1+jitter,size=n)
            self.switch_times=np.concatenate(([0.],np.cumsum(steps)))
            levels=np.asarray(d['levels'],dtype=float)
            self.step_levels=scale*levels[rng.permutation(len(levels))]
            self.step_jitter=rng.uniform(-float(d['level_jitter']),float(d['level_jitter']),size=(n+1,2))
            self.realized_params.update({'period_s':period,'level_scale':scale,
                'switch_times_s':[float(x) for x in self.switch_times if x<=active]})
        elif pattern=='sine':
            self.realized_params.update({k:sample(k,d[k],maximum=1. if k.endswith('_scale') else None)
                for k in ('ax_frequency_hz','steer_frequency_hz','ax_scale','steer_scale')})
            self.realized_params['phases_rad']=[float(self.phases[0]),float(self.phases[1])]
        elif pattern=='chirp':
            self.realized_params.update({
                'ax_start_frequency_hz':sample('start_frequency_hz',d['start_frequency_hz']),
                'steer_start_frequency_hz':sample('start_frequency_hz',d['start_frequency_hz']),
                'ax_frequency_slope_hz_per_s':sample('frequency_slope_hz_per_s',d['frequency_slope_hz_per_s']),
                'steer_frequency_slope_hz_per_s':sample('frequency_slope_hz_per_s',d['frequency_slope_hz_per_s']),
                'ax_scale':sample('ax_scale',d['ax_scale'],maximum=1.),
                'steer_scale':sample('steer_scale',d['steer_scale'],maximum=1.),
                'phases_rad':[float(self.phases[0]),float(self.phases[1])]})
        elif pattern=='prbs':
            period=sample('period_s',d['period_s'])
            period_jitter=_fraction(random_cfg,pattern,'period_jitter_frac')
            amp_jitter=_fraction(random_cfg,pattern,'amplitude_jitter_frac')
            ax_scale=sample('ax_scale',d['ax_scale'],maximum=1.)
            steer_scale=sample('steer_scale',d['steer_scale'],maximum=1.)
            n=max(8,int(np.ceil(active/(period*(1-period_jitter))))+5)
            switch_periods=period*rng.uniform(1-period_jitter,1+period_jitter,size=n)
            self.switch_times=np.concatenate(([0.],np.cumsum(switch_periods)))
            signs=rng.choice([-1.,1.],size=(n+1,2))
            amplitudes=rng.uniform(1-amp_jitter,1,size=(n+1,2))
            self.prbs_levels=signs*amplitudes*np.array([ax_scale,steer_scale])
            kmax=min(len(self.switch_times),int(np.searchsorted(self.switch_times,active,'right'))+1)
            self.realized_params.update({'period_s':period,'ax_scale':ax_scale,'steer_scale':steer_scale,
                'period_jitter_frac':period_jitter,'amplitude_jitter_frac':amp_jitter,
                'switch_times_s':[float(x) for x in self.switch_times[:kmax]],
                'levels':[row.tolist() for row in self.prbs_levels[:kmax]]})
        elif pattern=='combined':
            fj=_fraction(random_cfg,pattern,'frequency_jitter_frac')
            aj=_fraction(random_cfg,pattern,'amplitude_jitter_frac')
            self.combined={}
            for prefix in ('ax','steer'):
                f=np.asarray(d[prefix+'_frequencies_hz'],dtype=float)
                a=np.asarray(d[prefix+'_scales'],dtype=float)
                f=f*rng.uniform(1-fj,1+fj,size=len(f))
                a=a*rng.uniform(1-aj,1+aj,size=len(a))
                self.combined[prefix]=(f,a)
                self.realized_params[prefix+'_frequencies_hz']=f.tolist()
                self.realized_params[prefix+'_scales']=a.tolist()
            self.realized_params['phases_rad']=self.phases.tolist()

    def targets(self,t):
        active=float(t)-float(self.cfg['warmup_s'])
        if active<0:
            return np.zeros(2,dtype=float)
        max_ref=self.cfg['reference']
        wave=np.clip(self._wave(active),-1.,1.)
        return np.array([float(wave[0])*float(max_ref['ax_max_mps2']),
            np.deg2rad(float(wave[1])*float(max_ref['front_steer_max_deg']))],dtype=float)

    @staticmethod
    def _smooth(x):
        a=np.clip(x,0.,1.)
        return a*a*(3-2*a)

    def _switch(self,t):
        return max(0,min(len(self.switch_times)-2,int(np.searchsorted(self.switch_times,t,side='right')-1)))

    def _wave(self,t):
        d=self.params
        p=self.realized_params
        ph=self.phases
        if self.pattern=='step':
            k=self._switch(t)
            now=self.step_levels[k%len(self.step_levels)]+self.step_jitter[k]
            prev=np.zeros(2) if k==0 else self.step_levels[(k-1)%len(self.step_levels)]+self.step_jitter[k-1]
            fade=self._smooth((t-self.switch_times[k])/float(d['transition_s']))
            return (1.-fade)*prev+fade*now
        if self.pattern=='sine':
            return np.array([p['ax_scale']*np.sin(2*np.pi*p['ax_frequency_hz']*t+ph[0]),
                             p['steer_scale']*np.sin(2*np.pi*p['steer_frequency_hz']*t+ph[1])])
        if self.pattern=='chirp':
            phase_ax=2*np.pi*(p['ax_start_frequency_hz']*t+.5*p['ax_frequency_slope_hz_per_s']*t*t)
            phase_st=2*np.pi*(p['steer_start_frequency_hz']*t+.5*p['steer_frequency_slope_hz_per_s']*t*t)
            return np.array([p['ax_scale']*np.sin(phase_ax+ph[0]),
                             p['steer_scale']*np.sin(phase_st+ph[1])])
        if self.pattern=='prbs':
            k=self._switch(t)
            now=self.prbs_levels[k]
            prev=np.zeros(2) if k==0 else self.prbs_levels[k-1]
            fade=self._smooth((t-self.switch_times[k])/float(d['transition_s']))
            return (1.-fade)*prev+fade*now
        if self.pattern=='combined':
            values=[]
            for i,prefix in enumerate(('ax','steer')):
                f,a=self.combined[prefix]
                values.append(sum(float(amp)*np.sin(2*np.pi*float(freq)*t+ph[i*4+j])
                                  for j,(freq,amp) in enumerate(zip(f,a))))
            return np.clip(np.asarray(values,dtype=float),-float(d['clip']),float(d['clip']))
        raise ValueError(f'Unsupported excitation pattern: {self.pattern}')


def excitation_fingerprint(cfg, pattern, seed, episode_key=None):
    """Hash the precise scheduled 10 Hz reference sequence to prevent redundant seeds."""
    dt = float(cfg["controller_dt_s"])
    duration = float(cfg["duration_s"])
    generator = Excitation(cfg, pattern, seed, episode_key=episode_key)
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
    conditions = model_condition(model, cfg, run["speed_tier"])
    # Bind worker to the exact file signatures/labels chosen by the parent.
    # This prevents a mid-batch YAML edit or .sim/.dll replacement from
    # silently putting a different load under the original run ID.
    if "expected_condition" in run and run["expected_condition"] != conditions:
        raise RuntimeError(f"Run {run['id']} condition differs from parent request")
    if "expected_signature" in run and run["expected_signature"] != model_signature(model):
        raise RuntimeError(f"Run {run['id']} TruckSim model files differ from parent request")
    mode = "reference"
    excite = Excitation(cfg["excitation"], run["pattern"], run["seed"],
                        episode_key=run["id"])
    fingerprint = excitation_fingerprint(cfg["excitation"], run["pattern"], run["seed"],
                                          episode_key=run["id"])
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
                             "verify SV_VXS initialization; speed is not part of the .sim/.dll identity")
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
        # The stand-alone LTR probe saves data before closing its one-shot
        # session. Follow the same sequence here. This worker has exactly one
        # TruckSim simulation and will exit without reinitializing the DLL.
        path = out / (run["id"] + ".npz")
        # Write the NPZ archive to a temporary file, then publish atomically.
        npz_temp = path.with_name(path.name + ".writing")
        with npz_temp.open("wb") as npz_fh:
            np.savez_compressed(npz_fh,
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
            excitation_effective_seed=np.array(str(excite.effective_seed)),
            payload_kg=np.array(conditions["payload_kg"], dtype=float),
            road_mu=np.array(conditions["road_mu"], dtype=float))
            npz_fh.flush()
            os.fsync(npz_fh.fileno())
        os.replace(npz_temp, path)
        # CSV is human-readable; row i contains state at t[i], and native
        # command applied over [t[i], t[i+1]). Last row has no outgoing input.
        csv_path = out / (run["id"] + ".csv")
        csv_temp = csv_path.with_name(csv_path.name + ".writing")
        with csv_temp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", *SIGNALS,
                        "u_to_next_throttle_pct", "u_to_next_brake_mpa", "u_to_next_steer_wheel_deg",
                        "a_ref_to_next_mps2", "delta_ref_to_next_rad", "LTR_from_Fz"])
            for i in range(len(times)):
                u = native[i].tolist() if i < len(native) else [""]*3
                r = refs[i].tolist() if i < len(refs) else [""]*2
                w.writerow([times[i], *raw[i].tolist(), *u, *r, ltr[i]])
            f.flush()
            os.fsync(f.fileno())
        os.replace(csv_temp, csv_path)
        meta = {
            "run_id": run["id"], "model_id": model["id"],
            "pattern": run["pattern"], "seed": run["seed"],
            "excitation_sha256": fingerprint,
            "excitation_effective_seed": str(excite.effective_seed),
            "excitation_realized_params": excite.realized_params,
            "engine_map_sha256": expected_map_sha,
            "engine_map_source": map_origin,
            "excitation_config": cfg["excitation"],
            "mode": mode, "train_eligible": True,
            "native_shutdown_mode": "skip_terminate_one_shot_process",
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
            "notes": "Payload/friction select a shared .sim/.dll; per-run speed is set by SV_VXS BEFORE Initialize(). "
                     "Native actuation is generated ONLY via low_controller.py + inverse_dynamics.py; "
                     "direct actuation diagnostic is exclusive to Trucksim_Simulation_ltr.py.",
            "sample_alignment": "raw control[k] acts from raw_state[k] to raw_state[k+1]; similarly at 10Hz",
        }
        json_write(path.with_suffix(".json"), meta)
        audit = audit_npz(path)
        if not audit["valid"]:
            raise RuntimeError(f"Dataset audit failed: {audit['issues']}")
        # Write the completion record BEFORE optional plotting and DLL shutdown.
        # The parent independently audits all artifacts even if the worker
        # terminates abnormally after this point.
        result = {"status": "data_written", "run_id": run["id"], "npz": str(path),
                  "csv": str(csv_path), "duration_s": meta["duration_actual_s"],
                  "samples": len(times), "train_eligible": True,
                  "wall_time_s": round(time.monotonic()-t0_start,3),
                  "native_shutdown_mode": "skip_terminate_one_shot_process"}
        json_write(out / (run["id"] + ".result.json"), result)
        print(f"[WORKER] DATA COMPLETE {run['id']} {len(times)} raw / "
              f"{len(processed['t10_s'])} target samples", flush=True)
        # Plot is optional and cannot invalidate already-committed data.
        if bool(run.get("plot", False)):
            try:
                plot_all_exports(times, np.column_stack([raw, ltr]),
                                 out / (run["id"]+".png"),
                                 title=f"TruckSim TRACE | {run['id']} | {mode}")
            except Exception as exc:
                print(f"[WORKER] WARNING: plot failed: {exc}", flush=True)
        # Do not invoke vs_terminate_run, vs_terminate or vs_free_all. A worker
        # is single-use; the parent still validates data after its process exits.
        sim.stop(native_terminate=False)
        print(f"[WORKER] One-shot data saved: {run['id']}", flush=True)
        return result
    finally:
        if started and sim.is_running:
            # Error-path cleanup must not call the known-failing native API.
            # The parent accepts only fully verified artifacts, not exit codes.
            sim.stop(native_terminate=False)


def _decode_process_output(output):
    if output is None:
        return ""
    return output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)


def verify_run_artifacts(directory, job, cfg):
    """Check actual trajectory completeness independently of worker exit code.

    Only a complete set of parseable NPZ/CSV/metadata files with the exact
    requested settings can be accepted. PNG and worker result JSON are optional.
    """
    root = Path(directory)
    rid = job["id"]
    npz = root / (rid + ".npz")
    csv_path = root / (rid + ".csv")
    meta_path = root / (rid + ".json")
    issues = []
    names = (npz, csv_path, meta_path)
    for item in names:
        if not item.is_file() or item.stat().st_size == 0:
            issues.append(f"missing/empty {item.name}")
    if issues:
        return {"valid": False, "issues": issues, "files": [], "samples": 0}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = job["expected_condition"]
        if meta.get("run_id") != rid or meta.get("model_id") != job["model_id"]:
            issues.append("metadata run/model ID mismatch")
        if meta.get("pattern") != job["pattern"] or meta.get("seed") != job["seed"]:
            issues.append("metadata excitation pattern/seed mismatch")
        if meta.get("excitation_sha256") != job["excitation_sha256"]:
            issues.append("metadata excitation hash mismatch")
        expected_ex = Excitation(cfg["excitation"], job["pattern"], job["seed"], episode_key=rid)
        if meta.get("excitation_effective_seed") != str(expected_ex.effective_seed):
            issues.append("effective episode seed mismatch")
        if meta.get("excitation_realized_params") != expected_ex.realized_params:
            issues.append("realized excitation parameters mismatch")
        if meta.get("engine_map_sha256") != job["engine_map_sha256"]:
            issues.append("metadata engine calibration hash mismatch")
        for label in ("payload_tier", "friction_tier", "speed_tier"):
            if meta.get(label) != expected[label]:
                issues.append(f"metadata {label} mismatch")
        if meta.get("trucksim") != job["expected_signature"]:
            issues.append("TruckSim model .sim/.dll signature mismatch")
        if meta.get("model_units") != cfg["units"] or meta.get("mode") != "reference":
            issues.append("units/mode mismatch")
        if meta.get("train_eligible") is not True:
            issues.append("metadata does not mark reference training data")
        dt = float(meta["solver_dt_s"])
        target_dt = float(cfg["excitation"]["target_dt_s"])
        expected_duration = float(cfg["excitation"]["duration_s"])
        expected_n = int(np.floor(expected_duration/dt + 1e-7)) + 1
        expected_n10 = int(np.floor(expected_duration/target_dt+1e-7)) + 1
        if not np.isfinite(dt) or dt <= 0:
            issues.append("invalid solver step")
        if int(meta.get("raw_samples", -1)) != expected_n:
            issues.append("metadata raw sample count mismatch")
        if int(meta.get("target_samples", -1)) != expected_n10:
            issues.append("metadata 10Hz sample count mismatch")
        if not np.isclose(float(meta.get("duration_actual_s", -1)), expected_duration,
                          rtol=0, atol=max(dt, 1e-6)):
            issues.append("incomplete simulation duration")
        # np.load reads the real compressed archive. The archive test detects
        # truncated ZIP members / CRC errors, not just a nominal file size.
        with zipfile.ZipFile(npz, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                issues.append(f"NPZ CRC failed for {bad}")
        with np.load(npz, allow_pickle=False) as z:
            essential = ["t_raw_s", "trucksim_raw", "u_native_raw",
                         "u_ref_raw_si", "state_raw_si", "pose_raw_si",
                         "ltr_raw", "t10_s", "state10_si", "pose10_si",
                         "ltr10", "u_native10", "u_ref10_si", "train_eligible",
                         "model_id", "payload_tier", "friction_tier", "speed_tier",
                         "payload_kg", "road_mu", "vx_init_kmh", "excitation_sha256"]
            missing = [name for name in essential if name not in z.files]
            if missing:
                raise ValueError(f"NPZ missing keys: {missing}")
            tr = z["t_raw_s"]
            xr = z["trucksim_raw"]
            un = z["u_native_raw"]
            ur = z["u_ref_raw_si"]
            t10 = z["t10_s"]
            x10 = z["state10_si"]
            p10 = z["pose10_si"]
            u10 = z["u_native10"]
            r10 = z["u_ref10_si"]
            if tr.shape != (expected_n,) or xr.shape != (expected_n, 19):
                issues.append("raw state count/shape mismatch")
            if un.shape != (expected_n-1, 3) or ur.shape != (expected_n-1, 2):
                issues.append("raw control count/shape mismatch")
            if t10.shape != (expected_n10,) or x10.shape != (expected_n10, 5):
                issues.append("10Hz state count/shape mismatch")
            if p10.shape != (expected_n10, 3) or u10.shape != (expected_n10-1, 3) or r10.shape != (expected_n10-1, 2):
                issues.append("10Hz pose/control count/shape mismatch")
            if (z["state_raw_si"].shape != (expected_n, 5) or
                z["pose_raw_si"].shape != (expected_n, 3) or
                z["ltr_raw"].shape != (expected_n,) or
                z["ltr10"].shape != (expected_n10,)):
                issues.append("raw/10Hz auxiliary array shape mismatch")
            for array in (tr, xr, un, ur, t10, x10, p10, u10, r10, z["ltr_raw"], z["ltr10"]):
                if not np.isfinite(array).all():
                    issues.append("nonfinite data")
                    break
            if not issues:
                if not (np.all(np.diff(tr) > 0) and
                        np.allclose(np.diff(tr), dt, rtol=0, atol=max(1e-7, dt*1e-3))):
                    issues.append("raw simulation timeline is invalid")
                if not (np.all(np.diff(t10) > 0) and
                        np.allclose(np.diff(t10), target_dt, rtol=0, atol=1e-6)):
                    issues.append("10Hz timeline is invalid")
                if not np.isclose(tr[-1]-tr[0], expected_duration, atol=dt, rtol=0):
                    issues.append("truncated raw timeline")
                if not np.isclose(t10[-1]-t10[0], expected_duration, atol=target_dt/2, rtol=0):
                    issues.append("truncated 10Hz timeline")
                if not np.isclose(xr[0,2], expected["vx_init_kmh"], atol=float(cfg["trucksim"].get("initial_speed_tolerance_kmh",2)), rtol=0):
                    issues.append("initial speed does not match requested run speed tier")
            labels = {"model_id": job["model_id"], "payload_tier": expected["payload_tier"],
                      "friction_tier": expected["friction_tier"],
                      "speed_tier": expected["speed_tier"],
                      "excitation_sha256": job["excitation_sha256"]}
            for label, expected_value in labels.items():
                if str(z[label].item()) != str(expected_value):
                    issues.append(f"NPZ {label} mismatch")
            for label in ("payload_kg", "road_mu", "vx_init_kmh"):
                if float(z[label].item()) != float(expected[label]):
                    issues.append(f"NPZ {label} mismatch")
            if not bool(z["train_eligible"].item()):
                issues.append("NPZ not marked train eligible")
            samples = len(tr)
            # CSV must actually contain every raw output row, not only a header.
            with csv_path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                header = next(reader, None)
                if header is None or header[:20] != ["time_s", *SIGNALS]:
                    issues.append("raw CSV header mismatch")
                count = 0
                first_row = last_row = None
                for row in reader:
                    count += 1
                    if len(row) != 26:
                        issues.append(f"CSV row {count} has wrong column count")
                        break
                    if first_row is None:
                        first_row = row
                    last_row = row
                if count != expected_n:
                    issues.append("CSV sample count mismatch")
                if first_row is not None and last_row is not None and not issues:
                    for row, i in ((first_row, 0), (last_row, -1)):
                        actual = np.asarray([float(v) for v in row[:20]], dtype=float)
                        target = np.concatenate(([tr[i]], xr[i]))
                        if not np.allclose(actual, target, rtol=1e-7, atol=1e-7):
                            issues.append("CSV and NPZ raw data disagree")
                            break
        basic = audit_npz(npz)
        if not basic["valid"]:
            issues += basic["issues"]
        # Worker marker is optional: an interpreter may fail after all data
        # files are fully written but before writing the marker.
        result_path = root / (rid + ".result.json")
        if result_path.exists():
            info = json.loads(result_path.read_text(encoding="utf-8"))
            if info.get("run_id") != rid or info.get("train_eligible") is not True:
                issues.append("worker result marker mismatch")
        return {"valid": not issues, "issues": issues,
                "files": [p.name for p in names], "samples": samples}
    except Exception as exc:
        issues.append(f"verification exception: {type(exc).__name__}: {exc}")
        return {"valid": False, "issues": issues, "files": [], "samples": 0}


def promote_verified_artifacts(staging, output, job, allow_overwrite):
    """Expose verified files only after complete worker data validation."""
    staging, output = Path(staging), Path(output)
    rid = job["id"]
    published = []
    for ext in (".npz", ".csv", ".json", ".result.json", ".png"):
        src = staging / (rid + ext)
        if not src.is_file():
            continue
        dst = output / src.name
        if dst.exists() and not allow_overwrite:
            raise FileExistsError(f"Refusing overwrite: {dst}")
        os.replace(src, dst)
        published.append(dst.name)
    # Validated data are now in output; leave other crash artifacts in staging.
    try:
        staging.rmdir()
    except OSError:
        pass
    # The worker's result points at staging; update its file paths if present.
    result_file = output / (rid + ".result.json")
    if result_file.exists():
        result = json.loads(result_file.read_text(encoding="utf-8"))
        result["npz"] = str(output / (rid + ".npz"))
        result["csv"] = str(output / (rid + ".csv"))
        json_write(result_file, result)
    return published


def collect_batch(cfg, config_path, project_root, dry_run=False, model_id=None, speed_tier=None):
    """Spawn one isolated child per run; .sim/.dll only depend on payload/mu."""
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
                  "speed_tier=", run["speed_tier"], "vx_init_kmh=",
                  _tier_table(cfg["trucksim"], "speed_tiers_kmh")[run["speed_tier"]],
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
    batch_name = (f"{model_id}_{speed_tier}" if model_id is not None and speed_tier is not None
                  else model_id if model_id is not None
                  else f"speed_{speed_tier}" if speed_tier else "all")
    json_write(output / f"batch_config_{batch_name}.json", {
        "config": cfg, "model_signatures_before": signatures,
        "engine_map_sha256": engine_map_sha256,
        "engine_map_rows": len(batch_engine_map),
        "resolved_models": {mid: model_condition(m, cfg) for mid, m in models.items()},
        "runs": jobs})
    script = Path(project_root) / "scripts" / "run_koopman.py"
    if not script.exists():
        raise FileNotFoundError(script)
    manifest = []
    summary_path = output / f"manifest_{batch_name}.json"
    timeout = float(cfg["batch"].get("timeout_s_per_run", 120))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    # Helpful for diagnosing native crashes; acceptance never depends on logs.
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    allow_overwrite = bool(cfg["batch"].get("allow_overwrite", False))
    for i, initial_job in enumerate(jobs):
        job = dict(initial_job)
        for mid, m in models.items():
            if model_signature(m) != signatures[mid]:
                raise RuntimeError(f"TruckSim model {mid} changed mid-batch")
        job["expected_signature"] = signatures[job["model_id"]]
        job["expected_condition"] = model_condition(models[job["model_id"]], cfg, job["speed_tier"])
        job["engine_map_values"] = engine_map_values
        job["engine_map_sha256"] = engine_map_sha256
        job["plot"] = bool(cfg["batch"].get("plot_first_per_pattern", False)) and (
            int(job["seed"]) == int(cfg["batch"]["seeds"][0]))
        job_file = output / (job["id"] + ".request.json")
        json_write(job_file, job)
        # Older runs may have completed data even when their worker crashed
        # during interpreter shutdown. Adopt them only after FULL validation.
        existing = output / (job["id"] + ".npz")
        if existing.exists() and not allow_overwrite:
            checked = verify_run_artifacts(output, job, cfg)
            if checked["valid"]:
                details = {"id": job["id"], "model_id": job["model_id"],
                           "status": "ok", "source": "existing_verified",
                           "worker_returncode": None, "data_verified": True,
                           "artifacts": checked["files"], "samples": checked["samples"]}
                json_write(output / (job["id"] + ".accepted.json"), details)
                print(f"[RECOVER] {job['id']} existing dataset verified; no rerun", flush=True)
                manifest.append(details)
                json_write(summary_path, {"config": str(config_path), "runs": manifest})
                continue
            raise FileExistsError(
                f"Run {job['id']} exists but cannot be verified: {checked['issues']}. "
                "Set a new batch.output_dir or allow_overwrite=true to re-collect.")
        # Each child writes ONLY to an isolated staging directory. An incomplete
        # or crash-corrupted run can never appear in the final training folder.
        stage = output / ".staging" / (job["id"] + "__" + str(time.time_ns()))
        stage.mkdir(parents=True, exist_ok=False)
        cmd = [sys.executable, str(script), "worker", "--config",
               str(Path(config_path).resolve()), "--job", str(job_file),
               "--output", str(stage)]
        started = time.monotonic()
        print(f"[RUN {i+1}/{len(jobs)}] {job['id']} ({job['model_id']})", flush=True)
        return_code = None
        timed_out = False
        stdout = ""
        stderr = ""
        try:
            p = subprocess.run(cmd, cwd=str(project_root), env=env,
                               timeout=timeout, capture_output=True, text=True,
                               errors="replace")
            return_code = p.returncode
            stdout, stderr = p.stdout, p.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _decode_process_output(exc.stdout)
            stderr = _decode_process_output(exc.stderr)
        (output/(job["id"]+".stdout.log")).write_text(stdout, encoding="utf-8")
        (output/(job["id"]+".stderr.log")).write_text(stderr, encoding="utf-8")
        checked = verify_run_artifacts(stage, job, cfg)
        details = {"id": job["id"], "model_id": job["model_id"],
                   "worker_returncode": return_code, "worker_timed_out": timed_out,
                   "wall_time_s": round(time.monotonic()-started, 2),
                   "data_verified": checked["valid"],
                   "samples": checked.get("samples"),
                   "issues": checked["issues"]}
        if checked["valid"]:
            # Publish only after verifying the archived trajectory, raw samples,
            # CSV rows, metadata, labels, units, timing and request fingerprint.
            published = promote_verified_artifacts(stage, output, job, allow_overwrite)
            details["status"] = "ok"
            details["artifacts"] = published
            details["source"] = "worker_verified"
            details["native_exit_warning"] = (return_code != 0 or timed_out)
            json_write(output / (job["id"] + ".accepted.json"), details)
            if return_code == 0 and not timed_out:
                print(f"[OK] {job['id']} verified dataset; worker exit=0", flush=True)
            else:
                print(f"[DATA OK] {job['id']} verified and published; "
                      f"worker exit={return_code}, timeout={timed_out}. "
                      "Native shutdown issue recorded; continuing.", flush=True)
        else:
            details["status"] = "timeout" if timed_out else "invalid_or_incomplete"
            details["staging_dir"] = str(stage)
            print(f"[FAILED DATA] {job['id']} exit={return_code} "
                  f"issues={checked['issues'][:5]}; see staging/logs", flush=True)
            if stderr:
                print(stderr[-1200:], flush=True)
        manifest.append(details)
        json_write(summary_path, {"config": str(config_path), "runs": manifest})
        if details["status"] != "ok" and bool(cfg["batch"].get("stop_on_error", True)):
            raise RuntimeError(f"Incomplete or corrupt dataset: {job['id']}; see {output} logs")
    print(f"[DONE] {sum(x['status']=='ok' for x in manifest)}/{len(jobs)} "
          f"datasets verified; {summary_path}", flush=True)
    return manifest
