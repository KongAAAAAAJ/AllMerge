"""Canonical 19-signal TruckSim IO map and SI units for TRACE.

Raw commands u_native = [throttle_pct, brake_MPa, steering_wheel_deg].
State at t[k] and commanded transition u[k] for [t[k], t[k+1]) are
stored separately to prevent one-step label leakage/misalignment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SIGNALS = (
    "ax", "ay", "vx", "vy", "Xo", "Yo", "Roll", "AVx", "Pitch",
    "AVy", "Yaw", "AVz", "AV_Eng", "Steer_L1", "Steer_R1",
    "Fz_L1", "Fz_L2", "Fz_R1", "Fz_R2",
)
UNITS_RAW = {
    "ax": "g", "ay": "g", "vx": "km/h", "vy": "km/h",
    "Xo": "configurable", "Yo": "configurable", "Roll": "deg", "AVx": "deg/s",
    "Pitch": "deg", "AVy": "deg/s", "Yaw": "deg", "AVz": "deg/s",
    "AV_Eng": "rpm", "Steer_L1": "deg", "Steer_R1": "deg",
    "Fz_L1": "N", "Fz_L2": "N", "Fz_R1": "N", "Fz_R2": "N",
}
G0 = 9.80665


def _as_2d(raw):
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(SIGNALS):
        raise ValueError(f"Expected raw TruckSim outputs [T,19], received {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Nonfinite TruckSim export detected")
    return arr


def ltr_from_fz(raw, sign=1.0):
    """Whole vehicle (right minus left); valid only for positive compression.

    Four outputs must include all relevant vertical supports of this model.
    """
    a = _as_2d(raw)
    if sign not in (-1, 1):
        raise ValueError("Wheel load compression sign must be +1 or -1")
    fz = a[:, 15:19] * sign
    left, right = fz[:, 0] + fz[:, 1], fz[:, 2] + fz[:, 3]
    denom = left + right
    valid = (fz >= 0).all(axis=-1) & (denom > 1e-6)
    val = np.full(len(a), np.nan, dtype=np.float64)
    val[valid] = (right[valid] - left[valid]) / denom[valid]
    return val


def to_si(raw, units):
    """Convert raw responses without changing their original on-disk copy.

    State s=[vx,vy,yaw_rate,roll,roll_rate] in m/s,m/s,rad/s,rad,rad/s.
    Roll rate is Euler angle derivative, not raw body AVx, and is computed
    from p,q,r and roll,pitch. Pose=[X,Y,yaw] in m,m,rad.
    """
    a = _as_2d(raw)
    vel = units.get("velocity", "km/h")
    pos = units.get("position", "m")
    engine_unit = units.get("engine_speed", "rpm")
    if vel not in ("km/h", "m/s"):
        raise ValueError("units.velocity must be km/h or m/s")
    if pos != "m":
        raise ValueError("Only position unit m is supported, confirm TruckSim exports")
    if engine_unit not in ("deg/s", "rpm"):
        raise ValueError("engine_speed must be deg/s or rpm")
    scale = 1 / 3.6 if vel == "km/h" else 1.0
    roll, pitch = np.deg2rad(a[:, 6]), np.deg2rad(a[:, 8])
    p, q, r = np.deg2rad(a[:, 7]), np.deg2rad(a[:, 9]), np.deg2rad(a[:, 11])
    cos_pitch = np.cos(pitch)
    if np.any(np.abs(cos_pitch) < 1e-3):
        raise ValueError("Euler roll/yaw conversion near pitch singularity")
    roll_rate = p + (q * np.sin(roll) + r * np.cos(roll)) * np.tan(pitch)
    yaw_rate_euler = (q * np.sin(roll) + r * np.cos(roll)) / cos_pitch
    # For chassis dynamics r is the body-frame yaw angular velocity AVz.
    state = np.column_stack([a[:, 2] * scale, a[:, 3] * scale, r, roll, roll_rate])
    pose = np.column_stack([a[:, 4], a[:, 5], np.unwrap(np.deg2rad(a[:, 10]))])
    extras = {
        "ax_mps2": a[:, 0] * G0,
        "ay_mps2": a[:, 1] * G0,
        "pitch_rad": pitch,
        "body_p_radps": p,
        "body_q_radps": q,
        "yaw_euler_rate_radps": yaw_rate_euler,
        "engine_rpm": a[:, 12] / 6.0 if engine_unit == "deg/s" else a[:, 12],
        "delta_front_rad": np.deg2rad(0.5 * (a[:, 13] + a[:, 14])),
        "wheel_fz_n": a[:, 15:19].copy(),
    }
    return state, pose, extras


def _interp_matrix(t_src, x, t_dst):
    # Yaw is unwrapped before this step; all columns are ordinary real-valued.
    return np.column_stack([np.interp(t_dst, t_src, x[:, j]) for j in range(x.shape[1])])


def resample(t_raw, state, pose, ltr, u_native, u_ref, dt=0.1):
    """Resample states linearly and commands by zero-order hold.

    t_raw shape [N+1], u_native and u_ref [N,?].
    Returned t10 [M+1], u10 [M,?], where command[k] acts over [t10[k],t10[k+1]).
    """
    t_raw = np.asarray(t_raw, dtype=float)
    if len(t_raw) < 2 or np.any(np.diff(t_raw) <= 0):
        raise ValueError("Sample times must be strictly increasing")
    if len(u_native) != len(t_raw)-1 or len(u_ref) != len(t_raw)-1:
        raise ValueError("Transition control lengths must equal state samples minus one")
    n = int(np.floor((t_raw[-1] - t_raw[0]) / dt + 1e-7))
    if n < 1:
        raise ValueError("Trajectory too short for target sample interval")
    t = t_raw[0] + np.arange(n+1, dtype=float) * dt
    # floating point integration tolerances
    t = np.minimum(t, t_raw[-1])
    idx = np.clip(np.searchsorted(t_raw, t[:-1], side="right") - 1,
                  0, len(u_native)-1)
    return {
        "t10_s": t,
        "state10_si": _interp_matrix(t_raw, state, t),
        "pose10_si": _interp_matrix(t_raw, pose, t),
        "ltr10": np.interp(t, t_raw, ltr) if np.isfinite(ltr).all() else np.full(len(t), np.nan),
        "u_native10": np.asarray(u_native)[idx],
        "u_ref10_si": np.asarray(u_ref)[idx],
    }


def audit_npz(path):
    with np.load(path, allow_pickle=False) as z:
        t = z["t10_s"]
        s = z["state10_si"]
        p = z["pose10_si"]
        u = z["u_native10"]
        eligible = bool(z["train_eligible"].item())
        issues = []
        if s.shape != (len(t),5) or p.shape != (len(t),3):
            issues.append("state/pose shape mismatch")
        if u.shape != (len(t)-1,3):
            issues.append("native controls not aligned")
        if not (np.isfinite(s).all() and np.isfinite(p).all() and np.isfinite(u).all()):
            issues.append("nonfinite state/pose/control")
        if np.any(np.diff(t) <= 0):
            issues.append("nonmonotonic time")
        if eligible:
            r = z["u_ref10_si"]
            if r.shape != (len(t)-1,2) or not np.isfinite(r).all():
                issues.append("invalid reference controls for train eligible trajectory")
        ltr = z["ltr10"]
        if np.any(np.isfinite(ltr) & (np.abs(ltr) > 1.05)):
            issues.append("LTR out of expected range")
        return {"path": str(path), "valid": not issues, "issues": issues,
                "steps": len(t)-1, "train_eligible": eligible}
