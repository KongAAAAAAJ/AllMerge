"""TRACE TruckSim 2016.1 native API memory guard (DIAGNOSTIC ONLY).

Purpose: detect whether the VS DLL writes beyond Python-provided import/export
buffers or configuration scalars. This is not an exit workaround and does not
change the production solver implementation or its DLL termination behavior.

Put this file in D:\\KONG_Files\\AllMerge\\all_merge\\scripts and run:

    python scripts\\trace_trucksim_native_guard.py --self-test
    python -X faulthandler scripts\\trace_trucksim_native_guard.py ^
        --config configs\\koopman.yaml ^
        --job outputs\\koopman\\data\\raw\\multi_model\\L0_MU08_V60__step_seed000.request.json ^
        --output outputs\\koopman\\data\\native_guard

Without a TruckSim Windows DLL this tool can run only the self-test.
"""
from __future__ import annotations

import argparse
import ctypes
import faulthandler
import gc
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

MAGIC_LEFT = 0x7F89AC1EF102D477
MAGIC_RIGHT = 0x70F4BEEE892AC15D
PAD_LEN = 16
_COUNTS = {"CopyExportVars": 0, "IntegrateIO": 0, "ReadConfiguration": 0}
PERSISTENT_BUFFERS = False


def log(s: str) -> None:
    print(f"[NATIVE-GUARD] {s}", flush=True)


def frame_for(ctype, n: int, name: str):
    if n < 1 or n > 65536:
        raise ValueError(f"Invalid {name} count: {n}")

    class Frame(ctypes.Structure):
        _pack_ = 1  # no padding between values and trailing canary
        _fields_ = [
            ("before", ctypes.c_uint64 * PAD_LEN),
            ("values", ctype * n),
            ("after", ctypes.c_uint64 * PAD_LEN),
        ]

    item = Frame()
    for i in range(PAD_LEN):
        item.before[i] = MAGIC_LEFT
        item.after[i] = MAGIC_RIGHT

    def check():
        if any(v != MAGIC_LEFT for v in item.before):
            raise RuntimeError(f"NATIVE WRITE BEFORE {name} buffer ({n} elements)")
        if any(v != MAGIC_RIGHT for v in item.after):
            raise RuntimeError(f"NATIVE WRITE PAST {name} buffer ({n} elements)")

    ptr = ctypes.cast(ctypes.byref(item, Frame.values.offset), ctypes.POINTER(ctype))
    return item, ptr, check


def guarded_ReadConfiguration(self, path):
    _COUNTS["ReadConfiguration"] += 1
    items = [
        frame_for(ctypes.c_int32, 1, "vs_read_configuration.n_import"),
        # Keep the current project's c_int64 assumption until verified
        # against the exact TruckSim 2016.1 SDK header.
        frame_for(ctypes.c_int64, 1, "vs_read_configuration.n_export"),
        frame_for(ctypes.c_double, 1, "vs_read_configuration.t_start"),
        frame_for(ctypes.c_double, 1, "vs_read_configuration.t_stop"),
        frame_for(ctypes.c_double, 1, "vs_read_configuration.t_step"),
    ]
    if PERSISTENT_BUFFERS:
        self._trace_cfg_frames = items
    self.dll_handle.vs_read_configuration(
        self.get_char_pointer(path), *(item[1] for item in items))
    for _, __, check in items:
        check()
    cfg = dict(n_import=int(items[0][0].values[0]),
               n_export=int(items[1][0].values[0]),
               t_start=float(items[2][0].values[0]),
               t_stop=float(items[3][0].values[0]),
               t_step=float(items[4][0].values[0]))
    log(f"ReadConfiguration: {cfg}; ctypes frame canaries intact")
    return cfg


def guarded_CopyExportVars(self, n_export):
    count = _COUNTS["CopyExportVars"] = _COUNTS["CopyExportVars"] + 1
    if PERSISTENT_BUFFERS:
        if not hasattr(self, "_trace_copy_export_frame"):
            self._trace_copy_export_frame = frame_for(ctypes.c_double, int(n_export), "vs_copy_export_vars.export")
        item, ptr, check = self._trace_copy_export_frame
        if len(item.values) != n_export:
            raise ValueError("Changing n_export mid-session is unsafe")
    else:
        item, ptr, check = frame_for(ctypes.c_double, int(n_export), "vs_copy_export_vars.export")
    self.dll_handle.vs_copy_export_vars(ptr)
    check()
    if count == 1:
        log(f"CopyExportVars OK: n_export={n_export}")
    return list(item.values)


def guarded_IntegrateIO(self, t, imports, exports):
    ni, no = len(imports), len(exports)
    if PERSISTENT_BUFFERS:
        if not hasattr(self, "_trace_io_frames"):
            self._trace_io_frames = (
                frame_for(ctypes.c_double, ni, "vs_integrate_io.import"),
                frame_for(ctypes.c_double, no, "vs_integrate_io.export"),
            )
        (before_i, iptr, icheck), (before_o, optr, ocheck) = self._trace_io_frames
        if len(before_i.values) != ni or len(before_o.values) != no:
            raise ValueError("Changing import/export dimensions mid-session is unsafe")
    else:
        before_i, iptr, icheck = frame_for(ctypes.c_double, ni, "vs_integrate_io.import")
        before_o, optr, ocheck = frame_for(ctypes.c_double, no, "vs_integrate_io.export")
    for i, value in enumerate(imports):
        before_i.values[i] = float(value)
    for i, value in enumerate(exports):
        before_o.values[i] = float(value)
    k = _COUNTS["IntegrateIO"] = _COUNTS["IntegrateIO"] + 1
    if k in (1, 10, 100, 1000, 2000, 6000):
        log(f"Entering IntegrateIO #{k}: t={t:.5f}, ni={ni}, ne={no}")
    result = self.dll_handle.vs_integrate_io(ctypes.c_double(t), iptr, optr)
    icheck()
    ocheck()
    if k in (1, 10, 100, 1000, 2000, 6000):
        log(f"IntegrateIO #{k}: returned={result}, ctypes frame canaries intact")
    return result, list(before_o.values)


def self_test():
    for ctype in (ctypes.c_double, ctypes.c_int32, ctypes.c_int64):
        fr, ptr, check = frame_for(ctype, 3, "self-test")
        ptr[0], ptr[1], ptr[2] = 1, 2, 3
        check()
        # Deliberately overwrite the first trailing canary, staying within
        # our own allocated Frame; the checker must detect it.
        ptr[3] = 99
        try:
            check()
        except RuntimeError as exc:
            assert "PAST" in str(exc)
        else:
            raise AssertionError("Buffer overflow was not detected")
    log("SELF-TEST PASS: in-bounds writes accepted; 1-element overruns detected")


def main():
    p = argparse.ArgumentParser(description="Diagnostic canaries around TruckSim VS API buffers")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--config", default=str(PROJECT / "configs" / "koopman.yaml"))
    p.add_argument("--job")
    p.add_argument("--output", default=str(PROJECT / "outputs" / "koopman" / "data" / "native_guard"))
    p.add_argument("--no-plot", action="store_true", help="A/B isolation: disable post-run matplotlib for this diagnostic")
    p.add_argument("--duration-s", type=float, help="Diagnostic duration override; reference fingerprint is recomputed")
    p.add_argument("--persistent-buffers", action="store_true", help="Reuse pinned C input/output buffers for entire VS session")
    p.add_argument("--native-terminate", action="store_true", help="EXPERIMENTAL: call native vs_terminate_run after successful integration")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return 0
    if os.name != "nt":
        p.error("Real TruckSim SDK diagnosis must run on Windows")
    if not args.job:
        p.error("--job must point to an existing worker .request.json")
    faulthandler.enable(all_threads=True)
    global PERSISTENT_BUFFERS
    PERSISTENT_BUFFERS = bool(args.persistent_buffers)
    from highway_env.vehicle.trucksim_simulation import Simulation
    if args.native_terminate:
        from highway_env.vehicle.trucksim_simulation.Trucksim_Simulation_ltr import TrucksimSimulationLTR
        original_stop = TrucksimSimulationLTR.stop
        def force_native_terminate(self, t_current=None, *, native_terminate=True):
            log("Calling real vs_terminate_run; if this crashes, inspect exact SDK ABI")
            return original_stop(self, t_current, native_terminate=True)
        TrucksimSimulationLTR.stop = force_native_terminate
    # Monkeypatch only this diagnostic process, not the production files.
    cls = Simulation.VehicleSimulation
    cls.ReadConfiguration = guarded_ReadConfiguration
    cls.CopyExportVars = guarded_CopyExportVars
    cls.IntegrateIO = guarded_IntegrateIO
    from highway_env.vehicle.koopman.collect import read_yaml, run_worker, excitation_fingerprint
    cfg = read_yaml(args.config)
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    if args.duration_s is not None:
        if not (args.duration_s > 0 and args.duration_s <= 120):
            p.error("--duration-s must be in (0, 120]")
        cfg["excitation"]["duration_s"] = float(args.duration_s)
        job["excitation_sha256"] = excitation_fingerprint(
            cfg["excitation"], job["pattern"], job["seed"])
        log(f"Diagnostic duration override: {args.duration_s:g}s")
    if args.no_plot:
        job = dict(job, plot=False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log(f"process pid={os.getpid()} model={job.get('model_id')} job={job.get('id')}")
    log(f"Guarded VS API buffers active; persistent={PERSISTENT_BUFFERS}; native_terminate={args.native_terminate}; NO os._exit; NO production files changed")
    started = time.monotonic()
    result = run_worker(cfg, job, out)
    log(f"Worker returned: {result}; API counts={_COUNTS}; wall_s={time.monotonic()-started:.2f}")
    log("Executing explicit Python garbage collection to reproduce shutdown failure")
    gc.collect()
    log("GC completed; proceeding with normal Python interpreter exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
