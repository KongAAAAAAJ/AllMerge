"""TRACE TruckSim lateral/longitudinal I/O probe.

Use ``python -m highway_env.vehicle.trucksim_simulation.Trucksim_Simulation_ltr``
from the project root. Edit the parameters in ``main()`` only. The legacy
``Truck_Simulation.py`` is not modified or imported by this module.

The plot contains every raw exported channel, one subplot per channel. Channel
names and units MUST be verified against the current TruckSim export list.

This version works around a reported crash in vs_terminate_run when running
as a stand-alone, single-use probe: it saves data before shutdown, and skips
native termination by default. This is NOT a validated fix for the underlying
VehicleSim DLL/SDK incompatibility and is not suitable for persistent sessions.
"""

import csv
import ctypes
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__:
    from . import Simulation
else:
    import Simulation  # Supports direct python path\to\Trucksim_Simulation_ltr.py


# Fixed Export Variables layout of the new longitudinal/lateral TruckSim model.
# Names reflect the supplied ordering; numbers are in native TruckSim units.
# Do not assume deg/rad, N/kN or m/s/km/h without checking the .sim configuration.
OUTPUT_SIGNALS = (
    'ax', 'ay', 'vx', 'vy', 'Xo', 'Yo', 'Roll', 'AVx',
    'Pitch', 'AVy', 'Yaw', 'AVz', 'AV_Eng', 'Steer_L1',
    'Steer_R1', 'Fz_L1', 'Fz_L2', 'Fz_R1', 'Fz_R2',
)
N_INPUTS = 3
N_OUTPUTS = len(OUTPUT_SIGNALS)
LTR_FIELD = 'LTR_from_Fz'  # Derived diagnostic; NOT a TruckSim Export Variable.


def ltr_from_wheel_loads(exports, compression_sign=1.0):
    """Compute whole-vehicle load-transfer ratio from the four supplied Fz.

    LTR_(right-left) = (Fz_R1 + Fz_R2 - Fz_L1 - Fz_L2) / sum(Fz).
    All four loads must share the same units and represent compressive magnitudes.
    Use compression_sign=-1 if TruckSim exports negative compressive forces.
    Invalid samples are NaN (never silently clipped to plausible values).
    """
    exports = np.asarray(exports, dtype=float)
    if exports.ndim != 2 or exports.shape[1] != N_OUTPUTS:
        raise ValueError(f'LTR requires [T, {N_OUTPUTS}] raw exports')
    if compression_sign not in (-1.0, 1.0):
        raise ValueError('compression_sign must be 1.0 or -1.0')
    fz = exports[:, 15:19] * compression_sign
    left = fz[:, 0] + fz[:, 1]
    right = fz[:, 2] + fz[:, 3]
    total = left + right
    valid = np.isfinite(fz).all(axis=1) & (fz >= 0).all(axis=1) & (total > 1e-9)
    ltr = np.full(exports.shape[0], np.nan, dtype=float)
    ltr[valid] = (right[valid] - left[valid]) / total[valid]
    return ltr


class TrucksimSimulationLTR:
    """Raw TruckSim I/O bridge, independent of the legacy export layout."""

    def __init__(self, simfile_path, dll_path):
        self.simfile_path = os.path.abspath(simfile_path)
        self.dll_path = os.path.abspath(dll_path)
        self.vs = Simulation.VehicleSimulation()
        self.configuration = None
        self.current_time = 0.0
        self.is_running = False
        self._dll_ready = False
        self._has_integrated = False
        self._solver_faulted = False
        self._abandoned_native_session = False

    def _ensure_dll(self):
        if self._dll_ready:
            return
        if os.name != 'nt':
            raise OSError('TruckSim DLL interface requires Windows')
        if not os.path.isfile(self.dll_path):
            raise FileNotFoundError(self.dll_path)
        if not os.path.isfile(self.simfile_path):
            raise FileNotFoundError(self.simfile_path)
        handle = ctypes.WinDLL(self.dll_path)
        if not self.vs.get_api(handle):
            raise RuntimeError('TruckSim DLL does not expose the required VS API')
        self._dll_ready = True

    def _check_error(self):
        if self.vs.dll_handle.vs_error_occurred():
            getter = getattr(self.vs.dll_handle, 'vs_get_error_message', None)
            message = getter() if getter else b'unknown solver error'
            raise RuntimeError(f'TruckSim solver: {message!r}')

    def start(self, config=None):
        if self._abandoned_native_session:
            raise RuntimeError('This TruckSim session was released without native termination; '
                               'start a fresh Python process for another run')
        if self.is_running:
            self.stop()
        self._ensure_dll()
        self.configuration = self.vs.ReadConfiguration(self.simfile_path)
        self._check_error()
        t0 = self.vs.dll_handle.vs_setdef_and_read(
            self.vs.get_char_pointer(self.simfile_path), None, None)
        self._check_error()
        if config and config.get('vx_init') is not None:
            vx = float(config['vx_init'])
            if not np.isfinite(vx):
                raise ValueError('vx_init must be finite, in km/h')
            if self.vs.Statement('SV_VXS', f' {vx}') != 0:
                raise RuntimeError('TruckSim rejected SV_VXS')
            self._check_error()
        self.current_time = float(t0)
        self._has_integrated = False
        self._solver_faulted = False
        self.is_running = True
        try:
            self.vs.Initialize(t0)
            self._check_error()
        except Exception:
            self.stop()
            raise

    def reset(self, config=None):
        self.start(config)

    def get_export_array(self):
        if not self.is_running:
            raise RuntimeError('Start the TruckSim solver first')
        return np.asarray(
            self.vs.CopyExportVars(self.configuration['n_export']), dtype=float)

    def get_time_step(self):
        if self.configuration is None:
            raise RuntimeError('Start the TruckSim solver first')
        return float(self.configuration['t_step'])

    def run(self, t_current, import_array, export_array):
        if not self.is_running:
            raise RuntimeError('TruckSim solver is not running')
        imports = np.asarray(import_array, dtype=float)
        exports = np.asarray(export_array, dtype=float)
        if imports.shape != (self.configuration['n_import'],):
            raise ValueError(
                f'Expected {self.configuration["n_import"]} imports, got {imports.shape}')
        if exports.shape != (self.configuration['n_export'],):
            raise ValueError(
                f'Expected {self.configuration["n_export"]} exports, got {exports.shape}')
        if not (np.isfinite(t_current) and t_current > self.current_time):
            raise ValueError('Simulation time must be finite and strictly increasing')
        if not (np.isfinite(imports).all() and np.isfinite(exports).all()):
            raise ValueError('TruckSim I/O must be finite')
        try:
            status, result = self.vs.IntegrateIO(float(t_current), imports, exports)
            self._check_error()
        except Exception:
            # The native solver may be partially initialized after a failed call.
            # Do not enter another native API call from the cleanup path.
            self._solver_faulted = True
            raise
        self.current_time = float(t_current)
        self._has_integrated = True
        return status, np.asarray(result, dtype=float)

    def stop(self, t_current=None, *, native_terminate=True):
        """End this Python wrapper's session.

        For the ONE-SHOT standalone probe, call native_terminate=False:
        some user DLLs crash inside vs_terminate_run even after integration.
        Skipping it is a diagnostic workaround ONLY and makes this instance
        non-reusable. For persistent/multi-episode execution, native shutdown
        must be fixed and validated against the exact TruckSim SDK/DLL.
        """
        if not self.is_running:
            return
        should_terminate = native_terminate and self._has_integrated and not self._solver_faulted
        self.is_running = False
        if should_terminate:
            try:
                self.vs.TerminateRun(
                    self.current_time if t_current is None else t_current)
            except BaseException:
                self._abandoned_native_session = True
                raise
        else:
            self._abandoned_native_session = True
            print('[TruckSim LTR] Native vs_terminate_run SKIPPED for this one-shot probe. '
                  'This DLL session must not be reused in the current Python process.')


def plot_all_exports(times, exports, image_path, signal_names=None,
                     title='TruckSim raw output channels', save_pdf=False):
    """Plot ALL TruckSim exports on ONE figure and save without a GUI.

    Export values are plotted in their native units; no index-based unit
    conversion is performed because models can use different export mappings.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    times = np.asarray(times, dtype=float)
    exports = np.asarray(exports, dtype=float)
    if times.ndim != 1 or exports.ndim != 2 or exports.shape[0] != times.size:
        raise ValueError('Plot expects time [T] and exports [T, n_export]')
    n = exports.shape[1]
    if n < 1:
        raise ValueError('TruckSim exported no signals; configure Export Variables')

    labels = dict(enumerate(OUTPUT_SIGNALS))
    if n == N_OUTPUTS + 1:
        labels[N_OUTPUTS] = LTR_FIELD + ' (derived)'
    if signal_names:
        labels.update(signal_names)
    columns = 2
    rows = math.ceil(n / columns)
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(15, 2.6 * rows), sharex=True)
    for index, ax in enumerate(axes.flat):
        if index >= n:
            ax.set_visible(False)
            continue
        name = str(labels.get(index, f'export_{index}'))
        ax.plot(times, exports[:, index], linewidth=1.05)
        ax.set_title(f'{index:02d} | {name}', fontsize=10)
        ax.grid(True, alpha=0.28)
        ax.tick_params(labelsize=8)
        ax.set_ylabel('Raw value')
        if index // columns == rows - 1 or index + columns >= n:
            ax.set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=160, bbox_inches='tight')
    print(f'[TruckSim LTR] figure: {image_path.resolve()}')
    if save_pdf:
        pdf_path = image_path.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f'[TruckSim LTR] vector figure: {pdf_path.resolve()}')
    plt.close(fig)


def probe(args):
    """Run a fixed-input I/O test; return generated artifact paths."""
    sim = TrucksimSimulationLTR(args.simfile, args.dll)
    u = np.asarray([args.throttle, args.brake, args.steer_wheel_deg], dtype=float)
    if not np.isfinite(u).all():
        raise ValueError('Throttle, brake and steering inputs must be finite')
    duration = float(args.seconds)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Simulation duration must be a positive finite number')

    sim.start({'vx_init': args.vx_init_kmh}
              if args.vx_init_kmh is not None else None)
    try:
        config = sim.configuration
        n_import = int(config['n_import'])
        n_export = int(config['n_export'])
        print(f'[TruckSim LTR] imports={n_import}, exports={n_export}, '
              f'dt={config["t_step"]}, t_start={config["t_start"]}, '
              f't_stop={config["t_stop"]}')
        if n_import != N_INPUTS:
            raise ValueError(
                'This probe assumes 3 Import Variables ordered '
                '[throttle, brake, steering-wheel angle]. '
                f'TruckSim reported {n_import}; verify your .sim Import Variables.')
        if n_export != N_OUTPUTS:
            raise ValueError(
                f'Expected {N_OUTPUTS} exports in the specified order '
                f'{list(OUTPUT_SIGNALS)}; TruckSim reported {n_export}. '
                'Check the Export Variables list in the active .sim model.')
        export = sim.get_export_array()
        if export.shape != (n_export,):
            raise ValueError(f'Expected {n_export} exports, got {export.shape}')
        dt = sim.get_time_step()
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError(f'Invalid TruckSim time step {dt}')
        t0 = sim.current_time
        t_stop = float(config.get('t_stop', -1.0))
        if not np.isfinite(t_stop):
            raise ValueError('TruckSim t_stop must be finite')
        # In VS external-program mode, TSTOP=-1 means no fixed end time.
        # A Python-controlled run then uses the requested duration.
        if t_stop < 0.0:
            actual_duration = duration
            print(f'[TruckSim LTR] TSTOP={t_stop:g}: external-run mode; '
                  f'using requested duration={duration:g}s')
        else:
            actual_duration = min(duration, max(0.0, t_stop - t0))
        nsteps = max(0, int(np.floor(actual_duration / dt + 1e-8)))
        if nsteps == 0:
            raise ValueError('The requested duration has no solver steps before t_stop')

        times = [float(t0)]
        states = [export.copy()]
        for i in range(nsteps):
            t_next = float(t0 + (i + 1) * dt)
            status, export = sim.run(t_next, u, export)
            if export.shape != (n_export,):
                raise ValueError('TruckSim changed its export dimension during the run')
            times.append(sim.current_time)
            states.append(export.copy())
            if status:
                print(f'[TruckSim LTR] solver stopped with status={status} at t={sim.current_time:.4f}s')
                break
    except BaseException:
        # Never mask the original integration error with a second native error.
        sim.stop(native_terminate=False)
        raise

    print(f'[TruckSim LTR] Integration finished: {len(times) - 1} steps, '
          f'last_time={times[-1]:.4f}s. Saving data BEFORE DLL shutdown.')
    raw_exports = np.stack(states, axis=0)  # [T, 19], unchanged TruckSim units
    derive_ltr = bool(getattr(args, 'compute_ltr_from_fz', False))
    if derive_ltr:
        sign = float(getattr(args, 'fz_compression_sign', 1.0))
        ltr = ltr_from_wheel_loads(raw_exports, compression_sign=sign)
        plot_exports = np.column_stack([raw_exports, ltr])
        csv_fields = [*OUTPUT_SIGNALS, LTR_FIELD]
        invalid = int(np.count_nonzero(~np.isfinite(ltr)))
        if invalid:
            print(f'[TruckSim LTR] WARNING: {invalid} invalid derived LTR samples; '
                  'verify wheel-load sign, units and contact state.')
    else:
        plot_exports = raw_exports
        csv_fields = list(OUTPUT_SIGNALS)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['time_s', *csv_fields])
        writer.writerows([t, *row.tolist()] for t, row in zip(times, plot_exports))
    print(f'[TruckSim LTR] raw data: {csv_path.resolve()} '
          f'({len(times)} samples, {n_export} TruckSim channels'
          + (', 1 derived LTR channel)' if derive_ltr else ')'))

    meta_path = csv_path.with_suffix('.json')
    metadata = {
        'simfile': str(Path(args.simfile).resolve()),
        'dll': str(Path(args.dll).resolve()),
        'vx_init_kmh': args.vx_init_kmh,
        'constant_command': {
            'throttle': args.throttle,
            'brake': args.brake,
            'steer_wheel_deg': args.steer_wheel_deg,
        },
        'manual_case_label': getattr(args, 'case_label', ''),
        'notes': 'Payload and friction coefficient are set in TruckSim, '
                 'not via this script. Exports retain raw TruckSim units.',
        'n_import': n_import,
        'n_export': n_export,
        'export_signal_names': list(OUTPUT_SIGNALS),
        'export_order_note': 'Signal names are user-specified and must be confirmed '
                             'against actual TruckSim Export Variables.',
        'derived_ltr': {
            'enabled': derive_ltr,
            'name': LTR_FIELD,
            'sign': 'right minus left',
            'fz_compression_sign': float(getattr(args, 'fz_compression_sign', 1.0)),
            'assumptions': 'Four wheel loads cover the vehicle, same units, positive compression',
        },
        't_step': dt,
        'num_samples': len(times),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                         encoding='utf-8')
    print(f'[TruckSim LTR] metadata: {meta_path.resolve()}')

    image_path = Path(getattr(args, 'figure', None) or csv_path.with_suffix('.png'))
    try:
        plot_all_exports(times, plot_exports, image_path,
                         signal_names=getattr(args, 'signal_names', None),
                         title=f'TruckSim LTR I/O probe | '
                               f'{getattr(args, "case_label", "")}',
                         save_pdf=getattr(args, 'save_pdf', False))
    finally:
        # CLI runs one DLL simulation per process. Native termination is off by
        # default because the user's TruckSim DLL crashed at that exact call.
        # Do not use this workaround for long-lived or multi-episode execution.
        sim.stop(native_terminate=bool(getattr(args, 'native_terminate', False)))
    return {'csv': csv_path, 'png': image_path, 'metadata': meta_path}


def main():
    # ================== Only edit these parameters ==================
    # Use paths to the NEW combined longitudinal/lateral TruckSim model.
    # The examples below match the old Truck_Simulation.py project layout.
    # simfile = r'D:\Users\Public\Documents\TruckSim2016.1_Data_now_using\truck_1_lon_lat.sim'
    # dll = r'D:\Users\Public\Documents\TruckSim2016.1_Data_now_using\Extensions\Multi_vehicle_lon_lat\s_s_1.dll'
    simfile = r'D:\Users\Public\Documents\TruckSim2016.1_Data_now_using\L0_MU08_V60.sim'
    dll = r'D:\Users\Public\Documents\TruckSim2016.1_Data_now_using\Extensions\Multi_vehicle_lon_lat\L0_MU08_V60.dll'

    vx_init_kmh = 60.0         # TruckSim SV_VXS, km/h; None = use .sim initial speed
    throttle = 0.0             # Expected TruckSim import 0
    brake = 0.0                # Expected TruckSim import 1, model-specific units
    steer_wheel_deg = 60.0      # Expected TruckSim import 2, steering-wheel degrees
    seconds = 5.0              # Duration of the constant-input test

    case_label = 'manual_TruckSim_case'  # Label only; DOES NOT change payload or mu
    csv_path = r'outputs\koopman\trucksim_ltr_probe.csv'
    figure_path = r'outputs\koopman\trucksim_ltr_probe.png'
    save_pdf = False           # True: also save a vector PDF of the SAME figure
    compute_ltr_from_fz = False  # Enable only AFTER confirming wheel-load sign/units
    native_terminate = False     # IMPORTANT: one-shot probe only; avoid DLL shutdown crash
    fz_compression_sign = 1.0    # Set -1.0 if compression is negative in TruckSim

    # Output names default to the 19-signal user-specified order above.
    # Customize plot labels here if useful (e.g. add verified physical units).
    signal_names = {
        # 0: 'longitudinal acceleration [g]',
        # 1: 'lateral acceleration [g]',
        # 2: 'longitudinal velocity [km/h]',
        # ... replace these ONLY using the current TruckSim Export Variables ...
    }
    # ================== End editable parameters =====================

    args = SimpleNamespace(
        simfile=simfile, dll=dll, vx_init_kmh=vx_init_kmh,
        throttle=throttle, brake=brake, steer_wheel_deg=steer_wheel_deg,
        seconds=seconds, csv=csv_path, figure=figure_path,
        case_label=case_label, signal_names=signal_names, save_pdf=save_pdf,
        compute_ltr_from_fz=compute_ltr_from_fz,
        fz_compression_sign=fz_compression_sign,
        native_terminate=native_terminate)
    probe(args)


if __name__ == '__main__':
    main()
