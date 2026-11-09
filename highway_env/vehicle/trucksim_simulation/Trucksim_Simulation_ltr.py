"""Independent lateral/longitudinal TruckSim DLL bridge and I/O probe.

This wrapper intentionally does NOT import Truck_Simulation.py: its output layout
belongs to the legacy model. It shares only the raw, layout-agnostic Simulation API.

Run as a module on Windows, e.g.
python -m highway_env.vehicle.trucksim_simulation.Trucksim_Simulation_ltr \
    --simfile path/to/new.sim --dll path/to/new.dll --seconds 2 --csv io_probe.csv
"""

import argparse
import csv
import ctypes
import os
from pathlib import Path

import numpy as np

from . import Simulation


class TrucksimSimulationLTR:
    """Raw numerical TruckSim interface; no assumptions about export order."""

    def __init__(self, simfile_path, dll_path):
        self.simfile_path = os.path.abspath(simfile_path)
        self.dll_path = os.path.abspath(dll_path)
        self.vs = Simulation.VehicleSimulation()
        self.configuration = None
        self.current_time = 0.0
        self.is_running = False
        self._dll_ready = False

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
        return np.asarray(self.vs.CopyExportVars(self.configuration['n_export']), dtype=float)

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
            raise ValueError(f'Expected {self.configuration["n_import"]} imports, got {imports.shape}')
        if exports.shape != (self.configuration['n_export'],):
            raise ValueError(f'Expected {self.configuration["n_export"]} exports, got {exports.shape}')
        if not (np.isfinite(t_current) and t_current > self.current_time):
            raise ValueError('Simulation time must be finite and strictly increasing')
        if not (np.isfinite(imports).all() and np.isfinite(exports).all()):
            raise ValueError('TruckSim I/O must be finite')
        status, result = self.vs.IntegrateIO(float(t_current), imports, exports)
        self.current_time = float(t_current)
        self._check_error()
        return status, np.asarray(result, dtype=float)

    def stop(self, t_current=None):
        if self.is_running:
            try:
                self.vs.TerminateRun(self.current_time if t_current is None else t_current)
            finally:
                self.is_running = False


def probe(args):
    """Log raw exports by index; signal identification belongs to the local .sim dataset."""
    sim = TrucksimSimulationLTR(args.simfile, args.dll)
    sim.start({'vx_init': args.vx_init_kmh} if args.vx_init_kmh is not None else None)
    try:
        config = sim.configuration
        print(f'[TruckSim LTR probe] n_import={config["n_import"]} '
              f'n_export={config["n_export"]} t_step={config["t_step"]} '
              f't_start={config["t_start"]} t_stop={config["t_stop"]}')
        if config['n_import'] != 3:
            raise ValueError('Probe assumes three inputs ordered throttle, brake, steering; '
                             'inspect TruckSim procedure mapping first')
        u = np.array([args.throttle, args.brake, args.steer_wheel_deg], dtype=float)
        if not np.isfinite(u).all():
            raise ValueError('Non-finite probe command')
        output_path = Path(args.csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export = sim.get_export_array()
        dt = sim.get_time_step()
        rows = [[sim.current_time, *export.tolist()]]
        duration = float(args.seconds)
        if duration < 0 or not np.isfinite(duration):
            raise ValueError('--seconds must be nonnegative and finite')
        for _ in range(int(np.floor(duration / dt + 1e-8))):
            status, export = sim.run(sim.current_time + dt, u, export)
            rows.append([sim.current_time, *export.tolist()])
            if status:
                print(f'[TruckSim LTR probe] solver status={status}')
                break
        with output_path.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['time_s', *[f'export_{i}' for i in range(config['n_export'])]])
            writer.writerows(rows)
        print(f'[TruckSim LTR probe] saved {len(rows)} samples to {output_path}')
    finally:
        sim.stop()


def main():
    parser = argparse.ArgumentParser(description='Check raw Python–TruckSim I/O for the new LTR dataset')
    parser.add_argument('--simfile', required=True)
    parser.add_argument('--dll', required=True)
    parser.add_argument('--csv', default='outputs/koopman/trucksim_ltr_probe.csv')
    parser.add_argument('--seconds', type=float, default=2.0)
    parser.add_argument('--vx-init-kmh', type=float)
    parser.add_argument('--throttle', type=float, default=0.0)
    parser.add_argument('--brake', type=float, default=0.0)
    parser.add_argument('--steer-wheel-deg', type=float, default=0.0)
    probe(parser.parse_args())


if __name__ == '__main__':
    main()
