import os, re
import ctypes
if __package__:
    from . import Simulation
else:  # Support direct script execution without changing cwd.
    import Simulation
import numpy as np
import matplotlib.pyplot as plt


class TrucksimSimulation:
    """仿真顺序：start() --> reset()--> run() --> stop()"""
    def __init__(self, simfile_path, dll_path):
        self.simfile_path = os.path.abspath(simfile_path)
        self.dll_path = os.path.abspath(dll_path)
        # self.change_sim_par(config)  # set initial condition
        self.vs = Simulation.VehicleSimulation()
        self.vs_dll_exist_flag = self.vs_dll_is_exist()
        self.is_running = False
        self.current_time = 0.0

    def vs_dll_is_exist(self):
        # dll_path = self.vs.get_dll_path(self.simfile_path)
        dll_path = self.dll_path
        if dll_path is not None and os.path.exists(dll_path):
            # vs_dll = ctypes.cdll.LoadLibrary(dll_path)
            vs_dll = ctypes.WinDLL(dll_path)
            if self.vs.get_api(vs_dll):
                exist_flag = True
            else:
                exist_flag = False
                print(f'can not get dll api, please check the dll {dll_path}')
        else:
            exist_flag = False
            print(f'please check dll_path or simfile_path existence or not')
        return exist_flag

    def get_export_array(self):
        return self.vs.CopyExportVars(self.configuration.get('n_export'))

    def get_time_step(self):
        return self.configuration.get('t_step')

    def _check_error(self):
        if self.vs.dll_handle.vs_error_occurred():
            getter = getattr(self.vs.dll_handle, "vs_get_error_message", None)
            message = getter() if getter else b"TruckSim solver error"
            raise RuntimeError(f"{self.simfile_path}: {message!r}")

    def start(self, config=None):
        """Restart and apply initial speed (km/h) before solver initialization."""
        if self.is_running:
            self.stop()
        if not os.path.isfile(self.simfile_path):
            raise FileNotFoundError(self.simfile_path)
        if not self.vs_dll_exist_flag:
            raise RuntimeError(f"TruckSim DLL unavailable: {self.dll_path}")
        self.configuration = self.vs.ReadConfiguration(self.simfile_path)
        self._check_error()
        t0 = self.vs.dll_handle.vs_setdef_and_read(
            self.vs.get_char_pointer(self.simfile_path), None, None
        )
        self._check_error()
        if config is not None:
            result = self.vs.Statement('SV_VXS', f" {config['vx_init']}")
            if result != 0:
                raise RuntimeError("TruckSim rejected initial speed SV_VXS")
            self._check_error()
        self.current_time = float(t0)
        self.is_running = True
        try:
            self.vs.Initialize(t0)
            self._check_error()
        except Exception as error:
            try:
                self.stop()
            except Exception as cleanup_error:
                error.add_note(f"TruckSim cleanup also failed: {cleanup_error!r}")
            raise

    def stop(self, t_current=None):
        """Stop once, using the last solver time when no time is supplied."""
        if self.is_running:
            try:
                self.vs.TerminateRun(self.current_time if t_current is None else t_current)
            finally:
                self.is_running = False

    def run(self, t_current, import_array, export_array):
        if not self.is_running:
            raise RuntimeError("TruckSim is not running")
        if len(import_array) != self.configuration['n_import']:
            raise ValueError("TruckSim import array length does not match dataset")
        if len(export_array) != self.configuration['n_export']:
            raise ValueError("TruckSim export array length does not match dataset")
        if not np.isfinite(t_current) or not np.all(np.isfinite(import_array)):
            raise ValueError("TruckSim time and inputs must be finite")
        status, export_array = self.vs.IntegrateIO(t_current, import_array, export_array)
        self.current_time = float(t_current)
        self._check_error()
        return status, export_array

    """已弃用：通过在磁盘上修改par文件，实现初始条件修改"""
    def change_sim_par(self, config):
        """change .par file to modify initial conditions"""
        pos_x = config['x_init']
        vx = config['vx_init']
        simfile_path = self.simfile_path
        if not os.path.isfile(simfile_path):
            raise FileNotFoundError(f"SIM file not found: {simfile_path}")

        # Step 1: Read the SIM file and extract the .par file path
        par_file_path = None
        with open(simfile_path, 'r') as sim_file:
            for line in sim_file:
                if line.strip().startswith("INPUT"):
                    # INPUT 后跟路径
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        par_file_path = parts[1].strip()
                        break

        if not par_file_path:
            raise ValueError("No 'INPUT' line found in SIM file.")

        if not os.path.isfile(par_file_path):
            raise FileNotFoundError(f"PAR file not found: {par_file_path}")

        # Step 2: Read and modify the PAR file
        with open(par_file_path, 'r') as par_file:
            lines = par_file.readlines()

        updated_lines = []
        for line in lines:
            stripped = line.strip()
            if re.match(r'^SSTART\s', stripped):
                # 替换 SSTART 后的数字
                updated_lines.append(re.sub(r'^(SSTART\s+)([-+eE0-9.]+)', lambda m: m.group(1) + str(pos_x), line))

            elif re.match(r'^SV_VXS\s', stripped):
                # 替换 SV_VXS 后的数字
                updated_lines.append(re.sub(r'^(SV_VXS\s+)([-+eE0-9.]+)', lambda m: m.group(1) + str(vx), line))
            else:
                updated_lines.append(line)

        # Step 3: Write the updated content back to the PAR file
        with open(par_file_path, 'w') as par_file:
            par_file.writelines(updated_lines)

    """在内存中修改par文件中的变量，不实际改变磁盘上par文件的内容，更高效，能实现单实例复用"""
    def reset(self, config: dict = None):
        """Restart the solver with optional initial conditions."""
        self.start(config=config)


if __name__ == '__main__':
    # === Test Trucksim API lon===
    # car_num = 3
    # trucksim_models = []
    # for i in [0, 1, 2]:
    #     simfile_path = f"D:\\Users\\Public\\Documents\\TruckSim2016.1_Data_now_using\\truck_{i + 1}.sim"
    #     dll_path = f"D:\\Users\\Public\\Documents\\TruckSim2016.1_Data_now_using\\Extensions\\Multi_vehicle\\s_s{i + 1}_64.dll"
    #     trucksim_model = TrucksimSimulation(simfile_path=simfile_path, dll_path=dll_path)

    #     # config = {
    #     #     'x_init': 50 * i,
    #     #     'vx_init': 0,  # [km/h]
    #     # }
    #     config = None
    #     trucksim_model.start()
    #     trucksim_model.reset(config=config)
    #     trucksim_models.append(trucksim_model)

    # t_step = trucksim_models[0].get_time_step()
    # export_array = trucksim_models[0].get_export_array()  # 外部速度修改不会立即生效，经历vs.IntegrateIO()后生效
    # status = 0
    # T = 30
    # steps = int(T / t_step)

    # history = [[], [], []]
    # linear_history = [[], [], []]
    # t_current = 0

    # linear_x = export_array[4]
    # linear_v = export_array[2]

    # for i in range(steps):
    #     if export_array[2] >= 90:
    #         stop = 1
    #     # timer
    #     t_current = (i + 1) * t_step
    #     import_arrays = [[0.2, 0], [0.2, 0], [0.2, 0]]  # throttle, brake
    #     # import_arrays = [[target_vx * 3.6], [target_vx * 3.6], [target_vx * 3.6]]
    #     for j in range(car_num):
    #         status, export_array = trucksim_models[j].vs.IntegrateIO(t_current, import_arrays[j], export_array)
    #         # status, export_array = trucksim_models[j].run(t_current, import_arrays[j], export_array)
    #         history[j].append(export_array)
    #         if status:
    #             trucksim_models[j].stop(t_step)
    #             break

    #         # ------linear
    #         linear_a = 0.1  # [m/s^2]
    #         linear_v += linear_a * t_step
    #         linear_x += linear_v * t_step
    #         linear_history[j].append([linear_x, linear_v])

    #     print(f"simulating, t = {t_current}s ...")

    # linear_history = np.array(linear_history)

    # for i in range(car_num):
    #     trucksim_models[i].stop(t_current)

    # data, t = [], []
    # for i in range(car_num):
    #     temp_data = np.array(history[i])

    #     # === 单位转换 ===
    #     temp_data[:, 0] *= 9.8
    #     temp_data[:, 1] *= 9.8
    #     temp_data[:, 2] *= 1 / 3.6
    #     temp_data[:, 3] *= 1 / 3.6
    #     temp_data[:, 6:] *= np.pi / 180

    #     data.append(temp_data)
    #     t.append(np.arange(len(temp_data)) * t_step)

    # labels = ["ax", "ay", "vx", "vy", "x", "y", "roll", "roll_dot", "pitch", "pitch_dot", "yaw", "yaw_dot"]
    # fig, axs = plt.subplots(6, 2, figsize=(14, 12))
    # axs = axs.flatten()
    # category = "Truck"
    # for i, ax in enumerate(axs):
    #     for j in range(car_num):
    #         ax.plot(t[j], data[j][:, i], label=f"{category} {j + 1}")
    #         if i == 2:
    #             ax.plot(t[j], linear_history[j][:, 1], '--')
    #         if i == 4:
    #             ax.plot(t[j], linear_history[j][:, 0], '--')
    #     ax.set_title(labels[i])
    #     ax.legend()
    #     ax.grid(True)
    # plt.tight_layout()
    # # plt.savefig("Results\\Multi-vehicle dynamics outputs.pdf")
    # plt.show()


    # === Test Trucksim API lon & lat ===
        car_num = 3
        trucksim_models = []
        for i in [0, 1, 2]:
            simfile_path = f"D:\\Users\\Public\\Documents\\TruckSim2016.1_Data_now_using\\truck_{i + 1}_lon_lat.sim"
            dll_path = f"D:\\Users\\Public\\Documents\\TruckSim2016.1_Data_now_using\\Extensions\\Multi_vehicle_lon_lat\\s_s_{i + 1}.dll"
            trucksim_model = TrucksimSimulation(simfile_path=simfile_path, dll_path=dll_path)
    
            # config = {
            #     'x_init': 50 * i,
            #     'vx_init': 0,  # [km/h]
            # }
            config = None
            trucksim_model.start()
            trucksim_model.reset(config=config)
            trucksim_models.append(trucksim_model)
    
        t_step = trucksim_models[0].get_time_step()
        export_array = trucksim_models[0].get_export_array()  # 外部速度修改不会立即生效，经历vs.IntegrateIO()后生效
        status = 0
        T = 30
        steps = int(T / t_step)
    
        history = [[], [], []]
        linear_history = [[], [], []]
        t_current = 0
    
        linear_x = export_array[4]
        linear_v = export_array[2]
    
        for i in range(steps):
            if export_array[2] >= 90:
                stop = 1
            # timer 
            t_current = (i + 1) * t_step
            import_arrays = [[0, -5, 80], [0, -5, 80], [0, -5, 80]]  # throttle, brake, steer
            # import_arrays = [[target_vx * 3.6], [target_vx * 3.6], [target_vx * 3.6]]
            for j in range(car_num):
                status, export_array = trucksim_models[j].vs.IntegrateIO(t_current, import_arrays[j], export_array)
                # status, export_array = trucksim_models[j].run(t_current, import_arrays[j], export_array)
                history[j].append(export_array)
                if status:
                    trucksim_models[j].stop(t_step)
                    break
    
                # ------linear
                linear_a = 0.1  # [m/s^2]
                linear_v += linear_a * t_step
                linear_x += linear_v * t_step
                linear_history[j].append([linear_x, linear_v])
    
            print(f"simulating, t = {t_current}s ...")
    
        linear_history = np.array(linear_history)
    
        for i in range(car_num):
            trucksim_models[i].stop(t_current)
    
        data, t = [], []
        for i in range(car_num):
            temp_data = np.array(history[i])
    
            # === 单位转换 ===
            temp_data[:, 0] *= 9.8
            temp_data[:, 1] *= 9.8
            temp_data[:, 2] *= 1 / 3.6
            temp_data[:, 3] *= 1 / 3.6
            temp_data[:, 6:] *= np.pi / 180
    
            data.append(temp_data)
            t.append(np.arange(len(temp_data)) * t_step)
    
        labels = ["ax", "ay", "vx", "vy", "x", "y", "roll", "roll_dot", "pitch", "pitch_dot", "yaw", "yaw_dot"]
        fig, axs = plt.subplots(6, 2, figsize=(14, 12))
        axs = axs.flatten()
        category = "Truck"
        for i, ax in enumerate(axs):
            for j in range(car_num):
                ax.plot(t[j], data[j][:, i], label=f"{category} {j + 1}")
                if i == 2:
                    ax.plot(t[j], linear_history[j][:, 1], '--')
                if i == 4:
                    ax.plot(t[j], linear_history[j][:, 0], '--')
            ax.set_title(labels[i])
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        # plt.savefig("Results\\Multi-vehicle dynamics outputs.pdf")
        plt.show()