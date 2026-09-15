import os, re
import ctypes
from trucksim_simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt


class TrucksimSimulation:
    def __init__(self, simfile_path, dll_path):
        self.simfile_path = simfile_path
        self.dll_path = dll_path
        # self.change_sim_par(config)  # set initial condition
        self.vs = Simulation.VehicleSimulation()
        self.vs_dll_exist_flag = self.vs_dll_is_exist()
        self.is_running = False

        # read configuration


        self.reset()

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

    def stop(self, t_current):
        if self.is_running:
            self.vs.TerminateRun(t_current)

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
        """reset initial conditions"""
        if config is None:
            config = {
                'x_init': 60,
                'vx_init': 36,
            }
        self.configuration = self.vs.ReadConfiguration(self.simfile_path)
        if config is not None:
            # pos_x = config['x_init']
            vx = config['vx_init']
            self.vs.Statement(keyword='SV_VXS', rest_of_line=f' {vx}')
            # self.vs.Statement(keyword='SSTART', rest_of_line=f' {pos_x}')  # 改不了
        self.is_running = True


if __name__ == '__main__':
    # === multi-vehicle dynamics koop_model ===
    car_num = 3
    trucksim_models = []
    for i in [0, 1, 2]:
        simfile_path = f"D:\\Users\\Public\\Documents\\TruckSim2016.1_Data_now_using\\truck_{i + 1}.sim"
        dll_path = f"D:\\Users\\Public\\Documents\\TruckSim2016.1_Data_now_using\\Extensions\\Multi_vehicle\\s_s{i + 1}_64.dll"
        trucksim_model = TrucksimSimulation(simfile_path=simfile_path, dll_path=dll_path)
        trucksim_models.append(trucksim_model)

    t_step = trucksim_models[0].get_time_step()
    export_array = trucksim_models[0].get_export_array()  # 外部速度修改不会立即生效，经历vs.IntegrateIO()后生效
    status = 0
    T = 10
    steps = int(T / t_step)

    # === reference path ===
    # reference_path = np.array([[i * 2.0, 3.0 * np.sin(i * 0.1)] for i in range(steps)])

    # === communication ===
    communication_delay = 5  # 单位：steps
    from collections import deque
    delayed_path_queue = deque(maxlen=communication_delay)

    history = [[], [], []]
    target_ax = -1
    v0 = 60  # km/h
    target_vx = v0 / 3.6
    for i in range(steps):
        # timer
        t_current = (i + 1) * t_step

        # target_x, target_y = reference_path[i]
        target_vx += target_ax * t_step  # [m/s]
        import_arrays = [[0.1, 0], [0.2, 0], [0.8, 0]]  # throttle, brake
        # import_arrays = [[target_vx * 3.6], [target_vx * 3.6], [target_vx * 3.6]]
        for j in range(car_num):
            status, export_array = trucksim_models[j].vs.IntegrateIO(t_current, import_arrays[j], export_array)
            history[j].append(export_array)
            if status:
                trucksim_models[j].stop(t_step)
                break

        print(f"simulating, t = {t_current}s ...")

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

    labels = ["dvx", "dvy", "vx", "vy", "x", "y", "roll", "roll_dot", "pitch", "pitch_dot", "yaw", "yaw_dot"]
    fig, axs = plt.subplots(6, 2, figsize=(14, 12))
    axs = axs.flatten()
    category = "Truck"
    for i, ax in enumerate(axs):
        for j in range(car_num):
            ax.plot(t[j], data[j][:, i], label=f"{category} {j + 1}")
        ax.set_title(labels[i])
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    # plt.savefig("Results\\Multi-vehicle dynamics outputs.pdf")
    plt.show()