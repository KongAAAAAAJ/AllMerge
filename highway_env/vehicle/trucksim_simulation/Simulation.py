import ctypes
import sys,string,os,re


class VehicleSimulation:
    def __init__(self):
        self.dll_handle = None

    def get_api(self, dll_handle):
        self.dll_handle = dll_handle

        try:
            # === 绑定所有需要的 DLL 函数签名 ===

            dll_handle.vs_run.argtypes = [ctypes.c_char_p]
            dll_handle.vs_run.restype = ctypes.c_int

            dll_handle.vs_setdef_and_read.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]
            dll_handle.vs_setdef_and_read.restype = ctypes.c_double

            dll_handle.vs_install_echo_function.argtypes = [ctypes.c_void_p]
            dll_handle.vs_install_echo_function.restype = None

            dll_handle.vs_initialize.argtypes = [
                ctypes.c_double,
                ctypes.c_void_p,
                ctypes.c_void_p
            ]
            dll_handle.vs_initialize.restype = None

            dll_handle.vs_read_configuration.argtypes = [
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)
            ]
            dll_handle.vs_read_configuration.restype = None

            dll_handle.vs_integrate_io.argtypes = [
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)
            ]
            dll_handle.vs_integrate_io.restype = ctypes.c_int

            dll_handle.vs_copy_export_vars.argtypes = [ctypes.POINTER(ctypes.c_double)]
            dll_handle.vs_copy_export_vars.restype = None

            dll_handle.vs_terminate_run.argtypes = [ctypes.c_double]
            dll_handle.vs_terminate_run.restype = None

            dll_handle.vs_set_sym_real.argtypes = [ctypes.c_char_p, ctypes.c_double]
            dll_handle.vs_set_sym_real.restype = ctypes.c_int

            dll_handle.vs_statement.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            dll_handle.vs_statement.restype = ctypes.c_int

            dll_handle.vs_error_occurred.argtypes = []
            dll_handle.vs_error_occurred.restype = ctypes.c_int

            dll_handle.vs_road_l.argtypes = [ctypes.c_double, ctypes.c_double]
            dll_handle.vs_road_l.restype = ctypes.c_double

            # === 可选函数（如需要） ===
            if hasattr(dll_handle, "vs_get_error_message"):
                dll_handle.vs_get_error_message.argtypes = []
                dll_handle.vs_get_error_message.restype = ctypes.c_char_p

            return True
        except AttributeError as e:
            print(f"[错误] 加载 DLL 函数失败: {e}")
            return False

    def get_char_pointer(self, python_string):
        # python version is greater or equal to 3.0 then we need to define the encoding when converting a string to
        # bytes. Once that is done we can convert the the python string to a char*.
        if sys.version_info >= (3, 0):
            char_pointer = ctypes.c_char_p(bytes(python_string, 'UTF-8'))
        else:
            char_pointer = ctypes.c_char_p(bytes(python_string))
        return char_pointer

    def get_parameter_value(self, line):
        index = line.find(' ')
        if index >= 0:
            return line[index:].strip()
        else:
            return None

    def get_dll_path(self, path_to_sim_file):
        dll_path = None
        prog_dir = None
        veh_code = None
        product_name = None
        product_ver = None
        library_name = None
        bitness_suffix = '_64' if ctypes.sizeof(ctypes.c_voidp) == 8 else '_32'
        platform = sys.platform

        sim_file = open(path_to_sim_file, 'r')
        for line in sim_file:
            if line.lstrip().startswith('PROGDIR'):
                prog_dir = self.get_parameter_value(line)
                if prog_dir == '.':
                  prog_dir = os.getcwd()
            elif line.lstrip().startswith('DLLFILE'):
                dll_path = self.get_parameter_value(line)
            elif line.lstrip().startswith('VEHICLE_CODE'):
                veh_code = self.get_parameter_value(line)
            elif line.lstrip().startswith('PRODUCT_ID'):
                product_name = self.get_parameter_value(line)
            elif line.lstrip().startswith('PRODUCT_VER'):
                product_ver = self.get_parameter_value(line)

        sim_file.close()

        if "tire" in veh_code:
            if platform == 'linux':
              library_name = 'libtire.so.%s'%(product_ver)
            else:
              library_name = "tire" + bitness_suffix
        elif product_name == "CarSim":
            if platform == 'linux':
              library_name = 'libcarsim.so.%s'%(product_ver)
            else:
              library_name = "carsim" + bitness_suffix
        elif product_name == "TruckSim":
            if platform == 'linux':
              library_name = 'libtrucksim.so.%s'%(product_ver)
            else:
              library_name = "trucksim" + bitness_suffix
        else:
            if platform == 'linux':
              library_name = 'libcarsim.so.%s'%(product_ver)
            else:
              library_name = veh_code + bitness_suffix

        if dll_path is None:
            if sys.platform == 'linux':
              dll_path = os.path.join(prog_dir, library_name)
            else:
              dll_path = os.path.join(prog_dir, "Programs", "Solvers", library_name + ".dll")
        return dll_path

    def run(self, path_to_sim_file):
        error_occurred = 1
        path_to_sim_file_ptr = self.get_char_pointer(path_to_sim_file)

        if path_to_sim_file_ptr is not None:
            error_occurred = self.dll_handle.vs_run(path_to_sim_file_ptr)  # 相当于手动点击"run"按键

        return error_occurred
        
    def print_error(self):
      error_string = ctypes.c_char_p(self.dll_handle.vs_get_error_message())
      print(error_string.value.decode('ascii'))

    def ReadConfiguration(self, path_to_sim_file):
        path_to_sim_file_ptr = self.get_char_pointer(path_to_sim_file)
        platform = sys.platform
        if path_to_sim_file_ptr is not None:
            if platform == 'linux':
              ref_n_import = ctypes.c_long()
              ref_n_export = ctypes.c_longlong()
            else:
              ref_n_import = ctypes.c_int32()
              ref_n_export = ctypes.c_int64()
            ref_t_start = ctypes.c_double()
            ref_t_stop = ctypes.c_double()
            ref_t_step = ctypes.c_double()
            self.dll_handle.vs_read_configuration(path_to_sim_file_ptr,
                                                  ctypes.byref(ref_n_import),
                                                  ctypes.byref(ref_n_export),
                                                  ctypes.byref(ref_t_start),
                                                  ctypes.byref(ref_t_stop),
                                                  ctypes.byref(ref_t_step))
            configuration = {'n_import': ref_n_import.value,
                             'n_export': ref_n_export.value,
                             't_start': ref_t_start.value,
                             't_stop': ref_t_stop.value,
                             't_step': ref_t_step.value}
            return configuration

    def CopyExportVars(self, n_export):
        export_array = (ctypes.c_double * n_export)()
        self.dll_handle.vs_copy_export_vars(ctypes.cast(export_array, ctypes.POINTER(ctypes.c_double)))
        export_list = [export_array[i] for i in range(n_export)]
        return export_list

    def GetRoadL(self, x, y):
        x_c_double = ctypes.c_double(x)
        y_c_double = ctypes.c_double(y)
        c_double_return = self.dll_handle.vs_road_l(x_c_double, y_c_double)
        return float(c_double_return)

    def IntegrateIO(self, t_current, import_array, export_array):
        t_current_c_double = ctypes.c_double(t_current)
        import_c_double_array = (ctypes.c_double * len(import_array))(*import_array)
        export_c_double_array = (ctypes.c_double * len(export_array))(*export_array)

        # c_integer_return = self.dll_handle.vs_integrate_io(t_current_c_double,
        #                                                    ctypes.byref(import_c_double_array),
        #                                                    ctypes.byref(export_c_double_array))
        c_integer_return = self.dll_handle.vs_integrate_io(
            t_current_c_double,
            import_c_double_array,  # 直接传数组，不要 byref
            export_c_double_array
        )

        export_array = [export_c_double_array[i] for i in range(len(export_array))]
        return c_integer_return, export_array

    def Initialize(self, t):
        t_c_double = ctypes.c_double(t)
        self.dll_handle.vs_initialize(t_c_double, None, None)

    def TerminateRun(self, t):
        t_c_double = ctypes.c_double(t)
        # print("vs_terminate_run ptr:", hex(ctypes.cast(self.dll_handle.vs_terminate_run, ctypes.c_void_p).value))
        self.dll_handle.vs_terminate_run(t_c_double)

    def Statement(self, keyword: str, rest_of_line: str, stop_error: int = 1) -> int:
        """
        调用 TruckSim 的 vs_statement 接口，在 setdef 阶段覆盖 .par 文件中已有参数。

        :param keyword:     Parsfile 中的关键字（如 "SSTART" 或 "SV_VXS"）
        :param rest_of_line: 紧跟关键字后的内容（例如 " 100.0"）
        :param stop_error:  如果关键字无效是否报错（1=报错，0=忽略）
        :return:            DLL 返回值，0 表示成功，-1/-2 等表示失败
        """
        kw_ptr  = self.get_char_pointer(keyword)
        text_ptr = self.get_char_pointer(rest_of_line)
        result = self.dll_handle.vs_statement(kw_ptr, text_ptr, ctypes.c_int(stop_error))
        if result != 0:
            print(f"[警告] vs_statement 失败: {keyword}{rest_of_line} (返回码: {result})")
        return result
