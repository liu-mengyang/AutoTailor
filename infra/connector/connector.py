import os
import importlib
import json
import warnings
import time
import logging
import subprocess
from loguru import logger

from ppadb.client import Client as AdbClient

from .backends.interface import __BUILTIN_BACKENDS__


class BackendConnector:
    def __init__(self, config):
        self.device_name = config["DEVICE_NAME"]
        
        self.config_dict = config
        
        backend_key = config["backend_key"]
        self.backend_name = config["backend_name"]
        
        if self.device_name == "localhost":
            if backend_key in __BUILTIN_BACKENDS__:
                self.backend_info = __BUILTIN_BACKENDS__[backend_key]
            else:
                raise ValueError(f"Unsupported backend name: {self.backend_name}. Please register the backend first.")
            self.model_dir = self.config_dict['MODEL_DIR']
            self.benchmark_dir = self.config_dict['BENCHMARK_DIR']
            self.default_benchmark_path = self.backend_info["default_benchmark_path"]
            self.config_dict["default_benchmark_path"] = self.default_benchmark_path
            self.backend = self._connect_backend()
            self.benchmark_fp32_model_path = self.backend.benchmark_fp32_model_path
            self.benchmark_fp16_model_path = self.backend.benchmark_fp16_model_path
            self.benchmark_int8_model_path = self.backend.benchmark_int8_model_path
            return
        else:
            if backend_key in __BUILTIN_BACKENDS__:
                self.backend_info = __BUILTIN_BACKENDS__[backend_key]
            else:
                raise ValueError(f"Unsupported backend name: {self.backend_name}. Please register the backend first.")
        
            self.default_benchmark_path = self.backend_info["default_benchmark_path"]
            self.config_dict["default_benchmark_path"] = self.default_benchmark_path

            self.client = AdbClient(host="127.0.0.1", port=5037)
            self.backend = self._connect_backend()
            self.model_dir = self.config_dict['REMOTE_MODEL_DIR']
            self.benchmark_dir = self.config_dict['BENCHMARK_DIR']
        
            if self.config_dict['DEVICE_SERIAL']:
                device = self.config_dict['DEVICE_SERIAL']
                if len(device.split('.')) != 1:
                    ##192.168
                    self.client.remote_connect(device,5555)
                    self.device = self.client.device(f"{device}:5555")
                else:
                    self.device = self.client.device(self.config_dict['DEVICE_SERIAL'])
            else:
                self.device = self.client.devices()[0]
        
            self.benchmark_fp32_model_path = self.backend.benchmark_fp32_model_path
            self.benchmark_fp16_model_path = self.backend.benchmark_fp16_model_path
            self.benchmark_int8_model_path = self.backend.benchmark_int8_model_path
        
    def _connect_backend(self):
        """ 
        Return the required backend class, and feed params to the backend. Supporting backend: tflite_cpu, tflite_gpu, openvino_vpu.
        
        Available backend and corresponding configs: 
        - For backend based on NCNN and TFLite platform: {
            'REMOTE_MODEL_DIR': path to the folder (on mobile device) where temporary models will be copied to.
            'BENCHMARK_MODEL_PATH': path (on android device) where the binary file `benchmark_model` is deployed.
            'DEVICE_SERIAL': if there are multiple adb devices connected to your host, you need to provide the \\
                            corresponding serial id. Set to '' if there is only one device connected to your host.
            'KERNEL_PATH': path (on mobile device) where the kernel implementations will be dumped.
        }
        - For backend based on OpenVINO platform: {
            'OPENVINO_ENV': path to openvino virtualenv (./docs/requirements/openvino_requirements.txt is provided)
            'OPTIMIZER_PATH': path to openvino optimizer
            'OPENVINO_RUNTIME_DIR': directory to openvino runtime
            'DEVICE_SERIAL': serial id of the device
            'DATA_TYPE': data type of the model (e.g., fp16, fp32)
        }
        
        The config can be declared and modified after create a workspace. Users could follow guidance from ./docs/builder/backend.md
        
        @params:
        backend_name: name of backend (subclass instance of `BaseBackend`). 
        """
        

        module = self.backend_info["class_module"]
        name = self.backend_info["class_name"]
        backend_module = importlib.import_module(module)   
        backend_cls = getattr(backend_module, name)
        
        return backend_cls(self.config_dict)

    def run_command_local(self, command):
        try:
            result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            res = [result.stderr, result.stdout]
        except subprocess.CalledProcessError:
            print(subprocess.PIPE)
            return "Segmentation Fault"
        return res


    def run_command(self, command):
        return self.device.shell(command)
    
    def send_benchmark(self, benchmark_file_path):
        """
        Sending benhmark file to the backend

        Args:
            benchmark_file_path (str): the local path of the benchmark file
        """
        self._send_file(benchmark_file_path, self.benchmark_dir)
        bench_name = os.path.basename(benchmark_file_path)
        self.run_command('chmod +x '+os.path.join(self.benchmark_dir, bench_name))
    
    def send_executor(self, executor_file_path):
        """
        Sending executor binary file to the backend

        Args:
            executor_file_path (str): the local path of the executor binary file
        """
        self._send_file(executor_file_path, '/data/local/tmp')
        file_name = os.path.basename(executor_file_path)
        self.run_command('chmod +x '+os.path.join('/data/local/tmp', file_name))
        
    def list_benchmarks(self):
        """
        List benchmarks of the backend

        Returns:
            str: result of listing
        """
        return self.run_command("ls " + self.benchmark_dir)
    
    def check_disk(self):
        """
        Check how many ratio the free disk, if not raising warnning
        """
        return 80
        return int(self.run_command("df -h | grep /data").split(' ')[-2].split('%')[0])
    
    def check_battery(self):
        """
        Check how many ratio the rest battery, if not raising warnning
        """
        if self.device_name != "localhost":
            level = int(self.run_command("dumpsys battery | grep level").split(': ')[-1])
            scale = int(self.run_command("dumpsys battery | grep scale").split(': ')[-1])
            return level / scale
        else:
            return 100
    
    def send_model(self, model_file_path):
        """
        Sending model file to the backend

        Args:
            model_file_path (str): the local path of the model
        """
        self._send_file(model_file_path, self.model_dir)
    
    def list_models(self):
        """
        List models of the backend

        Returns:
            str: result of listing
        """
        return self.run_command("ls " + self.model_dir)
    
    def _send_file(self, dst_local, dst_remote_dir):
        
        # check disk
        usage_ratio = self.check_disk()
        if usage_ratio >= 95:
            raise OSError('Disk free space is not enough.')
        elif usage_ratio >= 80:
            warnings.warn(f"Disk is in high usage ratio: {usage_ratio}")
        
        basename = os.path.basename(dst_local)
        self.device.push(dst_local, os.path.join(dst_remote_dir, basename))
        
    def profile(self, model_path, configs, enable_latency_constraint=True, latency_constraint=2000, specific_benchmark=None, verbose=False):
        """
        Profiling model performance on connected backend

        Available backend and corresponding configs: 
        - For backend based on NCNN platform: {
            'taskset': processor mask
            'loop_count': the number of loops for averaging multiple profilings
            'num_threads': the number of threads for cpu execution
            'powersave': 0=all cores, 1=little cores only, 2=big cores only
            'gpu_device': -1=cpu-only, 0=gpu0, 1=gpu1 ...
            'cooling_down': whether or not cooling down in each benchmark
            'HW': the height and width of input shape
            'CIN': the channel size of input shape
        }
        - For backend based on TFLite platform: {
            'taskset': processor mask
            'num_runs': the number of loops for averaging multiple profilings
            'num_threads': the number of threads for cpu execution
            'warm_ups': the number of pre-running for warming up
            'use_gpu': whether or not using gpu
            'enable_op_profiling': whether or not enabling operator-wise profiling
            'dst_kernel_path': other kernel
            'close_xnnpack': whther or not closing xnnpack
        }

        Args:
            model_path (str): the remote path of the profiled model
            configs (dict): the benchmark configs
            enable_latency_constraint (bool): whether or not enable the latency constraint. Defaults to True.
            latency_constraints (int): the latency constraints of running a single kernel in millisecond. Defaults to 2000.
            specific_benchmark (str, optional): the path of the specific benchmark file. Defaults to None.

        Raises:
            str: benchmark return messages
        """
        
        taskset_cmd = f"taskset {configs['taskset']} " if configs['taskset'] else '' 
        
        if specific_benchmark is None and configs["use_fp32"]:
            benchmark_model_path = os.path.join(self.benchmark_dir, self.default_benchmark_path)
        elif specific_benchmark is None and configs["use_fp16"]:
            benchmark_model_path = os.path.join(self.benchmark_dir, self.benchmark_fp16_model_path)
        elif specific_benchmark is None and configs["use_int8"]:
            benchmark_model_path = os.path.join(self.benchmark_dir, self.benchmark_int8_model_path)
        else:
            benchmark_model_path = os.path.join(self.benchmark_dir, specific_benchmark)
        
        # check battery
        if self.device_name != "localhost":
            battery_ratio = self.check_battery()
            while battery_ratio < 0.5:
                time.sleep(600)
                print(f'The battery is in low level: {battery_ratio*100} %, device cold down for 10 minutes.')
                battery_ratio = self.check_battery()
        if self.backend_name == 'ncnn':
            basic_cmd = f" {taskset_cmd}{benchmark_model_path}" \
                        f" {configs['loop_count']}" \
                        f" {configs['num_threads']}" \
                        f" {configs['powersave']}" \
                        f" {configs['gpu_device']}" \
                        f" {configs['cooling_down']}" \
                        f" model_name={model_path}" \
                        f" shape={configs['shape']}"
            if self.device_name == "localhost":
                print(basic_cmd)
                res = self.run_command_local(basic_cmd)
                logger.info(res[0])
                logger.info(res[1])
                res = res[1]
            else:
                if enable_latency_constraint:
                    cmd = f"/data/local/tmp/executor \"{basic_cmd}\" {str(latency_constraint)}"
                    res = self.run_command(cmd)
                else:
                    res = self.run_command(basic_cmd)
            
            # if verbose:
            #     logger.info(res)
        elif self.backend_name == 'tflite':
            kernel_cmd = f"--kernel_path={configs['dst_kernel_path']}" if configs['dst_kernel_path'] else ''
            basic_cmd = f" {taskset_cmd} {benchmark_model_path} {kernel_cmd}" \
                        f" --num_threads={configs['num_threads']}" \
                        f" --num_runs={configs['num_runs']}" \
                        f" --run_delay={configs['run_delay']}" \
                        f" --warmup_runs={configs['warm_ups']}" \
                        f" --graph={model_path}" \
                        f" --enable_op_profiling={'true' if configs['enable_op_profiling'] else 'false'}" \
                        f" --use_gpu={'true' if configs['use_gpu'] else 'false'}" \
                        f" --gpu_precision_loss_allowed={'true' if configs['use_fp16'] else 'false'}" \
                        f" --gpu_experimental_enable_quant={'true' if configs['use_int8'] else 'false'}" \
                        f" --gpu_backend={configs['gpu_backend']}" \
                        f" --use_xnnpack={'true' if configs['use_xnnpack'] else 'false'}" \
                        f" --use_nnapi={'true' if configs['use_nnapi'] else 'false'}" \
                        f" --disable_nnapi_cpu={'false' if configs['nnapi_on_cpu'] else 'true'}" \
                        f" --nnapi_allow_fp16={'true' if configs['use_fp16'] else 'false'}" \
                        f" --use_hexagon={'true' if configs['use_hexagon'] else 'false'}"
            if enable_latency_constraint:
                cmd = f"/data/local/tmp/executor \"{basic_cmd}\" {str(latency_constraint)}"
                res = self.run_command(cmd)
            else:
                res = self.run_command(basic_cmd)
    
            if verbose:
                logger.info(res)
        
        elif self.backend_name == 'onnx':
            if configs['use_nnapi']:
                nnapi_part_cmd = f" \"{'NNAPI_FLAG_USE_FP16' if configs['use_fp16'] else ' '} {'NNAPI_FLAG_CPU_DISABLED' if configs['use_gpu'] else 'NNAPI_FLAG_CPU_ONLY'} {'NNAPI_FLAG_USE_NCHW' if configs['use_nchw_layout'] else ' '}\"" 
            else:
                nnapi_part_cmd = " "
            basic_cmd = f" {taskset_cmd} {benchmark_model_path} -m \'times\'" \
                        f" -s -e \"{'nnapi' if configs['use_nnapi'] else 'cpu'}\"" \
                        f" { '-i' if configs['use_nnapi'] else ' '}" \
                        f" {nnapi_part_cmd}" \
                        f" -r {configs['num_runs']}" \
                        f" -c 1 '-x' { '0' if configs['use_gpu'] else configs['max_parallels_nums']}" \
                        f" -S 1 -I" \
                        f" {model_path}" 
            res = self.run_command(basic_cmd)
    
            if verbose:
                logger.info(res)

        elif self.backend_name == 'pytorch':
            basic_cmd = f" {taskset_cmd} {benchmark_model_path}" \
                        f" --model={model_path}" \
                        f" --input_dims={configs['input_dims']}" \
                        f" --iter={configs['iter']}" \
                        f" --warmup={configs['warmup']}" \
                        f" --input_type={configs['input_type']}" \
                        f" --vulkan={'true' if configs['vulkan'] else 'false'}" \
                        f" --report_pep={'true' if configs['report_pep'] else 'false'}"    
            logger.info('basic cmd: '+basic_cmd)
            res = self.run_command(basic_cmd)


            if verbose:
                logger.info(res)
        return res
    
    def parse(self, message, mode=None):
        """
        Parsing the profiled result

        Args:
            message (str): result from the profiling

        Returns:
            dict: result dictionary from ProfiledResult
        """
        latency = (self.backend.parser.parse(message, mode=mode)).results.get('latency')
        return latency
