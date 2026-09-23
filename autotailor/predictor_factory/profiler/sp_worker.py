from multiprocessing import Process
import os
import statistics
import time

import coremltools as ct
from loguru import logger
import numpy as np
import torch

from .utils import timestamp


class spWorker(Process):
    def __init__(self,
                 wp_task_queue,
                 out_queue,
                 backend_config,
                 command_config,
                 connector,
                 enable_latency_constraint=True,
                 latency_constraint=2000,
                 verbose=False,
                 workspace=os.getenv("WORKSPACE"),
                 gc=False,
                 debug=False):
        super(spWorker, self).__init__()
        self.wp_task_queue = wp_task_queue
        self.out_queue = out_queue
        self.connector = connector

        # extract backend information
        self.device_name = backend_config["DEVICE_NAME"]
        self.backend = command_config["backend"]
        # extract quantization information
        self.enable_int8 = command_config["use_int8"]
        self.enable_fp16 = command_config["use_fp16"]
        
        self.command_config = command_config
        self.enable_latency_constraint = enable_latency_constraint
        self.latency_constraint = latency_constraint
        self.verbose = verbose
        self.workspace = workspace
        self.gc_mode = gc
        
        self.debug_mode = debug

    def run(self):
        
        timestamp('spWorker', 'start')
        error_save_path = os.path.join(self.workspace, 'results_spWorker', 'profile_error.log')

        if self.debug_mode:
            while True:
                kernel_config = self.wp_task_queue.get()
                timestamp('spWorker', 'get task')
                kernel_config['latency'] = 0.1
                logger.info(f"profile latency {kernel_config['latency']}")
                time.sleep(0.05) # cold down
                try:
                    num_tasks = self.out_queue.qsize()
                except NotImplementedError:
                    # Fallback mechanism for macOS
                    num_tasks = len(self.out_queue._buffer)
                
                while num_tasks > 20:
                    time.sleep(1)
                    continue
                self.out_queue.put(kernel_config)
                timestamp('spWorker', 'send and profile over')
        
        while True:
            kernel_config = self.wp_task_queue.get()
            timestamp('spWorker', 'get task')
            
            try:
                if self.device_name != 'localhost':
                # send
                    if self.backend == 'ncnn':
                        base_dir = kernel_config['converted_model']
                        if self.enable_int8:
                            ncnn_param_path = base_dir + '_tracing_int8.ncnn.param'
                            ncnn_bin_path = base_dir + '_tracing_int8.ncnn.bin'
                        else:
                            ncnn_param_path = base_dir + '_tracing.ncnn.param'
                            ncnn_bin_path = base_dir + '_tracing.ncnn.bin'
                        self.connector.send_model(ncnn_param_path)
                        self.connector.send_model(ncnn_bin_path)
                        # remove local converted model after sending
                        if self.gc_mode:
                            os.remove(ncnn_param_path)
                            os.remove(ncnn_bin_path)
                    elif self.backend == 'tflite':
                        base_dir = kernel_config['converted_model']
                        tflite_path = base_dir + '.tflite'
                        self.connector.send_model(tflite_path)
                        # remove local converted model after sending
                        if self.gc_mode:
                            os.remove(tflite_path)
                    elif self.backend == 'onnx':
                        base_dir = kernel_config['converted_model']
                        onnx_path = base_dir + '.onnx'
                        self.connector.send_model(onnx_path)
                        # remove local converted model after sending
                        if self.gc_mode:
                            os.remove(onnx_path)
                    elif self.backend == 'pytorch':
                        base_dir = kernel_config['converted_model']
                        pytorch_path = base_dir + '.pt'
                        logger.info("base_dir "+base_dir) ### test
                        self.connector.send_model(pytorch_path)
                        # remove local converted model after sending
                        if self.gc_mode:
                            os.remove(pytorch_path)

                timestamp('spWorker', 'send over')
                # profile
                model_name = kernel_config['converted_model'].split('/')[-1]
                if self.backend == 'tflite':
                    model_name = model_name + '.tflite'
                elif self.backend == 'ncnn':
                    if self.enable_int8:
                        model_name = model_name + '_tracing_int8.ncnn'
                    else:
                        model_name = model_name + '_tracing.ncnn'
                elif self.backend == 'onnx':
                    model_name = model_name + '.onnx'
                elif self.backend == 'pytorch':
                    model_name = model_name + '.pt'
                elif self.backend == "cudnn":
                    model_name = model_name + ".pt"
                elif self.backend == "coreml":
                    model_name = model_name + ".mlmodel"
                else:
                    raise NotImplementedError
                
                if self.device_name == "localhost":
                    if self.backend == 'ncnn':
                        model_path = kernel_config['converted_model'] + '_tracing.ncnn'
                else:
                    model_path = os.path.join(self.connector.model_dir, model_name)
                
                configs = self.command_config
                input_shape = kernel_config["input_shape"]
                if isinstance(input_shape, dict):
                    str_shapes = ""
                    for k,v in input_shape.items():
                        if len(v) == 4:
                            b, c, h, w = v
                            if len(str_shapes) == 0:
                                str_shapes += f'[{h},{w},{c}]'
                            else:
                                str_shapes += f',[{h},{w},{c}]'
                        elif len(v) == 3:
                            b, c, hw = v
                            if len(str_shapes) == 0:
                                str_shapes += f'[{hw}, {c}]'
                            else:
                                str_shapes += f',[{hw}, {c}]'
                        elif len(v) == 2:
                            b, c = v
                            if len(str_shapes) == 0:
                                str_shapes += f'[{c}]'
                            else:
                                str_shapes += f',[{c}]'
                    configs['shape'] = str_shapes
                elif len(input_shape)==4:
                    self.command_config["HW0"] = input_shape[2]
                    self.command_config["HW1"] = input_shape[3]
                    self.command_config["CIN"] = input_shape[1]
                    
                    configs['shape'] = f'[{input_shape[2]},{input_shape[3]},{input_shape[1]}]'
                else:
                    configs['shape'] = f"[{','.join(map(str, input_shape))}]"
                # if 'K' in unit_kernel_config:
                #     configs['shape'] = f'[1,{unit_kernel_config["M"]},{unit_kernel_config["N"]}]'
                # elif 'KERNEL_SIZE' not in unit_kernel_config:
                #     configs['shape'] = f'[1,{unit_kernel_config["HW"]},{unit_kernel_config["CIN"]}]'
                # elif 'CIN' in unit_kernel_config:
                #     configs['shape'] = f'[{unit_kernel_config["HW"]},{unit_kernel_config["HW"]},{unit_kernel_config["CIN"]}]'
                # elif 'CHANNEL_SIZE' in unit_kernel_config:
                #     configs['shape'] = f'[{unit_kernel_config["HW"]},{unit_kernel_config["HW"]},{unit_kernel_config["CHANNEL_SIZE"]}]'
                
                if self.backend == "cudnn":
                    num_runtimes = self.command_config["loop_count"]
                    trace_model_path = kernel_config['converted_model']
                    assert torch.cuda.is_available()
                    
                    model = torch.load(trace_model_path)
                    model = model.cuda()
                    model.eval()
                    
                    if isinstance(input_shape, dict):
                        x = []
                        for k, v in input_shape.items():
                            x.append(torch.rand(v))
                        if len(x) == 1:
                            x = x[0].cuda()
                        else:
                            x = torch.stack(x, dim=0).cuda()
                    else:
                        x = torch.rand(input_shape).cuda()
                    with torch.no_grad():
                        # warmup
                        for i in range(8):
                            output = model(x)
                        
                        inference_times = []
                        for i in range(num_runtimes):
                            torch.cuda.synchronize()
                            start_time = time.time()
                            
                            output = model(x)

                            torch.cuda.synchronize()
                            end_time = time.time()
                            
                            inference_times.append(end_time - start_time)

                    average_inference_time = sum(inference_times) / num_runtimes
                    min_inference_time = min(inference_times)
                    max_inference_time = max(inference_times)
                    std_dev_inference_time = statistics.stdev(inference_times)
                    
                    latency = f"{average_inference_time*1000} +- {std_dev_inference_time*1000}"
                    kernel_config['latency'] = latency
                elif self.backend == "coreml":
                    num_runtimes = self.command_config["loop_count"]
                    model_path = kernel_config["converted_model"]
                    
                    processor_type = self.command_config["processor_type"]
                    if processor_type == "cpu":
                        cp_type = ct.ComputeUnit.CPU_ONLY
                    elif processor_type == "ne":
                        cp_type = ct.ComputeUnit.CPU_AND_NE
                    elif processor_type == "gpu":
                        cp_type = ct.ComputeUnit.CPU_AND_GPU
                    
                    model = ct.models.MLModel(model_path, compute_units=cp_type)
                    
                    x = {}
                    if isinstance(input_shape, dict):
                        x = {}
                        for k, v in input_shape.items():
                            x[str(k)] = np.random.rand(*list(v))
                    else:
                        x["input"] = np.random.rand(*list(input_shape))
                    with torch.no_grad():
                        # warmup
                        for i in range(8):
                            output = model.predict(x)
                        
                        inference_times = []
                        for i in range(num_runtimes):
                            start_time = time.time()
                            
                            output = model.predict(x)
                        
                            end_time = time.time()
                            
                            inference_times.append(end_time - start_time)

                    average_inference_time = sum(inference_times) / num_runtimes
                    min_inference_time = min(inference_times)
                    max_inference_time = max(inference_times)
                    std_dev_inference_time = statistics.stdev(inference_times)
                    
                    latency = f"{average_inference_time*1000} +- {std_dev_inference_time*1000}"
                    kernel_config['latency'] = latency
                else:
                    if self.enable_fp16:
                        logger.info("Use fp16 benchmark")
                        specific_benchmark = self.connector.benchmark_fp16_model_path
                    elif self.enable_int8:
                        logger.info("Use int8 benchmark")
                        specific_benchmark = self.connector.benchmark_int8_model_path
                    else:
                        logger.info("Use fp32 benchmark")
                        specific_benchmark = self.connector.benchmark_fp32_model_path 

                    res = self.connector.profile(model_path=model_path,
                                                configs=configs,
                                                specific_benchmark=specific_benchmark,
                                                enable_latency_constraint=self.enable_latency_constraint,
                                                latency_constraint=self.latency_constraint,
                                                verbose=self.verbose)
                    # remove remote converted model after profiling
                    if self.gc_mode and self.device_name!='localhost':
                        to_remove_files = []
                        if self.backend == 'tflite':
                            to_remove_files.append(model_path)
                        if self.backend == 'onnx':
                            to_remove_files.append(model_path)
                        elif self.backend == 'pytorch':
                            to_remove_files.append(model_path)
                        elif self.backend == 'ncnn':
                            to_remove_files.append(model_path+'.bin')
                            to_remove_files.append(model_path+'.param')
                        for file_path in to_remove_files:
                            self.connector.run_command('rm ' + file_path)
                    if 'Segmentation fault' in res:
                        kernel_config['latency'] = 'Segmentation fault'
                    elif 'Timeout' in res:
                        kernel_config['latency'] = 'Timeout'
                    elif 'Notgreen' in res:
                        kernel_config['latency'] = 'Notgreen'
                    elif 'ANEURALNETWORKS_OP_FAILED' in res:
                        kernel_config['latency'] = 'ANEURALNETWORKS_OP_FAILED'
                    else:
                        kernel_config['latency'] = self.connector.parse(res)['latency']
                logger.info(f"profile latency {kernel_config['latency']}")
                time.sleep(0.2) # cold down
                try:
                    num_tasks = self.out_queue.qsize()
                except NotImplementedError:
                    # Fallback mechanism for macOS
                    num_tasks = len(self.out_queue._buffer)
                while num_tasks > 20:
                    time.sleep(1)
                    continue
                self.out_queue.put(kernel_config)
                timestamp('spWorker', 'send and profile over')
            
            except Exception as e:
                os.makedirs(os.path.join(self.workspace, 'results_spWorker'), exist_ok=True)
                open(error_save_path, 'a').write(f'{model_name}: {e}\n')
