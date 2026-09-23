import os
import shutil
from multiprocessing import Process
import time

from loguru import logger

from infra.generator.generator import KernelGenerator

from .utils import timestamp


class gcWorker(Process):
    def __init__(self,
                 gc_task_queue,
                 wp_task_queue,
                 backend_config,
                 command_config,
                 save_dir,
                 convertor,
                 generator_framework="torch",
                 verbose=False,
                 workspace=os.getenv("WORKSPACE"),
                 gc=True,
                 debug=False,
                 ):
        super(gcWorker, self).__init__()
        self.gc_task_queue = gc_task_queue
        self.wp_task_queue = wp_task_queue
        self.save_dir = save_dir
        self.convertor = convertor
        self.kernelGenerator = KernelGenerator()
        self.generator_framework = generator_framework
        
        # extract backend information
        self.backend = command_config["backend"]
        
        # extract quantization information
        self.enable_fp16 = command_config["use_fp16"]
        self.enable_int8 = command_config["use_int8"]
        self.enable_fp32 = command_config["use_fp32"]

        self.verbose = verbose
        self.workspace = workspace
        self.gc_mode = gc
        self.debug_mode = debug
        
    def run(self):
        timestamp('gcWorker', 'start')
        error_save_path = os.path.join(self.workspace, 'results_gcWorker', 'generate_error.log')
        
        if self.debug_mode:
            while True:
                config_dict = self.gc_task_queue.get()

                config = config_dict['config']
                model_path = config_dict['model_path']
                kernel_type = config_dict['kernel_type']
                model_name = os.path.basename(model_path).split('.')[0]
                converted_path = os.path.join(self.save_dir, model_name)
                input_tensor_shape = [1,3,224,224]
                kernel_config = {
                    'converted_model': converted_path,
                    'model': model_path,
                    'shapes': input_tensor_shape,
                    'config': config
                    }
                try:
                    num_tasks = self.wp_task_queue.qsize()
                except NotImplementedError:
                    # Fallback mechanism for macOS
                    num_tasks = len(self.wp_task_queue._buffer)
                while num_tasks > 20:
                    time.sleep(1)
                    continue
                self.wp_task_queue.put(kernel_config)
                timestamp('gcWorker', 'generate and convert over')
                
        try:
            while True:
                config_dict = self.gc_task_queue.get()
                config = config_dict['config']
                model_path = config_dict['model_path']
                kernel_type = config_dict['kernel_type']
                
                # generate
                generator_framework = self.generator_framework
                
                save_model = False
                
                model, input_shape = self.kernelGenerator.generate_model_for_kernel(
                        kernel_type,
                        config,
                        save_path=model_path,
                        framework=generator_framework,
                        save_model=save_model)
                # convert
                model_name = os.path.basename(model_path).split('.')[0]
                converted_path = os.path.join(self.save_dir, model_name)
                if len(input_shape) == 2 and isinstance(input_shape[0], list):
                    input_shape = {
                        "input_0": tuple(input_shape[0]),
                        "input_1": tuple(input_shape[1])
                    }
                if self.backend == 'ncnn':
                    self.convertor.torch2ncnn(model=model,
                                            model_name=model_name,
                                            data_shape=input_shape,
                                            save_dir=self.save_dir,
                                            enable_int8=self.enable_int8,
                                            verbose=False)
                    
                    # remove source onnx file after converting
                    if self.gc_mode:
                        # pnnx itermediate files
                        os.remove(os.path.join(self.save_dir, (model_name+'_tracing.pnnx.onnx')))
                        os.remove(os.path.join(self.save_dir, (model_name+'_tracing.pnnx.param')))
                        os.remove(os.path.join(self.save_dir, (model_name+'_tracing.pnnx.bin')))
                        os.remove(os.path.join(self.save_dir, (model_name+'_tracing_pnnx.py')))
                        os.remove(os.path.join(self.save_dir, (model_name+'_tracing_ncnn.py')))
                        # onnx file
                        if save_model:
                            os.remove(model_path)
                        # itermediate pytorch file
                        os.remove(os.path.join(self.save_dir, (model_name+'_tracing.pt')))
                        
                elif self.backend == 'tflite':
                    self.convertor.torch2tflite(model=model,
                                            model_name=model_name,
                                            data_shape=input_shape,
                                            save_dir=self.save_dir,
                                            enable_fp32=self.enable_fp32,
                                            enable_fp16=self.enable_fp16,
                                            enable_int8=self.enable_int8,
                                            verbose=True)
                    option = '_float32' if self.enable_fp32 else '_float16'
                    converted_path = self.save_dir +'/'+ model_name + option
                    # remove source onnx file after converting
                    # if self.gc_mode:
                        # itermediate tf dir
                        # os.remove(os.path.join(self.save_dir, model_path))
                elif self.backend == 'trt':
                    self.convertor.torch2trt(model=model,
                                            model_name=model_name,
                                            data_shape=input_shape,
                                            save_dir=self.save_dir,
                                            enable_fp16=self.enable_fp16,
                                            verbose=False)
                    converted_path = os.path.join(self.save_dir, f"{model_name}.engine")
                elif self.backend == "cudnn":
                    self.convertor.torch2script(model=model,
                                                model_name=model_name,
                                                data_shape=input_shape,
                                                save_dir=self.save_dir)
                    converted_path = os.path.join(self.save_dir, f"{model_name}_tracing.pt")
                elif self.backend == "coreml":
                    self.convertor.torch2coreml(model=model,
                                                model_name=model_name,
                                                data_shape=input_shape,
                                                save_dir=self.save_dir)
                    converted_path = os.path.join(self.save_dir, f"{model_name}.mlmodel")
                else:
                    raise NotImplementedError
                kernel_config = {
                    'converted_model': converted_path,
                    'model': model_path,
                    'input_shape': input_shape,
                    'config': config
                }
                print(kernel_config)
                try:
                    num_tasks = self.wp_task_queue.qsize()
                except NotImplementedError:
                    # Fallback mechanism for macOS
                    num_tasks = len(self.wp_task_queue._buffer)
                
                while num_tasks > 20:
                    time.sleep(1)
                    continue
                self.wp_task_queue.put(kernel_config)
                timestamp('gcWorker', 'generate and convert over')
        except Exception as e:
            logger.error(e)
            os.makedirs(os.path.join(self.workspace, 'results_gcWorker'), exist_ok=True)
            open(os.path.join(error_save_path), 'a').write(f"{id}: {e}\n")
