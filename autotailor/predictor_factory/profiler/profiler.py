import os
import torch
import torch.nn as nn
import multiprocessing as mp
import statistics
import time
import random
import ast

import coremltools as ct
import numpy as np

from loguru import logger
from infra.connector.connector import BackendConnector
from infra.convertor.convertor import Convertor
from infra.generator.generator import KernelGenerator
from .gc_worker import gcWorker
from .sp_worker import spWorker
from .utils import *


class ReshapeGelu(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        # x is gelu inputshape [1, shape[3], shape[1]*shape[2]]
        x = x.reshape(self.shape)
        x = nn.functional.gelu(x)
        return x


class Profiler(object):
    def __init__(self, backend_config, command_config):
        self.backend_config = backend_config
        self.command_config = command_config

        backend_config["backend_name"] = command_config["backend"]
        backend_config["backend_key"] = (
            f"{command_config['backend']}_{command_config['processor_type']}"
        )
        if backend_config["DEVICE_NAME"] == "localhost":
            tmp = backend_config["backend_key"]
            backend_config["backend_key"] = f"{tmp}_local"
        print(backend_config["backend_key"])
        self.connector = BackendConnector(backend_config)
        self.convertor = Convertor()

    def generate_and_convert(
        self,
        kernel_type,
        mark,
        kernel_configs,
        convert_configs=None,
        generator_framework="torch",
        verbose=False,
        workspace="workspace/",
        gc=True,
        enable_fp16=True,
        enable_int8=False,
    ):
        error_save_path = os.path.join(
            workspace, "results_gcWorker", "generate_error.log"
        )
        model_save_dir = os.path.join(os.getenv("MODEL_HOME"), mark)
        config_save_path = f"results/convert/{mark}_{kernel_type}.json"
        kernelGenerator = KernelGenerator()
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs("results/convert", exist_ok=True)

        try:
            # convert configs to profiled sample format, list to dict, and save
            if convert_configs is None:
                convert_configs = {}
                converted_index = 0
                for config in kernel_configs:
                    id = f"{converted_index:06d}"
                    obj = ast.literal_eval(config)
                    config_lst = list(obj)
                    config_dict = {}
                    config_dict["config"] = config_lst
                    model_path = os.path.join(
                        self.convertor.save_root,
                        "onnx",
                        ("_".join([kernel_type, mark, id]) + ".onnx"),
                    )
                    os.makedirs(
                        os.path.join(self.convertor.save_root, "onnx"), exist_ok=True
                    )
                    model_path = model_path.replace("-", "_")
                    config_dict["model_path"] = model_path
                    config_dict["kernel_type"] = kernel_type
                    convert_configs[id] = config_dict
                    converted_index += 1

            # save

            save_dir = os.path.join(
                self.convertor.save_root, "ncnn", ("_".join([kernel_type, mark]))
            )
            os.makedirs(save_dir, exist_ok=True)
            index = 0
            config_num = len(convert_configs)

            for id, config_dict in convert_configs.items():
                index += 1
                if "converted_model" in config_dict:
                    continue
                save_model = False
                model_path = config_dict["model_path"]
                if (
                    kernel_type == "Mul"
                    and config_dict["config"][0] == config_dict["config"][1]
                ):
                    config_dict["config"][0] = [
                        config_dict["config"][0],
                        config_dict["config"][0],
                    ]
                if kernel_type == "Gelu":
                    input_shape = [
                        config_dict["config"][0][0],
                        config_dict["config"][0][3],
                        config_dict["config"][0][1] * config_dict["config"][0][2],
                    ]
                    input_reshape = [
                        config_dict["config"][0][0],
                        config_dict["config"][0][3],
                        config_dict["config"][0][1],
                        config_dict["config"][0][2],
                    ]
                    # print(input_shape)
                    model = ReshapeGelu(input_reshape)
                else:
                    model, input_shape = kernelGenerator.generate_model_for_kernel(
                        kernel_type,
                        config_dict["config"],
                        save_path=model_path,
                        framework=generator_framework,
                        save_model=save_model,
                    )

                # convert

                model_name = os.path.basename(model_path).split(".")[0]
                converted_path = os.path.join(save_dir, model_name + ".ncnn")
                converted_path_in_table = os.path.join(
                    "models/ncnn", ("_".join([kernel_type, mark])), model_name + ".ncnn"
                )

                if len(input_shape) == 2 and isinstance(input_shape[0], list):
                    input_shape = {
                        "input_0": tuple(input_shape[0]),
                        "input_1": tuple(input_shape[1]),
                    }

                self.convertor.torch2ncnn(
                    model=model,
                    model_name=model_name,
                    data_shape=input_shape,
                    save_dir=save_dir,
                    enable_int8=enable_int8,
                    verbose=False,
                )
                logger.info(f"Convert {index} | {config_num}")
                # remove source onnx file after converting
                if gc:
                    # pnnx itermediate files
                    os.remove(
                        os.path.join(save_dir, (model_name + "_tracing.pnnx.onnx"))
                    )
                    os.remove(
                        os.path.join(save_dir, (model_name + "_tracing.pnnx.param"))
                    )
                    os.remove(
                        os.path.join(save_dir, (model_name + "_tracing.pnnx.bin"))
                    )
                    os.remove(os.path.join(save_dir, (model_name + "_tracing_pnnx.py")))
                    os.remove(os.path.join(save_dir, (model_name + "_tracing_ncnn.py")))
                    # onnx file
                    if save_model:
                        os.remove(model_path)
                    # itermediate pytorch file
                    os.remove(os.path.join(save_dir, (model_name + "_tracing.pt")))

                config_dict["input_shape"] = input_shape
                config_dict["converted_model"] = converted_path_in_table
                if index % 20 == 0:
                    save_profiled_results(
                        convert_configs, detail=True, save_path=config_save_path
                    )
            save_profiled_results(
                convert_configs, detail=True, save_path=config_save_path
            )

        except Exception as e:
            logger.error(e)
            os.makedirs(os.path.join(workspace, "results_gcWorker"), exist_ok=True)
            open(os.path.join(error_save_path), "a").write(f"{id}: {e}\n")

    def profile(
        self,
        kernel_type,
        mark,
        kernel_configs,
        enable_pipeline=True,
        enable_gc=True,
        enable_latency_constraint=False,
        latency_constraint=60000,
        existed_profiled_results=None,
        save_results=True,
    ):
        model_save_dir = os.path.join(os.getenv("MODEL_HOME"), mark)
        profile_save_path = f"results/{mark}_{kernel_type}.json"

        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs("results", exist_ok=True)

        model_configs = []
        profiled_model_configs = {}

        # resume profiling
        if existed_profiled_results:
            profiled_model_configs = existed_profiled_results

        if enable_pipeline:
            self.gc_task_queue = mp.Queue()
            self.wp_task_queue = mp.Queue()
            self.out_queue = mp.Queue()

            gc_worker = gcWorker(
                self.gc_task_queue,
                self.wp_task_queue,
                self.backend_config,
                self.command_config,
                model_save_dir,
                self.convertor,
            )

            gc_worker.start()

            sp_worker = spWorker(
                self.wp_task_queue,
                self.out_queue,
                self.backend_config,
                self.command_config,
                self.connector,
                enable_latency_constraint=enable_latency_constraint,
                latency_constraint=latency_constraint,
            )

            sp_worker.start()

        num_kernels = len(kernel_configs)
        profiled_index = 0

        if enable_pipeline:
            converted_index = 0
            while profiled_index < num_kernels:
                try:
                    num_tasks = self.gc_task_queue.qsize()
                except NotImplementedError:
                    # Fallback mechanism for macOS
                    num_tasks = len(self.gc_task_queue._buffer)
                if converted_index < num_kernels and num_tasks <= 20:
                    # avoid queue size surpass the limitation of Python
                    id = f"{converted_index:06d}"
                    if id in profiled_model_configs:
                        # existed profiled, skip
                        converted_index += 1
                        profiled_index += 1
                        logger.info("Skip profiled model: " + id)
                        continue
                    config_dict = {}
                    config_dict["config"] = kernel_configs[converted_index]
                    # generate and convert
                    model_path = os.path.join(
                        self.convertor.save_root,
                        "onnx",
                        ("_".join([kernel_type, mark, id]) + ".onnx"),
                    )
                    os.makedirs(
                        os.path.join(self.convertor.save_root, "onnx"), exist_ok=True
                    )
                    model_path = model_path.replace("-", "_")
                    # if backend == 'tflite':
                    #    model_path = os.path.join(self.convertor.save_root, "tf", ("_".join([kernel_type, mark, id])))
                    config_dict["model_path"] = model_path
                    config_dict["kernel_type"] = kernel_type

                    self.gc_task_queue.put(config_dict)
                    converted_index += 1
                    logger.info(
                        "Generating: " + str(converted_index) + " | " + str(num_kernels)
                    )

                # check out queue
                if not self.out_queue.empty():
                    id = f"{profiled_index:06d}"
                    kernel_config = self.out_queue.get()
                    timestamp("master", "consume 1 task")
                    profiled_model_configs[id] = kernel_config
                    profiled_index += 1
                    logger.info(
                        "Returning: " + str(profiled_index) + " | " + str(num_kernels)
                    )

                    if profiled_index > 0 and profiled_index % 5 == 0 and save_results:
                        # save configs
                        save_profiled_results(
                            profiled_model_configs,
                            detail=True,
                            save_path=profile_save_path,
                        )

        if save_results:
            save_profiled_results(
                profiled_model_configs, detail=True, save_path=profile_save_path
            )

        if enable_pipeline:
            gc_worker.terminate()
            sp_worker.terminate()
        return profiled_model_configs

    def profile_batch_models(self, model_configs, op_type=None, mode=None):
        # send
        for key, model_config in model_configs.items():
            model_name = model_config["converted_model"].split(".ncnn")[0]
            ncnn_param_path = model_name + "_tracing.ncnn.param"
            ncnn_bin_path = model_name + "_tracing.ncnn.bin"
            if self.backend_config["DEVICE_NAME"] != "localhost":
                self.connector.send_model(ncnn_param_path)
                self.connector.send_model(ncnn_bin_path)
            # shape process
            model_name = model_name.split("/")[-1]
            model_name = f"{model_name}_tracing.ncnn"
            if self.backend_config["DEVICE_NAME"] == "localhost":
                model_path = (
                    model_config["converted_model"].split(".ncnn")[0] + "_tracing.ncnn"
                )
            else:
                model_path = os.path.join(self.connector.model_dir, model_name)
            configs = self.command_config
            if op_type == "AveragePool" or op_type == "MaxPool":
                input_shape = model_config["config"][2]
            elif (
                op_type == "Linear"
                or op_type == "LinearMatMul"
                or op_type == "GlobalAveragePool"
                or op_type == "Transpose"
                or op_type == "Reshape"
                or op_type == "Softmax"
                or op_type == "LayerNormalization"
            ):
                input_shape = model_config["config"][1]
            elif (
                op_type == "Add"
                or op_type == "Relu"
                or op_type == "ReduceMean"
                or op_type == "HardSwish"
                or op_type == "HardSigmoid"
                or op_type == "BiasAdd"
                or op_type == "Flatten"
                or op_type == "BatchNormalization"
            ):
                input_shape = model_config["config"][0]
            elif op_type == "Mul" or op_type == "MatMul" or op_type == "ScaleMul":
                if model_config["config"][0] == model_config["config"][1]:
                    input_shape = {
                        "input_1": model_config["config"][0],
                        "input_2": model_config["config"][0],
                    }
                else:
                    input_shape = {
                        "input_1": model_config["config"][0][0],
                        "input_2": model_config["config"][0][1],
                    }
            elif op_type == "Gelu":
                config = model_config["config"][0]
                input_shape = [config[3], config[1] * config[2]]
            else:
                input_shape = model_config["config"][4]
            if isinstance(input_shape, dict):
                str_shapes = ""
                print(input_shape)
                for k, v in input_shape.items():
                    if op_type == "MatMul":
                        if len(v) == 4:
                            b, c, h, w = v
                            if len(str_shapes) == 0:
                                str_shapes += f"[{c},{h},{w}]"
                            else:
                                str_shapes += f",[{c},{w},{h}]"
                        else:
                            raise NotImplementedError
                    elif len(v) == 4:
                        b, c, h, w = v
                        if len(str_shapes) == 0:
                            str_shapes += f"[{h},{w},{c}]"
                        else:
                            str_shapes += f",[{h},{w},{c}]"
                    elif len(v) == 3:
                        b, c, hw = v
                        if len(str_shapes) == 0:
                            str_shapes += f"[{hw},{c}]"
                        else:
                            str_shapes += f",[{hw},{c}]"
                    elif len(v) == 2:
                        b, c = v
                        if len(str_shapes) == 0:
                            str_shapes += f"[{c}]"
                        else:
                            str_shapes += f",[{c}]"
                configs["shape"] = str_shapes
            elif (
                op_type == "BiasAdd"
                or op_type == "Transpose"
                or op_type == "Reshape"
                or op_type == "LinearMatMul"
            ):
                if len(input_shape) == 4:
                    self.command_config["HW0"] = input_shape[2]
                    self.command_config["HW1"] = input_shape[3]
                    self.command_config["CIN"] = input_shape[1]

                    configs["shape"] = (
                        f"[{input_shape[3]},{input_shape[2]},{input_shape[1]},{input_shape[0]}]"
                    )
                elif len(input_shape) == 3:
                    # FIXME
                    if op_type == "LinearMatMul":
                        self.command_config["HW0"] = input_shape[0]
                        self.command_config["HW1"] = input_shape[1]
                        self.command_config["CIN"] = input_shape[2]
                        configs["shape"] = (
                            f"{input_shape[0]},{input_shape[1]},{input_shape[2]}"
                        )
                    else:
                        self.command_config["HW0"] = input_shape[1]
                        self.command_config["HW1"] = input_shape[2]
                        self.command_config["CIN"] = input_shape[0]

                        configs["shape"] = (
                            f"[{input_shape[1]},{input_shape[2]},{input_shape[0]}]"
                        )
            elif len(input_shape) == 4:
                self.command_config["HW0"] = input_shape[2]
                self.command_config["HW1"] = input_shape[3]
                self.command_config["CIN"] = input_shape[1]

                configs["shape"] = (
                    f"[{input_shape[2]},{input_shape[3]},{input_shape[1]}]"
                )
            else:
                configs["shape"] = f"[{','.join(map(str, input_shape))}]"

            res = self.connector.profile(
                model_path=model_path,
                configs=configs,
                enable_latency_constraint=False,
                verbose=True,
            )
            latency = self.connector.parse(res, mode=mode)["latency"]
            print(latency)
            model_config["latency"] = latency
        return model_configs

    def profile_model(
        self, model, input_shape, model_name=None, op_type=None, ret_convert_time=False
    ):
        model = model.eval()
        if model_name == None:
            model_name = "test"
        with torch.no_grad():
            # convert
            convert_st = time.time()
            os.makedirs("models/test", exist_ok=True)
            if self.backend_config["backend_name"] == "ncnn":
                self.convertor.torch2ncnn(
                    model=model,
                    model_name=model_name,
                    data_shape=input_shape,
                    save_dir="models/test",
                    verbose=False,
                )
            elif self.backend_config["backend_name"] == "cudnn":
                self.convertor.torch2script(
                    model=model,
                    model_name="test",
                    data_shape=input_shape,
                    save_dir="models/test",
                )
            elif self.backend_config["backend_name"] == "coreml":
                self.convertor.torch2coreml(
                    model=model,
                    model_name="test",
                    data_shape=input_shape,
                    save_dir="models/test",
                )
            elif self.backend_config["backend_name"] == "tflite":
                self.convertor.torch2tflite(
                    model=model,
                    model_name="test",
                    data_shape=input_shape,
                    save_dir="models/test",
                    enable_fp32=self.command_config["use_fp32"],
                    enable_fp16=self.command_config["use_fp16"],
                    enable_int8=self.command_config["use_int8"],
                    verbose=True,
                )
            else:
                raise NotImplementedError
            convert_ed = time.time()
            config_dict = {
                "converted_model": f"models/test/{model_name}",
                "model": model_name,
                "shapes": input_shape,
            }

            # send
            try:
                if self.backend_config["backend_name"] == "ncnn":
                    base_dir = config_dict["converted_model"]
                    base_dir = base_dir.replace("-", "_")
                    ncnn_param_path = base_dir + "_tracing.ncnn.param"
                    ncnn_bin_path = base_dir + "_tracing.ncnn.bin"
                    if self.backend_config["DEVICE_NAME"] != "localhost":
                        self.connector.send_model(ncnn_param_path)
                        self.connector.send_model(ncnn_bin_path)
                if self.backend_config["backend_name"] == "tflite":
                    tag = "_float16" if self.command_config["use_fp16"] else "_float32"
                    base_dir = config_dict["converted_model"]
                    tflite_model_path = base_dir + f"{tag}.tflite"
                    self.connector.send_model(tflite_model_path)

            except Exception as e:
                raise e

            # profile
            if self.backend_config["backend_name"] == "ncnn":
                model_name = f"{model_name}_tracing.ncnn"
                if self.backend_config["DEVICE_NAME"] == "localhost":
                    model_path = os.path.join("models/test", model_name)
                else:
                    model_path = os.path.join(self.connector.model_dir, model_name)
                configs = self.command_config
                if isinstance(input_shape, dict):
                    str_shapes = ""
                    print(input_shape)
                    for k, v in input_shape.items():
                        if op_type == "MatMul":
                            if len(v) == 4:
                                b, c, h, w = v
                                if len(str_shapes) == 0:
                                    str_shapes += f"[{c},{h},{w}]"
                                else:
                                    str_shapes += f",[{c},{w},{h}]"
                            else:
                                raise NotImplementedError
                        elif len(v) == 4:
                            b, c, h, w = v
                            if len(str_shapes) == 0:
                                str_shapes += f"[{h},{w},{c}]"
                            else:
                                str_shapes += f",[{h},{w},{c}]"
                        elif len(v) == 3:
                            b, c, hw = v
                            if len(str_shapes) == 0:
                                str_shapes += f"[{hw},{c}]"
                            else:
                                str_shapes += f",[{hw},{c}]"
                        elif len(v) == 2:
                            b, c = v
                            if len(str_shapes) == 0:
                                str_shapes += f"[{c}]"
                            else:
                                str_shapes += f",[{c}]"
                    configs["shape"] = str_shapes
                elif (
                    op_type == "BiasAdd"
                    or op_type == "Transpose"
                    or op_type == "Reshape"
                ):
                    if len(input_shape) == 4:
                        self.command_config["HW0"] = input_shape[2]
                        self.command_config["HW1"] = input_shape[3]
                        self.command_config["CIN"] = input_shape[1]

                        configs["shape"] = (
                            f"[{input_shape[2]},{input_shape[3]},{input_shape[1]},{input_shape[0]}]"
                        )
                        # if op_type == "Gelu":
                        #     self.command_config["HW0"] = input_shape[2]
                        #     self.command_config["HW1"] = input_shape[3]
                        #     self.command_config["CIN"] = input_shape[1]

                        #     configs['shape'] = f'[{input_shape[2]},{input_shape[3]},{input_shape[1]}]'
                    elif len(input_shape) == 3:
                        self.command_config["HW0"] = input_shape[1]
                        self.command_config["HW1"] = input_shape[2]
                        self.command_config["CIN"] = input_shape[0]

                        configs["shape"] = (
                            f"[{input_shape[1]},{input_shape[2]},{input_shape[0]}]"
                        )
                elif op_type == "Gelu":
                    self.command_config["HW0"] = input_shape[3]
                    self.command_config["HW1"] = input_shape[1]
                    self.command_config["CIN"] = input_shape[2]
                    print(input_shape)
                    configs["shape"] = (
                        f"[{input_shape[3]},{input_shape[1]},{input_shape[2]}]"
                    )
                    print(configs["shape"])
                elif len(input_shape) == 4:
                    self.command_config["HW0"] = input_shape[2]
                    self.command_config["HW1"] = input_shape[3]
                    self.command_config["CIN"] = input_shape[1]

                    configs["shape"] = (
                        f"[{input_shape[2]},{input_shape[3]},{input_shape[1]}]"
                    )
                else:
                    configs["shape"] = f"[{','.join(map(str, input_shape))}]"
                print(configs["shape"])
                res = self.connector.profile(
                    model_path=model_path,
                    configs=configs,
                    enable_latency_constraint=False,
                    verbose=True,
                )
                print(res)
                if "Segmentation fault" in res:
                    return -1
                latency = self.connector.parse(res)["latency"]
            elif self.backend_config["backend_name"] == "cudnn":
                num_runtimes = self.command_config["loop_count"]
                trace_model_path = os.path.join("models/test", "test_tracing.pt")
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

                latency = f"{average_inference_time * 1000} +- {std_dev_inference_time * 1000}"
            elif self.backend_config["backend_name"] == "coreml":
                num_runtimes = self.command_config["loop_count"]
                model_path = os.path.join("models/test", "test.mlmodel")

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

                latency = f"{average_inference_time * 1000} +- {std_dev_inference_time * 1000}"

            elif self.backend_config["backend_name"] == "tflite":
                tag = "_float16" if self.command_config["use_fp16"] else "_float32"
                model_name = f"test{tag}.tflite"
                model_path = os.path.join(self.connector.model_dir, model_name)
                configs = self.command_config
                if isinstance(input_shape, dict):
                    str_shapes = ""
                    for k, v in input_shape.items():
                        if len(v) == 4:
                            b, c, h, w = v
                            if len(str_shapes) == 0:
                                str_shapes += f"[{h},{w},{c}]"
                            else:
                                str_shapes += f",[{h},{w},{c}]"
                        elif len(v) == 3:
                            b, c, hw = v
                            if len(str_shapes) == 0:
                                str_shapes += f"[{hw}, {c}]"
                            else:
                                str_shapes += f",[{hw}, {c}]"
                        elif len(v) == 2:
                            b, c = v
                            if len(str_shapes) == 0:
                                str_shapes += f"[{c}]"
                            else:
                                str_shapes += f",[{c}]"
                    configs["shape"] = str_shapes
                elif len(input_shape) == 4:
                    self.command_config["HW0"] = input_shape[2]
                    self.command_config["HW1"] = input_shape[3]
                    self.command_config["CIN"] = input_shape[1]

                    configs["shape"] = (
                        f"[{input_shape[2]},{input_shape[3]},{input_shape[1]}]"
                    )
                else:
                    configs["shape"] = f"[{','.join(map(str, input_shape))}]"
                res = self.connector.profile(
                    model_path=model_path,
                    configs=configs,
                    enable_latency_constraint=False,
                    verbose=True,
                )
                latency = self.connector.parse(res)["latency"]

            if ret_convert_time:
                convert_time = convert_ed - convert_st
                return latency, convert_time

            return latency
