import argparse
import json
import operator
import os
from pprint import pprint
import re
import sys
import time

import onnx
import pandas as pd
from tqdm import tqdm
import toml
from loguru import logger
import onnx_graphsurgeon as gs

AUTOTAILOR_HOME = os.getenv('AUTOTAILOR_HOME')
sys.path.append(AUTOTAILOR_HOME)

from autotailor.predictor_factory.profiler.profiler import Profiler
from infra.generator.nn_modules.torch_networks.blocks import TorchBlock, ConvBlock
from infra.generator.nn_modules.torch_networks.operators import Conv

class PreConv(TorchBlock):
    def __init__(self, config, conv_op):
        super().__init__(config)
        self.input_shape = config[4]
        pre_conv_op = Conv(config)
        self.pre_conv_op = pre_conv_op.get_model()
        self.conv_op = conv_op
        
    def get_model(self):
        return self.build_model([self.pre_conv_op, self.conv_op])

class DominantOPDetector:
    def __init__(self,
                 backend_config,
                 command_config,
                 dominance_constraint=0.1,
                 debug=False):
        self.dominance_constraint=dominance_constraint
        self.skip_list = ['Split']
        self.debug_list = ['Gelu']
        self.debug = debug
        self.profiler = Profiler(backend_config, command_config)

    def set_dominance_constraint(self, dominance_constraint):
        self.dominance_constraint = dominance_constraint

    def set_skip_list(self, skip_list):
        self.skip_list = skip_list
    
    def detect(self, model_tir, model_name, save_dir="configs/dominant_op"):
        model = model_tir.build()
        # profile full model
        if self.debug == False:
            model_latency_str = str(self.profiler.profile_model(model, model_tir.inp_shape, model_name=model_name))
            model_latency = 0
            if "+-" in model_latency_str:
                model_latency = float(model_latency_str.split(" +- ")[0])
            
        
        op_dict = model_tir.get_ops(drop_dup=False, disable_extracting=True)
        detect_buffer_dict = {}
        dominant_operators = []
        print("Operators:")
        pprint(list(op_dict.keys()))
        
        for op_type, op_list in tqdm(op_dict.items()):
            ### DEBUG
            # if op_type != "Linear":
            #     continue
            ###
            if op_type in self.skip_list:
                continue
            if self.debug and op_type not in self.debug_list:
                continue
            for op in op_list:
                if "fused" in op.features:
                    print(f"Pass {op.name}")
                    continue
                index = 0
                model = op.build()
                input_shape = op.features["in_shape"]
                if self.debug:
                    print(model)
                    print(input_shape)
                if op_type == "iConv" or op_type == "iDepthConv":
                    print("in")
                    config = [1, 1, 1, False, list(input_shape), list(input_shape), input_shape[1], input_shape[1]]
                    print(config)
                    model_block = PreConv(config=config, conv_op=model)
                    model_with_pre_conv = model_block.get_model()
                    pre_block = Conv(config=config)
                    model_pre_conv = pre_block.get_model()
                    latency_str = str(self.profiler.profile_model(model_with_pre_conv, input_shape))
                    latency_all = float(latency_str.split(" +- ")[0])
                    latency_str = str(self.profiler.profile_model(model, input_shape))
                    latency_single = float(latency_str.split(" +- ")[0])
                    latency_str = str(self.profiler.profile_model(model_pre_conv, input_shape))
                    latency_pre = float(latency_str.split(" +- ")[0])
                    offset = latency_pre + latency_single - latency_all
                    latency = latency_single - offset
                    print(f"offset : {offset}")
                    print(f"true latency: {latency}")
                elif op_type == "MatMul":
                    if isinstance(input_shape[0], list) or isinstance(input_shape[0], tuple):
                        input_shape = {
                                "input_1": input_shape[0],
                                "input_2": input_shape[1]
                            }
                    latency_str = str(self.profiler.profile_model(model, input_shape, op_type=op_type))
                    latency = 0
                    if "+-" in latency_str:
                        latency = float(latency_str.split(" +- ")[0])
                    elif "-1" in latency_str:
                        logger.info(f"unsupport matmul {input_shape}")    
                    else:
                        raise ValueError(f"Backend not support for kernel {op.info}")
                else:
                    if op_type == "Add":
                        input_shape = {
                            "input_1": input_shape,
                            "input_2": input_shape
                        }
                    if op_type == "Mul":
                        if isinstance(input_shape[0], list) or isinstance(input_shape[0], tuple):
                            input_shape = {
                                "input_1": input_shape[0],
                                "input_2": input_shape[1]
                            }
                        
                    latency_str = str(self.profiler.profile_model(model, input_shape, model_name=model_name, op_type=op_type))
                    print(f"Latency of {op.name}: {latency_str}")
                    latency = 0
                    if "+-" in latency_str:
                        latency = float(latency_str.split(" +- ")[0])
                    elif "-1" in latency_str:
                        print(model)
                        logger.info(f"unsupport {op_type} {input_shape}")    
                    else:
                        raise ValueError(f"Backend not support for kernel {op.info}")
                logger.info(f"{op_type} {index} cost {latency} ms, input shape {input_shape}")
                if op_type not in detect_buffer_dict:
                    detect_buffer_dict[op_type] = latency
                else:
                    detect_buffer_dict[op_type] += latency
        pprint(detect_buffer_dict)
        sum_of_latency = sum(list(detect_buffer_dict.values()))
        
        print(f"Direct latency: {model_latency}; Sum of latencies of operators: {sum_of_latency}; Absolute bias: {sum_of_latency-model_latency}; Related bias: {abs(sum_of_latency-model_latency)/model_latency}")
        
        for op_type, op_latency in sorted(detect_buffer_dict.items(), key=operator.itemgetter(1), reverse=True):
            p = op_latency / sum_of_latency
            if p >= self.dominance_constraint:
                dominant_operators.append(op_type)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, model_name + ".json")
        content = {"dominant_operators":dominant_operators}
        with open(file_path,'w') as fp:
            json.dump(content, fp=fp, indent=4)
        
        return detect_buffer_dict, dominant_operators