import argparse
import copy
import json
import math
import os
import os.path as osp
import sys

AUTOTAILOR_HOME = os.environ["AUTOTAILOR_HOME"]
sys.path.append(AUTOTAILOR_HOME)

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import toml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms



from autotailor.tir.tailor_ir import TailorIR
from autotailor.tir.graph_utils import shape_inference
from autotailor.tailor.tailor import Tailor
from autotailor.tailor.adaptor import Adaptor
import autotailor.tir.globvar as globvar
from autotailor.tir.blocks import BottleneckBlock, BottleneckResidualBlock, DepthConvOp



def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("onnx_path", type=str)
    arg_parser.add_argument("supernet_config_path", type=str)
    arg_parser.add_argument("predictor_path", type=str, help="LUT or MLP weight dict path")
    arg_parser.add_argument("--method", type=str, default="beam_evolution")
    arg_parser.add_argument("--efficiency_constraint", type=float, default=30)
    arg_parser.add_argument("--efficiency_metric", type=str, default="lut")
    arg_parser.add_argument("--accuracy_metric", type=str, default="flops")
    arg_parser.add_argument("--sensitivity_weight_path", type=str, default=None)
    arg_parser.add_argument("--log_path", type=str, default=None)
    arg_parser.add_argument("--mark", type=str, default=None)
    
    args = arg_parser.parse_args()
    return args


def main():
    args = parse_args()
    
    if args.log_path:
        logger.add(args.log_path)
    
    onnx_path = args.onnx_path
    supernet_config_dict = toml.load(args.supernet_config_path)

    onnx_model = onnx.load(onnx_path)
    globvar.shape_dict = shape_inference(onnx_model)
    globvar.onnx_graph = gs.import_onnx(onnx_model)
    globvar.onnx_graph.fold_constants()
    globvar.onnx_graph.cleanup().toposort()

    tir = TailorIR(supernet_config_dict)
    tir.parse_graph()

    tailor = Tailor(tir)
    
    adaptor = Adaptor(tailor)
    
    data_dict = json.load(open(args.predictor_path))
    
    best_code, best_acc, best_lat, acc_list_random = adaptor.adapt(efficiency_metric=args.efficiency_metric,
                                                                    efficiency_constraint=args.efficiency_constraint,
                                                                    mode=args.method,
                                                                    data_dict=data_dict,
                                                                    accuracy_metric=args.accuracy_metric,
                                                                    trans_weight_path=args.sensitivity_weight_path,
                                                                    )
    
    # export subnet
    if args.mark:
        tailor.transform(best_code)
        tailor.tir.build()
        torch_nn = tailor.tir.torch_nn
        torch_nn.eval()
        torch_nn = torch_nn.cpu()
        export_data = torch.randn((1,3,))
        
        from infra.convertor.convertor import Convertor
        convertor = Convertor()
        
        convertor.torch2ncnn(model=torch_nn,
                             model_name=args.mark,
                             data_shape=(1,3,best_code["resolution"],best_code["resolution"]),
                             save_dir='./')
    
if __name__ == "__main__":
    main()