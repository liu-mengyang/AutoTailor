# Parse given OFA onnx DAG into TailorIR

import argparse
import os
from pprint import pprint
import sys

AUTOTAILOR_HOME = os.getenv('AUTOTAILOR_HOME')
sys.path.append(AUTOTAILOR_HOME)

import toml

import onnx
import onnx_graphsurgeon as gs

from autotailor.tir.blocks import SingleOp
from autotailor.tir.tailor_ir import TailorIR
from autotailor.tailor.tailor import Tailor
from autotailor.tir.graph_utils import shape_inference
import autotailor.tir.globvar as globvar


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("onnx_path", type=str)
    arg_parser.add_argument("supernet_config_path", type=str)
    arg_parser.add_argument("--resolution", type=int, default=224)
    arg_parser.add_argument("--input_shape", type=int, nargs=4, default=(1, 3, 224, 224),
                            help="Input shape for the model, default is (1, 3, 224, 224)")
    
    args = arg_parser.parse_args()
    return args


def main():
    args = parse_args()
    model_path = args.onnx_path
    supernet_config_dict = toml.load(args.supernet_config_path)
    
    onnx_model = onnx.load(model_path)
    if "vit" in model_path:
        onnx_model.ir_version = 10
        onnx_model.opset_import[0].version = 20

    globvar.shape_dict = shape_inference(onnx_model, input_shape=args.input_shape)
    globvar.onnx_graph = gs.import_onnx(onnx_model)
    globvar.onnx_graph.fold_constants()
    globvar.onnx_graph.cleanup().toposort()
    
    tir = TailorIR(supernet_config_dict)
    tir.parse_graph(inp_shape=(1,3,args.resolution,args.resolution))
    
    tailor = Tailor(tir)
    
    for stage_id, stage in tir.stages.items():
        print(f"Stage {stage_id}: {stage.features}")
        for block_id, block in stage.flow.items():
            print(f"Block {block_id}: {block.features}")
            if "main_path" in block.features:
                for operator_id, operator in enumerate(block.main_path):
                    print(f"Operator {operator_id}: {operator.features}")
    # # analyze components
    # block_op_dict = {}
    # for stage_id, stage in tir.stages.items():
    #     for block_id, block in stage.flow.items():
    #         if block.type not in block_op_dict:
    #             block_op_dict[block.type] = []
    #         if isinstance(block, SingleOp):
    #             continue
    #         else:
    #             ops = block.get_ops()
    #             for op in ops:
    #                 if op.type not in block_op_dict[block.type]:
    #                     block_op_dict[block.type].append(op.type)
    # pprint(block_op_dict)
    print("Supernet:")
    pprint(tailor.supercode)
    
if __name__ == "__main__":
    main()