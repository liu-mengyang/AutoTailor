# Extract the design space of given onnx DAG

import argparse
import os
import sys
import random

AUTOTAILOR_HOME = os.getenv('AUTOTAILOR_HOME')
sys.path.append(AUTOTAILOR_HOME)

import json
from loguru import logger
from pprint import pprint
import toml

import onnx
import onnx_graphsurgeon as gs

from autotailor.tir.tailor_ir import TailorIR
from autotailor.tir.graph_utils import shape_inference
import autotailor.tir.globvar as globvar
from autotailor.tailor.tailor import Tailor
from autotailor.predictor_factory.design_space_extractor import Extractor
from autotailor.predictor_factory.extract_utils import *

logger.add(globvar.log_file)


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("onnx_path", type=str)
    arg_parser.add_argument("supernet_config_path", type=str)
    arg_parser.add_argument("--workers", type=int, default=None)
    arg_parser.add_argument("--save_path", type=str, default="test_designspace.json")
    arg_parser.add_argument("--block_level", action="store_true", default=False)
    arg_parser.add_argument("--pruning_kernel_size", action="store_true", default=False)
    arg_parser.add_argument("--resolution", type=int, default=224)
    
    args = arg_parser.parse_args()
    return args

def main():
    args = parse_args()
    onnx_path = args.onnx_path
    supernet_config_dict = toml.load(args.supernet_config_path)
    
    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if args.workers:
        num_workers = args.workers
    else:
        num_workers = multiprocessing.cpu_count()
    # num_workers = 32
    # num_workers = 2 # debug
    
    multiprocessing.set_start_method('spawn', force=True)
    
    onnx_model = onnx.load(onnx_path)
    globvar.shape_dict = shape_inference(onnx_model)
    globvar.onnx_graph = gs.import_onnx(onnx_model)
    globvar.onnx_graph.fold_constants()
    globvar.onnx_graph.cleanup().toposort()
    
    tir = TailorIR(supernet_config_dict)
    tir.parse_graph(inp_shape=(1,3,args.resolution,args.resolution))
    
    tailor = Tailor(tir)
    
    code_list = []
    supercode = tailor.supercode
    print(supercode)
    
    glob_candidates, stage_candidates, block_comb_candidates = generate_candidates(tailor, args.pruning_kernel_size)
    
    extractors = []
    for i in range(num_workers):
        tir = TailorIR(supernet_config_dict)
        tir.parse_graph()
        
        tailor = Tailor(tir)
        extractor = Extractor(tailor)
        extractors.append(extractor)
    
    total_op_dict = {}
    glob_count = 0
    for glob_candidate in tqdm(glob_candidates):
        stage_block_candidates = list(it.product(stage_candidates, block_comb_candidates))
        # for stage_candidate in tqdm(stage_candidates):
        code = {**glob_candidate}
        # split block_combs
        worker_tasks = []
        # map
        for i in range(num_workers):
            worker_tasks.append((i,
                                 extractors[i],
                                 stage_block_candidates[i::num_workers],
                                 copy.deepcopy(code),
                                 args.block_level))
        pool = multiprocessing.Pool(processes=num_workers)
        op_dict_list = pool.starmap(extractor_worker, worker_tasks)
        pool.close()
        pool.join()
        
        # reduce
        for op_dict in op_dict_list:
            for op_type, op_features in op_dict.items():
                if op_type in total_op_dict:
                    total_op_dict[op_type] |= op_features
                else:
                    total_op_dict[op_type] = op_features # set of tuples
        glob_count += 1
        logger.info(f"Count: {glob_count} / {len(glob_candidates)}")
    
    # Expand for kernel size var
    if args.pruning_kernel_size:
        kernel_size_candidates = []
        total_op_dict_bak = copy.deepcopy(total_op_dict)
        for block_type, var_dict in tailor.block_vars.items():
            if "kernel_size" in var_dict:
                for k in var_dict["kernel_size"]:
                    if k not in kernel_size_candidates:
                        kernel_size_candidates.append(k)
        for op_type, op_features in total_op_dict_bak.items():
            if op_type == "Conv" or op_type == "DepthConv":
                # op expand
                for op_feature in op_features:
                    if len(kernel_size_candidates) > 0:
                        if op_feature[0] == max(kernel_size_candidates):
                            for k in kernel_size_candidates:
                                if k != max(kernel_size_candidates):
                                    feature = list(copy.deepcopy(op_feature))
                                    feature[0] = k
                                    total_op_dict[op_type].add(tuple(feature))
            print(f"# of {op_type}: {len(total_op_dict[op_type])}")
    
    for op_type, op_features in total_op_dict.items():
        print(f"# of {op_type}: {len(op_features)}")
    
    json.dump(
        total_op_dict,
        open(args.save_path, "w"),
        cls=NpEncoder,
        indent=4)


if __name__ == "__main__":
    main()