import argparse
import json
import math
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

import onnx_graphsurgeon as gs

AUTOTAILOR_HOME = os.getenv('AUTOTAILOR_HOME')
sys.path.append(AUTOTAILOR_HOME)

from infra.convertor.convertor import Convertor
from autotailor.predictor_factory.profiler.profiler import Profiler
from autotailor.tir.blocks.single_op import SingleOp
from autotailor.tir.blocks.residual import ResidualBlock
from autotailor.tir.graph_utils import extract_sub_graph

class FusionRuleDetector:
    def __init__(self,
                 backend_config,
                 command_config,
                 fusion_strictness=0.8,
                 combine_strictness=0.5,
                 ):
        self.fusion_strictness = fusion_strictness
        self.combine_strictness = combine_strictness
        self.enable_scaling = True
        self.fusion_rules = {'combine':{}, 'eliminate':{}}
        self.detailed_fusion_rules = {'combine':{}, 'eliminate':{}}
        self.split_rules = []
        self.skip_list = ["Add", "Mul"]
        self.convertor = Convertor()
        self.profiler = Profiler(backend_config, command_config)

    def set_fusion_strictness(self, fusion_strictness):
        self.fusion_strictness = fusion_strictness

    def set_combine_strictness(self, combine_strictness):
        self.combine_strictness = combine_strictness

    def set_skip_list(self, skip_list):
        self.skip_list = skip_list
    
    def detect(self, model_tir, model_name, save_dir="configs/fusion_rules"):
        self._detect_on_tir(model_tir)
        
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, model_name + "_fusion_rules.json")
        content = {
            "fusion_rules": self.fusion_rules,
            "detailed_fusion_rules": self.detailed_fusion_rules
            }
        with open(file_path,'w') as fp:
            json.dump(content, fp=fp, indent=4)
        
        return self.fusion_rules

    def _detect_on_tir(self, model_tir):
            
        def detect_on_path(path):
            compared_nodes = []
            compared_nodes_type = []
            for block in path:
                if not isinstance(block, SingleOp):
                    # iterate into new path
                    if len(compared_nodes) >= 2:
                        compared_nodes, compared_nodes_type = self._sequential_detect(compared_nodes, compared_nodes_type)
                    # clear buffer
                    compared_nodes = []
                    compared_nodes_type = []
                    if block.paths is not None:
                        for path_key, block_path in block.paths.items():
                            detect_on_path(block_path)
                    elif block.flow is not None:
                        detect_on_path(block.flow)
                    else:
                        raise NotImplementedError
                    compared_nodes = []
                    compared_nodes_type = []
                else:
                    # update buffer
                    compared_nodes.append(block.name)
                    compared_nodes_type.append(block.type)
                    if len(compared_nodes) >= 2:
                        compared_nodes, compared_nodes_type = self._sequential_detect(compared_nodes, compared_nodes_type)
                    
        prev_type = None
        for stage_id, stage in model_tir.stages.items():
            stage_path = []
            for block_id, block in stage.flow.items():
                stage_path.append(block)
            detect_on_path(stage_path)
                    

    def _sequential_detect(self, compared_nodes, compared_nodes_type):
        str_candidate = '&'.join(compared_nodes)
        str_candidate_type = '&'.join(compared_nodes_type)
        if str_candidate in self.detailed_fusion_rules["eliminate"] or str_candidate in self.detailed_fusion_rules["combine"]:
            return compared_nodes, compared_nodes_type
        # if str_candidate not in self.fusion_counts:
        #     self.fusion_counts[str_candidate] = 0
        #     self.split_lives[str_candidate] = 0
        # filter out detected kernels
        # if str_candidate in list(self.fusion_rules["eliminate"]) or str_candidate in self.fusion_rules["combine"]:
        #     return compared_nodes, [str_candidate]
        # elif str_candidate in self.split_rules:
        #     return [], []

        # if self.backend=='ncnn' and self.processor_type=='gpu':
        #     if 'ReduceMean' in compared_nodes_type:
        #         # NCNN bug with single reducemean on GPU
        #         return [], []
        # elif self.backend=='tflite':
        #     if 'Div' in compared_nodes_type:
        #         # TFLite bug with single div on GPU
        #         return [], []
        # FIXME: compromise for elasticvit
        if "Split" in compared_nodes_type:
            return [], []
        elif "Add" in compared_nodes_type:
            return [], []
        elif "Mul" in compared_nodes_type:
            return [], []
        elif "Concat" in compared_nodes_type:
            return [], []
        print(f"Existing fusion rules: {self.fusion_rules}")
        print(compared_nodes)
        fuse_type, is_fused, max_node_id = self._is_fusable(compared_nodes[:-1],
                                                      [compared_nodes[-1]],
                                                      alpha=self.fusion_strictness,
                                                      enable_scaling=self.enable_scaling)
        if is_fused:
            self.detailed_fusion_rules[fuse_type][str_candidate] = max_node_id
            self.fusion_rules[fuse_type][str_candidate_type] = max_node_id
            
            print(f"New fusion rule: {str_candidate}")
            return compared_nodes, [str_candidate_type]
        else:
            self.split_rules.append(str_candidate)
            print(f"New split rule: {str_candidate}")
            return [compared_nodes[-1]], [compared_nodes_type[-1]]
    
    def _is_fusable(self, start_node, next_node, alpha=0.5, beta=10, gamma=0.1, enable_scaling=True):
        save_dir = 'weights'
        model_name = '_sub_graph_model'
        onnx_path = os.path.join(save_dir, model_name+'.onnx')
        
        sub_graphs = [start_node+next_node,
                        start_node,
                        next_node]
        latency_lst = []
        var_lst = []

        for sub_graph in sub_graphs:
            onnx_sub_graph = extract_sub_graph(sub_graph)
            sub_graph_model = gs.export_onnx(onnx_sub_graph)
            os.makedirs(os.path.join(os.getenv('AUTOTAILOR_HOME'), save_dir), exist_ok=True)
            onnx.save(sub_graph_model, onnx_path)
            # got shapes of inputs
            inputs_shape_dict = {}
            for i in sub_graph_model.graph.input:
                if len(re.findall("(input\d)", i.name))!=0:
                    inputs_shape_dict[i.name] = [x.dim_value for x in i.type.tensor_type.shape.dim]
            model = self.convertor.onnx2torch(onnx_path, model_name=model_name, save_dir=save_dir)
            latency = 0.0
            var = 0,0
            latency_str = str(self.profiler.profile_model(model, inputs_shape_dict))
            if "+-" in latency_str:
                latency = float(latency_str.split(" +- ")[0])
                var = float(latency_str.split(" +- ")[1])
            else:
                try:
                    latency = float(latency_str)
                except:
                    raise ValueError(f"Backend not support for kernel {model}")
            print(f"Latency of {str(sub_graph)} : {latency} +- {var}")
            latency_lst.append(latency)
            var_lst.append(var)
        
        
        if len(start_node) > 1:
            node_names = []
            for node in start_node:
                node_names.append(node)
                start_node_key = "&".join(node_names)
                    
        max_node_id = None
        if latency_lst[1] > latency_lst[2]:
            if len(start_node) > 1:
                if start_node_key in self.detailed_fusion_rules["eliminate"]:
                    max_node_id = self.detailed_fusion_rules["eliminate"][start_node_key]
                elif start_node_key in self.detailed_fusion_rules["combine"]:
                    max_node_id = self.detailed_fusion_rules["combine"][start_node_key]
                else:
                    raise KeyError(f"{start_node_key} is an unexpected key")
            else:
                max_node_id = 0
        else:
            max_node_id = len(start_node)
            
        if enable_scaling:
            is_fused = False
            if min(latency_lst[1],  latency_lst[2]) < beta*sum(var_lst) and max(latency_lst[1],  latency_lst[2]) > beta*min(latency_lst[1],  latency_lst[2]):
                bias = sum(var_lst)
            else:
                bias = 0
            if latency_lst[1] + latency_lst[2] - latency_lst[0] > alpha * min(latency_lst[1],  latency_lst[2]) + bias:
                is_fused = True
                # delete redundant fusion rule
                if len(start_node) > 1:
                    if start_node_key in self.detailed_fusion_rules["eliminate"]:
                        self.detailed_fusion_rules["eliminate"].pop(start_node_key)
                    elif start_node_key in self.detailed_fusion_rules["combine"]:
                        self.detailed_fusion_rules["combine"].pop(start_node_key)
                if (latency_lst[0] < (1.0 + gamma) * latency_lst[1]) or (latency_lst[0] < (1.0 + gamma) * latency_lst[2]):
                    return "eliminate", is_fused, max_node_id
                else:
                    return "combine", is_fused, max_node_id
            else:
                is_fused = False
                return "split", is_fused, max_node_id
        else:
            if latency_lst[1] + latency_lst[2] - latency_lst[0] > alpha * min(latency_lst[1],  latency_lst[2]):
                is_fused = True
                # delete redundant fusion rule
                if len(start_node) > 1:
                    if start_node_key in self.detailed_fusion_rules["eliminate"]:
                        self.detailed_fusion_rules["eliminate"].pop(start_node_key)
                    elif start_node_key in self.detailed_fusion_rules["combine"]:
                        self.detailed_fusion_rules["combine"].pop(start_node_key)
                if (latency_lst[0] < (1.0 + gamma) * latency_lst[1]) or (latency_lst[0] < (1.0 + gamma) * latency_lst[2]):
                    return "eliminate", is_fused, max_node_id
                else:
                    return "combine", is_fused, max_node_id
            else:
                is_fused = False
                return "split", is_fused, max_node_id
    
    def fuse_tir(self, tir, fusion_rule):
    
        def fuse_on_path(path):
            compared_nodes = []
            compared_nodes_type = []
            for block in path:
                if not isinstance(block, SingleOp):
                    # iterate into new path
                    if len(compared_nodes_type) >= 2:
                        compared_nodes, compared_nodes_type = _sequential_fuse(compared_nodes, compared_nodes_type)
                    
                    if isinstance(block, ResidualBlock):
                        compared_nodes.append(block.start_node)
                        compared_nodes_type.append(block.start_node.type)
                    # clear buffer
                    compared_nodes = []
                    compared_nodes_type = []
                    if block.paths is not None:
                        for path_key, block_path in block.paths.items():
                            fuse_on_path(block_path)
                    elif block.flow is not None:
                        fuse_on_path(block.flow)
                    else:
                        raise NotImplementedError
                    if isinstance(block, ResidualBlock):
                        if block.last_node.type != "Split":
                            compared_nodes.append(block.last_node)
                            compared_nodes_type.append(block.reduce_type)
                else:
                    if block.type == "Split":
                        continue
                    # update buffer
                    compared_nodes.append(block)
                    compared_nodes_type.append(block.type)
                    if len(compared_nodes_type) >= 2:
                        compared_nodes, compared_nodes_type = _sequential_fuse(compared_nodes, compared_nodes_type)
                    
        def _sequential_fuse(compared_nodes, compared_nodes_type):
            str_candidate_type = '&'.join(compared_nodes_type)
            if str_candidate_type in fusion_rule:
                max_id = fusion_rule[str_candidate_type]
                for i in range(len(compared_nodes)):
                    if i != max_id:
                        compared_nodes[i].features["fused"] = True
                        
                        nodes_name = []
                        for node in compared_nodes:
                            nodes_name.append(node.name)
                        print(f"Fuse {nodes_name}")
                return compared_nodes, compared_nodes_type
            else:
                compared_nodes = [compared_nodes[-1]]
                compared_nodes_type = [compared_nodes_type[-1]]
            return compared_nodes, compared_nodes_type

        
        for stage_id, stage in tir.stages.items():
            stage_path = []
            for block_id, block in stage.flow.items():
                stage_path.append(block)
            fuse_on_path(stage_path)
    
        return tir
    