from collections import OrderedDict
import copy
import time

from loguru import logger
import torch
import torch.nn as nn

import autotailor.tir.parser as parser
import autotailor.tir.globvar as globvar
from .blocks import *
from .blocks.bottleneck import BottleneckBlock
from .graph_utils import *
from .mp_utils import build_worker
from .stage import Stage


__BUILTIN_BLOCKS__ = {
    "BottleneckBlock": "parse_bottleneck",
    "MBBlock": "parse_mb",
    "CNResidualBlock": "parse_cnresidual",
    "AttentionBlock": "parse_attention",
    "FFNBlock": "parse_ffn",
    "TransformerBlock": "parse_transformer",
    "LinearMatMulOp": "parse_linearmatmul"
}


class BranchNetwork(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 heads: list):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList(heads)
        self.inter_res = None
        self.inter_res_detached_dict = {}
        self.no_backbone = False
        if len(self.backbone.stages) == 0:
            self.no_backbone = True

    def forward(self, x: torch.Tensor, head_id=None):
        if head_id is None:
            ys = []
            if not self.no_backbone:
                self.inter_res = self.backbone(x)

                for i, head in enumerate(self.heads):
                    inter_res_detached = copy.deepcopy(self.inter_res.detach().requires_grad_(True))
                    self.inter_res_detached_dict[i] = inter_res_detached
                    ys.append(head(self.inter_res_detached_dict[i]))
            else:
                for i, head in enumerate(self.heads):
                    ys.append(head(x))
            return ys
        elif head_id == -1:
            if not self.no_backbone:
                self.inter_res = self.backbone(x)

                for i, head in enumerate(self.heads):
                    inter_res_detached = copy.deepcopy(self.inter_res.detach().requires_grad_(True))
                    self.inter_res_detached_dict[i] = inter_res_detached
            return None
        else:
            if not self.no_backbone:
                y = self.heads[head_id](self.inter_res_detached_dict[head_id])
            else:
                y = self.heads[head_id](x)
            return y

    def forward_backbone(self, x: torch.Tensor):
        if not self.no_backbone:
            self.inter_res = self.backbone(x)

            for i, head in enumerate(self.heads):
                inter_res_detached = copy.deepcopy(self.inter_res.detach().requires_grad_(True))
                self.inter_res_detached_dict[i] = inter_res_detached

    def forward_head(self, head_id, x=None):
        if not self.no_backbone:
            y = self.heads[head_id](self.inter_res_detached_dict[head_id])
        else:
            y = self.heads[head_id](x)
        return y


class TorchNetwork(nn.Module):
    def __init__(self, stage_modules: list):
        super().__init__()
        self.stages = nn.Sequential(*stage_modules)

    def forward(self, x: torch.Tensor):
        for stage in self.stages:
            x = stage(x)
        return x


class TailorIR(object):
    def __init__(self,
                 supernet_cfg_dict):
        self.supernet_cfg_dict = supernet_cfg_dict
        globvar.model_name = supernet_cfg_dict["title"]

        self.stages = None
        self.op_dict = {}
        self.op_list = []
        # self.op_type_dict = {}
        # self.op_type_idx = 0

        self.inp_shape = None

        self.verbose_level = None
        self.dominant_op_types = []

        self.shared_backbone = None
        self.dynamic_heads = []

    def parse_graph(self, inp_shape = (1,3,224,224)):
        """_summary_

        Args:
            input_cin (int, optional): _description_. Defaults to 3.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        self.inp_shape = inp_shape

        cur_node = get_first_node(globvar.onnx_graph)

        # First parsing: find residual blocks and singleop blocks
        logger.info("Initial parsing")
        new_stage = parser.first_parse(cur_node, self.supernet_cfg_dict, self.inp_shape)

        for j, block in new_stage.flow.items():
            logger.info(f"{block.type} {str(block.features['out_channel'])}")


        for block_name in self.supernet_cfg_dict["arch"]["blocks"]:
            if block_name in __BUILTIN_BLOCKS__:
                parser_function = getattr(parser, __BUILTIN_BLOCKS__[block_name])

                logger.info(f"{block_name} parsing")
                new_stage = parser_function(new_stage, self.supernet_cfg_dict, self.inp_shape)

                for j, block in new_stage.flow.items():
                    logger.info(f"{block.type} {str(block.features['out_channel'])}")

        # Stage detection
        logger.info("Stage detection")
        stages = OrderedDict()

        cur_stage_id = 0
        stages[cur_stage_id] = Stage()
        if len(inp_shape) == 4:
            cur_width = inp_shape[1]
        else:
            cur_width = inp_shape[-1]
        cur_hw = inp_shape[-1]
        dims = len(inp_shape)
        for block_id, block in new_stage.flow.items():
            if block.features['out_channel'] != cur_width:
                # Detect stage detection by width
                # width changed
                logger.info(f"Width changed from {block.type}")
                if len(stages[cur_stage_id].flow) != 0:
                    cur_stage_id += 1
                    stages[cur_stage_id] = Stage()
            elif len(block.features["out_shape"]) != dims:
                # Detect stage detection by dim#
                # dim changed
                logger.info(f"Dim changed from {block.type}")
                if len(stages[cur_stage_id].flow) != 0:
                    cur_stage_id += 1
                    stages[cur_stage_id] = Stage()
            elif block.features["out_shape"][-1] != cur_hw:
                # Detect stage detection by hw
                # hw changed
                logger.info(f"HW changed from {block.type}")
                if len(stages[cur_stage_id].flow) != 0:
                    cur_stage_id += 1
                    stages[cur_stage_id] = Stage()
            cur_width = block.features['out_channel']
            cur_hw = block.features["out_shape"][-1]
            dims = len(block.features["out_shape"])
            stages[cur_stage_id].add_block(block)

        for i, stage in stages.items():
            logger.info(f'##### STAGE {i} #####')
            for j, block in stage.flow.items():
                if j == 0:
                    stage.update({"in_shape": block.features["in_shape"],
                                  "in_channel": block.features["in_channel"]})
                logger.info(f"{block.type} {str(block.features['out_channel'])}")
            stage.update({"out_shape": block.features["out_shape"], "out_channel": block.features["out_channel"]})

        self.stages = stages
        self.num_stages = len(stages)
        self.stage_modules = []

        return stages

    def build(self,
              branching=False,
              share_stage_id=None,
              share_block_id=None,
              branch_stage_id=None,
              branch_block_id=None,
              branching_id=None,
              num_heads=None,
              cache=None):
        # TODO: make branching by default
        if branching:
            assert share_stage_id is not None
            assert share_block_id is not None
            assert branch_stage_id is not None
            assert branch_block_id is not None
            assert branching_id is not None
            assert num_heads is not None
            # Create backbone
            if branching_id == 0:
                self.shared_backbone = []
                self.dynamic_heads = []
                if share_stage_id != -1:
                    # backbone exist
                    backbone_modules = []
                    for stage_id, stage in self.stages.items():
                        if stage_id == share_stage_id:
                            # meet the last stage
                            block_modules = []
                            for block_id, block in stage.flow.items():
                                if block_id == share_block_id:
                                    # meet the last block
                                    if block.is_active():
                                        block_modules.append(block.build(cache=cache[0] if cache else None, block_id=f"{stage_id}-{block_id}"))
                                    break
                                else:
                                    assert block_id < share_block_id
                                    if block.is_active():
                                        block_modules.append(block.build(cache=cache[0] if cache else None, block_id=f"{stage_id}-{block_id}"))
                            if len(block_modules) != 0:
                                stage_module = nn.Sequential(*block_modules)
                                backbone_modules.append(stage_module)
                            break
                        else:
                            assert stage_id < share_stage_id
                            stage_module = stage.build(cache=cache[0] if cache else None, stage_id=stage_id)
                            backbone_modules.append(stage_module)
                    self.shared_backbone = TorchNetwork(backbone_modules)
                else:
                    # No backbone
                    self.shared_backbone = TorchNetwork([])
            # Create branch
            dynamic_head_modules = []
            for stage_id, stage in self.stages.items():
                if stage_id < branch_stage_id:
                    # Before meeting branching stage
                    continue
                elif stage_id == branch_stage_id:
                    # meet branch stage, looking for branch block
                    block_modules = []
                    cnt = 0

                    for block_id, block in stage.flow.items():
                        if block_id < branch_block_id:
                            # Before meeting branch block
                            continue
                        else:
                            # meeted branch block
                            if block.is_active():
                                block_modules.append(block.build(cache=cache[branching_id] if cache else None, block_id=f"{stage_id}-{block_id}"))
                    if len(block_modules) == 0:
                        # branch stage is empty
                        continue
                    stage_module = nn.Sequential(*block_modules)
                    dynamic_head_modules.append(stage_module)
                else:
                    stage_module = stage.build(cache=cache[branching_id] if cache else None, stage_id=stage_id)
                    dynamic_head_modules.append(stage_module)
            assert len(dynamic_head_modules) != 0
            self.dynamic_heads.append(TorchNetwork(dynamic_head_modules))

            if num_heads == branching_id+1:
                # Create BranchNetwork
                torch_nn = BranchNetwork(self.shared_backbone,
                                        self.dynamic_heads)
                # print(torch_nn)
                self.torch_nn = torch_nn
                return torch_nn
            else:
                return None
        else:
            self.stage_modules = []
            for i, stage in self.stages.items():
                stage_module = stage.build(cache=cache[0] if cache else None, stage_id=i)
                self.stage_modules.append(stage_module)

            ## Build
            torch_nn = TorchNetwork(self.stage_modules)
            self.torch_nn = torch_nn
            return torch_nn

    def buildv0(self,
              branching=False,
              branch_stage_id=None,
              branch_block_id=None,
              branching_id=None,
              determine_feature=None,
              num_heads=None,
              half_start=False,
              half_end=False,
              cache=None):
        # DEPRECATED
        # TODO: make branching by default
        # half options are only used in supernet tuning for overlapping CPU
        # building cost with GPU forward and backward computation cost.
        # 2.5 is the threshold to control forward/backward building ratio.
        # we think that backward is 1.5 times of forward so dividing by 2.5.
        if branching:
            assert branch_stage_id is not None
            assert branch_block_id is not None
            assert branching_id is not None
            assert num_heads is not None
            if not half_end:
                # Create backbone
                if branching_id == 0:
                    self.shared_backbone = []
                    self.dynamic_heads = []

                    backbone_modules = []
                    for i, stage in self.stages.items():
                        if i == branch_stage_id:
                            cnt = 0
                            if determine_feature == "width_mult":
                                # branch from the start point
                                break
                            block_modules = []
                            for j, block in stage.flow.items():
                                if isinstance(block, BottleneckResidualBlock):
                                    cnt += 1
                                if cnt >= (branch_block_id+1):
                                    break
                                else:
                                    if block.is_active():
                                        block_modules.append(block.build(cache=cache, block_id=f"{i}-{j}"))
                            if len(block_modules) != 0:
                                stage_module = nn.Sequential(*block_modules)
                                backbone_modules.append(stage_module)
                            break
                        else:
                            stage_module = stage.build(cache=cache, stage_id=i)
                            backbone_modules.append(stage_module)
                    self.shared_backbone = TorchNetwork(backbone_modules)
            if not half_start:
                # Create branch
                dynamic_head_modules = []
                for j, stage in self.stages.items():
                    if j < branch_stage_id:
                        continue
                    elif j == branch_stage_id:
                        block_modules = []
                        cnt = 0
                        if determine_feature == "width_mult":
                            # branch from the start point
                            assert branch_block_id == 0
                            cnt = 1
                            # print("Direct branching out")
                        for k, block in stage.flow.items():
                            if isinstance(block, BottleneckResidualBlock):
                                cnt += 1
                            if cnt < (branch_block_id+1):
                                continue
                            else:
                                if block.is_active():
                                    block_modules.append(block.build(cache=cache, block_id=f"{j}-{k}"))
                        if len(block_modules) != 0:
                            stage_module = nn.Sequential(*block_modules)
                            dynamic_head_modules.append(stage_module)
                    else:
                        stage_module = stage.build(cache=cache, stage_id=j)
                        dynamic_head_modules.append(stage_module)
                self.dynamic_heads.append(TorchNetwork(dynamic_head_modules))

            if num_heads == branching_id+1:
                # Create BranchNetwork
                torch_nn = BranchNetwork(self.shared_backbone,
                                        self.dynamic_heads)
                # print(torch_nn)
                self.torch_nn = torch_nn
                return torch_nn
            else:
                return None
        else:
            if not half_end:
                self.stage_modules = []
            for i, stage in self.stages.items():
                if half_end and i <= self.num_stages//2.5:
                    continue
                stage_module = stage.build(cache=cache, stage_id=i)
                self.stage_modules.append(stage_module)
                if half_start and i == self.num_stages//2.5:
                    break

            if not half_start:
                ## Build
                torch_nn = TorchNetwork(self.stage_modules)
                self.torch_nn = torch_nn
                return torch_nn

    def bind_weight(self):
        for i, stage in self.stages.items():
            stage.bind_weight()

    def transform(self, kv_features):
        """Transform features of the stages in tir.

        TIR can transform input resolution

        Args:
          kv_features: the dictionary of to transform features.

        Raises:
          KeyError: the error occured in updating non exist feature or not
                    not support to transform this dimension.
        """
        for k, v in kv_features.items():
            if k == "resolution":
                # logger.info(f"Update inp shape to {v}")
                self.inp_shape = (self.inp_shape[0], self.inp_shape[1], v, v)

                inp_shape = self.inp_shape
                # Update shape information
                for stage_id, stage in self.stages.items():
                    for block_id, block in stage.flow.items():
                        block.update({"in_shape": inp_shape})
                        inp_shape = block.features["out_shape"]
            else:
                raise KeyError

    def get_weights(self):
        net_params = []
        for op_type, op_list in self.get_ops(drop_dup=False, disable_extracting=True).items():
            for op in op_list:
                if op.features["has_weights"]:
                    params = op.get_weights()
                    for param in params:
                        net_params.append(param)
        return net_params

    def update_weights(self, parameters_iterator):
        # TODO: now is param dict to tir, but tir to param dict is the right thing
        for name, param in parameters_iterator:
            tag_list = name.split(".")
            num_tags = len(tag_list)
            stage_id = int(tag_list[1])
            block_id = int(tag_list[2])
            if num_tags == 4:
                # single op block
                weight_name = tag_list[3]
                op = self.stages[stage_id].flow[block_id]
                op.update_weights(weight_name, param)
            elif num_tags == 5:
                # flow block
                layer_id = int(tag_list[3])
                weight_name = tag_list[4]
                op = self.stages[stage_id].flow[block_id].flow[layer_id]
                op.update_weights(weight_name, param)
            elif num_tags == 6:
                # residual block
                attr_str = tag_list[3]
                layer_id = int(tag_list[4])
                weight_name = tag_list[5]
                op = getattr(self.stages[stage_id].flow[block_id], attr_str)[layer_id]
                op.update_weights(weight_name, param)
            elif num_tags == 7:
                # flow residual block
                layer_id = int(tag_list[3])
                attr_str = tag_list[4]
                sub_layer_id = int(tag_list[5])
                weight_name = tag_list[6]
                op = getattr(self.stages[stage_id].flow[block_id].flow[layer_id], attr_str)[sub_layer_id]
                op.update_weights(weight_name, param)
            elif num_tags == 8 and "qkv" in name:
                # qkv residual block
                layer_id = int(tag_list[3])
                attr_str = tag_list[4]
                sub_attr_str = tag_list[5]
                sub_layer_id = int(tag_list[6])
                weight_name = tag_list[7]
                op = getattr(getattr(self.stages[stage_id].flow[block_id].flow[layer_id], attr_str), sub_attr_str)[sub_layer_id]
                op.update_weights(weight_name, param)
            elif num_tags == 8:
                # 1 nested residual block
                attr_str = tag_list[3]
                layer_id = int(tag_list[4])
                sub_attr_str = tag_list[5]
                sub_layer_id = int(tag_list[6])
                weight_name = tag_list[7]
                op = getattr(getattr(self.stages[stage_id].flow[block_id], attr_str)[layer_id], sub_attr_str)[sub_layer_id]
                op.update_weights(weight_name, param)
            else:
                raise NotImplementedError(f"Not support for param_name with {num_tags} tags: {name}")

    def update_grad(self):
        for stage_id, stage in self.stages.items():
            for block_id, block in stage.flow.items():
                if block.is_active():
                    block.update_grad()

    def update_gradv0(self,
                    parameters_iterator,
                    branched=False,
                    branch_stage_id=None,
                    branch_block_id=None,
                    no_backbone=False,
                    determine_feature=None,
                    branch_stages=None,
                    weights=False):
        # DEPRECATED
        # The layer skipping will not skip intermediate layers.
        # So that the stage/block/layer id of subTIR is also correct for superTIR.
        # TODO: Carefully check block id
        cur_head = -1
        block_stage_offset = 0
        for name, param in parameters_iterator:
            stage_offset = 0
            block_offset = 0
            if branched:
                # preprocess name for branched model
                # print(name)
                if "backbone" == name.split('.')[0]:
                    name = name.split("backbone.")[1]
                    tag_list = name.split(".")
                    tag_list = name.split(".")
                    num_tags = len(tag_list)
                    stage_id = int(tag_list[1])
                    block_id = int(tag_list[2])
                elif "heads" == name.split('.')[0]:
                    name = name.split("heads.")[1]
                    heads_id = name.split(".")[0]

                    # # detect new head
                    # if cur_head != heads_id:
                    #     cur_head = heads_id

                    name = name[2:]
                    tag_list = name.split(".")
                    num_tags = len(tag_list)
                    block_id = int(tag_list[2])
                    # print(f"Stage offset: {branch_stages}")
                    stage_id = int(tag_list[1]) + branch_stage_id + branch_stages[int(heads_id)]
                    # if determine_feature == "width_mult":
                    #     # offset one stage for stage-level transformation
                    #     stage_id += 1
                    if int(tag_list[1]) == 0 and branch_stages[int(heads_id)] == 0:
                        # find first target block for the first stage in heads
                        # TODO: currently only support ResNet50
                        cnt = 0
                        if determine_feature == "width_mult":
                            # branch from the start point
                            assert branch_block_id == 0
                            cnt = 1
                        if num_tags == 6:
                            # print(self.stages[stage_id].flow)
                            # print(f"Looking for block id from {block_id}")
                            for i, block in self.stages[stage_id].flow.items():
                                if isinstance(block, BottleneckResidualBlock):
                                    cnt += 1
                                if cnt < (branch_block_id+1):
                                    continue
                                if cnt >= (block_id+1):
                                    block_id = i
                                    # print(f"Found {block_id}")
                                    break
                            assert cnt != 0
                        else:
                            # now are other op blocks
                            break
                            print(name)
                            raise NotImplementedError
                    else:
                        block_id = int(tag_list[2])
            else:
                tag_list = name.split(".")
                num_tags = len(tag_list)
                stage_id = int(tag_list[1])
                block_id = int(tag_list[2])

            # print(stage_id)
            # print(block_id)

            weight_name = None
            op = None

            if num_tags == 4:
                # single op block
                weight_name = tag_list[3]
                op = self.stages[stage_id].flow[block_id]
            elif num_tags == 5:
                # flow block
                layer_id = int(tag_list[3])
                weight_name = tag_list[4]
                op = self.stages[stage_id].flow[block_id].flow[layer_id]
            elif num_tags == 6:
                # residual block
                attr_str = tag_list[3]
                layer_id = int(tag_list[4])
                weight_name = tag_list[5]
                # print(self.stages[stage_id].flow)
                op = getattr(self.stages[stage_id].flow[block_id], attr_str)[layer_id]
            elif num_tags == 8:
                # 1 nested residual block
                attr_str = tag_list[3]
                layer_id = int(tag_list[4])
                sub_attr_str = tag_list[5]
                sub_layer_id = int(tag_list[6])
                weight_name = tag_list[7]
                op = getattr(getattr(self.stages[stage_id].flow[block_id], attr_str)[layer_id], sub_attr_str)[sub_layer_id]
            else:
                raise NotImplementedError(f"Not support for param_name with {num_tags} tags")
            if not weights:
                op.update_grad(weight_name, param.grad)
            else:
                op.update_weights(weight_name, param)

    def reorganize_weight(self, cfgs):
        logger.info(f"Reorganize weight with {cfgs}")
        for stage_id, stage in self.stages.items():
            stage.reorganize_weight(cfgs)

    def get_ops(self, drop_dup=True, disable_extracting=False, enable_id=False):
        op_dict = {}
        ops = []
        ids = []
        self.recount()
        for stage_id, stage in self.stages.items():
            if enable_id:
                stage_ids, stage_ops = stage.get_ops(enable_id=enable_id, stage_id=stage_id)
            else:
                stage_ops = stage.get_ops()
            # print(stage_ops)
            ops += stage_ops
            if enable_id:
                ids += stage_ids

        # iterate collected op list
        for i, op in enumerate(ops):
            # if op.type == "Reshape":
            #     print(op.info)
            if disable_extracting:
                if op.type not in op_dict:
                    op_dict[op.type] = [op]
                else:
                    op_dict[op.type].append(op)
            else:
                if enable_id:
                    features = f"{ids[i]}-{op.info}"
                else:
                    features = f"{op.info}"
                if drop_dup:
                    # use set
                    if op.type not in op_dict:
                        op_dict[op.type] = {features} # set of tuples
                    else:
                        op_dict[op.type].add(features)
                else:
                    # use list
                    if op.type not in op_dict:
                        op_dict[op.type] = [features] # list of tuples
                    else:
                        op_dict[op.type].append(features)
        return op_dict
    
    def recount(self):
        for stage_id, stage in self.stages.items():
            stage.recount()

    def pre_build(self, built_op_dict, block_building=False, skip_recount=False):
        op_dict = {}
        ops = []
        ids = []
        if not skip_recount or getattr(self, '_depth_dirty', True):
            self.recount()
            self._depth_dirty = False
        for stage_id, stage in self.stages.items():
            stage_ids, stage_ops = stage.get_ops(enable_id=True, stage_id=stage_id)
            ops += stage_ops
            ids += stage_ids

        # iterate collected op list
        for i, op in enumerate(ops):
            if op.type not in built_op_dict:
                built_op_dict[op.type] = {}

            features = str(ids[i])+"-"+str(op.info)

            if features not in built_op_dict[op.type]:
                if not block_building:
                    built_op_dict[op.type][features] = op.build()
                else:
                    built_op_dict[op.type][features] = None

    def get_blocks(self, drop_dup=True, disable_extracting=False):
        block_dict = {}
        blocks = []
        for stage_id, stage in self.stages.items():
            stage_blocks = stage.get_blocks()
            blocks += stage_blocks

        # iterate collected block list
        for block in blocks:
            # print(block.type)
            if disable_extracting:
                if block.type not in block_dict:
                    block_dict[block.type] = [block]
                else:
                    block_dict[block.type].append(block)
            else:
                features = block.info
                # print(features)
                if drop_dup:
                    # use set
                    if block.type not in block_dict:
                        block_dict[block.type] = {features} # set of tuples
                    else:
                        block_dict[block.type].add(features)
                else:
                    # use list
                    if block.type not in block_dict:
                        block_dict[block.type] = [features] # list of tuples
                    else:
                        block_dict[block.type].append(features)
        return block_dict

    def count_flops_params(self, end_stage_id=None, end_block_id=None, start_stage_id=None, start_block_id=None):
        flops = 0
        params = 0
        for stage_id, stage in self.stages.items():
            if start_stage_id is not None and start_block_id is not None:
                if stage_id < start_stage_id:
                    continue
                elif stage_id == start_stage_id:
                    stage_flops, stage_params = stage.count_flops_params(start_block_id=start_block_id)
                else:
                    stage_flops, stage_params = stage.count_flops_params()
            elif end_stage_id is not None and end_block_id is not None:
                if stage_id == end_stage_id:
                    stage_flops, stage_params = stage.count_flops_params(end_block_id=end_block_id)
                else:
                    stage_flops, stage_params = stage.count_flops_params()
            else:
                stage_flops, stage_params = stage.count_flops_params()
            flops += stage_flops
            params += stage_params
            if end_stage_id:
                if stage_id == end_stage_id:
                    break

        return flops, params

    def set_verbose_level(self, verbose_level):
        def set_on_path(path, verbose_level):
            for block in path:
                if not isinstance(block, SingleOp):
                    if block.paths is not None:
                        for path_key, block_path in block.paths.items():
                            set_on_path(block_path, verbose_level)
                    elif block.flow is not None:
                        set_on_path(block.flow, verbose_level)
                    else:
                        raise NotImplementedError
                    block.verbose_level = verbose_level
                else:
                    block.verbose_level = verbose_level


        for i, stage in self.stages.items():
            stage_path = []
            for block_id, block in stage.flow.items():
                stage_path.append(block)
            set_on_path(stage_path, verbose_level)
