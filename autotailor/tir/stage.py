from collections import OrderedDict

import torch.nn as nn

from autotailor.tir.blocks import SingleOp
from autotailor.tir.blocks.tailor_utils import make_divisible


class Stage(object):
    def __init__(self, verbose_level='simple', divisor=8):
        self.flow = OrderedDict()
        self.num_blocks = 0
        self.features = {
            "in_shape": (),
            "out_shape": (),
            "in_channel": 0,
            "out_channel": 0,
        }

        self.module = None

        self.update_dimensions = ["in_shape", "out_shape", "in_channel", "out_channel"]
        self.dynamic_dimensions = ["width_mult", "reduce_depth"]

        self.super_depth = 0
        self.divisor = divisor

        self.verbose_level = verbose_level

    def add_block(self, block):
        self.flow[self.num_blocks] = block
        self.num_blocks += 1

    def update(self, kv_features: dict) -> None:
        """Update features of the stage.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
        """
        for k, v in kv_features.items():
            if k in self.features and k in self.update_dimensions:
                self.features[k] = v
            else:
                raise KeyError
        # cur_width = self.features["in_channel"]
        # cur_shape = self.features["in_shape"]
        # for block_id, block in self.flow.items():
        #     block.update({"in_channel": cur_width,
        #                   "in_shape": cur_shape})
        #     cur_width = block.features["out_channel"]
        #     cur_shape = block.features["out_shape"]

    def build(self, cache=None, stage_id=None):
        block_modules = []
        for i, block in self.flow.items():
            if block.is_active():
                block_module = block.build(cache=cache, block_id=f"{stage_id}-{i}")
                block_modules.append(block_module)

        self.module = nn.Sequential(*block_modules)
        return self.module

    def bind_weight(self):
        for block_id, block in self.flow.items():
            block.bind_weight()

    def cnt_blocks(self):
        for block_id, block in self.flow.items():
            if block.is_transformable():
                self.super_depth += 1

    def transform(self, kv_features):
        """Transform features of the blocks in stage.

        Stage can transform width ratio and mask depth

        Args:
          kv_features: the dictionary of to transform features.

        Raises:
          KeyError: the error occured in updating non exist feature or not
                    not support to transform this dimension.
        """
        for k, v in kv_features.items():
            if k == "width_mult":
                cur_width = self.features["in_channel"]
                cur_shape = self.features["in_shape"]
                for block_id, block in self.flow.items():
                    out_channel = block.features["max_out_channel"]
                    block.update({"out_channel": make_divisible(out_channel * v,
                                                                self.divisor),
                                  "in_channel": cur_width,
                                  "in_shape": cur_shape})
                    cur_width = block.features["out_channel"]
                    cur_shape = block.features["out_shape"]
                self.update({"out_channel": cur_width, "out_shape": cur_shape})

            elif k == "reduce_depth":
                # unmask first
                for block_id, block in self.flow.items():
                    block.unmask()
                max_block_cnt = self.super_depth + v
                block_cnt = 0
                start_mask = False
                for block_id, block in self.flow.items():
                    if start_mask:
                        block.mask()
                    elif block.is_transformable():
                        block_cnt += 1
                        if block_cnt > max_block_cnt:
                            # Until the next dynamic block
                            block.mask()
                            start_mask = True
                            self.update({
                                "out_channel": block.features["in_channel"],
                                "out_shape": block.features["in_shape"]})
                # print(f"{self.super_depth}:{v}:{block_cnt}")
            else:
                raise KeyError

    def reorganize_weight(self, cfgs):
        for block_id, block in self.flow.items():
            block.reorganize_weight(cfgs)

    def get_ops(self, enable_id=False, stage_id=None):
        ids = []
        ops = []
        for block_id, block in self.flow.items():
            if block.features['activated']:
                if isinstance(block, SingleOp):
                    ops.append(block)
                    if enable_id:
                        ids.append(f"{stage_id}-{block_id}")
                else:
                    block_ops = block.get_ops()
                    for op_id, op in enumerate(block_ops):
                        ops.append(op)
                        if enable_id:
                            ids.append(f"{stage_id}-{block_id}-{op.id}")
            else:
                break
        if enable_id:
            return ids, ops
        else:
            return ops

    def recount(self):
        for i, block in enumerate(self.flow.values()):
            if not isinstance(block, SingleOp):
                block.recount()

    def get_blocks(self):
        blocks = []
        for block_id, block in self.flow.items():
            blocks.append(block)
        return blocks

    def count_flops_params(self, end_block_id=None, start_block_id=None):
        flops = 0
        params = 0
        for block_id, block in self.flow.items():
            if start_block_id is not None:
                if block_id < start_block_id:
                    continue
            elif end_block_id is not None:
                if block_id == end_block_id:
                    break
            if block.features["activated"]:
                block_flops, block_params = block.count_flops_params()
                flops += block_flops
                params += block_params


        return flops, params

    @property
    def info_dict(self):
        flow_info_dict = OrderedDict()
        for i in range(len(self.flow)):
            flow_info_dict[i] = self.flow[i].info_dict

        return {
            'flow': flow_info_dict,
            'features': self.features
        }
