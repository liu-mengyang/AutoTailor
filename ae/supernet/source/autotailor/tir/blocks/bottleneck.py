from loguru import logger
import torch
import torch.nn as nn

from .block import Block
from .single_op import *
from .tailor_utils import make_divisible, closest


class BottleneckBlock(Block):
    def __init__(self, flow, all_expand_ratios, expand_base_on, divisor=8):
        """Build bottleneck block

        Args:
            flow (TIR Flow): the list of nodes
            all_expand_ratios: the list of variable expand ratios
            expand_base_on: expand ratio dynamic mode
        """
        super().__init__()
        self.type = 'BottleneckBlock'
        assert len(flow) > 0
        self.flow = flow

        # extract information
        first_conv = None
        middle_conv = None
        last_conv = None
        for i, op in enumerate(self.flow):
            if isinstance(op, ConvOp):
                if first_conv is None:
                    first_conv = op
                elif middle_conv is None:
                    middle_conv = op
                elif last_conv is None:
                    last_conv = op
                else:
                    raise NotImplementedError
            op.id = i

        expand_ratio_list = []
        if isinstance(all_expand_ratios[0], list):
            for exp_lst in all_expand_ratios:
                for exp in exp_lst:
                    if exp not in expand_ratio_list:
                        expand_ratio_list.append(exp)
            expand_ratio_list.sort()
        else:
            expand_ratio_list = all_expand_ratios

        if expand_base_on == "in":
            expand_base_width = first_conv.features['in_channel']
        else:
            expand_base_width = last_conv.features['out_channel']

        max_expand_ratio = closest(
            middle_conv.features['in_channel'] / expand_base_width,
            expand_ratio_list)
        self.expand_ratio_list = expand_ratio_list
        self.features = {
            'activated': True,
            'trans_type': 'active',
            'in_shape': (),
            'out_shape': (),
            'max_in_channel': first_conv.features['in_channel'],
            'max_out_channel': last_conv.features['out_channel'],
            'max_kernel_size': middle_conv.features['kernel_size'],
            'max_expand_ratio': max_expand_ratio,
            'in_channel': first_conv.features['in_channel'],
            'middle_width': middle_conv.features['in_channel'],
            "max_middle_width": middle_conv.features["in_channel"] * max_expand_ratio,
            'out_channel': last_conv.features['out_channel'],
            'kernel_size': middle_conv.features['kernel_size'],
            'expand_ratio': max_expand_ratio,
            'expand_base_on': expand_base_on,
            'divisor': divisor,
        }

        self.dynamic_dimensions.append('kernel_size')
        self.dynamic_dimensions.append('expand_ratio')


    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Only in shape and width can be updated.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        super().update(kv_features)
        inp_shape = self.features['in_shape']
        # TODO: bottleneck updating
        tensor_shape = inp_shape
        cin = self.features['in_channel']
        cout = self.features["out_channel"]

        expand_ratio = self.features["expand_ratio"]
        base_width = cin if self.features["expand_base_on"] == "in" else cout
        middle_width = make_divisible(base_width * expand_ratio,
                                      self.features["divisor"])
        self.features["middle_width"] = middle_width

        cur_width = cin
        cur_shape = inp_shape
        first_conv = None
        middle_conv = None
        last_conv = None
        for node in self.flow:
            # update each node in flow
            if isinstance(node, ConvOp):
                if first_conv is None:
                    first_conv = node
                    node.update({"in_channel": cur_width,
                                 "out_channel": middle_width,
                                 "in_shape": cur_shape})
                elif middle_conv is None:
                    middle_conv = node
                    node.update({"in_channel": cur_width,
                                 "out_channel": cur_width,
                                 "in_shape": cur_shape})
                elif last_conv is None:
                    last_conv = node
                    node.update({"in_channel": cur_width,
                                 "out_channel": cout,
                                 "in_shape": cur_shape})
                else:
                    raise NotImplementedError
            else:
                node.update({"in_channel": cur_width,
                             "in_shape": cur_shape})
            cur_width = node.features["out_channel"]
            cur_shape = node.features["out_shape"]

        self.features["out_channel"] = cur_width
        self.features['out_shape'] = cur_shape

    def build(self, cache=None, block_id=None) -> nn.Module:
        """Build torch module for bottleneck block.

        Build operator one by one in each path.

        Retruns:
          Torch module of bottleneck.
        """
        op_modules = []
        for op in self.flow:
            op_module = op.build(cache=cache, block_id=f"{block_id}-{op.id}")
            op_modules.append(op_module)

        self.module = nn.Sequential(*op_modules)
        return self.module

    def bind_weight(self):
        for op in self.flow:
            op.bind_weight()

    def reorganize_weight(self, cfgs):
        first_conv = None
        middle_conv = None
        last_conv = None
        for node in self.flow:
            if isinstance(node, ConvOp):
                if first_conv is None:
                    first_conv = node

                elif middle_conv is None:
                    middle_conv = node

                elif last_conv is None:
                    last_conv = node
                else:
                    raise NotImplementedError
        assert first_conv and middle_conv and last_conv

        if "expand_ratio" in cfgs:
            # reorganize weights for dynamic expand ratio in progressive training
            importance = torch.sum(
                torch.abs(last_conv.super_weights.data), dim=(0, 2, 3)
            )

            width_list = []

            for expand_ratio in self.expand_ratio_list:
                base_width = self.features["max_in_channel"] if self.features["expand_base_on"] == "in" else self.features["max_out_channel"]
                width = make_divisible(base_width * expand_ratio,
                                      self.features["divisor"])
                width_list.append(width)

            mini_ratio_index = min(cfgs["expand_ratio"])
            mini_width = width_list[mini_ratio_index]

            right = len(importance)
            num_stages = len(width_list) - mini_ratio_index - 1
            base = -num_stages * 1e5
            for i in range(num_stages):
                left = width_list[len(width_list)-2-i]
                importance[left:right] += base
                logger.info(f"Reorganize weight of {self.type} from {left} to {right} with {base} offset")
                base += 1e5
                right = left

            sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)

            # reorganize the first conv in cout dim
            logger.info("reorganize weight of first conv")
            first_conv.super_weights.data = torch.index_select(
                first_conv.super_weights.data, 0, sorted_idx
            )
            if first_conv.features["has_bias"]:
                first_conv.super_bias.data = torch.index_select(
                    first_conv.super_bias.data, 0, sorted_idx
                )

            # reorganize the middle conv in channel dim
            logger.info("reorganize weight of middle conv")
            middle_conv.super_weights.data = torch.index_select(
                middle_conv.super_weights.data, 0, sorted_idx
            )
            if middle_conv.features["has_bias"]:
                middle_conv.super_bias.data = torch.index_select(
                    middle_conv.super_bias.data, 0, sorted_idx
                )

            # reorganize the last conv in cin dim
            logger.info("reorganize weight of last conv")
            last_conv.super_weights.data = torch.index_select(
                last_conv.super_weights.data, 1, sorted_idx
            )

            bn_cnt = 0
            for node in self.flow:
                if isinstance(node, BNOp):
                    logger.info("reorganize weight of bn")
                    bn_cnt += 1
                    if bn_cnt >= 3:
                        continue
                    node.super_weights.data = torch.index_select(node.super_weights.data, 0, sorted_idx)
                    node.super_bias.data = torch.index_select(node.super_bias.data, 0, sorted_idx)
                    node.super_mean.data = torch.index_select(node.super_mean.data, 0, sorted_idx)
                    node.super_var.data = torch.index_select(node.super_var.data, 0, sorted_idx)


    def transform(self, kv_features: dict) -> None:
        """Transform features of the block.

        Bottleneck block can transform kernel size and expand ratio

        Args:
          kv_features: the dictionary of to transform features.

        Raises:
          KeyError: the error occured in updating non exist feature or not
                    not support to transform this dimension.
        """
        super().transform(kv_features)

        # update the kernel_size of conv in main_path
        if "kernel_size" in kv_features.keys():
            for op in self.flow:
                if isinstance(op, ConvOp):
                    if op.features["kernel_size"] != 1:
                        op.update({"kernel_size": self.features['kernel_size']})

        if "expand_ratio" in kv_features.keys():
            cin = self.features['in_channel']
            cout = self.features['out_channel']
            expand_ratio = self.features['expand_ratio']

            if self.features['expand_base_on'] == 'in':
                expand_base_width = cin
            else:
                expand_base_width = cout

            middle_width = make_divisible(expand_base_width * expand_ratio,
                                          self.features['divisor'])
            # update middle width of middle conv in main_path
            self.features['middle_width'] = middle_width

            inp_shape = self.features["in_shape"]
            cur_shape = inp_shape
            cur_width = cin
            first_conv = None
            middle_conv = None
            last_conv = None
            for op in self.flow:
                # update conv op in flow
                if isinstance(op, ConvOp):
                    if first_conv is None:
                        first_conv = op
                        op.update({"in_channel": cur_width,
                                "in_shape": cur_shape,
                                "out_channel": middle_width})
                    elif middle_conv is None:
                        middle_conv = op
                        op.update({"in_channel": cur_width,
                                "in_shape": cur_shape,
                                "out_channel": cur_width})
                    elif last_conv is None:
                        last_conv = op
                        op.update({"in_channel": cur_width,
                                "in_shape": cur_shape,
                                "out_channel": cout})
                    else:
                        raise NotImplementedError
                else:
                    op.update({"in_channel": cur_width,
                            "in_shape": cur_shape,
                            "out_channel": cur_width})
                cur_shape = op.features["out_shape"]
                cur_width = op.features["out_channel"]
            assert first_conv and middle_conv and last_conv

    def count_flops_params(self):
        params = 0
        flops = 0
        for op in self.flow:
            op_flops, op_params = op.count_flops_params()
            flops += op_flops
            params += op_params
        return flops, params

    def get_ops(self):
        op_lst = []
        for op in self.flow:
            if not isinstance(op, SingleOp):
                # nested block
                ops = op.get_ops()
                for subop in ops:
                    subop.id = str(op.id)+'-'+str(subop.id)
                    op_lst.append(subop)
            else:
                op_lst.append(op)
        return op_lst

    def update_grad(self):
        for op in self.flow:
            op.update_grad()

    @property
    def info_dict(self):
        ret_dict = super().info_dict

        ret_dict['flow'] = []

        for op in self.flow:
            ret_dict['flow'].append(op.info_dict)
        return ret_dict

    @property
    def info(self):
        info_list = []
        for op in self.flow:
            info_list.append(op.info)
        info_tuple = tuple(info_list)
        return info_tuple
