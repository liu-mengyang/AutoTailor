import copy

import torch.nn as nn

from .residual import ResidualBlock
from .single_op import *


class ConvResidualModule(nn.Module):
    def __init__(self, main_path, residual_path, reduce_type):
        super().__init__()
        self.main_path = main_path
        self.residual_path = residual_path
        self.reduce_type = reduce_type

    def forward(self, x):
        y = self.main_path(x)
        ### debug
        # y = x
        # print(y.shape)
        # # print(y)
        # for op in self.main_path:
        #     print(op)
        #     y = op(y)
        #     print(y.shape)
        #     # print(y)
        ###
        x = self.residual_path(x)
        if self.reduce_type == "Add":
            y = y + x
        elif self.reduce_type == "Mul":
            y = y * x
        elif self.reduce_type == "Concat":
            y = torch.concat(x, y, dim=1)
        return y


class ConvResidualBlock(ResidualBlock):
    def __init__(self, start_node, last_node, first_path, second_path):
        """Build residual block

        Args:
            start_node (TIR Node): the start node of the block
            last_node (TIR Node): the second node of the block
            first_path (list): the first path of nodes
            second_path (list): the second path of nodes
        """
        super().__init__(start_node,
                         last_node,
                         first_path,
                         second_path)
        self.type = 'ConvResidualBlock'

        # extract information
        first_conv = None
        last_conv = None
        conv_count = 0
        for op in self.main_path:
            if isinstance(op, ConvOp):
                conv_count += 1
                if first_conv is None:
                    first_conv = op
                last_conv = op
        if conv_count == 0:
            # conv may exist in residual path
            for op in self.residual_path:
                if isinstance(op, ConvOp):
                    conv_count += 1
                    if first_conv is None:
                        first_conv = op
                    last_conv = op
            if conv_count != 0:
                # swp main residual
                tmp = copy.deepcopy(self.main_path)
                self.main_path = copy.deepcopy(self.residual_path)
                self.residual_path = tmp
        assert conv_count != 0

        self.features['trans_type'] = 'active'
        self.features['max_in_channel'] = first_conv.features['in_channel']
        self.features['max_out_channel'] = last_conv.features['out_channel']
        self.features['max_kernel_size'] = first_conv.features['kernel_size']
        self.features['in_channel'] = first_conv.features['in_channel']
        self.features['out_channel'] = last_conv.features['out_channel']
        self.features['kernel_size'] = first_conv.features['kernel_size']
        self.features['conv_count'] = conv_count

        self.dynamic_dimensions.append('kernel_size')

    def build(self, cache=None, block_id=None) -> nn.Module:
        """Build torch module for conv residual block.

        Build operator one by one in each path.

        Retruns:
          Torch module of conv residual.
        """
        reduce_type = self.features["reduce_type"]

        op_modules = []
        for op in self.main_path:
            op_module = op.build(cache=cache, block_id=f"{block_id}-{op.id}")
            op_modules.append(op_module)

        main_path_module = nn.Sequential(*op_modules)
        op_modules = []
        for op in self.residual_path:
            op_module = op.build(cache=cache, block_id=f"{block_id}-{op.id}")
            op_modules.append(op_module)

        residual_path_module = nn.Sequential(*op_modules)

        self.module = ConvResidualModule(main_path_module,
                                         residual_path_module, reduce_type)
        return self.module

    def transform(self, kv_features: dict) -> None:
        """Transform features of the block.

        Conv block can transform kernel size

        Args:
          kv_features: the dictionary of to transform features.

        Raises:
          KeyError: the error occured in updating non exist feature or not
                    not support to transform this dimension.
        """
        super().transform(kv_features)

        # update the kernel_size of conv in main_path
        if "kernel_size" in kv_features.keys():
            for op in self.main_path:
                if isinstance(op, ConvOp):
                    if op.features["kernel_size"] != 1:
                        op.update({"kernel_size": self.features['kernel_size']})
