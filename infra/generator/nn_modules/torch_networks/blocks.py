# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import re
import torch
import torch.nn as nn
from .operators import *
from .utils import get_inputs_by_shapes
from ..interface import BaseBlock
logging = logging.getLogger("autotailor")


class TorchBlock(BaseBlock):
    def __init__(self, config):
        self.config = config
        # if 'K' in config:
        #     self.input_shape = [1,config['M'],config['N']]
        # elif 'KERNEL_SIZE' not in config:   
        #     #gemm
        #     self.input_shape = [1,config['HW'],config['CIN']]
        # elif 'CIN' in config:
        #     self.input_shape = [config["CIN"], config["HW"], config["HW"]]

        # else:
        #     self.input_shape = [config["CHANNEL_SIZE"], config["HW"], config["HW"]]
        # self.input_tensor_shape = [self.input_shape]

    def save_model(self, save_path):
        logging.info("Here is torchblock")
        model = self.get_model()
        model.eval()
        torch.onnx.export(
            model,
            self.input_shape,
            save_path,
            input_names=['input'],
            output_names=['output'],
            verbose=False,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
        )

    def build_model(self, ops):
        ''' convert a list of operators to torch model.
        '''
        
        class Model(nn.Module):
            def __init__(self, ops):
                super().__init__()
                self.layers = nn.Sequential(*ops)

            def forward(self, inputs):
                x = self.layers(inputs)
                return x
        
        class DualModel(nn.Module):
            def __init__(self, ops):
                super().__init__()
                self.layers = nn.Sequential(*ops)

            def forward(self, inputs0, inputs1):
                # x = self.layers(inputs0, inputs1)
                x = self.layers[0](inputs0, inputs1)
                for op in self.layers[1:]:
                    x = op(x)
                return x
        if hasattr(ops[0], "get_is_two_inputs") and ops[0].get_is_two_inputs():
            return DualModel(ops)
        else:
            return Model(ops)

    def get_model(self):
        raise NotImplementedError


# class ConvBnRelu(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         conv_op = Conv(self.input_shape, config)
#         self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

#         bn_op = BN(out_shape, config)
#         self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
#         relu_op = Relu(out_shape, config)
#         self.relu_op = relu_op.get_model()

#     def get_model(self):
#         return self.build_model([self.conv_op, self.bn_op, self.relu_op])



# class ConvSwish(TorchBlock):
#     def __init__(self, config, batch_size):
#         super().__init__(config, batch_size)
#         conv_op = Conv(self.input_shape, config)
#         self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()
        
#         bn_op = BN(out_shape, config)
#         self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
#         swish_op = Swish(out_shape, config)
#         self.swish_op = swish_op.get_model()

#     def get_model(self):
#         return self.build_model([self.conv_op, self.bn_op, self.swish_op])


# class DWConvSwish(nn.Module):
#     def __init__(self, config, batch_size):
#         super().__init__(config, batch_size)
#         dwconv_op = DwConv(self.input_shape, config)
#         self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()
        
#         bn_op = BN(out_shape, config)
#         self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
#         swish_op = Swish(out_shape, config)
#         self.swish_op = swish_op.get_model()

#     def get_model(self):
#         return self.build_model([self.dwconv_op, self.bn_op, self.swish_op])


# class ConvBnRelu6(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         conv_op = Conv(self.input_shape, config)
#         self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

#         bn_op = BN(out_shape, config)
#         self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
#         relu6_op = Relu6(out_shape, config)
#         self.relu6_op = relu6_op.get_model()

#     def get_model(self):
#         return self.build_model([self.conv_op, self.bn_op, self.relu6_op])


# class DWConvBnRelu6(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         dwconv_op = DwConv(self.input_shape, config)
#         self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

#         bn_op = BN(out_shape, config)
#         self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
#         relu6_op = Relu6(out_shape, config)
#         self.relu6_op = relu6_op.get_model()

#     def get_model(self):
#         return self.build_model([self.dwconv_op, self.bn_op, self.relu6_op])


# class ConvBn(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         conv_op = Conv(self.input_shape, config)
#         self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

#         bn_op = BN(out_shape, config)
#         self.bn_op = bn_op.get_model()

#     def get_model(self):
#         return self.build_model([self.conv_op, self.bn_op])


# class ConvBn(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         dwconv_op = DwConv(self.input_shape, config)
#         self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

#         bn_op = BN(out_shape, config)
#         self.bn_op = bn_op.get_model()

#     def get_model(self):
#         return self.build_model([self.dwconv_op, self.bn_op])


# class ConvRelu(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         conv_op = Conv(self.input_shape, config)
#         self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()
        
#         relu_op = Relu(out_shape, config)
#         self.relu_op = relu_op.get_model()

#     def get_model(self):
#         return self.build_model([self.conv_op, self.relu_op])


# class ConvRelu6(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         conv_op = Conv(self.input_shape, config)
#         self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()
        
#         relu6_op = Relu6(out_shape, config)
#         self.relu6_op = relu6_op.get_model()

#     def get_model(self):
#         return self.build_model([self.conv_op, self.relu6_op])


# class ConvHswish(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         conv_op = Conv(self.input_shape, config)
#         self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()
        
#         hswish_op = Hswish(out_shape, config)
#         self.hswish_op = hswish_op.get_model()

#     def get_model(self):
#         return self.build_model([self.conv_op, self.hswish_op])


class ConvBlock(TorchBlock):
    # 0.kernel_size 1.stride 2.group 3.has_bias 4.in_shape 5.out_shape 6.in_channel 7.out_channel
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config[4]
        conv_op = Conv(config)
        self.conv_op = conv_op.get_model()

    def get_model(self):
        return self.build_model([self.conv_op])

# class GEMMBlock(TorchBlock):
#     def __init__(self, config, batch_size=1):
#         super().__init__(config, batch_size)

#         gemm_op = GEMM(self.input_shape, config)
#         self.gemm_op = gemm_op.get_model()
    
#     def get_model(self):
#         return self.build_model([self.gemm_op])


class MatMulBlock(TorchBlock):
    # in_shape, out_shape, in_channel, out_channel
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config[0]
        gemm_op = MatMul(config)
        self.gemm_op = gemm_op.get_model()
    
    def get_model(self):
        return self.build_model([self.gemm_op])
    

# class ConvBnHswish(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         conv_op = Conv(self.input_shape, config)
#         self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

#         bn_op = BN(out_shape, config)
#         self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
#         hswish_op = Hswish(out_shape, config)
#         self.hswish_op = hswish_op.get_model()

#     def get_model(self):
#         return self.build_model([self.conv_op, self.bn_op, self.hswish_op])


# class ConvBnReluMaxPool(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         conv_op = Conv(self.input_shape, config)
#         self.conv_op, out_shape = conv_op.get_model(), conv_op.get_output_shape()

#         bn_op = BN(out_shape, config)
#         self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
#         relu_op = Relu(out_shape, config)
#         self.relu_op, out_shape = relu_op.get_model(), relu_op.get_output_shape()
        
#         maxpool_op = MaxPool(out_shape, config)
#         self.maxpool_op = maxpool_op.get_model()

#     def get_model(self):
#         return self.build_model([self.conv_op, self.bn_op, self.relu_op, self.maxpool_op])


# class DwConvBn(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         dwconv_op = DwConv(self.input_shape, config)
#         self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

#         bn_op = BN(out_shape, config)
#         self.bn_op = bn_op.get_model()

#     def get_model(self):
#         return self.build_model([self.dwconv_op, self.bn_op])


# class DwConvRelu(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         dwconv_op = DwConv(self.input_shape, config)
#         self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()
        
#         relu_op = Relu(out_shape, config)
#         self.relu_op = relu_op.get_model()

#     def get_model(self):
#         return self.build_model([self.dwconv_op, self.relu_op])


# class DwConvRelu6(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         dwconv_op = DwConv(self.input_shape, config)
#         self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()
        
#         relu6_op = Relu6(out_shape, config)
#         self.relu6_op = relu6_op.get_model()

#     def get_model(self):
#         return self.build_model([self.dwconv_op, self.relu6_op])


# class DwConvBnRelu(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         dwconv_op = DwConv(self.input_shape, config)
#         self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

#         bn_op = BN(out_shape, config)
#         self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
#         relu_op = Relu(out_shape, config)
#         self.relu_op = relu_op.get_model()

#     def get_model(self):
#         return self.build_model([self.dwconv_op, self.bn_op, self.relu_op])


# class DwConvBnRelu6(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         dwconv_op = DwConv(self.input_shape, config)
#         self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

#         bn_op = BN(out_shape, config)
#         self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
#         relu6_op = Relu6(out_shape, config)
#         self.relu6_op = relu6_op.get_model()

#     def get_model(self):
#         return self.build_model([self.dwconv_op, self.bn_op, self.relu6_op])


class DepthConvBlock(TorchBlock):
    # 0.kernel_size 1.stride 2.group 3.has_bias 4.in_shape 5.out_shape 6.in_channel 7.out_channel
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config[4]
        depthconv_op = DepthConv(config)
        self.depthconv_op = depthconv_op.get_model()

    def get_model(self):
        return self.build_model([self.depthconv_op])


# class DwConvBnHswish(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         dwconv_op = DwConv(self.input_shape, config)
#         self.dwconv_op, out_shape = dwconv_op.get_model(), dwconv_op.get_output_shape()

#         bn_op = BN(out_shape, config)
#         self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
#         hswish_op = Hswish(out_shape, config)
#         self.hswish_op = hswish_op.get_model()

#     def get_model(self):
#         return self.build_model([self.dwconv_op, self.bn_op, self.hswish_op])


class MaxPoolBlock(TorchBlock):
    # kernel_size, stride, in_shape, out_shape, in_channel, out_channel
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_shape=config[2]
        maxpool_op = MaxPool(config)
        self.maxpool_op = maxpool_op.get_model()

    def get_model(self):
        return self.build_model([self.maxpool_op])


class AvgPoolBlock(TorchBlock):
    # kernel_size, stride, in_shape, out_shape, in_channel, out_channel
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_shape=config[2]
        avgpool_op = AvgPool(config)
        self.avgpool_op = avgpool_op.get_model()

    def get_model(self):
        return self.build_model([self.avgpool_op])


class GlobalAvgPoolBlock(TorchBlock):
    # 0.in_shape 1.out_shape 2.in_channel 3.out_channel
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_shape=config[0]
        globalavgpool_op = GlobalAvgpool(config)
        self.globalavgpool_op = globalavgpool_op.get_model()

    def get_model(self):
        return self.build_model([self.globalavgpool_op])


class FCBlock(TorchBlock):
    # 0.has_bias 1.in_shape 2.out_shape 3.in_channel 4.out_channel
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_shape=config[1]
        linear_op = FC(config)
        self.linear_op = linear_op.get_model()

    def get_model(self):
        return self.build_model([self.linear_op])


# class ConcatBlock(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         self.config = config
#         self.input_shape = [[cin, config["HW"], config["HW"]]
#                        for cin in [config['CIN1'], config['CIN2'], config['CIN3'], config['CIN4']]
#                        if cin != 0]
#         self.input_tensor_shape = self.input_shape
#         self.batch_size = batch_size
        
#         concat_op = Concat(self.input_shape, config)
#         self.concat_op = concat_op.get_model()

#     def get_model(self):
#         class Model(nn.Module):
#             def __init__(self, concat_op):
#                 super().__init__()
#                 self.concat = concat_op

#             def forward(self, first_input, second_input):
#                 return self.concat(first_input, second_input)

#         return Model(self.concat_op)


# class SplitBlock(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         split_op = Split(self.input_shape, config)
#         self.split_op = split_op.get_model()

#     def get_model(self):
#         class Model(nn.Module):
#             def __init__(self, split_op):
#                 super().__init__()
#                 self.split = split_op

#             def forward(self, inputs):
#                 return self.split(inputs)

#         return Model(self.split_op)


# class ChannelShuffle(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#     def get_model(self):
#         class Model(nn.Module):
#             def __init__(self):
#                 super().__init__()

#             def forward(self, inputs):
#                 _, c, h, w = list(inputs.shape)
#                 x = torch.reshape(inputs, [-1, c // 2, 2, h, w])
#                 x = torch.transpose(x, 2, 1)
#                 x = torch.reshape(x, [-1, c, h, w])
#                 return x

#         return Model()


# class SEBlock(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         se_op = SE(self.input_shape, config)
#         self.se_op = se_op.get_model()

#     def get_model(self):
#         return self.build_model([self.se_op])


# class SESwishBlock(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         se_op = SE_swish(self.input_shape, config)
#         self.se_op = se_op.get_model()

#     def get_model(self):
#         return self.build_model([self.se_op])


class MeanBlock(TorchBlock):
    # 0.in_shape 1.out_shape 2.in_channel 3.out_channel
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config[0]
        mean_op = Mean(config)
        self.mean_op = mean_op.get_model()

    def get_model(self):
        return self.build_model([self.mean_op])


# class BnRelu(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         bn_op = BN(self.input_shape, config)
#         self.bn_op, out_shape = bn_op.get_model(), bn_op.get_output_shape()
        
#         relu_op = Relu(out_shape, config)
#         self.relu_op = relu_op.get_model()

#     def get_model(self):
#         return self.build_model([self.bn_op, self.relu_op])


class BNBlock(TorchBlock):
    # 0.in_shape 1.out_shape 2.in_channel 3.out_channel
    def __init__(self, config):
        super().__init__(config)
        assert config[0] == config[1]
        self.input_shape = config[0]
        bn_op = BN(config)
        self.bn_op = bn_op.get_model()

    def get_model(self):
        return self.build_model([self.bn_op])


class LNBlock(TorchBlock):
    # dims, in_shape, out_shape, in_channel, out_channel
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config[1]
        ln_op = LN(config)
        self.ln_op = ln_op.get_model()

    def get_model(self):
        return self.build_model([self.ln_op])


class HswishBlock(TorchBlock):
    # 0.in_shape 1.out_shape 2.in_channel 3.out_channel
    def __init__(self, config):
        super().__init__(config)
        assert config[0] == config[1]
        self.input_shape = config[0]
        hswish_op = Hswish(config)
        self.hswish_op = hswish_op.get_model()

    def get_model(self):
        return self.build_model([self.hswish_op])


class HsigmoidBlock(TorchBlock):
    # 0.in_shape 1.out_shape 2.in_channel 3.out_channel
    def __init__(self, config):
        super().__init__(config)
        assert config[0] == config[1]
        self.input_shape = config[0]
        hsigmoid_op = Hsigmoid(config)
        self.hsigmoid_op = hsigmoid_op.get_model()

    def get_model(self):
        return self.build_model([self.hsigmoid_op])

# class SwishBlock(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         swish_op = Swish(self.input_shape, config)
#         self.swish_op = swish_op.get_model()

#     def get_model(self):
#         return self.build_model([self.swish_op])


class ReluBlock(TorchBlock):
    # 0.in_shape 1.out_shape 2.in_channel 3.out_channel
    def __init__(self, config):
        super().__init__(config)
        assert config[0] == config[1]
        self.input_shape = config[0]
        relu_op = Relu(config)
        self.relu_op = relu_op.get_model()

    def get_model(self):
        return self.build_model([self.relu_op])

class GeluBlock(TorchBlock):
    # 0.in_shape 1.out_shape 2.in_channel 3.out_channel
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config[0]
        assert config[0] == config[1]
        gelu_op = Gelu(config)
        self.gelu_op = gelu_op.get_model()

    def get_model(self):
        return self.build_model([self.gelu_op])

# class AddRelu(TorchBlock):
#     def __init__(self, config, batch_size = 1):
#         super().__init__(config, batch_size)

#         add_op = Add(self.input_shape, config)
#         self.add_op, out_shape = add_op.get_model(), add_op.get_output_shape()

#         relu_op = Relu(out_shape, config)
#         self.relu_op = relu_op.get_model()

#     def get_model(self):
#         class Model(nn.Module):
#             def __init__(self, add_op, relu_op):
#                 super().__init__()
#                 self.add = add_op
#                 self.relu = relu_op

#             def forward(self, first_input, second_input):
#                 x = self.add(first_input, second_input)
#                 x = self.relu(x)
#                 return x

#         return Model(self.add_op, self.relu_op)

class ReshapeBlock(TorchBlock):
    # perm, in_shape, out_shape, in_channel, out_channel
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config[1]
        reshape_op = Reshape(config)
        self.reshape_op = reshape_op.get_model(config)

    def get_model(self):
        return self.build_model([self.reshape_op])


class TransposeBlock(TorchBlock):
    # perm, in_shape, out_shape, in_channel, out_channel
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config[1]
        transpose_op = Transpose(config)
        self.transpose_op = transpose_op.get_model(config)

    def get_model(self):
        return self.build_model([self.transpose_op])


class FlattenBlock(TorchBlock):
    # in_shape, out_shape, in_channel, out_channel
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config[0]
        flatten_op = Flatten(config)
        self.flatten_op = flatten_op.get_model()

    def get_model(self):
        return self.build_model([self.flatten_op])
    

class SoftmaxBlock(TorchBlock):
    # 0.axis 1.in_shape 2.out_shape 3.in_channel 4.out_channel
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config[1]
        softmax_op = Softmax(config)
        self.softmax_op = softmax_op.get_model(config)

    def get_model(self):
        return self.build_model([self.softmax_op])


class AddBlock(TorchBlock):
    # 0.in_shape 1.out_shape 3.in_channel 4.out_channel
    def __init__(self, config):
        super().__init__(config)
        assert config[0] == config[1]
        self.input_shape = config[0]
        add_op = Add(config)
        self.add_op = add_op.get_model()

    def get_model(self):
        return self.build_model([self.add_op])


class MulBlock(TorchBlock):
    # 0.in_shape 1.out_shape 3.in_channel 4.out_channel
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config[0]
        assert len(self.input_shape) == 2
        mul_op = Mul(config)
        self.mul_op = mul_op.get_model()

    def get_model(self):
        return self.build_model([self.mul_op])
    
    def get_is_two_inputs(self):
        return True
    

class ScaleMulBlock(TorchBlock):
    # 0.in_shape 1.out_shape 3.in_channel 4.out_channel
    def __init__(self, config):
        super().__init__(config)
        self.input_shape = config[0]
        mul_op = ScaleMul(config)
        self.mul_op = mul_op.get_model()

    def get_model(self):
        return self.build_model([self.mul_op])
    
    
class ClipBlock(TorchBlock):
    # 0.in_shape 1.out_shape 2.in_channel 3.out_channel
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_shape=config[0]
        clip_op = Clip(config)
        self.clip_op = clip_op.get_model()

    def get_model(self):
        return self.build_model([self.clip_op])

class PadBlock(TorchBlock):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_shape = config[0]
        pad_op = Pad(config)
        self.pad_op = pad_op.get_model()

    def get_model(self):
        return self.build_model([self.pad_op])