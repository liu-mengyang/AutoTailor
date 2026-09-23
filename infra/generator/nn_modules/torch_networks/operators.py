# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import numpy as np
import torch.nn as nn
from .utils import get_padding
from ..interface import BaseOperator

'''
This file contains the torch implementation of operators
'''

#---------------------- convolution layer ----------------------#

class Conv(BaseOperator):
    # 0.kernel_size 1.stride 2.group 3.has_bias 4.in_shape 5.out_shape 6.in_channel 7.out_channel
    def get_model(self):
        kernel_size, stride, group, has_bias, in_shape, out_shape, in_channel,out_channel = self.config
        assert len(in_shape)==4
        padding = get_padding(kernel_size, stride, in_shape[-1])
        return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                         stride=stride,padding=padding, groups=group,
                         bias=has_bias)

    def get_output_shape(self):
        return self.config[5]

# class GEMM(BaseOperator):
#     def get_model(self):
#         cin = self.config['CIN']
#         cout = self.config['COUT']
#         return nn.Linear(cin, cout, bias=False)
    

class MatMul(BaseOperator):
    # in_shape, out_shape, in_channel, out_channel
    def get_model(self):
        
        class MatMulModule(nn.Module):
            def __init__(self):
                super(MatMulModule, self).__init__()
            
            def forward(self, x1, x2):
                y = x1 @ x2
                return y
            
            def get_is_two_inputs(self):
                return True
            
        return MatMulModule()

    def get_output_shape(self):
        return self.config[1]
    
    def get_is_two_inputs(self):
        return True


class DepthConv(BaseOperator):
    # 0.kernel_size 1.stride 2.group 3.has_bias 4.in_shape 5.out_shape 6.in_channel 7.out_channel
    def get_model(self):
        kernel_size, stride, group, has_bias, in_shape, out_shape, in_channel,out_channel = self.config
        assert len(in_shape)==4
        padding = get_padding(kernel_size, stride, in_shape[-1])
        assert in_channel==out_channel
        return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=group,
                         bias=has_bias)

    def get_output_shape(self):
        return self.config[5]


# class ConvTrans(BaseOperator):
#     def get_model(self):
#         cin = self.input_shape[0]
#         cout = cin if "COUT" not in self.config else self.config["COUT"]
#         padding = get_padding(self.config["KERNEL_SIZE"], self.config["STRIDES"], self.input_shape[1])
#         output_padding = self.config["STRIDES"] + 2 * padding - self.config["KERNEL_SIZE"]
#         return nn.ConvTranspose2d(cin, cout, kernel_size=self.config["KERNEL_SIZE"], stride=self.config["STRIDES"], padding=padding, output_padding=output_padding)

#     def get_output_shape(self):
#         cout = self.input_shape[0] if "COUT" not in self.config else self.config["COUT"]
#         return [cout, self.input_shape[1] * self.config["STRIDES"], self.input_shape[2] * self.config["STRIDES"]]

# #------------------ normalization and pooling ------------------#

class BN(BaseOperator):
    # 0.in_shape 1.out_shape 2.in_channel 3.out_channel
    def get_model(self):
        cin = self.config[-2]
        if len(self.config[0]) == 3 or len(self.config[0]) == 2:
            return nn.BatchNorm1d(cin)
        
        return nn.BatchNorm2d(cin)

    def get_output_shape(self):
        return self.config[1]


class LN(BaseOperator):
    # 0.dims 1.in_shape 2.out_shape 3.in_channel 4.out_channel
    def get_model(self):
        dims = self.config[0]
        return nn.LayerNorm(dims)

    def get_output_shape(self):
        return self.config[2]
    

class GlobalAvgpool(BaseOperator):
    # 0.in_shape 1.out_shape 2.in_channel 3.out_channel
    def get_model(self):
        return nn.AdaptiveAvgPool2d([1, 1])
    
    def get_output_shape(self):
        return self.config[1]


class Pad(BaseOperator):
    def get_model(self):
        
        class PadModule(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return nn.functional.pad(x, (0,0,0,0,0,0,0,0))
            
        return PadModule()

    def get_output_shape(self):
        return self.config[1]


class Mean(BaseOperator):
    # 0.in_shape 1.out_shape 3.in_channel 4.out_channel
    def get_model(self):
        print(self.config)
        in_shape = self.config[0]
        out_shape = self.config[1]
        dims = []
        
        
        if len(in_shape) == len(out_shape):
            for dim in range(len(in_shape)):
                if out_shape[dim] != in_shape[dim]:
                    dims.append(dim)
        else:
            for dim in range(len(in_shape)):
                if in_shape[dim] not in out_shape:
                    dims.append(dim)
            if len(dims) == 0:
                dims.append(len(dims)-1)
        print(len(dims))
        
        assert len(dims) == 1
        
        keepdims = len(in_shape) == len(out_shape)
        
        class MeanModule(nn.Module):
            def __init__(self, dims, keepdims):
                super().__init__()
                self.dims = dims
                self.keepdims = keepdims
            
            def forward(self, x):
                return torch.mean(x, self.dims, self.keepdims)
        return MeanModule(dims[0], keepdims)

    def get_output_shape(self):
        return self.config[1]


class MaxPool(BaseOperator):
    # kernel_size, stride, in_shape, out_shape, in_channel, out_channel
    def get_model(self):
        ks = self.config[0]
        stride = self.config[1]
        hw = self.config[2][-1]
        padding = get_padding(ks, stride, hw)
        return nn.MaxPool2d(ks, stride, padding=padding)

    def get_output_shape(self):
        return self.config[3]


class AvgPool(BaseOperator):
    # kernel_size, stride, in_shape, out_shape, in_channel, out_channel
    def get_model(self):
        ks = self.config[0]
        stride = self.config[1]
        hw = self.config[2][-1]
        padding = get_padding(ks, stride, hw)
        return nn.AvgPool2d(ks, stride, padding=padding)

    def get_output_shape(self):
        return self.config[3]

# #------------------------ other modules ------------------------#

# class SE(BaseOperator):
#     def get_model(self):
#         class SE(nn.Module):
#             def __init__(self, num_channels, se_ratio=0.25):
#                 super().__init__()
#                 mid_channels = int(num_channels * se_ratio)
#                 self.squeeze = nn.Conv2d(num_channels, mid_channels, kernel_size=1, padding=0)
#                 self.relu = nn.ReLU()
#                 self.excite = nn.Conv2d(mid_channels, num_channels, kernel_size=1, padding=0)
#                 self.hswish = nn.Hardswish()

#             def _scale(self, x):
#                 x = x.mean(3, keepdim=True).mean(2, keepdim=True)
#                 x = self.squeeze(x)
#                 x = self.relu(x)
#                 x = self.excite(x)
#                 x = self.hswish(x)
#                 return x

#             def forward(self, x):
#                 scale = self._scale(x)
#                 return scale * x
#         return SE(self.input_shape[0])


# class SE_swish(BaseOperator):
#     def get_model(self):
#         class SE_swish(nn.Module):
#             def __init__(self, num_channels, se_ratio=0.25):
#                 super().__init__()
#                 mid_channels = int(num_channels * se_ratio)
#                 self.squeeze = nn.Conv2d(num_channels, mid_channels, kernel_size=1, padding=0)
#                 self.relu = nn.ReLU()
#                 self.excite = nn.Conv2d(mid_channels, num_channels, kernel_size=1, padding=0)
#                 self.swish = nn.SiLU()

#             def _scale(self, x):
#                 x = x.mean(3, keepdim=True).mean(2, keepdim=True)
#                 x = self.squeeze(x)
#                 x = self.relu(x)
#                 x = self.excite(x)
#                 x = self.swish(x)
#                 return x

#             def forward(self, x):
#                 scale = self._scale(x)
#                 return scale * x
#         return SE_swish(self.input_shape[0])


class FC(BaseOperator):
    # 0.has_bias 1.in_shape 2.out_shape 3.in_channel 4.out_channel
    def get_model(self):
        has_bias = self.config[0]
        cin = self.config[3]
        cout = self.config[4]
        return nn.Linear(cin, cout, bias=has_bias)
        
    def get_output_shape(self):
        return self.config[2]

# #-------------------- activation function --------------------#

class Clip(BaseOperator):
    def get_model(self):
        class ClipModule(nn.Module):
            def forward(self, x):
                x = torch.clamp(x, 0, 6)
                return x
        return ClipModule()

    def get_output_shape(self):
        return self.config[1]


class Relu(BaseOperator):
    # 0.in_shape 1.out_shape 3.in_channel 4.out_channel
    def get_model(self):
        return nn.ReLU()

    def get_output_shape(self):
        return self.config[1]

class Gelu(BaseOperator):
    # 0.in_shape 1.out_shape 3.in_channel 4.out_channel
    def get_model(self):
        return nn.GELU()

    def get_output_shape(self):
        return self.config[1]

# class Relu6(BaseOperator):
#     def get_model(self):
#         return nn.ReLU6()


# class Sigmoid(BaseOperator):
#     def get_model(self):
#         return nn.Sigmoid()

class Reshape(BaseOperator):
    def get_model(self, config):
        shape = config[0]
        class ReshapeModule(nn.Module):
            def __init__(self, shape):
                super(ReshapeModule, self).__init__()
                self.shape = shape
            
            def forward(self, x):
                return torch.reshape(x, self.shape)

        return ReshapeModule(shape)
    
    def get_output_shape(self):
        return self.config[2]


class Transpose(BaseOperator):
    def get_model(self, config):
        perm = config[0]
        class PermuteModule(nn.Module):
            def __init__(self, perm):
                super(PermuteModule, self).__init__()
                self.perm = perm
            
            def forward(self, x):
                return torch.permute(x, self.perm)

        return PermuteModule(perm)
    
    def get_output_shape(self):
        return self.config[2]


class Flatten(BaseOperator):
    def get_model(self):
        class FlattenModule(nn.Module):
            def __init__(self):
                super(FlattenModule, self).__init__()
            
            def forward(self, x):
                return torch.flatten(x)

        return FlattenModule()
    
    def get_output_shape(self):
        return self.config[2]


class Softmax(BaseOperator):
    def get_model(self, config):
        axis = config[0]
        return nn.Softmax(dim=axis)

    def get_output_shape(self):
        return self.config[2]


class Hswish(BaseOperator):
    # 0.in_shape 1.out_shape 3.in_channel 4.out_channel
    def get_model(self):
        return nn.Hardswish()
    
    def get_output_shape(self):
        return self.config[1]


class Hsigmoid(BaseOperator):
    # 0.in_shape 1.out_shape 3.in_channel 4.out_channel
    def get_model(self):
        return nn.Hardsigmoid()
    
    def get_output_shape(self):
        return self.config[1]


# class Swish(BaseOperator):
#     def get_model(self):
#         class Swish(nn.Module):
#             def forward(self, x):
#                 return x * torch.sigmoid(x)
#         return Swish()


# #---------------------- basic operation ----------------------#

# class Reshape(BaseOperator):
#     def get_model(self):
#         if len(self.input_shape) == 3:
#             self.output_shape = [self.input_shape[1], self.input_shape[2], self.input_shape[0]]
#             def func(inputs):
#                 return torch.reshape(inputs, [1] + self.output_shape)
#         else:
#             self.output_shape = [1, 2, int(self.input_shape[0] / 2)]
#             def func(inputs):
#                 return torch.reshape(inputs, [1] + self.output_shape)
#         return func

#     def get_output_shape(self):
#         return self.output_shape


class Add(BaseOperator):
    # 0.in_shape 1.out_shape 3.in_channel 4.out_channel
    def get_model(self):
        in_shape = self.config[0]
        assert len(in_shape)==4 or len(in_shape)==3
        class AddModule(nn.Module):
            def forward(self, first_input):
                return torch.add(first_input, first_input)
        return AddModule()

    def get_output_shape(self):
        return self.config[1]


class Mul(BaseOperator):
    # 0.in_shape 1.out_shape 3.in_channel 4.out_channel
    def get_model(self):
        class MulModule(nn.Module):
            def forward(self, inputs0, inputs1):
                return torch.mul(inputs0, inputs1)

            def get_is_two_inputs(self):
                return True

        return MulModule()

    def get_output_shape(self):
        return self.config[1]

    def get_is_two_inputs(self):
        return True


class ScaleMul(BaseOperator):
    # 0.in_shape 1.out_shape 3.in_channel 4.out_channel
    def get_model(self):
        class MulModule(nn.Module):
            def forward(self, x):
                return torch.mul(x, x)

        return MulModule()

    def get_output_shape(self):
        return self.config[1]


# class Concat(BaseOperator):
#     def get_model(self):
#         def func(first_input, second_input):
#             return torch.cat((first_input, second_input), dim=1)
#         return func

#     def get_output_shape(self):
#         if len(self.input_shape) > 1 and type(self.input_shape[0]) == list: # e.g. [[3, 28, 28], [5, 28, 28]] -> [8, 28, 28]
#             output_shape = [sum([i[0] for i in self.input_shape])] + self.input_shape[0][1:]
#         elif len(self.input_shape) == 3: # e.g. [4, 28, 28] -> [8, 28, 28]
#             output_shape = [self.input_shape[0] * 2] + self.input_shape[1:]
#         else: # e.g. [1024] -> [2048]
#             output_shape = [self.input_shape[0] * 2]
#         return output_shape

#     def get_is_two_inputs(self):
#         return True


# class Flatten(BaseOperator):
#     def get_model(self):
#         return nn.Flatten()

#     def get_output_shape(self):
#         return [int(np.prod(self.input_shape))]


# class Split(BaseOperator):
#     def get_model(self):
#         cin = self.input_shape[0]
#         def func(inputs):
#             return torch.split(inputs, [cin // 2, cin - cin // 2], dim=1)
#         return func

#     def get_output_shape(self):
#         return [self.input_shape[0] // 2, self.input_shape[1], self.input_shape[2]]
